"""AI chatbot backend (stdlib only): auth, tenant isolation, validation, rate limiting,
LLM retries/timeouts, SSE streaming, SQLite persistence, idempotent retries."""
import base64, hashlib, hmac, itertools, json, logging, os, sqlite3, threading, time, uuid
import urllib.error, urllib.request
from collections import defaultdict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

log = logging.getLogger("chat")
MAX_MSG, MAX_BODY, HISTORY_LIMIT = 4000, 16384, 20
SYSTEM_PROMPT = os.environ.get(
    "SYSTEM_PROMPT",
    "You are the in-app assistant. If you do not know a business-specific fact, say so instead of guessing.")


class ApiError(Exception):
    def __init__(self, status, code, msg):
        super().__init__(msg); self.status, self.code, self.msg = status, code, msg


class ProviderError(Exception):
    def __init__(self, msg, retryable=True):
        super().__init__(msg); self.retryable = retryable


# ---------- auth (HMAC-signed bearer token; swap for the platform's existing auth) ----------
def _b64(b): return base64.urlsafe_b64encode(b).decode().rstrip("=")
def _unb64(s): return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))
def _sig(secret, p): return _b64(hmac.new(secret.encode(), p.encode(), hashlib.sha256).digest())

def issue_token(secret, user, tenant, ttl=3600):
    p = _b64(json.dumps({"u": user, "t": tenant, "exp": int(time.time()) + ttl}).encode())
    return f"{p}.{_sig(secret, p)}"

def verify_token(secret, token):
    try:
        p, s = token.split(".")
        if not hmac.compare_digest(s, _sig(secret, p)): raise ValueError
        d = json.loads(_unb64(p))
        if d["exp"] < time.time(): raise ValueError
        return d["u"], d["t"]
    except Exception:
        raise ApiError(401, "unauthorized", "Invalid or missing credentials.")


# ---------- LLM providers ----------
class MockProvider:
    """Deterministic provider for tests/dev, with failure injection."""
    def __init__(self): self.fail_next, self.always_fail, self.delay = 0, False, 0
    def stream(self, messages, timeout):
        if self.always_fail: raise ProviderError("provider down")
        if self.fail_next > 0:
            self.fail_next -= 1; raise ProviderError("rate limited")
        time.sleep(self.delay)
        text = f"You said: {messages[-1]['content']} (context: {len(messages)} messages)"
        for w in text.split(" "): yield w + " "


class AnthropicProvider:
    """Server-side only; key comes from env. Non-streaming call emitted as one chunk
    (swap in SSE streaming from the API for lower time-to-first-token)."""
    def __init__(self, key=None, model=None):
        self.key = key or os.environ["ANTHROPIC_API_KEY"]
        self.model = model or os.environ.get("LLM_MODEL", "claude-sonnet-5-5")
    def stream(self, messages, timeout):
        body = json.dumps({"model": self.model, "max_tokens": 1000,
                           "system": SYSTEM_PROMPT, "messages": messages}).encode()
        req = urllib.request.Request("https://api.anthropic.com/v1/messages", body, {
            "content-type": "application/json", "x-api-key": self.key, "anthropic-version": "2023-06-01"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r: data = json.load(r)
        except urllib.error.HTTPError as e:
            raise ProviderError(f"http {e.code}", retryable=e.code in (429, 500, 502, 503, 529))
        except Exception as e:
            raise ProviderError(str(e))
        yield "".join(b.get("text", "") for b in data["content"])


class GeminiProvider:
    """Google AI Studio key. Server-side only; key comes from env."""
    def __init__(self, key=None, model=None):
        self.key = key or os.environ["GEMINI_API_KEY"]
        self.model = model or os.environ.get("LLM_MODEL", "gemini-2.5-flash")
    def stream(self, messages, timeout):
        body = json.dumps({
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"role": "model" if m["role"] == "assistant" else "user",
                          "parts": [{"text": m["content"]}]} for m in messages]}).encode()
        req = urllib.request.Request(
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent", body,
            {"content-type": "application/json", "x-goog-api-key": self.key})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r: data = json.load(r)
        except urllib.error.HTTPError as e:
            raise ProviderError(f"http {e.code}", retryable=e.code in (429, 500, 502, 503))
        except Exception as e:
            raise ProviderError(str(e))
        try:
            yield "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
        except (KeyError, IndexError):
            raise ProviderError("empty response", retryable=False)


class RateLimiter:
    def __init__(self, limit, window=60):
        self.limit, self.window, self.hits, self.lock = limit, window, defaultdict(deque), threading.Lock()
    def check(self, key):
        now = time.monotonic()
        with self.lock:
            q = self.hits[key]
            while q and now - q[0] > self.window: q.popleft()
            if len(q) >= self.limit:
                raise ApiError(429, "rate_limited", "Too many requests. Please retry shortly.")
            q.append(now)


SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations(id TEXT PRIMARY KEY, tenant TEXT NOT NULL, user TEXT NOT NULL, created REAL);
CREATE TABLE IF NOT EXISTS messages(id INTEGER PRIMARY KEY AUTOINCREMENT, conv_id TEXT NOT NULL,
  role TEXT NOT NULL, content TEXT NOT NULL, idem TEXT, created REAL);
CREATE UNIQUE INDEX IF NOT EXISTS ux_idem ON messages(conv_id, idem) WHERE idem IS NOT NULL;
"""


class ChatService:
    def __init__(self, provider, db_path=":memory:", rate_limit=30, retries=2, timeout=20, backoff=0.2):
        self.provider, self.retries, self.timeout, self.backoff = provider, retries, timeout, backoff
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.executescript(SCHEMA)
        self.lock, self.limiter = threading.Lock(), RateLimiter(rate_limit)

    def _owned(self, ident, cid):
        user, tenant = ident
        if not isinstance(cid, str) or not self.db.execute(
                "SELECT 1 FROM conversations WHERE id=? AND tenant=? AND user=?", (cid, tenant, user)).fetchone():
            raise ApiError(404, "not_found", "Conversation not found.")

    def list_conversations(self, ident):
        with self.lock:
            rows = self.db.execute("SELECT id, created FROM conversations WHERE tenant=? AND user=? ORDER BY created DESC",
                                   (ident[1], ident[0])).fetchall()
        return [{"id": r[0], "created": r[1]} for r in rows]

    def get_conversation(self, ident, cid):
        with self.lock:
            self._owned(ident, cid)
            rows = self.db.execute("SELECT role, content FROM messages WHERE conv_id=? ORDER BY id", (cid,)).fetchall()
        return {"id": cid, "messages": [{"role": r, "content": c} for r, c in rows]}

    def _generate(self, history):
        for attempt in range(self.retries + 1):
            try:
                it = self.provider.stream(history, self.timeout)
                try: first = next(it)
                except StopIteration: return iter([""])
                return itertools.chain([first], it)
            except ProviderError as e:
                log.warning("provider error attempt=%d retryable=%s err=%s", attempt, e.retryable, e)
                if not e.retryable or attempt == self.retries:
                    raise ApiError(502, "provider_unavailable", "The assistant is temporarily unavailable. Please try again.")
                time.sleep(self.backoff * 2 ** attempt)

    def chat(self, ident, cid, message, idem=None):
        """Generator of ('meta'|'delta'|'done', value). Errors raise ApiError before the first yield."""
        if not isinstance(message, str) or not message.strip():
            raise ApiError(400, "invalid_input", "Message must not be empty.")
        if len(message) > MAX_MSG:
            raise ApiError(400, "invalid_input", f"Message exceeds {MAX_MSG} characters.")
        if idem is not None and (not isinstance(idem, str) or len(idem) > 64):
            raise ApiError(400, "invalid_input", "Invalid idempotency key.")
        self.limiter.check(ident[0])
        with self.lock:
            if cid is None:
                cid = uuid.uuid4().hex
                self.db.execute("INSERT INTO conversations VALUES(?,?,?,?)", (cid, ident[1], ident[0], time.time()))
            else:
                self._owned(ident, cid)
            existing = idem and self.db.execute(
                "SELECT id FROM messages WHERE conv_id=? AND idem=?", (cid, idem)).fetchone()
            if existing:  # retried request: never duplicate the user message
                r = self.db.execute("SELECT content FROM messages WHERE conv_id=? AND role='assistant' AND id>? "
                                    "ORDER BY id LIMIT 1", (cid, existing[0])).fetchone()
                if r:
                    yield "meta", cid; yield "delta", r[0]; yield "done", r[0]; return
            else:
                self.db.execute("INSERT INTO messages(conv_id,role,content,idem,created) VALUES(?,?,?,?,?)",
                                (cid, "user", message, idem, time.time()))
            self.db.commit()
            rows = self.db.execute("SELECT role, content FROM messages WHERE conv_id=? ORDER BY id DESC LIMIT ?",
                                   (cid, HISTORY_LIMIT)).fetchall()
        history = [{"role": r, "content": c} for r, c in reversed(rows)]
        stream = self._generate(history)
        yield "meta", cid
        parts = []
        try:
            for chunk in stream:
                parts.append(chunk); yield "delta", chunk
        except ProviderError as e:
            log.warning("mid-stream provider error: %s", e)
            raise ApiError(502, "provider_unavailable", "The response was interrupted. Please retry.")
        full = "".join(parts)
        with self.lock:
            self.db.execute("INSERT INTO messages(conv_id,role,content,created) VALUES(?,?,?,?)",
                            (cid, "assistant", full, time.time()))
            self.db.commit()
        yield "done", full


def make_server(svc, secret, host="127.0.0.1", port=0):
    widget = Path(__file__).parent.parent / "static" / "index.html"

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass

        def _send(self, status, body, ctype="application/json"):
            if not isinstance(body, bytes): body = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers(); self.wfile.write(body)

        def _ident(self):
            h = self.headers.get("Authorization", "")
            if not h.startswith("Bearer "): raise ApiError(401, "unauthorized", "Invalid or missing credentials.")
            return verify_token(secret, h[7:])

        def _guard(self, fn):
            rid = uuid.uuid4().hex[:8]
            try: fn()
            except ApiError as e:
                self._send(e.status, {"error": {"code": e.code, "message": e.msg, "request_id": rid}})
            except Exception:
                log.exception("unhandled rid=%s", rid)
                self._send(500, {"error": {"code": "internal", "message": "Something went wrong.", "request_id": rid}})

        def do_GET(self): self._guard(self._get)
        def do_POST(self): self._guard(self._post)

        def _get(self):
            p = self.path.split("?")[0]
            if p == "/healthz": return self._send(200, {"ok": True})
            if p == "/": return self._send(200, widget.read_bytes(), "text/html; charset=utf-8")
            ident = self._ident()
            if p == "/api/conversations": return self._send(200, svc.list_conversations(ident))
            if p.startswith("/api/conversations/"):
                return self._send(200, svc.get_conversation(ident, p.rsplit("/", 1)[1]))
            raise ApiError(404, "not_found", "Not found.")

        def _post(self):
            if self.path != "/api/chat": raise ApiError(404, "not_found", "Not found.")
            ident = self._ident()
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_BODY: raise ApiError(413, "too_large", "Request too large.")
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
                assert isinstance(body, dict)
            except Exception:
                raise ApiError(400, "invalid_input", "Body must be a JSON object.")
            events = svc.chat(ident, body.get("conversation_id"), body.get("message"), body.get("idempotency_key"))
            first = next(events)  # validation/auth/provider errors surface here, before headers
            if not body.get("stream"):
                cid, full = first[1], ""
                for kind, val in events:
                    if kind == "done": full = val
                return self._send(200, {"conversation_id": cid, "reply": full})
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream"); self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            def ev(name, data): self.wfile.write(f"event: {name}\ndata: {json.dumps(data)}\n\n".encode()); self.wfile.flush()
            ev("meta", {"conversation_id": first[1]})
            try:
                for kind, val in events:
                    ev(kind, {"text": val} if kind == "delta" else {"reply": val})
            except ApiError as e:
                ev("error", {"code": e.code, "message": e.msg})

    return ThreadingHTTPServer((host, port), H)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    provider = (GeminiProvider() if os.environ.get("GEMINI_API_KEY") else
                AnthropicProvider() if os.environ.get("ANTHROPIC_API_KEY") else MockProvider())
    svc = ChatService(provider, os.environ.get("DB_PATH", "chat.db"), int(os.environ.get("RATE_LIMIT_PER_MIN", 30)))
    srv = make_server(svc, os.environ["AUTH_SECRET"], os.environ.get("HOST", "0.0.0.0"), int(os.environ.get("PORT", 8080)))
    log.info("listening on %s (provider=%s)", srv.server_address, type(provider).__name__)
    srv.serve_forever()
