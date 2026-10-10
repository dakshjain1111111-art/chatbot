import json, threading, time, unittest, urllib.error, urllib.request
from pathlib import Path
from app.server import ChatService, MockProvider, issue_token, make_server

SECRET = "test-secret"


class T(unittest.TestCase):
    @classmethod
    def setUpClass(c):
        c.provider = MockProvider()
        c.svc = ChatService(c.provider, rate_limit=10**6, backoff=0.01)
        c.srv = make_server(c.svc, SECRET)
        c.port = c.srv.server_address[1]
        threading.Thread(target=c.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(c): c.srv.shutdown(); c.srv.server_close()

    def setUp(self):
        self.provider.fail_next, self.provider.always_fail = 0, False
        self.svc.limiter.limit = 10**6; self.svc.limiter.hits.clear()

    def call(self, method, path, body=None, user="alice", tenant="acme", token=None, raw=False):
        tok = issue_token(SECRET, user, tenant) if token is None else token
        h = {"Content-Type": "application/json"}
        if tok: h["Authorization"] = "Bearer " + tok
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}",
                                     None if body is None else json.dumps(body).encode(), h, method=method)
        try:
            with urllib.request.urlopen(req) as r:
                d = r.read(); return r.status, d.decode() if raw else json.loads(d)
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())


    # A. functional
    def test_multi_turn_and_history(self):
        s, a = self.call("POST", "/api/chat", {"message": "hi"})
        self.assertEqual(s, 200); self.assertIn("You said: hi", a["reply"])
        s, b = self.call("POST", "/api/chat", {"message": "again", "conversation_id": a["conversation_id"]})
        self.assertIn("context: 3 messages", b["reply"])
        s, conv = self.call("GET", "/api/conversations/" + a["conversation_id"])
        self.assertEqual([m["role"] for m in conv["messages"]], ["user", "assistant", "user", "assistant"])

    def test_streaming(self):
        s, text = self.call("POST", "/api/chat", {"message": "stream me", "stream": True}, raw=True)
        self.assertEqual(s, 200)
        for k in ("event: meta", "event: delta", "event: done"): self.assertIn(k, text)

    def test_empty_and_long_input(self):
        self.assertEqual(self.call("POST", "/api/chat", {"message": "   "})[0], 400)
        self.assertEqual(self.call("POST", "/api/chat", {"message": "x" * 4001})[0], 400)
        self.assertEqual(self.call("POST", "/api/chat", {"message": 5})[0], 400)

    # C. security
    def test_auth_required(self):
        self.assertEqual(self.call("POST", "/api/chat", {"message": "x"}, token="")[0], 401)
        self.assertEqual(self.call("GET", "/api/conversations", token="")[0], 401)
        self.assertEqual(self.call("GET", "/api/conversations", token="forged.token")[0], 401)
        self.assertEqual(self.call("GET", "/api/conversations", token=issue_token("other", "a", "b"))[0], 401)
        self.assertEqual(self.call("GET", "/api/conversations", token=issue_token(SECRET, "a", "b", ttl=-5))[0], 401)

    def test_tenant_and_user_isolation(self):
        cid = self.call("POST", "/api/chat", {"message": "secret plan"})[1]["conversation_id"]
        for who in ({"user": "bob", "tenant": "acme"}, {"user": "alice", "tenant": "other"}):
            self.assertEqual(self.call("GET", "/api/conversations/" + cid, **who)[0], 404)
            self.assertEqual(self.call("POST", "/api/chat", {"message": "x", "conversation_id": cid}, **who)[0], 404)
            self.assertNotIn(cid, json.dumps(self.call("GET", "/api/conversations", **who)[1]))

    def test_rate_limit(self):
        self.svc.limiter.limit = 2
        codes = [self.call("POST", "/api/chat", {"message": "x"}, user="rl")[0] for _ in range(4)]
        self.assertEqual(codes, [200, 200, 429, 429])

    def test_no_secrets_in_frontend(self):
        html = (Path(__file__).parent.parent / "static" / "index.html").read_text().lower()
        for bad in ("x-api-key", "anthropic", "sk-ant"): self.assertNotIn(bad, html)

    # D. reliability
    def test_retry_then_success(self):
        self.provider.fail_next = 2
        self.assertEqual(self.call("POST", "/api/chat", {"message": "x"})[0], 200)

    def test_provider_down_safe_error_and_no_duplicates_on_retry(self):
        cid = self.call("POST", "/api/chat", {"message": "one"})[1]["conversation_id"]
        self.provider.always_fail = True
        s, err = self.call("POST", "/api/chat", {"message": "two", "conversation_id": cid, "idempotency_key": "k1"})
        self.assertEqual(s, 502); self.assertNotIn("provider down", json.dumps(err)); self.assertIn("request_id", err["error"])
        self.provider.always_fail = False
        s, ok = self.call("POST", "/api/chat", {"message": "two", "conversation_id": cid, "idempotency_key": "k1"})
        self.assertEqual(s, 200)
        msgs = self.call("GET", "/api/conversations/" + cid)[1]["messages"]
        self.assertEqual([m["content"] for m in msgs if m["role"] == "user"], ["one", "two"])
        self.assertEqual(len(msgs), 4)
        s, again = self.call("POST", "/api/chat", {"message": "two", "conversation_id": cid, "idempotency_key": "k1"})
        self.assertEqual(again["reply"], ok["reply"])
        self.assertEqual(len(self.call("GET", "/api/conversations/" + cid)[1]["messages"]), 4)

    def test_200_requests_success_rate_and_overhead(self):
        ok, times = 0, []
        for i in range(200):
            if i % 10 == 0: self.provider.fail_next = 1  # transient provider errors, absorbed by retry
            t = time.perf_counter()
            ok += self.call("POST", "/api/chat", {"message": f"q{i}"})[0] == 200
            times.append(time.perf_counter() - t)
        times.sort()
        p95 = times[int(len(times) * .95) - 1]
        print(f"\n  success={ok}/200  p95 API overhead={p95*1000:.1f} ms")
        self.assertGreaterEqual(ok / 200, 0.99)
        self.assertLess(p95, 0.3)


if __name__ == "__main__":
    unittest.main()
