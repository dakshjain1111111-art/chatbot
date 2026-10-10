# AI Chatbot Integration – MVP reference implementation

Stdlib-only Python backend + embeddable widget. Run tests: `python3 -m unittest -v`
Run server: `AUTH_SECRET=... [ANTHROPIC_API_KEY=... LLM_MODEL=...] python3 -m app.server`
(without a key it uses the mock provider). Widget is served at `/`; the host app sets `window.CHAT_TOKEN`.

## API
- `POST /api/chat` `{message, conversation_id?, idempotency_key?, stream?}` -> JSON `{conversation_id, reply}` or SSE (`meta`/`delta`/`done`/`error`)
- `GET /api/conversations`, `GET /api/conversations/{id}`, `GET /healthz`

## Acceptance criteria status
| Area | Criterion | Status |
|---|---|---|
| Functional | Multi-turn context, history persistence, empty/long input, error + retry UI | Automated tests pass; UI states implemented in widget (manual check) |
| Performance | API overhead p95 < 300 ms | Measured 11 ms (mock LLM, local, 200 requests) |
| Performance | First token p95 < 3 s / full response p95 < 10 s | **Needs real provider + client workload.** Current Anthropic adapter returns one chunk, so TTFT = full latency; add provider SSE streaming to meet TTFT |
| Security | No cross-user/tenant access, auth on all protected endpoints, no client-side keys, validation, rate limits, safe errors | Automated tests pass. Token scheme is a stand-in: replace `verify_token` with the platform's real auth |
| Reliability | >= 99% of 200 requests, retries/timeouts, no duplicate messages | 200/200 with injected transient failures; idempotency tested |
| AI quality | >= 90% on 50 client prompts, rubric scoring | **Not run.** Needs client prompt set and rubric (relevance, correctness, instruction-following, grounding) and the real model |
| Testing/delivery | CI green, >= 80% coverage, docs | Tests pass; `coverage` is not installed here so line coverage is unmeasured. Run `coverage run -m unittest && coverage report` in CI. |

## Deployment and rollback
Env vars: `AUTH_SECRET`, `ANTHROPIC_API_KEY`, `LLM_MODEL`, `DB_PATH`, `RATE_LIMIT_PER_MIN`, `SYSTEM_PROMPT`, `PORT`.
Deploy: build/ship the `app/` and `static/` directories, set the env vars, run behind TLS, then check `/healthz` and send one test chat.
Rollback: redeploy the previous release. The schema is additive and only creates tables if missing, so no down-migration is needed. To disable the feature without a deploy, hide the widget or unset the key (the service falls back to the mock provider only if started that way, so prefer removing the route at the gateway).

## Known gaps vs. the plan
No RAG or business-data connectors (Phase 4, optional), SQLite instead of the platform DB, in-memory per-process rate limiter (use Redis for multi-instance), no real streaming from the LLM yet.
