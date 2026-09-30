# Feature: docker-http-api

## Objective
Run PageIndex in Docker against a custom OpenAI-compatible provider, with
separately selectable index and chat models, exposed as an HTTP API.

## Problem / Why
The repository only ships a library and an index-only CLI (`run_pageindex.py`).
The user needs a containerized service where documents are indexed with one
model and queried (chat) with another, configured from `.env`.

## Scope
- Docker image + compose for the CLI indexer (done, T1).
- FastAPI HTTP service wrapping `PageIndexClient` in local mode.
- Separate `PAGEINDEX_INDEX_MODEL` / `PAGEINDEX_CHAT_MODEL`, optional
  per-role base URL / API key overrides, falling back to `OPENAI_*`.
- Persistent document storage via a volume.

Out of scope: streaming responses, cloud mode, changes to the `pageindex`
package itself, web UI.

## Constraints
- Do not modify `pageindex/` (SDK) — the service is a thin adapter.
- Secrets only via `.env`, never baked into the image.
- Default port binding on 127.0.0.1; optional bearer token.
- Artifacts in English.

## Tasks
- [x] T1 — Docker image, compose, `.env.example`, `.dockerignore` for the CLI indexer. Route: inline (4 small config files, already understood). Checks: image builds; `--help` runs; model reaches LiteLLM; requests hit `OPENAI_BASE_URL` (observed in container).
- [x] T2 — FastAPI service (`server/`) + tests: health, upload+index PDF, list/get documents, chat with separate chat model. Route: delegated writer (2+ non-trivial files). Checks: pytest `tests/test_server.py` inside Docker (host lacks deps); RED observed before GREEN.
- [x] T3 — Wire service into Docker/compose (`api` service, storage volume, port, CLI under a profile), update `.env.example`, add `docs/docker.md`. Route: delegated writer. Checks: `docker compose config`, image build, container `/health` responds.

## Acceptance criteria
- `docker compose up api` serves the API; `POST /documents` indexes with the index model; `POST /chat` answers with the chat model.
- Index and chat models (and optionally endpoints) selectable independently from `.env`.
- Indexed documents survive container restarts.

## Delivery
- Strategy: ask-on-risk. Forecast ~450 authored changed lines (T1 ~50, T2 ~300, T3 ~100).
- Branch: `feat/docker-http-api`.

## Progress / Evidence
- T1: committed `1099663`. RDD assess: high (process_boundary, Dockerfile) -> consent granted -> 4-lens review approved, acknowledged (lineage review-5cfffa733b81615f, authority burned). Reviewed boundary advances to `1099663`.
- T2: `server/app.py` (app factory `create_app(client=None, env=None)`, lazy client, env -> `client_kwargs`), `server/requirements.txt`, `tests/test_server.py`. Endpoints: `GET /health`, `POST|GET /documents`, `GET|DELETE /documents/{doc_id}`, `POST /chat`. Errors: SDK rejections 400, "not found" 404, provider errors (litellm/openai/httpx or `LLMRetriesExhausted` anywhere in the cause chain) 502 with the exception class only. Concurrency: plain `def` endpoints (threadpool); indexing serialized with a lock to bound provider load (the SDK store already locks its own writes); chat runs concurrently. RED: `ModuleNotFoundError: No module named 'server'` (collection error). GREEN: `tests/test_server.py` 21 passed; regression `tests/test_client.py tests/test_package_surface.py` 267 passed (both inside `pageindex:local`). Real `PageIndexClient` accepts the mapped kwargs (smoke). Commit: `81eb6a1`.
- T3: Dockerfile installs `server/requirements.txt` and copies `server/` (still root; root-owned `./results` documented instead of a non-root user, since bind mounts created by the daemon would not be writable). Compose: `api` service (uvicorn entrypoint, `127.0.0.1:${PAGEINDEX_PORT:-8000}`, named volume `storage` at `/app/storage`, Python healthcheck, `restart: unless-stopped`), CLI moved to service `cli` under profile `cli`, shared `env_file` with `required: false`. `.env.example` ships models and overrides commented out. `tests/test_docker_entrypoint.py` runs the real ENTRYPOINT string with a stub `python` (characterization of T1 behavior, so no RED; skipped on Windows). `docs/docker.md` added. Checks: `docker compose config` (no `.env`) exit 0; `docker compose build` ok; `docker compose up -d api` -> `/health` = `{"status":"ok","index_model":null,"chat_model":null,"auth":false}`, container healthcheck `healthy`; `docker compose --profile cli run --rm cli --help` prints usage; `tests/test_server.py tests/test_docker_entrypoint.py` 23 passed in the rebuilt image. Commit: see git log (`build(docker): ...`).
- T1 advisory findings folded into T2/T3: `.env.example` must mark models optional and ship them commented out; compose `env_file` should be `required: false`; `.dockerignore` excludes `tests/` (test runs mount the repo instead); document root-owned bind mounts; entrypoint arg handling gets a test in T3.

## Next step
All tasks done. Parent: RDD assess T2 (`81eb6a1`) and T3 against boundary `1099663`; live end-to-end check (upload + chat against a real provider) still pending.
