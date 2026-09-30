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
- [ ] T2 — FastAPI service (`server/`) + tests: health, upload+index PDF, list/get documents, chat with separate chat model. Route: delegated writer (2+ non-trivial files). Checks: pytest `tests/test_server.py` inside Docker (host lacks deps); RED observed before GREEN.
- [ ] T3 — Wire service into Docker/compose (`api` service, storage volume, port, CLI under a profile), update `.env.example`, add `docs/docker.md`. Route: delegated writer. Checks: `docker compose config`, image build, container `/health` responds.

## Acceptance criteria
- `docker compose up api` serves the API; `POST /documents` indexes with the index model; `POST /chat` answers with the chat model.
- Index and chat models (and optionally endpoints) selectable independently from `.env`.
- Indexed documents survive container restarts.

## Delivery
- Strategy: ask-on-risk. Forecast ~450 authored changed lines (T1 ~50, T2 ~300, T3 ~100).
- Branch: `feat/docker-http-api`.

## Progress / Evidence
- T1: verified before branching (see checks above). Commit: pending.

## Next step
Commit T1, then delegate T2.
