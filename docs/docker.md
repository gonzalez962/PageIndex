# Run PageIndex in Docker

Run PageIndex as an HTTP API against any OpenAI-compatible provider, with one
model for indexing and another for chat, configured from `.env`. A one-shot
CLI indexer is available under a compose profile.

## Quick path

1. Configure the provider:

   ```bash
   cp .env.example .env
   # set OPENAI_API_KEY, OPENAI_BASE_URL, and optionally
   # PAGEINDEX_INDEX_MODEL / PAGEINDEX_CHAT_MODEL / PAGEINDEX_API_TOKEN
   ```

2. Start the API:

   ```bash
   docker compose up -d --build api
   ```

3. Check it:

   ```bash
   curl -s http://127.0.0.1:8000/health
   # {"status":"ok","index_model":"...","chat_model":"...","auth":true}
   ```

## Use the API

Set `TOKEN` to your `PAGEINDEX_API_TOKEN`; drop the `Authorization` header
when no token is configured. `/health` never needs it.

| Action | Request |
|--------|---------|
| Index a PDF | `curl -s -H "Authorization: Bearer $TOKEN" -F file=@report.pdf http://127.0.0.1:8000/documents` |
| List documents | `curl -s -H "Authorization: Bearer $TOKEN" "http://127.0.0.1:8000/documents?limit=50&offset=0"` |
| Get one | `curl -s -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8000/documents/<doc_id>` |
| Delete one | `curl -s -X DELETE -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8000/documents/<doc_id>` |
| Ask | see below |

```bash
curl -s -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"question": "What are the key findings?", "doc_id": "<doc_id>"}' \
  http://127.0.0.1:8000/chat
# {"answer":"..."}
```

`doc_id` accepts one id, a list of ids, or `null` (the whole library).
Indexing is synchronous: `POST /documents` returns once the document is
indexed, which can take minutes for large PDFs. Interactive docs are served
at `http://127.0.0.1:8000/docs`.

| Status | Meaning |
|--------|---------|
| 400 | The SDK rejected the input (e.g. a blank PDF) |
| 401 | Missing or wrong bearer token |
| 404 | Unknown document |
| 413 | Upload over `PAGEINDEX_MAX_UPLOAD_MB` |
| 415 | Not a `.pdf` file |
| 502 | The model provider failed (model name, base URL, or key) |

## Configuration

All settings live in `.env`; see `.env.example` for the annotated list.

| Variable | Default | Purpose |
|----------|---------|---------|
| `OPENAI_API_KEY`, `OPENAI_BASE_URL` | required | Provider connection for both roles |
| `PAGEINDEX_INDEX_MODEL` | SDK default | Indexing model (API and CLI) |
| `PAGEINDEX_CHAT_MODEL` | SDK default | Chat model (API) |
| `PAGEINDEX_INDEX_BASE_URL`, `PAGEINDEX_INDEX_API_KEY` | `OPENAI_*` | Separate provider for indexing (API) |
| `PAGEINDEX_CHAT_BASE_URL`, `PAGEINDEX_CHAT_API_KEY` | `OPENAI_*` | Separate provider for chat (API) |
| `PAGEINDEX_API_TOKEN` | unset (no auth) | Bearer token for every endpoint but `/health` |
| `PAGEINDEX_PORT` | `8000` | Host port, bound on `127.0.0.1` only |
| `PAGEINDEX_MAX_UPLOAD_MB` | `50` | Upload size limit |

Restart after editing `.env`: `docker compose up -d api`.

## Persistence

Indexed documents live in the named volume `storage` (`/app/storage` in the
container). They survive `docker compose down`, restarts, and rebuilds;
`docker compose down -v` deletes them.

The API runs one uvicorn worker and indexes one document at a time; chat
requests run concurrently.

## CLI indexer

Put PDFs in `./data`; tree structures are written to `./results`.

```bash
docker compose --profile cli run --rm cli --pdf_path data/report.pdf
docker compose --profile cli run --rm cli --help
```

`PAGEINDEX_INDEX_MODEL` becomes `--index-model`; the CLI uses `OPENAI_*`
only (per-role overrides apply to the API).

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Settings are ignored | `.env` must sit next to `docker-compose.yml`; compose runs without it, silently using defaults. Check with `docker compose config`. |
| Connection refused to a local provider | Inside the container `localhost` is the container. Use `http://host.docker.internal:<port>/v1` (Linux: add `extra_hosts: ["host.docker.internal:host-gateway"]` to the service). |
| `LLM Provider NOT provided` or wrong provider | Model ids containing `/` need the `openai/` prefix, e.g. `openai/meta-llama/Llama-3.3-70B-Instruct`. |
| A placeholder model name reaches the provider | Model variables are used literally; leave them commented out to use the SDK defaults. |
| `./results` files owned by root (Linux) | The image runs as root. Run `sudo chown -R "$USER" results`, or pass `--user "$(id -u):$(id -g)"` to `docker compose run`. |
| 502 from `/chat` or `/documents` | Provider rejected the call; check `docker compose logs api`, the model name, base URL, and key. |
