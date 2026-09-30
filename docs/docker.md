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
| Index a PDF | see [Index a PDF](#index-a-pdf) |
| List jobs | `curl -s -H "Authorization: Bearer $TOKEN" "http://127.0.0.1:8000/jobs?status=failed"` |
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
Interactive docs are served at `http://127.0.0.1:8000/docs`.

### Index a PDF

Uploads are queued: `POST /documents` checks the file and answers `202` right
away; indexing runs in the background. Poll the job until it is `done`, then
use its `doc_id`.

```bash
curl -s -H "Authorization: Bearer $TOKEN" -F file=@report.pdf \
  http://127.0.0.1:8000/documents
# {"job_id":"job-3f2a...","status":"queued","name":"report.pdf"}

curl -s -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8000/jobs/job-3f2a...
# {"id":"job-3f2a...","status":"done","doc_id":"pi-...","attempts":1,...}
```

| Job status | Meaning |
|------------|---------|
| `queued` | Waiting for a worker |
| `processing` | Being indexed (minutes for large PDFs) |
| `done` | Indexed; `doc_id` is set |
| `failed` | `error` says why; the PDF is kept for a retry |

Retry a failed job after fixing the cause (e.g. the provider key):

```bash
curl -s -X POST -H "Authorization: Bearer $TOKEN" \
  http://127.0.0.1:8000/jobs/job-3f2a.../retry
```

`GET /jobs` lists jobs newest first (`status`, `limit`, `offset` filters), and
`/health` reports `"queue": {"queued": n, "processing": n}`.

| Status | Meaning |
|--------|---------|
| 202 | Upload or retry accepted; follow the `Location` header to the job |
| 400 | Not a PDF, or the SDK rejected the input |
| 401 | Missing or wrong bearer token |
| 404 | Unknown document or job |
| 409 | Retry of a job that is not `failed` |
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
| `PAGEINDEX_INDEX_WORKERS` | `1` | Documents indexed in parallel |

Restart after editing `.env`: `docker compose up -d api`.

## Persistence

Indexed documents live in the named volume `storage` (`/app/storage` in the
container). They survive `docker compose down`, restarts, and rebuilds;
`docker compose down -v` deletes them.

### Indexing queue

Each upload is stored as a job in `/app/storage/jobs/<job_id>/` (`job.json`
plus the PDF), so the queue lives in the same volume as the documents:

- **Restarts:** on startup, `queued` jobs and jobs interrupted mid-indexing
  resume in upload order; nothing needs re-uploading. A job whose document
  was already stored before the interruption is linked to it, not indexed
  twice.
- **Cleanup:** a job's PDF is deleted once it is `done`; failed jobs keep it
  for retries. Job records are kept.
- **Workers:** `PAGEINDEX_INDEX_WORKERS` sets how many documents are indexed at
  once. Each document already makes many concurrent model calls, so raising it
  multiplies provider load and the chance of rate limits; keep `1` unless your
  provider has headroom.

Chat, listing, and `/health` stay responsive while documents are indexed.

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
| 502 from `/chat`, or a job `failed` with an upstream provider error | Provider rejected the call; check `docker compose logs api`, the model name, base URL, and key, then retry the job. |
