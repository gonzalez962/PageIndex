# Run PageIndex in Docker

Run PageIndex as an HTTP API against any OpenAI-compatible provider, with one
model for indexing and another for chat, configured from `.env`. A one-shot
CLI indexer is available under a compose profile.

Requires Docker Compose >= 2.24 (`env_file` uses `required: false`); check
with `docker compose version`.

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
   # {"status":"ok","client":"ok","index_model":"...","chat_model":"...","auth":true,...}
   ```

   `/health` builds the model client; if that fails it answers 503 with
   `"status":"error","client":"error"` and the container turns unhealthy.
   The cause is in `docker compose logs api`.

## Use the API

Set `TOKEN` to your `PAGEINDEX_API_TOKEN`; drop the `Authorization` header
when no token is configured. `/health` never needs it.

| Action | Request |
|--------|---------|
| Index a PDF or image | see [Index a PDF](#index-a-pdf) |
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

Two optional flags return where the answer came from, in fields of their own
(both default to `false`; with both off the response is just `answer`):

- `"citations": true` asks the model to cite each claim. The answer keeps
  `[n]` markers and `citations` lists one entry per distinct source:
  `document`, `doc_id`, `page`, and `section`, the title of the deepest tree
  node covering that page (for Markdown, the heading of its section).
  Citations are written by the model, so a claim can go uncited; `doc_id` is
  `null` when the cited name matches no document, and `section` is `null`
  when it cannot be found.
- `"sources_read": true` lists the pages the agent actually read with
  `get_page_content`, per document: `document`, `doc_id`, `pages`. Reading a
  page does not mean the answer used it. `doc_id` is `null` when several
  documents share the name.

Both come from the same single chat run.

```bash
curl -s -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"question": "What is the payment term?", "doc_id": "<doc_id>", "citations": true, "sources_read": true}' \
  http://127.0.0.1:8000/chat
# {"answer":"The term is 30 days [1].",
#  "citations":[{"index":1,"document":"notes.md","doc_id":"pi-...","page":3,"section":"Payment terms"}],
#  "sources_read":[{"document":"notes.md","doc_id":"pi-...","pages":[2,3]}]}
```

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

Retry a failed job after fixing the cause (e.g. the provider key). A job
stuck in `processing` that no worker is running (its final state could not be
saved) can be retried the same way:

```bash
curl -s -X POST -H "Authorization: Bearer $TOKEN" \
  http://127.0.0.1:8000/jobs/job-3f2a.../retry
```

PNG and JPEG images (`png`, `jpg`, `jpeg`) upload the same way
(`-F file=@scan.png`); other image formats are refused. The API also accepts
UTF-8 Markdown (`.md`, `.markdown`) and plain text (`.txt`), case-insensitively.
A UTF-8 BOM is fine; text containing NUL bytes or only whitespace is rejected.
Markdown with recognized headings (`#` through `######`, or a whole-line
`**bold**` heading; headings inside fenced code blocks do not count) uses its
heading tree as the document structure, with each section becoming a page and
any text before the first heading placed in a “Preamble” node. No LLM call
builds that structure, but the index model still generates node summaries and
the document description. Plain text and Markdown without headings are split
into pages of about 1,000 tokens along paragraph boundaries and indexed by the
standard pipeline, where the LLM infers the structure. Text documents use
standard mode even when flash is requested, and OCR settings do not apply;
they are accepted even when OCR is off.

The extension picks the kind and the content must match it: a PDF header,
valid UTF-8 text, or a PNG or JPEG image. Only the first frame is read, so an
animated PNG is one page. Scanned pages and images are read by OCR; see
[OCR](#ocr).

An uploaded image is stored, and indexed, under a fresh UUID name
(`<32 hex digits>.png` or `.jpg`), so uploading `scan.png` twice never
collides. The name you uploaded is kept as `original_name` in the upload
response, the job, and the document's `metadata`:

```bash
curl -s -H "Authorization: Bearer $TOKEN" -F file=@scan.png   http://127.0.0.1:8000/documents
# {"job_id":"job-7c1e...","status":"queued","name":"5b0e...c4.png","original_name":"scan.png"}
```

PDF, Markdown, and text uploads keep their file name and have no
`original_name`; only image uploads are renamed to UUIDs.

`GET /jobs` lists jobs newest first (`status`, `limit`, `offset` filters), and
`/health` reports `"queue": {"queued": n, "processing": n}`.

| Status | Meaning |
|--------|---------|
| 202 | Upload or retry accepted; follow the `Location` header to the job |
| 400 | Content that does not match its PDF, PNG, JPEG, or text extension (including invalid UTF-8, NUL bytes, or blank text), an invalid `Content-Length`, or the SDK rejected the input |
| 401 | Missing or wrong bearer token |
| 404 | Unknown document or job |
| 409 | Retry of a job that is queued, running, or already done |
| 413 | Upload over `PAGEINDEX_MAX_UPLOAD_MB`; a declared `Content-Length` over the limit is refused before the body is read |
| 415 | Not a `.pdf`, `.png`, `.jpg`, `.jpeg`, `.md`, `.markdown` or `.txt` file |
| 502 | The model provider failed (model name, base URL, or key); a `NotFoundError` adds a hint about the `openai/` prefix, and an OCR image rejection says the model may not support image input |
| 503 | The model client cannot be built (check the logs), or the upload could not be stored (retry later) |

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
| `PAGEINDEX_MAX_UPLOAD_MB` | `50` | Upload size limit; a finite number > 0 that is at least 1 byte (e.g. `0.5`) |
| `PAGEINDEX_INDEX_WORKERS` | `1` | Documents indexed in parallel; an integer >= 1 |
| `PAGEINDEX_OCR` | `auto` | OCR mode: `off`, `auto` or `force`, any letter case (API and CLI) |
| `PAGEINDEX_OCR_MODEL` | index model | Vision model for OCR (API only); uses the indexing provider |

An invalid `PAGEINDEX_MAX_UPLOAD_MB`, `PAGEINDEX_INDEX_WORKERS` or
`PAGEINDEX_OCR` stops the API at startup with a message naming the variable.

### OCR

Scanned PDFs, image uploads and figure-heavy pages are read by the indexing
model through its vision input, so that model (or `PAGEINDEX_OCR_MODEL`)
**must accept images**.

- `auto` (default): pages without a usable text layer are transcribed; text
  pages mostly covered by images get a figure description appended. Text-only
  PDFs make no extra calls.
- `force`: every page is transcribed.
- `off`: text layer only; scanned PDFs fail as blank and image uploads fail. Markdown and text uploads are accepted; OCR settings do not apply to them.

Each OCR'd or described page costs one vision call. Documents that needed OCR
are indexed in standard mode (slower than flash).

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
- **Single process:** the queue, the running-job set, and the startup cleanup
  of half-written job directories are per-process. Run the API as one
  process: do not add uvicorn `--workers` and do not scale the `api` service.
  Raise `PAGEINDEX_INDEX_WORKERS` instead.

Chat, listing, and `/health` stay responsive while documents are indexed.

## CLI indexer

Put PDFs or PNG/JPEG images in `./data`; tree structures are written to `./results`.

```bash
docker compose --profile cli run --rm cli --pdf_path data/report.pdf
docker compose --profile cli run --rm cli --help
```

`PAGEINDEX_INDEX_MODEL` becomes `--index-model`; the CLI uses `OPENAI_*`
only (per-role overrides apply to the API). For `--pdf_path` runs,
`PAGEINDEX_OCR` becomes `--ocr`; a flag you pass explicitly wins, and
`--md_path` runs ignore it. `PAGEINDEX_OCR_MODEL` is not passed to the CLI,
because it names a model on the API's indexing provider; the CLI OCRs with
its index model unless you pass `--ocr-model` yourself.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Settings are ignored | `.env` must sit next to `docker-compose.yml`; compose runs without it, silently using defaults. Check with `docker compose config`. |
| Connection refused to a local provider | Inside the container `localhost` is the container. Use `http://host.docker.internal:<port>/v1` (Linux: add `extra_hosts: ["host.docker.internal:host-gateway"]` to the service). |
| `LLM Provider NOT provided` or wrong provider | Model ids containing `/` need the `openai/` prefix, e.g. `openai/meta-llama/Llama-3.3-70B-Instruct`. |
| A placeholder model name reaches the provider | Model variables are used literally; leave them commented out to use the SDK defaults. |
| `./results` files owned by root (Linux) | The image runs as root. Run `sudo chown -R "$USER" results`, or pass `--user "$(id -u):$(id -g)"` to `docker compose run`. |
| 503 "Model client is not configured", or `/health` 503 | The client could not be built from `.env`; `docker compose logs api` has the cause. Fix `.env`, then `docker compose up -d api`. A running API retries a failed build at most every 30 s. |
| 502 with `NotFoundError` | The provider does not know the model. If its id contains `/`, prefix it with `openai/` so it goes to `OPENAI_BASE_URL`. |
| A job `failed`: the OCR model may not support image input | Point `PAGEINDEX_OCR_MODEL` (or `PAGEINDEX_INDEX_MODEL`) at a vision-capable model, or set `PAGEINDEX_OCR=off` for text-only PDFs. |
| 502 from `/chat`, or a job `failed` with an upstream provider error | Provider rejected the call; check `docker compose logs api`, the model name, base URL, and key, then retry the job. |
