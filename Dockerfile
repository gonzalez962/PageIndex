FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt .
COPY server/requirements.txt server/requirements.txt
RUN pip install -r requirements.txt -r server/requirements.txt

COPY pageindex ./pageindex
COPY run_pageindex.py .
COPY server ./server

# Runs as root on purpose: files written to bind mounts (./results) are then
# root-owned on Linux hosts; see docs/docker.md. The HTTP API (compose
# service "api") overrides this entrypoint with uvicorn.

# PAGEINDEX_INDEX_MODEL (optional) becomes --index-model. For --pdf_path runs,
# PAGEINDEX_OCR / PAGEINDEX_OCR_MODEL (optional) become --ocr / --ocr-model,
# placed before the given arguments so explicit flags win; other runs (e.g.
# --md_path, which refuses OCR flags) never get them. Every other argument
# passes through to run_pageindex.py unchanged.
ENTRYPOINT ["sh", "-c", "case \" $* \" in *\" --pdf_path\"*) set -- ${PAGEINDEX_OCR:+--ocr \"$PAGEINDEX_OCR\"} ${PAGEINDEX_OCR_MODEL:+--ocr-model \"$PAGEINDEX_OCR_MODEL\"} \"$@\";; esac; exec python run_pageindex.py ${PAGEINDEX_INDEX_MODEL:+--index-model \"$PAGEINDEX_INDEX_MODEL\"} \"$@\"", "--"]
CMD ["--help"]
