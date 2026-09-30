FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY pageindex ./pageindex
COPY run_pageindex.py .

# PAGEINDEX_INDEX_MODEL (optional) becomes --index-model; every other argument
# passes through to run_pageindex.py unchanged.
ENTRYPOINT ["sh", "-c", "exec python run_pageindex.py ${PAGEINDEX_INDEX_MODEL:+--index-model \"$PAGEINDEX_INDEX_MODEL\"} \"$@\"", "--"]
CMD ["--help"]
