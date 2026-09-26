"""C7: FastAPI service. Run: python -m uvicorn app.api.main:app --port 8000  (docs at /docs)."""
import time
from collections.abc import Iterator
from contextlib import asynccontextmanager
from typing import Literal

import psycopg
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from psycopg_pool import PoolTimeout
from pydantic import BaseModel

from app.config import EMBED_MODEL, RERANK, RERANK_TOP
from app.db.connection import create_pool, embedding_dim
from app.log import get_logger
from app.search import hybrid

log = get_logger("app.api")

# One pool per process, opened at startup; requests borrow connections instead of opening new ones.
pool = create_pool()


@asynccontextmanager
async def lifespan(_: FastAPI):
    t0 = time.perf_counter()
    pool.open(wait=False)
    try:
        pool.wait(timeout=10)
    except PoolTimeout:
        log.warning("DB not reachable at startup; pool keeps retrying in the background")
    model = hybrid._model()  # load the embedding model once, not on the first request
    model_dim = model.get_sentence_embedding_dimension()
    try:
        with pool.connection(timeout=5) as conn:
            db_dim = embedding_dim(conn)
    except (PoolTimeout, psycopg.OperationalError):
        db_dim = None
        log.warning("embedding dimension check skipped: DB not reachable")
    if db_dim is not None and db_dim != model_dim:
        # Fail at startup with a clear message instead of an opaque pgvector error on every search.
        msg = (f"embedding model {EMBED_MODEL} is {model_dim}-d but chunks.embedding is vector({db_dim}); "
               "update schema.sql, migrate the column and re-ingest")
        log.error(msg)
        raise RuntimeError(msg)
    log.info("embedding dimension check: model=%d db=%s", model_dim, db_dim)
    # Loading isn't enough: the first encode() and the first queries on a fresh connection are slow (~1.5 s),
    # so run one real search now instead of on the first user request.
    try:
        with pool.connection(timeout=5) as conn:
            hybrid.search("warm up", conn=conn)
    except Exception:
        log.warning("warm-up search failed (DB down?); warming up the model only", exc_info=True)
        hybrid.embed(model, ["warm up"])
    if RERANK:
        # The warm-up query yields < 2 candidates, so search() never reranks it: load + run the reranker explicitly.
        hybrid._reranker().predict([("warm up", "warm up")])
    log.info("API ready in %.1fs (pool min=%d max=%d from DB_POOL_SIZE/DB_POOL_MAX_OVERFLOW, model warmed up, "
             "reranker=%s)", time.perf_counter() - t0, pool.min_size, pool.max_size,
             f"on(top {RERANK_TOP})" if RERANK else "off")
    yield
    pool.close()
    log.info("API shutdown: pool closed")


def get_db() -> Iterator[psycopg.Connection]:
    with pool.connection() as conn:
        yield conn


app = FastAPI(title="Transcript Hybrid Search", version="0.1.0", lifespan=lifespan)


class Turn(BaseModel):
    speaker: str
    text: str


class SearchHit(BaseModel):
    recording_id: str
    chunk_index: int
    speaker: str
    start_s: float
    end_s: float
    text: str
    score: float
    keyword_rank: int | None
    vector_rank: int | None
    prev: Turn | None
    next: Turn | None


class SearchResponse(BaseModel):
    query: str
    speaker: str | None
    recording: str | None
    count: int
    took_ms: float
    results: list[SearchHit]


class Recording(BaseModel):
    id: str
    duration_s: float
    status: str
    chunks: int


@app.exception_handler(psycopg.OperationalError)
@app.exception_handler(PoolTimeout)
async def db_unavailable(_: Request, exc: Exception):
    log.error("database unavailable: %s: %s", type(exc).__name__, exc)
    return JSONResponse(status_code=503, content={"detail": "database unavailable"})


@app.exception_handler(RequestValidationError)
async def validation_failed(request: Request, exc: RequestValidationError):
    # Log 422s (e.g. empty recording=) so they're traceable in logs/app.log; the response stays FastAPI's default.
    log.info("422 %s %s: %s", request.method, request.url.path,
             [(".".join(map(str, e["loc"])), e["msg"]) for e in exc.errors()])
    return await request_validation_exception_handler(request, exc)


@app.get("/health")
def health(conn: psycopg.Connection = Depends(get_db)) -> dict:
    conn.execute("SELECT 1")
    stats = pool.get_stats()
    return {"status": "ok", "db": "ok", "pool": {"size": stats.get("pool_size"), "available": stats.get("pool_available")}}


@app.get("/recordings", response_model=list[Recording])
def recordings(conn: psycopg.Connection = Depends(get_db)) -> list[Recording]:
    rows = conn.execute(
        "SELECT r.id, r.duration_s, r.status, count(c.id) FROM recordings r "
        "LEFT JOIN chunks c ON c.recording_id = r.id GROUP BY r.id ORDER BY r.id"
    ).fetchall()
    return [Recording(id=r[0], duration_s=r[1], status=r[2], chunks=r[3]) for r in rows]


@app.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(..., min_length=1, max_length=500, description="Keywords (AND, \"phrase\", -exclude) or a question"),
    speaker: Literal["A", "B"] | None = Query(None, description="Requires `recording` (labels are per recording)"),
    # min_length=1: an empty recording= would otherwise skip the 404 check and silently filter on id ''.
    recording: str | None = Query(None, min_length=1, description="Recording id, e.g. rec04_support_billing"),
    n: int = Query(10, ge=1, le=50),
    conn: psycopg.Connection = Depends(get_db),
) -> SearchResponse:
    if speaker and not recording:
        raise HTTPException(422, "speaker filter requires a recording filter (speaker labels are per recording)")
    if recording and not conn.execute("SELECT 1 FROM recordings WHERE id = %s", (recording,)).fetchone():
        log.info("GET /search unknown recording=%r -> 404", recording)
        raise HTTPException(404, detail="unknown recording")
    t0 = time.perf_counter()
    hits = hybrid.search(q, speaker=speaker, recording=recording, n=n, conn=conn)
    took = round((time.perf_counter() - t0) * 1e3, 1)
    log.info("GET /search q=%r speaker=%s recording=%s n=%d -> %d hits in %sms", q, speaker, recording, n, len(hits), took)
    return SearchResponse(query=q, speaker=speaker, recording=recording, count=len(hits), took_ms=took,
                          results=[SearchHit(**{k: v for k, v in h.items() if k != "chunk_id"}) for h in hits])
