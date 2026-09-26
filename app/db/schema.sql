CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS recordings (
    id              TEXT PRIMARY KEY,
    file_path       TEXT NOT NULL,
    content_sha256  TEXT NOT NULL UNIQUE,
    duration_s      REAL NOT NULL,
    status          TEXT NOT NULL CHECK (status IN ('processing', 'completed', 'failed')),
    error           TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id            BIGSERIAL PRIMARY KEY,
    recording_id  TEXT NOT NULL REFERENCES recordings(id) ON DELETE CASCADE,
    chunk_index   INT  NOT NULL,
    speaker       TEXT NOT NULL,
    start_s       REAL NOT NULL,
    end_s         REAL NOT NULL,
    text          TEXT NOT NULL,
    tsv           TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    embedding     VECTOR(384) NOT NULL,
    UNIQUE (recording_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING GIN (tsv);
-- No vector index on purpose: exact scan at this scale (L11). Production: HNSW (vector_cosine_ops).
