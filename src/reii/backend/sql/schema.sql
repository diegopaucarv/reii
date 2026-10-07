-- ============================================================================
-- REII · SQLite schema v1
-- ============================================================================
PRAGMA foreign_keys  = ON;
PRAGMA journal_mode  = WAL;
PRAGMA synchronous   = NORMAL;
PRAGMA temp_store    = MEMORY;
PRAGMA mmap_size     = 268435456;   -- 256 MB
PRAGMA page_size     = 4096;

-- ── schema versioning ───────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS _schema_version (
    version     INTEGER PRIMARY KEY,
    applied_at  TEXT NOT NULL
);

-- ── documents ───────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS documents (
    doc_id          TEXT PRIMARY KEY,
    doc_idx         INTEGER NOT NULL,
    orden           INTEGER,
    origen          TEXT,
    metadata_json   TEXT NOT NULL DEFAULT '{}',
    texto_completo  TEXT,
    n_uces          INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_documents_orden ON documents(orden);

-- ── UCEs ────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS uces (
    uce_id                  TEXT PRIMARY KEY,
    doc_id                  TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    local_idx               INTEGER NOT NULL,
    section_id              INTEGER NOT NULL DEFAULT 0,
    seccion                 TEXT,
    texto                   TEXT NOT NULL,
    n_tokens                INTEGER,

    -- Fase inductiva
    cluster_id              INTEGER,
    is_stable               INTEGER NOT NULL DEFAULT 0,
    stability_method        TEXT,
    is_terminal_consolidated INTEGER NOT NULL DEFAULT 0,

    -- Fase abductiva (NUNCA contamina cluster_id)
    projected_cluster_id    INTEGER,
    projection_distance     REAL,
    projection_margin       REAL,
    projection_ratio        REAL,

    phi_score               REAL,

    -- Payload heterogéneo: verbos, pronombres, adverbios, negaciones, etc.
    linguistic_json         TEXT NOT NULL DEFAULT '{}',
    -- metricas_lexicas, complejidad_sintactica, etc.
    metrics_json            TEXT NOT NULL DEFAULT '{}',

    created_at              TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_uces_doc        ON uces(doc_id);
CREATE INDEX IF NOT EXISTS idx_uces_doc_local  ON uces(doc_id, local_idx);
CREATE INDEX IF NOT EXISTS idx_uces_seccion    ON uces(seccion);
CREATE INDEX IF NOT EXISTS idx_uces_cluster
    ON uces(cluster_id) WHERE cluster_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_uces_stable
    ON uces(is_stable) WHERE is_stable = 1;
CREATE INDEX IF NOT EXISTS idx_uces_projected
    ON uces(projected_cluster_id) WHERE projected_cluster_id IS NOT NULL;

-- ── UCs ─────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS ucs (
    uc_id         TEXT PRIMARY KEY,
    doc_id        TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    texto         TEXT NOT NULL,
    cluster_id    INTEGER,
    uce_ids_json  TEXT NOT NULL DEFAULT '[]',
    lemmas_json   TEXT NOT NULL DEFAULT '[]',
    n_lemmas      INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_ucs_doc     ON ucs(doc_id);
CREATE INDEX IF NOT EXISTS idx_ucs_cluster ON ucs(cluster_id) WHERE cluster_id IS NOT NULL;

-- ── clusters ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS clusters (
    cluster_id      INTEGER PRIMARY KEY,
    cluster_name    TEXT,
    member_count    INTEGER NOT NULL DEFAULT 0,
    top_terms_json  TEXT NOT NULL DEFAULT '[]',
    centroid_json   TEXT,
    created_at      TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- ── terms (TermAnalyzer output) ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS terms (
    term               TEXT NOT NULL,
    cluster_id         INTEGER NOT NULL,
    frecuencia_global  INTEGER NOT NULL DEFAULT 0,
    frecuencia_cluster INTEGER NOT NULL DEFAULT 0,
    chi2_yates         REAL,
    p_valor            REAL,
    p_adj              REAL,
    phi                REAL,
    cramer_v           REAL,
    c_value            REAL,
    significativo      INTEGER NOT NULL DEFAULT 0,
    asignado_estricto  INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (term, cluster_id)
);
CREATE INDEX IF NOT EXISTS idx_terms_cluster
    ON terms(cluster_id);
CREATE INDEX IF NOT EXISTS idx_terms_sig
    ON terms(cluster_id, significativo) WHERE significativo = 1;

-- ── annotations (AI discourse) ──────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS annotations (
    annotation_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    uce_id         TEXT NOT NULL REFERENCES uces(uce_id) ON DELETE CASCADE,
    agent_name     TEXT NOT NULL,
    trait          TEXT,
    quote          TEXT,
    confidence     TEXT,
    subtype        TEXT,
    payload_json   TEXT NOT NULL DEFAULT '{}',
    created_at     TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_annotations_uce       ON annotations(uce_id);
CREATE INDEX IF NOT EXISTS idx_annotations_agent     ON annotations(agent_name);
CREATE INDEX IF NOT EXISTS idx_annotations_uce_agent ON annotations(uce_id, agent_name);

-- ── network edges ───────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS network_edges (
    src     TEXT NOT NULL,
    dst     TEXT NOT NULL,
    weight  REAL NOT NULL,
    PRIMARY KEY (src, dst)
);
CREATE INDEX IF NOT EXISTS idx_network_src ON network_edges(src);
CREATE INDEX IF NOT EXISTS idx_network_dst ON network_edges(dst);

-- ── key/value store (config, sintesis, multivariate, etc.) ──────────────────
CREATE TABLE IF NOT EXISTS kv_store (
    key         TEXT PRIMARY KEY,
    value_json  TEXT NOT NULL,
    updated_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- ════════════════════════════════════════════════════════════════════════════
-- Vistas
-- ════════════════════════════════════════════════════════════════════════════
CREATE VIEW IF NOT EXISTS v_cluster_summary AS
SELECT
    c.cluster_id,
    c.cluster_name,
    c.member_count                                 AS declared_members,
    COUNT(u.uce_id)                                AS uce_count,
    SUM(CASE WHEN u.is_stable = 1 THEN 1 ELSE 0 END) AS stable_count,
    AVG(u.phi_score)                               AS mean_phi
FROM clusters c
LEFT JOIN uces u ON u.cluster_id = c.cluster_id
GROUP BY c.cluster_id;

CREATE VIEW IF NOT EXISTS v_annotation_counts AS
SELECT
    agent_name,
    COUNT(*)                    AS n_annotations,
    COUNT(DISTINCT uce_id)      AS n_uces
FROM annotations
GROUP BY agent_name;

CREATE VIEW IF NOT EXISTS v_uce_full AS
SELECT
    u.uce_id, u.doc_id, u.local_idx, u.seccion, u.texto,
    u.cluster_id, u.is_stable, u.phi_score,
    u.projected_cluster_id, u.projection_distance, u.projection_ratio,
    d.metadata_json AS doc_metadata_json
FROM uces u
JOIN documents d ON d.doc_id = u.doc_id;

-- ── workflow config (v2) ─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS workflow_config (
    key         TEXT PRIMARY KEY,
    value_json  TEXT NOT NULL,
    value_type  TEXT NOT NULL,
    category    TEXT,
    description TEXT,
    updated_by  TEXT,
    updated_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS workflow_config_history (
    history_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    key         TEXT NOT NULL,
    value_json  TEXT NOT NULL,
    value_type  TEXT NOT NULL,
    category    TEXT,
    description TEXT,
    updated_by  TEXT,
    updated_at  TEXT NOT NULL,
    operation   TEXT NOT NULL
);

CREATE TRIGGER IF NOT EXISTS trg_workflow_config_audit_update
AFTER UPDATE ON workflow_config
BEGIN
    INSERT INTO workflow_config_history
        (key, value_json, value_type, category, description, updated_by, updated_at, operation)
    VALUES
        (OLD.key, OLD.value_json, OLD.value_type, OLD.category, OLD.description, OLD.updated_by, OLD.updated_at, 'UPDATE');
END;

CREATE TRIGGER IF NOT EXISTS trg_workflow_config_audit_delete
AFTER DELETE ON workflow_config
BEGIN
    INSERT INTO workflow_config_history
        (key, value_json, value_type, category, description, updated_by, updated_at, operation)
    VALUES
        (OLD.key, OLD.value_json, OLD.value_type, OLD.category, OLD.description, OLD.updated_by, OLD.updated_at, 'DELETE');
END;

INSERT OR IGNORE INTO _schema_version (version, applied_at) VALUES (1, datetime('now'));
INSERT OR IGNORE INTO _schema_version (version, applied_at) VALUES (2, datetime('now'));
