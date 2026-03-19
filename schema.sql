-- Parameter Golf — Observability Schema
-- SQLite database for post-hoc run analysis, bottleneck detection, and kernel benchmarking.
--
-- Data flow:
--   train_gpt.py → JSONL (streaming, in-flight)
--   ingest_runs.py → runs.db (post-hoc, queryable)
--   benchmark scripts → kernel_benchmarks JSON → runs.db
--
-- Usage:
--   sqlite3 runs.db < schema.sql          # create empty DB
--   uv run python ingest_runs.py          # ingest all JSONL logs
--   sqlite3 runs.db "SELECT * FROM v_leaderboard"

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- ============================================================
-- RUNS — one row per completed training run
-- Source: "final" event in JSONL
-- ============================================================

CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    git_hash        TEXT,
    created_at      TEXT DEFAULT (datetime('now')),

    -- final metrics
    steps           INTEGER,
    wall_ms         REAL,
    step_avg_ms     REAL GENERATED ALWAYS AS (wall_ms / MAX(steps, 1)) STORED,
    vl_prequant     REAL,       -- val loss before quantization
    vb_prequant     REAL,       -- val BPB before quantization
    vl_postquant    REAL,       -- val loss after quantization
    vb_postquant    REAL,       -- val BPB after quantization (THE competition metric)
    vl_sw_postquant REAL,       -- sliding window val loss (post-quant, stride=EVAL_STRIDE)
    vb_sw_postquant REAL,       -- sliding window val BPB (post-quant) — best eval metric

    -- artifact size
    bytes_total     INTEGER,    -- code + compressed model
    bytes_model     INTEGER,
    bytes_code      INTEGER,
    model_params    INTEGER,

    -- early stop
    early_stopped   INTEGER DEFAULT 0,
    early_stop_reason TEXT,

    -- key config columns (most-queried, avoids JSON parsing)
    num_layers      INTEGER,
    model_dim       INTEGER,
    num_heads       INTEGER,
    num_kv_heads    INTEGER,
    mlp_mult        INTEGER,
    vocab_size      INTEGER,
    train_seq_len   INTEGER,
    iterations      INTEGER,
    matrix_lr       REAL,
    scalar_lr       REAL,
    embed_lr        REAL,
    muon_momentum   REAL,
    logit_softcap   REAL,
    seed            INTEGER,

    -- full config snapshot (all hyperparameters as JSON)
    config_json     TEXT,

    -- probe metadata
    hypothesis      TEXT,       -- what this run is testing (from PROBE_HYPOTHESIS env var)
    notes           TEXT        -- free-text notes (agent or human can annotate)
);

-- ============================================================
-- VAL_CHECKPOINTS — every validation measurement
-- Source: "val" events in JSONL
-- Enables: cross-run BPB curve comparison, convergence analysis
-- ============================================================

CREATE TABLE IF NOT EXISTS val_checkpoints (
    run_id      TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    step        INTEGER NOT NULL,
    val_loss    REAL,
    val_bpb     REAL,
    ref_bpb     REAL,           -- reference curve value at this step (NULL if no ref)
    status      TEXT,           -- ON_TRACK, BEATING, BEHIND, KILL_*
    elapsed_ms  REAL,
    PRIMARY KEY (run_id, step)
);

-- ============================================================
-- TRAIN_CHECKPOINTS — training loss at logged steps
-- Source: "train" events in JSONL (emitted at TRAIN_LOG_EVERY cadence)
-- Enables: training loss curves, LR schedule visualization, grad norm monitoring
-- ============================================================

CREATE TABLE IF NOT EXISTS train_checkpoints (
    run_id      TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    step        INTEGER NOT NULL,
    train_loss  REAL,
    lr          REAL,           -- learning rate scale at this step
    grad_norm   REAL,           -- total gradient norm (NULL until grad norm logging enabled)
    elapsed_ms  REAL,
    PRIMARY KEY (run_id, step)
);

-- ============================================================
-- STEP_PROFILES — per-step section timing breakdown
-- Source: "profile" events in JSONL (emitted when PROFILE_SECTIONS=1)
-- Sampled at TRAIN_LOG_EVERY cadence (default every 200 steps)
--
-- Sections:
--   data      — batch loading (next_batch)
--   fwd_bwd   — forward + backward pass (all micro steps)
--   optimizer  — all optimizer.step() calls
--   misc      — zero_grad, lr schedule, grad clip
-- ============================================================

CREATE TABLE IF NOT EXISTS step_profiles (
    run_id      TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    step        INTEGER NOT NULL,
    section     TEXT NOT NULL,
    ms          REAL NOT NULL,
    PRIMARY KEY (run_id, step, section)
);

-- ============================================================
-- KERNEL_BENCHMARKS — isolated kernel performance measurements
-- Source: benchmark script JSON output
-- Stores both the candidate AND its baseline for self-contained comparison
-- ============================================================

CREATE TABLE IF NOT EXISTS kernel_benchmarks (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at          TEXT DEFAULT (datetime('now')),
    git_hash            TEXT,
    gpu_name            TEXT,           -- e.g. 'H100', 'RTX_3070', 'RTX_4090'

    -- what we're benchmarking
    kernel_name         TEXT NOT NULL,  -- e.g. 'cross_entropy', 'rmsnorm_linear', 'rope', 'muon_ns'
    variant             TEXT NOT NULL,  -- e.g. 'pytorch_baseline', 'fused_ce_softcap_v1'

    -- input shape (JSON for flexibility across kernel types)
    -- e.g. {"batch": 512, "seq_len": 1024, "vocab_size": 1024, "dim": 512}
    input_json          TEXT,

    -- measurements
    latency_ms          REAL,
    bandwidth_gbs       REAL,           -- achieved memory bandwidth (GB/s)
    tflops              REAL,           -- achieved compute throughput

    -- baseline comparison (same kernel_name, baseline variant, same input)
    baseline_latency_ms REAL,
    speedup             REAL GENERATED ALWAYS AS (
                            baseline_latency_ms / NULLIF(latency_ms, 0)
                        ) STORED,

    notes               TEXT
);

CREATE INDEX IF NOT EXISTS idx_kb_kernel_variant
    ON kernel_benchmarks(kernel_name, variant);

CREATE INDEX IF NOT EXISTS idx_kb_gpu
    ON kernel_benchmarks(gpu_name);

-- ============================================================
-- VIEWS
-- ============================================================

-- Leaderboard: runs ranked by post-quant BPB (lower = better)
CREATE VIEW IF NOT EXISTS v_leaderboard AS
SELECT
    run_id,
    git_hash,
    vb_postquant,
    vb_sw_postquant,
    vb_prequant,
    step_avg_ms,
    steps,
    model_dim,
    num_layers,
    num_heads,
    bytes_total,
    early_stopped,
    hypothesis,
    notes,
    created_at
FROM runs
ORDER BY vb_postquant ASC;

-- Bottleneck finder: avg section time per run, with % of total step
CREATE VIEW IF NOT EXISTS v_bottlenecks AS
SELECT
    sp.run_id,
    sp.section,
    ROUND(AVG(sp.ms), 3) AS avg_ms,
    ROUND(
        100.0 * AVG(sp.ms) / SUM(AVG(sp.ms)) OVER (PARTITION BY sp.run_id),
        1
    ) AS pct_of_step,
    COUNT(*) AS n_samples
FROM step_profiles sp
GROUP BY sp.run_id, sp.section
ORDER BY sp.run_id, avg_ms DESC;

-- Kernel speedup comparison: best variant per kernel per GPU
CREATE VIEW IF NOT EXISTS v_kernel_speedups AS
SELECT
    kernel_name,
    variant,
    gpu_name,
    ROUND(AVG(latency_ms), 4) AS avg_latency_ms,
    ROUND(AVG(baseline_latency_ms), 4) AS avg_baseline_ms,
    ROUND(AVG(speedup), 2) AS avg_speedup,
    COUNT(*) AS n_measurements
FROM kernel_benchmarks
GROUP BY kernel_name, variant, gpu_name
ORDER BY kernel_name, avg_speedup DESC;

-- Run pairs: compare two runs side-by-side (use with WHERE clause)
-- Usage: SELECT * FROM v_run_compare WHERE run_id IN ('run_a', 'run_b')
CREATE VIEW IF NOT EXISTS v_run_compare AS
SELECT
    run_id,
    vb_postquant,
    step_avg_ms,
    steps,
    wall_ms,
    model_dim || 'd_' || num_layers || 'L_' || num_heads || 'H' AS arch,
    matrix_lr,
    scalar_lr,
    bytes_total,
    hypothesis,
    notes
FROM runs;

-- ============================================================
-- EXAMPLE AGENT QUERIES
-- ============================================================
--
-- "What's the current best BPB?"
--   SELECT vb_postquant, run_id, notes FROM v_leaderboard LIMIT 1;
--
-- "What section is the bottleneck in the latest run?"
--   SELECT section, avg_ms, pct_of_step
--   FROM v_bottlenecks
--   WHERE run_id = (SELECT run_id FROM runs ORDER BY created_at DESC LIMIT 1);
--
-- "How much did fused CE save end-to-end?"
--   SELECT r.run_id, r.step_avg_ms, r.vb_postquant, r.notes
--   FROM runs r
--   WHERE r.notes LIKE '%fused_ce%' OR r.notes LIKE '%baseline%'
--   ORDER BY r.created_at;
--
-- "What's the isolated speedup of fused CE on H100?"
--   SELECT variant, avg_latency_ms, avg_speedup
--   FROM v_kernel_speedups
--   WHERE kernel_name = 'cross_entropy' AND gpu_name = 'H100';
--
-- "Show BPB convergence curve for a run"
--   SELECT step, val_bpb, ref_bpb, status
--   FROM val_checkpoints
--   WHERE run_id = ?
--   ORDER BY step;
--
-- "Compare step profiles between two runs"
--   SELECT sp.run_id, sp.section, ROUND(AVG(sp.ms), 2) AS avg_ms
--   FROM step_profiles sp
--   WHERE sp.run_id IN (?, ?)
--   GROUP BY sp.run_id, sp.section
--   ORDER BY sp.section, sp.run_id;
--
-- "Which kernel is worth fusing next?" (combine bottleneck + benchmark data)
--   -- 1. Find biggest section:
--   SELECT section, pct_of_step FROM v_bottlenecks
--   WHERE run_id = ? ORDER BY pct_of_step DESC LIMIT 1;
--   -- 2. Check if a fused variant exists and what speedup it gets:
--   SELECT variant, avg_speedup FROM v_kernel_speedups
--   WHERE kernel_name = ? AND gpu_name = ?;
--   -- 3. Estimate wall-time savings:
--   --    saved_ms_per_step = section_avg_ms * (1 - 1/speedup)
--   --    total_steps_gained = saved_ms_per_step * steps / step_avg_ms
