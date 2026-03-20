"""
Ingest JSONL run logs and kernel benchmark results into SQLite.

Usage:
    # Ingest all JSONL logs in logs/
    uv run python ingest_runs.py

    # Ingest specific log files
    uv run python ingest_runs.py logs/abc123.jsonl logs/def456.jsonl

    # Ingest kernel benchmark JSON
    uv run python ingest_runs.py --benchmarks bench_fused_ce.json

    # Custom DB path
    uv run python ingest_runs.py --db my_runs.db
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


DB_DEFAULT = "runs.db"
SCHEMA_FILE = "schema.sql"
LOGS_DIR = "logs"


def ensure_schema(conn: sqlite3.Connection) -> None:
    schema = Path(SCHEMA_FILE).read_text(encoding="utf-8")
    conn.executescript(schema)


def ingest_jsonl(conn: sqlite3.Connection, jsonl_path: Path) -> dict[str, int]:
    """Ingest a single JSONL file. Returns counts of events ingested by type."""
    counts: dict[str, int] = {"config": 0, "val": 0, "train": 0, "profile": 0, "kernels": 0, "final": 0}
    config_data: dict = {}

    events = []
    for line in jsonl_path.read_text(encoding="utf-8").strip().splitlines():
        if not line.strip():
            continue
        events.append(json.loads(line))

    # --- Pass 1: collect config and insert the runs row (from "final" event) ---
    # The runs row must exist before val_checkpoints / step_profiles inserts
    # because those tables have FK constraints referencing runs(run_id).
    for ev in events:
        t = ev.get("t")

        if t == "config":
            config_data = ev
            counts["config"] += 1

        elif t == "final":
            run_id = ev.get("run_id", jsonl_path.stem)
            conn.execute(
                """INSERT OR REPLACE INTO runs
                   (run_id, git_hash, steps, wall_ms,
                    vl_prequant, vb_prequant, vl_postquant, vb_postquant,
                    vl_sw_postquant, vb_sw_postquant,
                    bytes_total, bytes_model, bytes_code, model_params,
                    early_stopped, early_stop_reason,
                    num_layers, model_dim, num_heads, num_kv_heads,
                    mlp_mult, vocab_size, train_seq_len, iterations,
                    matrix_lr, scalar_lr, embed_lr, muon_momentum,
                    logit_softcap, seed,
                    config_json, hypothesis, notes)
                   VALUES (?, ?, ?, ?,
                           ?, ?, ?, ?,
                           ?, ?,
                           ?, ?, ?, ?,
                           ?, ?,
                           ?, ?, ?, ?,
                           ?, ?, ?, ?,
                           ?, ?, ?, ?,
                           ?, ?,
                           ?, ?, ?)""",
                (
                    run_id,
                    ev.get("git"),
                    ev.get("steps"),
                    ev.get("wall_ms"),
                    ev.get("vl_prequant"),
                    ev.get("vb_prequant"),
                    ev.get("vl_postquant"),
                    ev.get("vb_postquant"),
                    ev.get("vl_sw_postquant"),
                    ev.get("vb_sw_postquant"),
                    ev.get("bytes_total"),
                    ev.get("bytes_model"),
                    ev.get("bytes_code"),
                    ev.get("model_params"),
                    int(ev.get("early_stopped", False)),
                    ev.get("early_stop_reason"),
                    # Config fields from the config event
                    config_data.get("num_layers"),
                    config_data.get("model_dim"),
                    config_data.get("num_heads"),
                    config_data.get("num_kv_heads"),
                    config_data.get("mlp_mult"),
                    config_data.get("vocab_size"),
                    config_data.get("train_seq_len"),
                    config_data.get("iterations"),
                    config_data.get("matrix_lr"),
                    config_data.get("scalar_lr"),
                    config_data.get("embed_lr"),
                    config_data.get("muon_momentum"),
                    config_data.get("logit_softcap"),
                    config_data.get("seed"),
                    json.dumps(config_data) if config_data else None,
                    config_data.get("hypothesis"),
                    config_data.get("notes"),
                ),
            )
            counts["final"] += 1

    # If no "final" event was found, the runs row doesn't exist — skip child inserts
    if counts["final"] == 0:
        return counts

    # --- Pass 2: insert val checkpoints and step profiles (FK parent now exists) ---
    run_id = config_data.get("run_id", jsonl_path.stem)

    for ev in events:
        t = ev.get("t")

        if t == "val":
            conn.execute(
                """INSERT OR REPLACE INTO val_checkpoints
                   (run_id, step, val_loss, val_bpb, ref_bpb, status, elapsed_ms)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    ev.get("s"),
                    ev.get("vl"),
                    ev.get("vb"),
                    ev.get("ref"),
                    ev.get("status"),
                    ev.get("ms"),
                ),
            )
            counts["val"] += 1

        elif t == "train":
            conn.execute(
                """INSERT OR REPLACE INTO train_checkpoints
                   (run_id, step, train_loss, lr, grad_norm, elapsed_ms)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    ev.get("s"),
                    ev.get("tl"),
                    ev.get("lr"),
                    ev.get("gn"),
                    ev.get("ms"),
                ),
            )
            counts["train"] += 1

        elif t == "profile":
            step = ev.get("s")
            for section in ("data", "fwd_bwd", "optimizer", "misc"):
                if section in ev:
                    conn.execute(
                        """INSERT OR REPLACE INTO step_profiles
                           (run_id, step, section, ms)
                           VALUES (?, ?, ?, ?)""",
                        (run_id, step, section, ev[section]),
                    )
            counts["profile"] += 1

        elif t == "kernels":
            for entry in ev.get("entries", []):
                conn.execute(
                    """INSERT OR REPLACE INTO kernel_profiles
                       (run_id, kernel_name, calls, cuda_time_us, cpu_time_us, pct_cuda)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        run_id,
                        entry.get("kernel_name"),
                        entry.get("calls"),
                        entry.get("cuda_time_us"),
                        entry.get("cpu_time_us"),
                        entry.get("pct_cuda"),
                    ),
                )
            counts["kernels"] += 1

    return counts


def ingest_benchmarks(conn: sqlite3.Connection, bench_path: Path) -> int:
    """Ingest kernel benchmark results from a JSON file.

    Expected format (single or array):
    {
        "kernel_name": "cross_entropy",
        "variant": "fused_ce_softcap_v1",
        "gpu_name": "H100",
        "git_hash": "abc1234",
        "input": {"batch": 512, "seq_len": 1024, "vocab_size": 1024},
        "latency_ms": 0.42,
        "bandwidth_gbs": 312.5,
        "tflops": null,
        "baseline_latency_ms": 0.97,
        "notes": "first working version"
    }
    """
    data = json.loads(bench_path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = [data]

    for entry in data:
        conn.execute(
            """INSERT INTO kernel_benchmarks
               (git_hash, gpu_name, kernel_name, variant,
                input_json, latency_ms, bandwidth_gbs, tflops,
                baseline_latency_ms, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                entry.get("git_hash"),
                entry.get("gpu_name"),
                entry["kernel_name"],
                entry["variant"],
                json.dumps(entry.get("input")) if entry.get("input") else None,
                entry.get("latency_ms"),
                entry.get("bandwidth_gbs"),
                entry.get("tflops"),
                entry.get("baseline_latency_ms"),
                entry.get("notes"),
            ),
        )
    return len(data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest JSONL logs and benchmarks into SQLite")
    parser.add_argument("files", nargs="*", help="JSONL log files to ingest (default: all in logs/)")
    parser.add_argument("--db", default=DB_DEFAULT, help=f"SQLite database path (default: {DB_DEFAULT})")
    parser.add_argument("--benchmarks", nargs="*", default=[], help="Kernel benchmark JSON files to ingest")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    ensure_schema(conn)

    # Ingest JSONL logs
    jsonl_files = [Path(f) for f in args.files] if args.files else sorted(Path(LOGS_DIR).glob("*.jsonl"))
    total = {"config": 0, "val": 0, "train": 0, "profile": 0, "kernels": 0, "final": 0}
    for jf in jsonl_files:
        if not jf.exists():
            print(f"skip: {jf} (not found)")
            continue
        counts = ingest_jsonl(conn, jf)
        for k, v in counts.items():
            total[k] += v
        print(f"ingested: {jf.name} — {counts}")

    # Ingest kernel benchmarks
    bench_count = 0
    for bf in args.benchmarks:
        bp = Path(bf)
        if not bp.exists():
            print(f"skip: {bp} (not found)")
            continue
        n = ingest_benchmarks(conn, bp)
        bench_count += n
        print(f"ingested: {bp.name} — {n} benchmark(s)")

    conn.commit()
    conn.close()

    print(f"\ntotal: {sum(total.values())} events from {len(jsonl_files)} log(s), {bench_count} benchmark(s)")
    print(f"  runs: {total['final']}, val: {total['val']}, profile: {total['profile']}, kernels: {total['kernels']}")
    print(f"db: {args.db}")


if __name__ == "__main__":
    main()
