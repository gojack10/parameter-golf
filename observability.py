"""
Training run observability: structured JSONL logging + early-stop state machine.

Keeps train_gpt.py clean — all monitoring logic lives here.
train_gpt.py calls ~5 methods; this module handles the rest.

StepProfiler: optional per-section CUDA event timing (PROFILE_SECTIONS=1).
"""

from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path

import torch


class StepProfiler:
    """CUDA event-based section timer. Near-zero overhead when disabled.

    Usage in training loop:
        profiler = StepProfiler(enabled=True)

        # each step:
        profiler.mark("data")           # start timing 'data' section
        x, y = loader.next_batch(...)
        profiler.mark("fwd_bwd")        # end 'data', start 'fwd_bwd'
        loss = model(x, y); loss.backward()
        profiler.mark("optimizer")
        for opt in optimizers: opt.step()
        profiler.mark("end")            # end last section

        # on log steps, harvest the timings:
        sections = profiler.collect()    # {"data": 1.2, "fwd_bwd": 15.3, ...}
    """

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self._marks: list[tuple[str, torch.cuda.Event]] = []

    def mark(self, name: str) -> None:
        if not self.enabled:
            return
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        self._marks.append((name, ev))

    def collect(self) -> dict[str, float] | None:
        """Sync and return section timings in ms. Returns None if disabled or <2 marks."""
        if not self.enabled or len(self._marks) < 2:
            self._marks.clear()
            return None
        torch.cuda.synchronize()
        sections: dict[str, float] = {}
        for i in range(len(self._marks) - 1):
            name = self._marks[i][0]
            start_ev = self._marks[i][1]
            end_ev = self._marks[i + 1][1]
            sections[name] = start_ev.elapsed_time(end_ev)
        self._marks.clear()
        return sections

    def reset(self) -> None:
        self._marks.clear()


class KernelProfiler:
    """torch.profiler-based CUDA kernel profiling for a window of training steps.

    Captures detailed kernel-level timing (every CUDA kernel launch) for a
    configurable window of steps after compile warmup. Exports:
    1. key_averages() summary -> JSONL "kernels" event -> SQLite kernel_profiles table
    2. Chrome trace JSON -> logs/{run_id}_kernels.json (for visual inspection)

    Usage in training loop:
        kp = KernelProfiler(enabled=True, run_id="abc", start_step=20, num_steps=5)

        for step in range(total_steps):
            kp.step_begin(step)
            # ... training step ...
            kp.step_end(step)

        # After training, get results:
        summary = kp.finish()  # returns list of kernel dicts, exports chrome trace
    """

    def __init__(
        self,
        enabled: bool = False,
        run_id: str = "",
        log_dir: str = "logs",
        start_step: int = 20,
        num_steps: int = 5,
    ):
        self.enabled = enabled
        self.run_id = run_id
        self.log_dir = log_dir
        self.start_step = start_step
        self.end_step = start_step + num_steps
        self._profiler = None
        self._active = False
        self._finished = False
        self._summary: list[dict] = []

    def step_begin(self, step: int) -> None:
        """Call at the start of each training step."""
        if not self.enabled or self._finished:
            return
        if step == self.start_step:
            # Start profiling
            from torch.profiler import profile, ProfilerActivity
            self._profiler = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=False,
                with_stack=False,
            )
            self._profiler.__enter__()
            self._active = True

    def step_end(self, step: int) -> None:
        """Call at the end of each training step."""
        if not self.enabled or not self._active or self._finished:
            return
        if step >= self.end_step - 1:
            # Stop profiling and collect results
            self._profiler.__exit__(None, None, None)
            self._active = False
            self._finished = True
            self._collect()

    def _collect(self) -> None:
        """Extract key_averages and export chrome trace."""
        if self._profiler is None:
            return

        # Export chrome trace
        trace_path = os.path.join(self.log_dir, f"{self.run_id}_kernels.json")
        try:
            self._profiler.export_chrome_trace(trace_path)
        except Exception:
            pass  # Best-effort

        # Extract key_averages
        # PyTorch 2.x uses device_time_total; older uses cuda_time_total
        def _cuda_us(e: object) -> float:
            for attr in ("cuda_time_total", "device_time_total", "self_cuda_time_total", "self_device_time_total"):
                v = getattr(e, attr, None)
                if v is not None and v > 0:
                    return float(v)
            return 0.0

        ka = self._profiler.key_averages()
        total_cuda = sum(_cuda_us(e) for e in ka)

        for entry in sorted(ka, key=_cuda_us, reverse=True):
            cuda_us = _cuda_us(entry)
            if cuda_us <= 0:
                continue
            self._summary.append({
                "kernel_name": entry.key,
                "calls": entry.count,
                "cuda_time_us": cuda_us,
                "cpu_time_us": getattr(entry, "cpu_time_total", 0),
                "pct_cuda": round(100.0 * cuda_us / max(total_cuda, 1), 4),
            })

    def finish(self) -> list[dict]:
        """Return kernel summary. Safe to call multiple times."""
        # If profiler is still active (e.g. training ended early), stop it
        if self._active and self._profiler is not None:
            self._profiler.__exit__(None, None, None)
            self._active = False
            self._finished = True
            self._collect()
        return self._summary


class RunMonitor:
    """Tracks a training run: emits JSONL, checks early-stop conditions."""

    def __init__(
        self,
        run_id: str,
        log_dir: str = "logs",
        early_stop: bool = True,
        ref_file: str = "",
        tolerance: float = 0.05,
        min_step: int = 500,
        patience: int = 3,
    ):
        self.run_id = run_id
        self.early_stop = early_stop
        self.tolerance = tolerance
        self.min_step = min_step
        self.patience = patience

        # JSONL output
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)
        self.jsonl_path = log_path / f"{run_id}.jsonl"

        # Reference curve
        self.ref_curve: dict[int, float] = {}
        self._ref_steps: list[int] = []
        self.ref_file = ref_file
        if ref_file:
            p = Path(ref_file)
            if p.exists():
                self.ref_curve = {int(k): float(v) for k, v in json.loads(p.read_text()).items()}
                self._ref_steps = sorted(self.ref_curve)

        # Early-stop state
        self._fail_count = 0
        self._regress_count = 0
        self._prev_bpb = float("inf")
        self.stop_reason: str | None = None

        # Probe metadata (from env vars, set via run_probe.sh KEY=VALUE)
        self.hypothesis = os.environ.get("PROBE_HYPOTHESIS", "")
        self.probe_notes = os.environ.get("PROBE_NOTES", "")

        # Git hash (best effort)
        self.git_hash = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=False,
        ).stdout.strip()

    # ------------------------------------------------------------------
    # JSONL emission
    # ------------------------------------------------------------------

    def _write(self, obj: dict) -> None:
        def _clean(v: object) -> object:
            if isinstance(v, float) and not math.isfinite(v):
                return None
            return v
        cleaned = {k: _clean(v) for k, v in obj.items()}
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(cleaned, separators=(",", ":")) + "\n")

    def emit_config(self, *, model_params: int, **kwargs: object) -> None:
        self._write({
            "t": "config", "run_id": self.run_id, "git": self.git_hash,
            "early_stop": self.early_stop, "ref_file": self.ref_file,
            "tolerance": self.tolerance, "model_params": model_params,
            "hypothesis": self.hypothesis or None,
            "notes": self.probe_notes or None,
            **kwargs,
        })

    # ------------------------------------------------------------------
    # Reference curve interpolation
    # ------------------------------------------------------------------

    def get_ref_bpb(self, step: int) -> float | None:
        if not self._ref_steps:
            return None
        if step <= self._ref_steps[0]:
            return self.ref_curve[self._ref_steps[0]]
        if step >= self._ref_steps[-1]:
            return self.ref_curve[self._ref_steps[-1]]
        for i in range(len(self._ref_steps) - 1):
            lo, hi = self._ref_steps[i], self._ref_steps[i + 1]
            if lo <= step <= hi:
                frac = (step - lo) / (hi - lo)
                return self.ref_curve[lo] * (1 - frac) + self.ref_curve[hi] * frac
        return None

    # ------------------------------------------------------------------
    # Early-stop check  (call at each val checkpoint)
    # ------------------------------------------------------------------

    def check_val(
        self,
        step: int,
        val_loss: float,
        val_bpb: float,
        elapsed_ms: float,
        is_last_step: bool = False,
    ) -> str:
        """Returns status string. Sets self.stop_reason on KILL.

        Statuses: ON_TRACK, BEATING, BEHIND, KILL_NAN, KILL_BEHIND, KILL_REGRESS
        """
        ref = self.get_ref_bpb(step)
        status = "ON_TRACK"

        if self.early_stop and not is_last_step and self.stop_reason is None:
            # NaN/Inf — immediate kill regardless of min_step
            if not math.isfinite(val_bpb):
                status = "KILL_NAN"
                self.stop_reason = "NAN_INF"
            elif step >= self.min_step:
                # Behind reference check
                if ref is not None:
                    if val_bpb > ref * (1 + self.tolerance):
                        self._fail_count += 1
                        status = "BEHIND"
                        if self._fail_count >= self.patience:
                            status = "KILL_BEHIND"
                            self.stop_reason = f"BEHIND_REF x{self._fail_count}"
                    else:
                        self._fail_count = 0
                        status = "BEATING" if val_bpb < ref else "ON_TRACK"

                # Regression check (val_bpb getting worse)
                if math.isfinite(val_bpb) and val_bpb > self._prev_bpb:
                    self._regress_count += 1
                    if self._regress_count >= self.patience and self.stop_reason is None:
                        status = "KILL_REGRESS"
                        self.stop_reason = f"REGRESSED x{self._regress_count}"
                else:
                    self._regress_count = 0

        # Always update prev (outside early-stop gate, so it's ready when we cross min_step)
        if math.isfinite(val_bpb):
            self._prev_bpb = val_bpb

        # Emit JSONL
        self._write({
            "t": "val", "s": step,
            "vl": round(val_loss, 6), "vb": round(val_bpb, 6),
            "ref": round(ref, 6) if ref is not None else None,
            "status": status, "ms": round(elapsed_ms),
        })
        return status

    # ------------------------------------------------------------------
    # Train step logging  (call at each logged train step)
    # ------------------------------------------------------------------

    def log_train(
        self, step: int, train_loss: float, elapsed_ms: float,
        lr_scale: float = 1.0, grad_norm: float | None = None,
    ) -> None:
        tl = train_loss if math.isfinite(train_loss) else None
        ev: dict = {
            "t": "train", "s": step,
            "tl": round(tl, 6) if tl is not None else None,
            "lr": round(lr_scale, 6),
            "ms": round(elapsed_ms),
            "avg": round(elapsed_ms / max(step, 1), 2),
        }
        if grad_norm is not None:
            ev["gn"] = round(grad_norm, 6)
        self._write(ev)

    # ------------------------------------------------------------------
    # Step profile logging  (call on logged steps when profiling enabled)
    # ------------------------------------------------------------------

    def log_profile(self, step: int, sections: dict[str, float]) -> None:
        """Emit a profile event with per-section timing in ms."""
        self._write({
            "t": "profile", "s": step,
            **{k: round(v, 3) for k, v in sections.items()},
            "total": round(sum(sections.values()), 3),
        })

    # ------------------------------------------------------------------
    # Kernel profile logging  (call after KernelProfiler.finish())
    # ------------------------------------------------------------------

    def log_kernels(self, kernels: list[dict]) -> None:
        """Emit kernel profile summary as a JSONL event."""
        if not kernels:
            return
        self._write({
            "t": "kernels",
            "run_id": self.run_id,
            "entries": kernels,
        })

    # ------------------------------------------------------------------
    # Final summary  (call once after serialization)
    # ------------------------------------------------------------------

    def emit_final(
        self,
        *,
        step: int,
        wall_ms: float,
        val_loss: float,
        val_bpb: float,
        q_val_loss: float,
        q_val_bpb: float,
        sw_val_loss: float | None = None,
        sw_val_bpb: float | None = None,
        bytes_total: int,
        bytes_model: int,
        bytes_code: int,
        model_params: int,
    ) -> None:
        ev: dict = {
            "t": "final", "run_id": self.run_id, "git": self.git_hash,
            "steps": step, "wall_ms": round(wall_ms),
            "vl_prequant": round(val_loss, 8), "vb_prequant": round(val_bpb, 8),
            "vl_postquant": round(q_val_loss, 8), "vb_postquant": round(q_val_bpb, 8),
            "bytes_total": bytes_total, "bytes_model": bytes_model, "bytes_code": bytes_code,
            "early_stopped": self.stop_reason is not None,
            "early_stop_reason": self.stop_reason,
            "model_params": model_params,
        }
        if sw_val_loss is not None:
            ev["vl_sw_postquant"] = round(sw_val_loss, 8)
        if sw_val_bpb is not None:
            ev["vb_sw_postquant"] = round(sw_val_bpb, 8)
        self._write(ev)
        self._auto_ingest()

    # ------------------------------------------------------------------
    # Auto-ingest into SQLite  (called at end of emit_final)
    # ------------------------------------------------------------------

    def _auto_ingest(self) -> None:
        """Ingest this run's JSONL into runs.db. Best-effort — never fails the run."""
        try:
            from ingest_runs import ensure_schema, ingest_jsonl
            import sqlite3
            conn = sqlite3.connect("runs.db")
            ensure_schema(conn)
            ingest_jsonl(conn, self.jsonl_path)
            conn.commit()
            conn.close()
        except Exception:
            pass  # DB ingest is nice-to-have, never worth crashing a run

    # ------------------------------------------------------------------
    # Status line for human-readable log
    # ------------------------------------------------------------------

    def status_line(self, step: int, val_bpb: float) -> str:
        """One-line status for log0() after a val check triggered early stop."""
        ref = self.get_ref_bpb(step)
        ref_str = f"{ref:.4f}" if ref is not None else "N/A"
        return (
            f"EARLY_STOP reason:{self.stop_reason} step:{step} "
            f"val_bpb:{val_bpb:.4f} ref:{ref_str}"
        )
