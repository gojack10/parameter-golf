"""Checkpoint save / load / eval-only for Parameter Golf.

Usage:
    # Save after training (reads final_model.pt or final_model.int8.ptz)
    uv run python checkpoints.py save --run-id my_run

    # Save from a specific file
    uv run python checkpoints.py save --run-id my_run --source final_model.int8.ptz

    # Eval-only on a saved checkpoint
    uv run python checkpoints.py eval --checkpoint checkpoints/my_run.pt --eval-stride 64

    # Eval with TTT
    uv run python checkpoints.py eval --checkpoint checkpoints/my_run.pt --eval-stride 256 \
        --ttt --ttt-lr 0.5 --ttt-min-doc-len 2048 --ttt-split-frac 0.5
"""
from __future__ import annotations

import argparse
import io
import math
import os
import sys
import time
import zlib

import torch

# ---------------------------------------------------------------------------
# Lazy imports from train_gpt — avoids pulling in CUDA / dist at import time
# ---------------------------------------------------------------------------

def _import_train_gpt():
    """Import symbols from train_gpt.py in the same directory."""
    import train_gpt as tg
    return tg


# ──────────────────────────────────────────────────────────────────────────────
# Save
# ──────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    run_id: str,
    source: str | None = None,
    out_dir: str = "checkpoints",
) -> str:
    """Save a checkpoint from a trained model artifact.

    Looks for model files in this priority order:
        1. Explicit *source* path (if given)
        2. final_model.int8.ptz  (quantised + compressed — smallest)
        3. final_model.pt        (full-precision state_dict)

    The checkpoint bundles the model weights together with the GPT constructor
    kwargs needed to reconstruct the architecture, so the checkpoint is
    fully self-contained.
    """
    tg = _import_train_gpt()

    # --- Resolve source file ---
    if source is None:
        for candidate in ("final_model.int8.ptz", "final_model.pt"):
            if os.path.isfile(candidate):
                source = candidate
                break
        if source is None:
            raise FileNotFoundError(
                "No model file found.  Run training first, or pass --source."
            )

    print(f"Loading weights from {source}")

    # --- Load weights ---
    if source.endswith(".ptz"):
        raw = open(source, "rb").read()
        try:
            import zstandard as zstd
            decompressed = zstd.ZstdDecompressor().decompress(raw)
        except Exception:
            decompressed = zlib.decompress(raw)
        quant_obj = torch.load(io.BytesIO(decompressed), map_location="cpu", weights_only=False)
        state_dict = tg.dequantize_state_dict_int8(quant_obj)
        quant_format = quant_obj.get("__quant_format__", "int8_clean_per_row_v1")
    elif source.endswith(".pt"):
        state_dict = torch.load(source, map_location="cpu", weights_only=False)
        quant_format = None
    else:
        raise ValueError(f"Unrecognised source extension: {source}")

    # --- Read current config (from env or defaults) ---
    args = tg.Hyperparameters()
    model_kwargs = dict(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        smeargate_enabled=args.smeargate_enabled,
        mtp_num_heads=0,  # MTP heads stripped from export
        mtp_loss_weight=0.0,
    )
    eval_defaults = dict(
        train_seq_len=args.train_seq_len,
        eval_stride=args.eval_stride,
        ttt_enabled=args.ttt_enabled,
        ttt_lr=args.ttt_lr,
        ttt_min_doc_len=args.ttt_min_doc_len,
        ttt_split_frac=args.ttt_split_frac,
        tokenizer_path=args.tokenizer_path,
        data_path=args.data_path,
    )

    # --- Bundle and write ---
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, f"{run_id}.pt")
    torch.save(
        dict(
            model_kwargs=model_kwargs,
            eval_defaults=eval_defaults,
            state_dict=state_dict,
            source=source,
            quant_format=quant_format,
        ),
        ckpt_path,
    )
    size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
    print(f"Checkpoint saved: {ckpt_path} ({size_mb:.1f} MiB)")
    return ckpt_path


# ──────────────────────────────────────────────────────────────────────────────
# Load
# ──────────────────────────────────────────────────────────────────────────────

def load_checkpoint(
    ckpt_path: str,
    device: str | torch.device = "cuda",
) -> tuple:
    """Load a checkpoint and reconstruct the model.

    Returns (model, model_kwargs, eval_defaults).
    """
    tg = _import_train_gpt()

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_kwargs = ckpt["model_kwargs"]
    eval_defaults = ckpt["eval_defaults"]
    state_dict = ckpt["state_dict"]

    model = tg.GPT(**model_kwargs)
    tg.load_export_state_dict_into_model(model, state_dict)

    model = model.to(device).bfloat16()
    for module in model.modules():
        if isinstance(module, tg.CastedLinear):
            module.float()
    tg.restore_low_dim_params_to_fp32(model)

    model.eval()
    return model, model_kwargs, eval_defaults


# ──────────────────────────────────────────────────────────────────────────────
# Eval
# ──────────────────────────────────────────────────────────────────────────────

def run_eval(
    ckpt_path: str,
    eval_stride: int | None = None,
    ttt: bool | None = None,
    ttt_lr: float | None = None,
    ttt_min_doc_len: int | None = None,
    ttt_split_frac: float | None = None,
    seq_len: int | None = None,
    batch_size: int = 32,
    device: str = "cuda",
) -> dict:
    """Load checkpoint, run eval, print results. Returns metrics dict."""
    tg = _import_train_gpt()
    import sentencepiece as spm

    print(f"Loading checkpoint: {ckpt_path}")
    model, model_kwargs, eval_defaults = load_checkpoint(ckpt_path, device=device)

    # Merge explicit args with checkpoint defaults
    seq_len = seq_len or eval_defaults["train_seq_len"]
    stride = eval_stride if eval_stride is not None else eval_defaults.get("eval_stride", 256)
    use_ttt = ttt if ttt is not None else eval_defaults.get("ttt_enabled", False)
    lr = ttt_lr if ttt_lr is not None else eval_defaults.get("ttt_lr", 0.5)
    min_doc = ttt_min_doc_len if ttt_min_doc_len is not None else eval_defaults.get("ttt_min_doc_len", 2048)
    split_frac = ttt_split_frac if ttt_split_frac is not None else eval_defaults.get("ttt_split_frac", 0.5)

    tokenizer_path = eval_defaults.get("tokenizer_path", "./data/tokenizers/fineweb_1024_bpe.model")
    data_path = eval_defaults.get("data_path", "./data/datasets/fineweb10B_sp1024")
    val_pattern = os.path.join(data_path, "fineweb_val_*.bin")

    print(f"Config: seq_len={seq_len} stride={stride} ttt={use_ttt}")
    if use_ttt:
        print(f"  TTT: lr={lr} min_doc_len={min_doc} split_frac={split_frac}")

    # Load tokenizer + LUTs
    sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
    dev = torch.device(device)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = tg.build_sentencepiece_luts(
        sp, model_kwargs["vocab_size"], dev,
    )

    # Load val data
    val_tokens = tg.load_validation_tokens(val_pattern, seq_len)
    print(f"Val tokens: {val_tokens.numel() - 1}")

    # Run eval
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    if stride > 0:
        if use_ttt:
            val_loss, val_bpb, n_ttt = tg.eval_val_ttt(
                model, seq_len, stride, dev, val_tokens,
                base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                ttt_lr=lr, ttt_min_doc_len=min_doc, ttt_split_frac=split_frac,
                sw_batch_size=batch_size,
            )
        else:
            val_loss, val_bpb = tg.eval_val_sliding(
                model, seq_len, stride, dev, val_tokens,
                base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                batch_size=batch_size,
            )
            n_ttt = 0
    else:
        # Non-sliding eval (stride=0 means standard chunked eval)
        args = tg.Hyperparameters()
        val_loss, val_bpb = tg.eval_val(
            args, model, 0, 1, dev, 1,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        )
        n_ttt = 0

    torch.cuda.synchronize()
    elapsed_ms = 1000.0 * (time.perf_counter() - t0)

    results = dict(
        val_loss=val_loss,
        val_bpb=val_bpb,
        eval_time_ms=elapsed_ms,
        stride=stride,
        ttt=use_ttt,
    )
    if use_ttt:
        results["n_ttt_docs"] = n_ttt

    print(f"\nResults:")
    print(f"  val_loss:  {val_loss:.8f}")
    print(f"  val_bpb:   {val_bpb:.8f}")
    print(f"  eval_time: {elapsed_ms:.0f}ms")
    if use_ttt:
        print(f"  ttt_docs:  {n_ttt}")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Checkpoint save/load/eval for Parameter Golf")
    sub = parser.add_subparsers(dest="command", required=True)

    # --- save ---
    p_save = sub.add_parser("save", help="Save a checkpoint from trained model artifacts")
    p_save.add_argument("--run-id", required=True, help="Unique run identifier")
    p_save.add_argument("--source", default=None, help="Model file (auto-detects final_model.int8.ptz or final_model.pt)")
    p_save.add_argument("--out-dir", default="checkpoints", help="Output directory")

    # --- eval ---
    p_eval = sub.add_parser("eval", help="Eval-only on a saved checkpoint")
    p_eval.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file")
    p_eval.add_argument("--eval-stride", type=int, default=None, help="Sliding window stride (0 = standard eval)")
    p_eval.add_argument("--seq-len", type=int, default=None, help="Sequence length (default: from checkpoint)")
    p_eval.add_argument("--batch-size", type=int, default=32, help="Sliding window batch size")
    p_eval.add_argument("--ttt", action="store_true", default=None, help="Enable test-time training")
    p_eval.add_argument("--no-ttt", action="store_false", dest="ttt", help="Disable test-time training")
    p_eval.add_argument("--ttt-lr", type=float, default=None)
    p_eval.add_argument("--ttt-min-doc-len", type=int, default=None)
    p_eval.add_argument("--ttt-split-frac", type=float, default=None)
    p_eval.add_argument("--device", default="cuda")

    # --- info ---
    p_info = sub.add_parser("info", help="Print checkpoint metadata")
    p_info.add_argument("--checkpoint", required=True)

    parsed = parser.parse_args()

    if parsed.command == "save":
        save_checkpoint(parsed.run_id, source=parsed.source, out_dir=parsed.out_dir)

    elif parsed.command == "eval":
        run_eval(
            parsed.checkpoint,
            eval_stride=parsed.eval_stride,
            ttt=parsed.ttt,
            ttt_lr=parsed.ttt_lr,
            ttt_min_doc_len=parsed.ttt_min_doc_len,
            ttt_split_frac=parsed.ttt_split_frac,
            seq_len=parsed.seq_len,
            batch_size=parsed.batch_size,
            device=parsed.device,
        )

    elif parsed.command == "info":
        ckpt = torch.load(parsed.checkpoint, map_location="cpu", weights_only=False)
        print("Model kwargs:")
        for k, v in ckpt["model_kwargs"].items():
            print(f"  {k}: {v}")
        print("\nEval defaults:")
        for k, v in ckpt["eval_defaults"].items():
            print(f"  {k}: {v}")
        print(f"\nSource: {ckpt.get('source', 'unknown')}")
        print(f"Quant format: {ckpt.get('quant_format', 'none')}")
        n_params = sum(p.numel() for p in ckpt["state_dict"].values())
        print(f"Parameters: {n_params:,}")


if __name__ == "__main__":
    main()
