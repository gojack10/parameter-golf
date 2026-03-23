"""Int5 roundtrip quality probe: requantize int6→int5, dequantize, eval BPB.

Measures the quality cost of int5 vs int6 quantization without training.
Also tests mixed strategies (int5 for early layers, int6 for late).

Usage: uv run python int5_roundtrip_eval.py [--device cuda|cpu]
"""

import io
import os
import sys
import time
from pathlib import Path

import torch
import zstandard

# Import model and eval from train_gpt
sys.path.insert(0, str(Path(__file__).parent))

# We need to set env vars before importing train_gpt
# Load config.yml defaults
import yaml
with open("config.yml") as f:
    cfg = yaml.safe_load(f)
for k, v in cfg.items():
    if k.isupper() and k not in os.environ:
        os.environ[k] = str(v)
# Override for local eval
os.environ.setdefault("EVAL_STRIDE", "0")  # no sliding window (too slow)
os.environ.setdefault("TORCH_COMPILE", "0")  # skip compile for speed
os.environ["VAL_BATCH_SIZE"] = "32768"  # small enough for 3070 8GB


def requantize_int5_tensor(q_int8: torch.Tensor, scale: torch.Tensor):
    """Requantize from int6 range [-32..31] to int5 range [-16..15]."""
    new_scale = scale.float() * 2.0
    q5 = torch.clamp(torch.round(q_int8.float() / 2.0), -16, 15).to(torch.int8)
    return q5, new_scale.to(scale.dtype)


def dequantize_q_s(q, s, orig_dtype):
    """Dequantize quantized values with scale."""
    if s.ndim > 0:
        return (q.float() * s.float().view(q.shape[0], *([1] * (q.ndim - 1)))).to(orig_dtype)
    return (q.float() * float(s.item())).to(orig_dtype)


def load_and_requantize(artifact_path, template_sd, strategy="all_int5", int5_layers=None):
    """Load artifact, requantize specified params to int5, dequantize to state dict.

    Strategies:
        "all_int5": all int6 params → int5
        "all_int6": no change (baseline)
        "mixed": early layers → int5, late layers → int6
        "mlp_only_int5": only MLP weights to int5, attention stays int6
    """
    with open(artifact_path, "rb") as f:
        blob = f.read()
    raw = zstandard.ZstdDecompressor().decompress(blob)
    state = torch.load(io.BytesIO(raw), map_location="cpu")
    w, m = state["w"], state["m"]

    out = {}
    int5_count = 0
    int6_count = 0

    for name, orig in template_sd.items():
        info = m.get(name)
        if info is None:
            continue
        orig_dtype = orig.dtype

        if info in ("passthrough", "passthrough_ctrl", "passthrough_fp16"):
            t = w[name]
            if t.dtype == torch.float16 and orig_dtype in (torch.float32, torch.bfloat16):
                t = t.to(orig_dtype)
            out[name] = t
            continue

        q, s = w[name + ".q"], w[name + ".scale"]

        if isinstance(info, dict) and info["type"] == "int6":
            # Decide whether to requantize this tensor to int5
            do_int5 = False
            if strategy == "all_int5":
                do_int5 = True
            elif strategy == "all_int6":
                do_int5 = False
            elif strategy == "mixed" and int5_layers is not None:
                # Check if this tensor belongs to an int5 layer
                for layer_idx in int5_layers:
                    if f"blocks.{layer_idx}." in name:
                        do_int5 = True
                        break
            elif strategy == "mlp_only_int5":
                do_int5 = "mlp" in name

            if do_int5:
                q, s = requantize_int5_tensor(q, s)
                int5_count += q.numel()
            else:
                int6_count += q.numel()

        out[name] = dequantize_q_s(q, s, orig_dtype)

    return out, int5_count, int6_count


def main():
    device = "cpu"
    for arg in sys.argv[1:]:
        if arg.startswith("--device"):
            device = arg.split("=")[1] if "=" in arg else sys.argv[sys.argv.index(arg) + 1]

    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    # Import after env vars set
    from train_gpt import (
        GPT, CastedLinear, Hyperparameters, eval_val,
        restore_low_dim_params_to_fp32,
        load_validation_tokens, build_sentencepiece_luts,
    )
    import sentencepiece as spm

    args = Hyperparameters()
    artifact_path = "final_model.int6.ptz"

    print(f"Device: {device}")
    print(f"Loading model architecture: {args.num_layers}L, dim={args.model_dim}, vocab={args.vocab_size}")

    # Build template model for state dict shapes
    template_model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        mtp_num_heads=0, mtp_loss_weight=0.0,
        bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n, rope_dims=args.rope_dims, ln_scale=args.ln_scale,
    )
    template_sd = template_model.state_dict()

    # Load validation data
    print("Loading validation data...")
    val_files = str(Path(args.data_path) / "fineweb_val_*.bin")
    val_tokens = load_validation_tokens(val_files, args.train_seq_len)
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    torch_device = torch.device(device)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, torch_device
    )
    effective_eval_seq_len = min(args.train_seq_len, 2048)

    # Define strategies to test
    strategies = [
        ("all_int6", "Baseline (int6, current)", {}),
        ("all_int5", "All int5 (worst case quality)", {}),
        ("mlp_only_int5", "MLP→int5, Attention→int6", {}),
        ("mixed", "Layers 0-5→int5, 6-10→int6", {"int5_layers": list(range(6))}),
        ("mixed", "Layers 0-3→int5, 4-10→int6", {"int5_layers": list(range(4))}),
        ("mixed", "Layers 0-8→int5, 9-10→int6", {"int5_layers": list(range(9))}),
    ]

    results = []
    for strategy, label, kwargs in strategies:
        print(f"\n{'='*60}")
        print(f"Strategy: {label}")
        print(f"{'='*60}")

        sd, int5_n, int6_n = load_and_requantize(
            artifact_path, template_sd, strategy=strategy, **kwargs
        )
        print(f"  int5 params: {int5_n:,}  |  int6 params: {int6_n:,}")

        model = GPT(
            vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
            num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
            tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
            logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
            mtp_num_heads=0, mtp_loss_weight=0.0,
            bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
            xsa_last_n=args.xsa_last_n, rope_dims=args.rope_dims, ln_scale=args.ln_scale,
        ).to(device).bfloat16()
        for m in model.modules():
            if isinstance(m, CastedLinear):
                m.float()
        restore_low_dim_params_to_fp32(model)
        model.load_state_dict(sd, strict=True)

        t0 = time.perf_counter()
        val_loss, val_bpb = eval_val(
            args, model, rank=0, world_size=1, device=device,
            grad_accum_steps=1, val_tokens=val_tokens,
            base_bytes_lut=base_bytes_lut,
            has_leading_space_lut=has_leading_space_lut,
            is_boundary_token_lut=is_boundary_token_lut,
            eval_seq_len=effective_eval_seq_len,
        )
        elapsed = time.perf_counter() - t0
        print(f"  val_loss: {val_loss:.6f}  val_bpb: {val_bpb:.6f}  ({elapsed:.1f}s)")
        results.append((label, val_bpb, int5_n, int6_n))

        del model, sd
        if device == "cuda":
            torch.cuda.empty_cache()

    # Summary
    baseline_bpb = results[0][1]
    print(f"\n{'='*80}")
    print(f"{'Strategy':<45} {'BPB':>10} {'Delta':>10} {'Int5 %':>8}")
    print(f"{'='*80}")
    for label, bpb, n5, n6 in results:
        delta = bpb - baseline_bpb
        pct5 = 100 * n5 / (n5 + n6) if (n5 + n6) > 0 else 0
        sign = "+" if delta > 0 else ""
        print(f"{label:<45} {bpb:>10.6f} {sign}{delta:>9.6f} {pct5:>7.1f}%")


if __name__ == "__main__":
    main()
