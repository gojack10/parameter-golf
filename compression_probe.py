"""Compression probe: measure bytes/param under different packing strategies.

Takes an existing final_model.int6.ptz and tries:
1. Baseline: current int8-container + torch.save + zstd-22
2. Packed int6 bitstream (no torch.save) + zstd-22
3. Packed int6 bitstream raw (no zstd) — floor measurement
4. Current tensors but no torch.save (raw concat) + zstd-22
5. Int5 quantization + packed + zstd-22 (quality unknown, size only)
6. Pruning: zero smallest 10% of weights, then current format + zstd-22

Usage: uv run python compression_probe.py [path_to_model]
"""

import io
import struct
import sys
import time

import numpy as np
import torch
import zstandard


def load_current_artifact(path: str):
    """Load existing int6 artifact and return raw components."""
    with open(path, "rb") as f:
        blob = f.read()
    raw = zstandard.ZstdDecompressor().decompress(blob)
    state = torch.load(io.BytesIO(raw), map_location="cpu")
    return state["w"], state["m"], len(blob), len(raw)


def pack_int6(values: np.ndarray) -> bytes:
    """Pack int6 values [-32..31] into a dense bitstream. 4 values = 3 bytes."""
    # Shift to unsigned [0..63]
    u = (values.astype(np.int8) + 32).astype(np.uint8)
    n = len(u)
    # Pad to multiple of 4
    pad = (4 - n % 4) % 4
    if pad:
        u = np.concatenate([u, np.zeros(pad, dtype=np.uint8)])
    # Pack 4 values into 3 bytes:
    # byte0 = v0[5:0] << 2 | v1[5:4]
    # byte1 = v1[3:0] << 4 | v2[5:2]
    # byte2 = v2[1:0] << 6 | v3[5:0]
    u = u.reshape(-1, 4)
    v0, v1, v2, v3 = u[:, 0], u[:, 1], u[:, 2], u[:, 3]
    b0 = (v0 << 2) | (v1 >> 4)
    b1 = ((v1 & 0x0F) << 4) | (v2 >> 2)
    b2 = ((v2 & 0x03) << 6) | v3
    packed = np.column_stack([b0, b1, b2]).astype(np.uint8).tobytes()
    return packed


def pack_int5(values: np.ndarray) -> bytes:
    """Pack int5 values [-16..15] into a dense bitstream. 8 values = 5 bytes."""
    u = (values.astype(np.int8).clip(-16, 15) + 16).astype(np.uint8)
    n = len(u)
    pad = (8 - n % 8) % 8
    if pad:
        u = np.concatenate([u, np.zeros(pad, dtype=np.uint8)])
    # Pack 8 values into 5 bytes using bit manipulation
    u = u.reshape(-1, 8)
    result = bytearray()
    for row in u:
        # 8 × 5 bits = 40 bits = 5 bytes
        bits = 0
        for i in range(8):
            bits = (bits << 5) | (int(row[i]) & 0x1F)
        result.extend(bits.to_bytes(5, "big"))
    return bytes(result)


def requantize_int5(q_int8: torch.Tensor, scale: torch.Tensor):
    """Re-quantize int6 values (in int8 container) down to int5 range [-16..15].
    Returns new q values and adjusted scale."""
    # Current: val ≈ q * scale, q in [-32..31]
    # New: val ≈ q5 * scale5, q5 in [-16..15]
    # scale5 = scale * 32/16 = scale * 2
    new_scale = scale.float() * 2.0
    q5 = torch.clamp(torch.round(q_int8.float() / 2.0), -16, 15).to(torch.int8)
    return q5, new_scale.to(scale.dtype)


def strategy_baseline(w, m, compressed_size, raw_size):
    """Strategy 0: Current format (just report existing numbers)."""
    return {
        "name": "Baseline (int8 container + torch.save + zstd-22)",
        "compressed": compressed_size,
        "raw": raw_size,
    }


def strategy_packed_int6_zstd(w, m):
    """Strategy 1: Pack int6 values into 6-bit bitstream, minimal header, zstd-22."""
    buf = bytearray()
    # Simple header: magic + param count for each tensor
    buf.extend(b"INT6")  # magic
    tensor_data = bytearray()
    header_entries = []

    for name, info in m.items():
        if isinstance(info, dict) and info["type"] == "int6":
            q = w[name + ".q"]
            s = w[name + ".scale"]
            q_np = q.numpy().flatten()
            packed = pack_int6(q_np)
            scale_bytes = s.numpy().tobytes()
            header_entries.append((name, q.shape, len(packed), len(scale_bytes)))
            tensor_data.extend(packed)
            tensor_data.extend(scale_bytes)
        elif isinstance(info, dict) and info["type"] == "int8":
            q = w[name + ".q"]
            s = w[name + ".scale"]
            tensor_data.extend(q.numpy().tobytes())
            tensor_data.extend(s.numpy().tobytes())
            header_entries.append((name, q.shape, q.numel(), s.numel() * s.element_size()))
        elif info in ("passthrough", "passthrough_ctrl", "passthrough_fp16"):
            t = w[name]
            tensor_data.extend(t.numpy().tobytes())
            header_entries.append((name, t.shape, t.numel() * t.element_size(), 0))

    # Minimal header: just count + shapes (for reconstruction)
    header_buf = io.BytesIO()
    header_buf.write(struct.pack("<I", len(header_entries)))
    for name, shape, data_len, scale_len in header_entries:
        name_bytes = name.encode("utf-8")
        header_buf.write(struct.pack("<H", len(name_bytes)))
        header_buf.write(name_bytes)
        header_buf.write(struct.pack("<B", len(shape)))
        for d in shape:
            header_buf.write(struct.pack("<I", d))
        header_buf.write(struct.pack("<II", data_len, scale_len))

    header_raw = header_buf.getvalue()
    full_raw = header_raw + bytes(tensor_data)
    compressed = zstandard.ZstdCompressor(level=22).compress(full_raw)
    return {
        "name": "Packed int6 bitstream + minimal header + zstd-22",
        "compressed": len(compressed),
        "raw": len(full_raw),
        "header_bytes": len(header_raw),
    }


def strategy_packed_int6_raw(w, m):
    """Strategy 2: Packed int6, no compression — absolute floor."""
    total = 0
    int6_params = 0
    int8_params = 0
    pass_bytes = 0
    scale_bytes = 0

    for name, info in m.items():
        if isinstance(info, dict) and info["type"] == "int6":
            q = w[name + ".q"]
            s = w[name + ".scale"]
            packed_bytes = (q.numel() * 6 + 7) // 8  # ceil(n*6/8)
            total += packed_bytes + s.numel() * s.element_size()
            int6_params += q.numel()
            scale_bytes += s.numel() * s.element_size()
        elif isinstance(info, dict) and info["type"] == "int8":
            q = w[name + ".q"]
            s = w[name + ".scale"]
            total += q.numel() + s.numel() * s.element_size()
            int8_params += q.numel()
            scale_bytes += s.numel() * s.element_size()
        elif info in ("passthrough", "passthrough_ctrl", "passthrough_fp16"):
            t = w[name]
            total += t.numel() * t.element_size()
            pass_bytes += t.numel() * t.element_size()

    return {
        "name": "Packed int6 raw (NO compression) — floor",
        "compressed": total,  # this IS the size, no further compression
        "raw": total,
        "breakdown": f"int6_packed={int6_params*6//8:,}  int8={int8_params:,}  pass={pass_bytes:,}  scales={scale_bytes:,}",
    }


def strategy_raw_concat_zstd(w, m):
    """Strategy 3: Raw tensor concat (no torch.save/pickle) + zstd-22."""
    buf = bytearray()
    for name, info in m.items():
        if isinstance(info, dict):
            buf.extend(w[name + ".q"].numpy().tobytes())
            buf.extend(w[name + ".scale"].numpy().tobytes())
        elif info in ("passthrough", "passthrough_ctrl", "passthrough_fp16"):
            buf.extend(w[name].numpy().tobytes())

    raw = bytes(buf)
    compressed = zstandard.ZstdCompressor(level=22).compress(raw)
    return {
        "name": "Raw tensor concat (no pickle) + zstd-22",
        "compressed": len(compressed),
        "raw": len(raw),
    }


def strategy_int5_packed_zstd(w, m):
    """Strategy 4: Requantize to int5, pack, zstd-22. Quality unknown — size only."""
    buf = bytearray()
    int5_params = 0

    for name, info in m.items():
        if isinstance(info, dict) and info["type"] == "int6":
            q = w[name + ".q"]
            s = w[name + ".scale"]
            q5, s5 = requantize_int5(q, s)
            packed = pack_int5(q5.numpy().flatten())
            buf.extend(packed)
            buf.extend(s5.numpy().tobytes())
            int5_params += q.numel()
        elif isinstance(info, dict) and info["type"] == "int8":
            q = w[name + ".q"]
            s = w[name + ".scale"]
            buf.extend(q.numpy().tobytes())
            buf.extend(s.numpy().tobytes())
        elif info in ("passthrough", "passthrough_ctrl", "passthrough_fp16"):
            t = w[name]
            buf.extend(t.numpy().tobytes())

    raw = bytes(buf)
    compressed = zstandard.ZstdCompressor(level=22).compress(raw)
    return {
        "name": f"Int5 requant ({int5_params:,} params) + packed + zstd-22  [QUALITY UNKNOWN]",
        "compressed": len(compressed),
        "raw": len(raw),
    }


def strategy_pruned_zstd(w, m):
    """Strategy 5: Zero out smallest 10% of weights, current int8 containers + zstd-22."""
    pruned_w = {}
    pruned_count = 0
    total_count = 0

    for name, info in m.items():
        if isinstance(info, dict):
            q = w[name + ".q"].clone()
            s = w[name + ".scale"].clone()
            if q.numel() > 1000:  # only prune large tensors
                flat = q.flatten().float()
                threshold = torch.quantile(flat.abs(), 0.10)
                mask = flat.abs() <= threshold
                pruned_count += mask.sum().item()
                total_count += flat.numel()
                flat[mask] = 0
                q = flat.to(torch.int8).reshape(q.shape)
            pruned_w[name + ".q"] = q
            pruned_w[name + ".scale"] = s
        else:
            pruned_w[name] = w[name]

    # Serialize with torch.save + zstd-22 (same as baseline)
    buf = io.BytesIO()
    torch.save({"w": pruned_w, "m": m}, buf)
    raw = buf.getvalue()
    compressed = zstandard.ZstdCompressor(level=22).compress(raw)
    return {
        "name": f"10% pruning ({pruned_count:,}/{total_count:,} zeroed) + torch.save + zstd-22",
        "compressed": len(compressed),
        "raw": len(raw),
    }


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "final_model.int6.ptz"
    print(f"Loading {path}...")
    w, m, compressed_size, raw_size = load_current_artifact(path)

    total_params = sum(
        w[name + ".q"].numel() if isinstance(info, dict) else w[name].numel()
        for name, info in m.items()
    )
    code_bytes_est = 60_000  # ~60KB for train_gpt.py

    print(f"Total params: {total_params:,}")
    print(f"Artifact budget: 16,000,000 bytes (minus ~{code_bytes_est:,} code = {16_000_000 - code_bytes_est:,} for model)")
    print(f"")

    strategies = [
        lambda: strategy_baseline(w, m, compressed_size, raw_size),
        lambda: strategy_raw_concat_zstd(w, m),
        lambda: strategy_packed_int6_zstd(w, m),
        lambda: strategy_packed_int6_raw(w, m),
        lambda: strategy_int5_packed_zstd(w, m),
        lambda: strategy_pruned_zstd(w, m),
    ]

    results = []
    for i, strat in enumerate(strategies):
        t0 = time.perf_counter()
        r = strat()
        elapsed = time.perf_counter() - t0
        r["elapsed_ms"] = elapsed * 1000
        results.append(r)

    model_budget = 16_000_000 - code_bytes_est

    print(f"{'Strategy':<65} {'Compressed':>12} {'Raw':>12} {'B/param':>8} {'Max params':>12} {'Time':>8}")
    print("=" * 125)
    for r in results:
        bpp = r["compressed"] / total_params
        max_params = int(model_budget / bpp) if bpp > 0 else 0
        print(
            f"{r['name']:<65} {r['compressed']:>12,} {r['raw']:>12,} {bpp:>8.3f} {max_params:>12,} {r['elapsed_ms']:>7.0f}ms"
        )
        if "breakdown" in r:
            print(f"  └─ {r['breakdown']}")
        if "header_bytes" in r:
            print(f"  └─ header: {r['header_bytes']:,} bytes")

    # Projection for H100 (fully trained model has less compressible weights)
    print(f"\n{'=' * 125}")
    print("NOTE: This model is undertrained (3070, 650 steps). H100 fully-trained weights")
    print("have higher entropy → worse compression ratios. H100 baseline: 15.5 MB compressed")
    print(f"from ~27 MB raw (1.74× ratio vs {raw_size / compressed_size:.1f}× here).")
    print(f"\nTo estimate H100 sizes, scale by: {raw_size / compressed_size:.1f}× / 1.74× = {compressed_size * (raw_size / compressed_size) / 1.74 / compressed_size:.2f}× this result")


if __name__ == "__main__":
    main()
