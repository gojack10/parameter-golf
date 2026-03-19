"""Extract a reference BPB curve from a run's JSONL log.

Usage:
    uv run python extract_reference.py logs/<run_id>.jsonl -o reference_bpb_<hardware>.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract reference BPB curve from JSONL")
    parser.add_argument("jsonl", type=Path, help="Path to run JSONL log")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output reference JSON path")
    args = parser.parse_args()

    curve: dict[str, float] = {}
    for line in args.jsonl.read_text().splitlines():
        event = json.loads(line)
        if event.get("t") == "val" and event.get("vb") is not None:
            curve[str(event["s"])] = event["vb"]

    if not curve:
        print(f"No val events found in {args.jsonl}")
        raise SystemExit(1)

    args.output.write_text(json.dumps(curve, indent=2) + "\n")
    print(f"Wrote {len(curve)} checkpoints to {args.output}")


if __name__ == "__main__":
    main()
