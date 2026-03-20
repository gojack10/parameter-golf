#!/usr/bin/env bash
# scout_prs.sh — Scrape parameter-golf PRs, diff against baseline, analyze with Claude
# Usage: ./scout_prs.sh [--top N] [--force] [--pr NUMBER]
set -euo pipefail

REPO="openai/parameter-golf"
CACHE_DIR=".scout_cache"
REPORT_DIR="scout_reports"
TOP_N=30
FORCE=0
SINGLE_PR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --top) TOP_N="$2"; shift 2 ;;
        --force) FORCE=1; shift ;;
        --pr) SINGLE_PR="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$CACHE_DIR/diffs" "$REPORT_DIR"

# Step 1: Fetch PR list (or single PR)
echo "=== Fetching PRs ==="
if [[ -n "$SINGLE_PR" ]]; then
    gh pr view "$SINGLE_PR" --repo "$REPO" --json number,title,state,mergedAt,author,body,files \
        | python3 -c "
import json, sys, re
pr = json.load(sys.stdin)
bpb_match = re.search(r'(?:val_bpb|bpb)\s*[=:]\s*([\d.]+)', pr['title'], re.I)
out = [{
    'number': pr['number'],
    'title': pr['title'],
    'state': pr['state'],
    'merged': pr.get('mergedAt', ''),
    'author': pr['author']['login'],
    'bpb_claimed': float(bpb_match.group(1)) if bpb_match else None,
    'body': (pr.get('body') or '')[:3000],
    'has_train_gpt': any(f['path'].endswith('train_gpt.py') for f in pr.get('files', []))
}]
json.dump(out, sys.stdout)
" > "$CACHE_DIR/pr_list.json"
else
    gh pr list --repo "$REPO" --state all --limit 200 \
        --search "record OR val_bpb OR bpb" \
        --json number,title,state,mergedAt,author,body,files \
        | python3 -c "
import json, sys, re

prs = json.load(sys.stdin)
results = []
for pr in prs:
    bpb_match = re.search(r'(?:val_bpb|bpb)\s*[=:]\s*([\d.]+)', pr['title'], re.I)
    bpb = float(bpb_match.group(1)) if bpb_match else None
    has_train = any(f['path'].endswith('train_gpt.py') for f in pr.get('files', []))
    results.append({
        'number': pr['number'],
        'title': pr['title'],
        'state': pr['state'],
        'merged': pr.get('mergedAt', ''),
        'author': pr['author']['login'],
        'bpb_claimed': bpb,
        'body': pr['body'][:3000] if pr.get('body') else '',
        'has_train_gpt': has_train
    })

results.sort(key=lambda x: (x['bpb_claimed'] or 99, -x['number']))
json.dump(results, sys.stdout, indent=2)
print(f'\n# {len(results)} PRs found, {sum(1 for r in results if r[\"has_train_gpt\"])} with train_gpt.py', file=sys.stderr)
" > "$CACHE_DIR/pr_list.json"
fi

PR_COUNT=$(python3 -c "import json; print(len(json.load(open('$CACHE_DIR/pr_list.json'))))")
echo "Found $PR_COUNT PRs"

# Step 2: For top N PRs with train_gpt.py, grab the diff via gh pr diff
echo "=== Fetching PR diffs ==="
python3 -c "
import json
prs = json.load(open('$CACHE_DIR/pr_list.json'))
candidates = [p for p in prs if p['has_train_gpt']][:$TOP_N]
for p in candidates:
    print(f'{p[\"number\"]}|{p[\"title\"][:60]}|{p[\"bpb_claimed\"] or \"?\"}')
" | while IFS='|' read -r pr_num pr_title pr_bpb; do
    DIFF_FILE="$CACHE_DIR/diffs/${pr_num}.diff"

    if [[ -f "$DIFF_FILE" && -s "$DIFF_FILE" && "$FORCE" -eq 0 ]]; then
        echo "  #$pr_num: cached ($(wc -l < "$DIFF_FILE") lines)"
        continue
    fi

    echo "  #$pr_num ($pr_bpb BPB): fetching diff..."

    # gh pr diff gives us the full PR diff — filter to train_gpt.py sections only
    # Use awk to extract diff hunks for files ending in train_gpt.py
    gh pr diff "$pr_num" --repo "$REPO" 2>/dev/null | awk '
        /^diff --git.*train_gpt\.py/ { capture=1 }
        capture { print }
        /^diff --git/ && !/train_gpt\.py/ { capture=0 }
    ' > "$DIFF_FILE" 2>/dev/null || { echo "    Failed"; continue; }

    DIFF_LINES=$(wc -l < "$DIFF_FILE")
    echo "    Diff: $DIFF_LINES lines"
    sleep 0.3
done

# Step 3: Analyze each diff with Claude
echo ""
echo "=== Analyzing diffs with Claude Sonnet ==="
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_FILE="$REPORT_DIR/scout_${TIMESTAMP}.md"

cat > "$REPORT_FILE" <<HEADER
# Parameter Golf PR Scout Report
Generated: $(date -Iseconds)

**Our current stack:** 9L/512d, MLP 3x, int6+zstd, fp16 embed, STE QAT, NorMuon, SWA,
SmearGate, MTP, sliding window (stride 256), TTT (full-param SGD, prefix/suffix split).
**Our best BPB:** 1.2244 (H100 baseline), 1.162 SW (4090 1800s).

HEADER

ANALYZED=0
python3 -c "
import json
prs = json.load(open('$CACHE_DIR/pr_list.json'))
candidates = [p for p in prs if p['has_train_gpt']][:$TOP_N]
for p in candidates:
    print(f'{p[\"number\"]}|{p[\"bpb_claimed\"] or \"?\"}|{p[\"state\"]}|{p[\"author\"]}|{p[\"title\"][:80]}')
" | while IFS='|' read -r pr_num pr_bpb pr_state pr_author pr_title; do
    DIFF_FILE="$CACHE_DIR/diffs/${pr_num}.diff"
    if [[ ! -f "$DIFF_FILE" || ! -s "$DIFF_FILE" ]]; then
        continue
    fi

    DIFF_LINES=$(wc -l < "$DIFF_FILE")
    if [[ "$DIFF_LINES" -lt 5 ]]; then
        continue
    fi

    PR_BODY=$(python3 -c "
import json
prs = json.load(open('$CACHE_DIR/pr_list.json'))
pr = next((p for p in prs if p['number'] == $pr_num), None)
print(pr['body'][:2000] if pr and pr.get('body') else 'No description')
")

    echo "  Analyzing #$pr_num ($pr_bpb BPB)..."

    # Truncate diff for context window
    DIFF_CONTENT=$(head -1500 "$DIFF_FILE")

    ANALYSIS=$(printf '%s' "You are analyzing a Parameter Golf competition PR (train best LM in 16MB, 10min on 8xH100, measured by bits-per-byte).

PR #$pr_num by $pr_author — $pr_title
State: $pr_state | Claimed BPB: $pr_bpb

PR Description:
$PR_BODY

Code diff (train_gpt.py changes only, truncated to 1500 lines):
$DIFF_CONTENT

OUR CURRENT STACK: 9L/512d, MLP 3x, int6+zstd, fp16 embed, STE QAT, NorMuon, SWA, SmearGate, MTP, sliding window eval (stride 256), TTT (full-param SGD prefix/suffix split), logit softcap 30.

Extract EXACTLY this format:

TECHNIQUES:
- [EVAL|TRAIN] technique_name: one-line description. Delta: X BPB if known

KEY_INNOVATIONS:
- What is novel or different from standard approaches (1-2 bullets max)

WE_DONT_HAVE:
- Techniques in this PR that are NOT in our current stack (be specific)

WE_SHOULD_STEAL:
- Top 1-2 things to adopt, with expected impact and difficulty (easy/medium/hard)" \
    | claude -p --model sonnet 2>/dev/null || echo "Analysis failed")

    cat >> "$REPORT_FILE" <<EOF

---
## PR #$pr_num — $pr_title
**Author:** $pr_author | **State:** $pr_state | **BPB:** $pr_bpb | **Diff:** $DIFF_LINES lines

$ANALYSIS

EOF

done

echo ""
echo "=== Generating summary ==="

# Final summary pass — combine report + prompt into one input
SUMMARY_PROMPT="Based on the PR analyses above, generate:

## Technique Inventory
| Technique | Type | Best BPB | PRs | In our stack? | Steal priority |
(one row per unique technique found across all PRs)

## Top 5 Things to Steal (priority order)
For each: technique, expected BPB delta, implementation difficulty, reference PR.

## Approaches Only 1-2 PRs Use (potential alpha)
Techniques that are rare and show strong results.

Be concise. This is a decision-making document, not a survey."

SUMMARY=$({ cat "$REPORT_FILE"; echo ""; echo "$SUMMARY_PROMPT"; } \
    | claude -p --model sonnet 2>/dev/null || echo "Summary failed")

echo "" >> "$REPORT_FILE"
echo "$SUMMARY" >> "$REPORT_FILE"

echo ""
echo "=== Done. Report: $REPORT_FILE ==="
echo ""
# Print just the summary section
echo "$SUMMARY"
