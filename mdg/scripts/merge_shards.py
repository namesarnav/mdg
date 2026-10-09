"""
Merge row-sharded attack outputs back into one result per (dataset, model, recipe).

When a cell is split across machines with --row-shard i/n, each machine writes
    <recipe>.sh<i>of<n>.jsonl   and   <recipe>.sh<i>of<n>_summary.json
covering rows i, i+n, i+2n, ... of the dataset. This concatenates the shards in
global row order into the plain <recipe>.jsonl and recomputes the summary from
the merged rows.

F1 is recomputed from the per-example records rather than averaged across
shards: macro F1 is not a weighted mean of per-shard macro F1, so averaging
would be quietly wrong.

    python -m mdg.scripts.merge_shards                 # merge every sharded cell
    python -m mdg.scripts.merge_shards --backfill      # also recompute F1 for
                                                       # old summaries that lack it
    python -m mdg.scripts.merge_shards --keep-shards   # don't delete shard files

Run this once all machines' results have been rsync'd together, before
consolidate_results.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from sklearn.metrics import f1_score

RESULTS = Path("mdg/adv_attack/results")
SHARD_RE = re.compile(r"^(?P<recipe>.+)\.sh(?P<i>\d+)of(?P<n>\d+)\.jsonl$")


def _f1(gold, pred, average) -> float:
    if not gold:
        return 0.0
    return round(float(f1_score(gold, pred, average=average, zero_division=0)), 4)


def summarize(recipe: str, rows: List[dict]) -> Dict:
    """Rebuild a summary dict from per-example records — the same numbers
    run_recipe writes, but computed over however many shards were merged."""
    golds, clean, attacked = [], [], []
    n_successful = n_failed = n_skipped = 0
    total_queries = 0

    for r in rows:
        rtype = r.get("result_type") or ""
        if "Successful" in rtype:
            n_successful += 1
        elif "Failed" in rtype:
            n_failed += 1
        else:
            n_skipped += 1
        q = r.get("num_queries")
        if q:
            total_queries += q
        gold = r.get("ground_truth")
        if gold is None:
            continue
        clean_pred = r.get("original_label")
        if clean_pred is None:
            continue
        pert = r.get("perturbed_label")
        golds.append(int(gold))
        clean.append(int(clean_pred))
        attacked.append(int(pert if pert is not None else clean_pred))

    n_total = len(rows)
    tried = n_successful + n_failed
    return {
        "recipe":              recipe,
        "row_shard":           None,
        "num_examples":        n_total,
        "clean_micro_f1":      _f1(golds, clean, "micro"),
        "clean_macro_f1":      _f1(golds, clean, "macro"),
        "attacked_micro_f1":   _f1(golds, attacked, "micro"),
        "attacked_macro_f1":   _f1(golds, attacked, "macro"),
        "n_successful":        n_successful,
        "n_failed":            n_failed,
        "n_skipped":           n_skipped,
        "attack_success_rate": round(n_successful / tried, 4) if tried else 0.0,
        "avg_queries":         round(total_queries / n_total, 2) if n_total else 0.0,
    }


def read_jsonl(path: Path) -> List[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                # A machine killed mid-write leaves one truncated final line.
                print(f"    [warn] dropping a truncated line in {path.name}")
    return out


def merge_dir(run_dir: Path, keep_shards: bool) -> int:
    """Merge every sharded recipe in one <split>/<model> directory."""
    groups: Dict[str, List[Path]] = defaultdict(list)
    for p in run_dir.glob("*.sh*of*.jsonl"):
        m = SHARD_RE.match(p.name)
        if m:
            groups[m.group("recipe")].append(p)

    merged = 0
    for recipe, parts in sorted(groups.items()):
        expected = {int(SHARD_RE.match(p.name).group("n")) for p in parts}
        if len(expected) != 1:
            print(f"  [skip] {run_dir}/{recipe}: mixed shard counts {sorted(expected)} "
                  f"— rerun these with one ROW_SHARDS value")
            continue
        n = expected.pop()
        have = sorted(int(SHARD_RE.match(p.name).group("i")) for p in parts)
        if have != list(range(n)):
            missing = sorted(set(range(n)) - set(have))
            print(f"  [wait] {run_dir}/{recipe}: {len(have)}/{n} shards "
                  f"(missing {missing}) — not merging yet")
            continue

        rows: List[dict] = []
        for p in sorted(parts, key=lambda q: int(SHARD_RE.match(q.name).group("i"))):
            rows.extend(read_jsonl(p))
        rows.sort(key=lambda r: r.get("idx", 0))
        for r in rows:
            r.pop("row_shard", None)

        out_jsonl = run_dir / f"{recipe}.jsonl"
        with open(out_jsonl, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        summary = summarize(recipe, rows)
        (run_dir / f"{recipe}_summary.json").write_text(json.dumps(summary, indent=2))

        if not keep_shards:
            for p in parts:
                p.unlink()
                sj = p.with_name(p.name[:-len(".jsonl")] + "_summary.json")
                if sj.exists():
                    sj.unlink()

        print(f"  [ok] {run_dir}/{recipe}: {n} shards → {len(rows)} rows  "
              f"ASR={summary['attack_success_rate']:.1%}  "
              f"microF1 {summary['clean_micro_f1']:.3f}→{summary['attacked_micro_f1']:.3f}")
        merged += 1
    return merged


def backfill_dir(run_dir: Path) -> int:
    """Older runs saved perturbed JSONL but no F1 (the metric predates them).
    Recompute from the records already on disk."""
    fixed = 0
    for jl in sorted(run_dir.glob("*.jsonl")):
        if SHARD_RE.match(jl.name):
            continue
        recipe = jl.stem
        sj = run_dir / f"{recipe}_summary.json"
        if sj.exists():
            try:
                d = json.loads(sj.read_text())
            except json.JSONDecodeError:
                d = {}
            if d.get("attacked_micro_f1") is not None:
                continue
        rows = read_jsonl(jl)
        if not rows or all(r.get("ground_truth") is None for r in rows):
            continue
        summary = summarize(recipe, rows)
        sj.write_text(json.dumps(summary, indent=2))
        print(f"  [backfill] {run_dir}/{recipe}: {len(rows)} rows  "
              f"microF1 {summary['clean_micro_f1']:.3f}→{summary['attacked_micro_f1']:.3f}")
        fixed += 1
    return fixed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(RESULTS), help="attack results root")
    ap.add_argument("--keep-shards", action="store_true",
                    help="leave the per-shard files in place after merging")
    ap.add_argument("--backfill", action="store_true",
                    help="also recompute F1 for unsharded runs whose summary lacks it")
    args = ap.parse_args()

    root = Path(args.results)
    if not root.is_dir():
        raise SystemExit(f"[FAIL] no results directory at {root}")

    run_dirs = sorted({p.parent for p in root.glob("*/*/*.jsonl")})
    merged = sum(merge_dir(d, args.keep_shards) for d in run_dirs)
    print(f"\nmerged {merged} sharded cell(s)")
    if args.backfill:
        fixed = sum(backfill_dir(d) for d in run_dirs)
        print(f"backfilled F1 for {fixed} older cell(s)")
    print("next: python -m mdg.scripts.consolidate_results")


if __name__ == "__main__":
    main()
