"""
Measure this machine's CPU training throughput and project how long each
remaining training task will take.

    python mdg/hpc/bench_train.py                 # all untrained tasks
    python mdg/hpc/bench_train.py --epochs 5
    python mdg/hpc/bench_train.py --threads 1 8 32 64

Transformer training on CPU is memory-bandwidth bound, not core bound: on an
8-core laptop, going from 1 to 10 threads bought 1.3x, not 10x. How a given box
scales is a property of its memory subsystem, so it has to be measured rather
than assumed -- the projection below is only as good as this measurement.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import torch

DATA = Path("mdg/finetune/data")
# (task id, dataset stem, hf model) for every pair in matrix.sh.
DATASETS = [
    "counterbench", "ac-reason", "bbh-causal-judgement", "causalbench_code",
    "causalbench_math", "causalbench_text", "corr2cause", "e-care",
    "fincausal-task1", "natquest", "Quriosity",
]
MODELS = ["bert-base-uncased", "roberta-base"]


def seq_stats(stem: str, tok, sample: int = 300) -> tuple[int, float]:
    """(train rows, mean token length) — padding is dynamic, so the mean
    length is what drives cost, not max_length."""
    p = DATA / f"namesarnav_{stem}__train.jsonl"
    if not p.exists():
        return 0, 0.0
    lines = p.read_text().splitlines()
    if not lines:
        return 0, 0.0
    random.seed(0)
    samp = random.sample(lines, min(sample, len(lines)))
    lens = []
    for l in samp:
        try:
            lens.append(min(len(tok(json.loads(l)["text"])["input_ids"]), 256))
        except (json.JSONDecodeError, KeyError):
            continue
    return len(lines), statistics.mean(lens) if lens else 0.0


def bench(model_name: str, seqlen: int, threads: int, batch: int, steps: int = 4) -> float:
    """Examples per second for one optimizer step at this shape."""
    from transformers import AutoModelForSequenceClassification
    torch.set_num_threads(threads)
    m = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=2e-5)
    ids = torch.randint(0, 20000, (batch, seqlen))
    lab = torch.randint(0, 2, (batch,))

    def step():
        opt.zero_grad()
        m(input_ids=ids, labels=lab).loss.backward()
        opt.step()

    for _ in range(2):
        step()
    t0 = time.perf_counter()
    for _ in range(steps):
        step()
    return batch / ((time.perf_counter() - t0) / steps)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threads", type=int, nargs="+", default=None,
                    help="thread counts to test (default: 1, cores/8, cores/2, cores)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--local-only", action="store_true",
                    help="ignore the Hub; count anything without a local checkpoint")
    ap.add_argument("--model", default="bert-base-uncased",
                    help="model to benchmark; roberta-base costs within a few %% of bert")
    args = ap.parse_args()

    import os
    cores = os.cpu_count() or 4
    threads = args.threads or sorted({1, max(1, cores // 8), max(1, cores // 2), cores})

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Which pairs still need training? A pair already pushed to the Hub counts
    # as done even with no local checkpoint, since that is what the attacks use.
    hub_ok = (lambda r: (False, ""))
    if not args.local_only:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).parent))
        from preflight_check import hub_state as hub_ok

    todo = []
    for mi, model in enumerate(MODELS):
        for di, stem in enumerate(DATASETS):
            task = mi * len(DATASETS) + di
            ckpt = Path(f"mdg/finetune/checkpoints/namesarnav_{stem}/{model}/config.json")
            if ckpt.exists():
                continue
            if hub_ok(f"namesarnav/{stem}-{model}")[0]:
                continue
            rows, mean_len = seq_stats(stem, tok)
            if rows:
                todo.append((task, stem, model, rows, mean_len))

    print(f"\n  machine: {cores} cores, torch default {torch.get_num_threads()} threads")
    print(f"  benchmarking {args.model}, batch {args.batch}\n")

    # Benchmark at the shortest and longest shape that actually occurs.
    lens = sorted({int(round(l / 8) * 8) for _, _, _, _, l in todo} or {128})
    probe = sorted({max(8, lens[0]), lens[-1]})
    rates: dict[tuple[int, int], float] = {}
    print(f"  {'threads':>8} {'seqlen':>7} {'ex/s':>9} {'ex/s/thread':>13} {'scaling':>9}")
    print("  " + "-" * 50)
    for t in threads:
        for sl in probe:
            r = bench(args.model, sl, t, args.batch)
            rates[(t, sl)] = r
            base = rates.get((threads[0], sl), r)
            print(f"  {t:>8} {sl:>7} {r:>9.2f} {r/t:>13.2f} {r/base:>8.2f}x")

    best_t = max(threads, key=lambda t: rates[(t, probe[-1])])

    def rate_at(seqlen: float, t: int) -> float:
        """Interpolate between the probed lengths; cost is near-linear in
        tokens at these lengths because the feed-forward blocks dominate."""
        lo, hi = probe[0], probe[-1]
        if hi == lo:
            return rates[(t, lo)]
        seqlen = max(lo, min(hi, seqlen))
        f = (seqlen - lo) / (hi - lo)
        # interpolate in seconds/example, which is what is linear in tokens
        s_lo, s_hi = 1 / rates[(t, lo)], 1 / rates[(t, hi)]
        return 1 / (s_lo + f * (s_hi - s_lo))

    print(f"\n  best thread count at the longest shape: {best_t}\n")
    print(f"  {'id':>3} {'dataset':<22}{'model':<18}{'rows':>8} {'tok':>5} "
          f"{'hours @'+str(best_t)+'t':>12} {'hours @1t':>10}")
    print("  " + "-" * 82)
    tot_best = tot_one = 0.0
    for task, stem, model, rows, mean_len in sorted(todo):
        steps = rows * args.epochs
        h_best = steps / rate_at(mean_len, best_t) / 3600
        h_one = steps / rate_at(mean_len, threads[0]) / 3600
        tot_best += h_best
        tot_one += h_one
        print(f"  {task:>3} {stem:<22}{model:<18}{rows:>8} {mean_len:>5.0f} "
              f"{h_best:>12.1f} {h_one:>10.1f}")
    print("  " + "-" * 82)
    print(f"  {len(todo)} tasks, {args.epochs} epochs: {tot_best:.1f} machine-hours "
          f"at {best_t} threads, {tot_one:.1f} at {threads[0]}\n")
    print("  One task at a time with every thread is the slowest way to use the box.")
    print("  Run several tasks at once instead -- but a single task cannot be split,")
    print("  so the largest one sets the floor no matter how many machines you have.\n")


if __name__ == "__main__":
    main()
