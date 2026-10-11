"""Live progress for training jobs whose output went to a log file.

    python mdg/hpc/watch_train.py                 # one table, then exit
    python mdg/hpc/watch_train.py --follow        # refresh until every task ends

run_parallel.sh redirects each task to its own log so parallel jobs do not
interleave. That is right on a cluster and useless in a notebook, where the
cell then prints nothing for an hour. This reads those logs back and prints one
line per task.

Colab runs cells one at a time, so this cannot be a second cell -- train_gpu.sh
runs the training in the background and this in the foreground.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# tqdm writes "\r 42%|####      | 420/1000 [01:23<01:54,  5.07it/s]".
BAR = re.compile(
    r"(\d+)%\|[^|]*\|\s*(\d+)/(\d+)\s*\[([\d:]+)<([\d:?]+),\s*([\d.]+)\s*(it/s|s/it)")
METRIC = re.compile(r"'eval_macro_f1':\s*([\d.]+)")
LOSS = re.compile(r"'loss':\s*([\d.]+)")
EPOCH = re.compile(r"'epoch':\s*([\d.]+)")


def log_dir() -> Path:
    if os.environ.get("MDG_LOG_DIR"):
        return Path(os.environ["MDG_LOG_DIR"])
    user = os.environ.get("USER") or "mdg"
    cluster = Path(f"/scratch/{user}/mdg-logs")
    return cluster if cluster.is_dir() else ROOT / "mdg/logs"


def read_tail(p: Path, n: int = 200_000) -> str:
    """Last n bytes. Training logs grow to megabytes of tqdm redraws and only
    the end is ever interesting."""
    with p.open("rb") as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - n))
        return f.read().decode("utf-8", "replace")


def parse(text: str) -> dict:
    """Latest state from one task's log."""
    st: dict = {"state": "starting", "pct": None, "eta": None, "rate": None,
                "f1": None, "loss": None, "epoch": None, "fit": None}
    if m := re.search(r"fit=(\S+)\s+eval=(\S+)\s+\((\d+)/(\d+)\)", text):
        st["fit"] = f"{int(m.group(3)):,}"
    # tqdm separates redraws with \r; the last bar in the file is current.
    bars = BAR.findall(text.replace("\r", "\n"))
    if bars:
        pct, cur, tot, el, eta, rate, unit = bars[-1]
        st.update(pct=int(pct), eta=eta, step=f"{cur}/{tot}", elapsed=el,
                  rate=f"{rate}{'it/s' if unit == 'it/s' else 's/it'}")
        st["state"] = "training"
    for rx, key, cast in ((METRIC, "f1", float), (LOSS, "loss", float),
                          (EPOCH, "epoch", float)):
        if hits := rx.findall(text):
            st[key] = cast(hits[-1])
    # Terminal states win over any bar left in the file.
    if "[SKIP]" in text:
        st["state"] = "skipped"
    if "[FAIL]" in text or "Traceback" in text:
        st["state"] = "FAILED"
    if "[OK] →" in text or "[OK] ->" in text:
        st["state"] = "pushed"
    return st


def table(tasks: list[int]) -> tuple[str, bool]:
    """(rendered table, every task reached a terminal state)."""
    d = log_dir()
    out = [f"  {'task':>5} {'rows':>9} {'progress':>9} {'step':>14} "
           f"{'elapsed':>9} {'eta':>9} {'rate':>10} {'loss':>7} {'F1':>6}  state"]
    out.append("  " + "-" * 104)
    done = True
    for t in tasks:
        p = d / f"train-task{t}.log"
        if not p.exists():
            out.append(f"  {t:>5} {'':>9} {'':>9} {'':>14} {'':>9} {'':>9} "
                       f"{'':>10} {'':>7} {'':>6}  queued")
            done = False
            continue
        st = parse(read_tail(p))
        if st["state"] in ("starting", "training"):
            done = False
        # Every field is rendered to a string first: half of them are None
        # until the first tqdm redraw lands, and None has no format spec.
        bar = f"{st['pct']}%" if st["pct"] is not None else ""
        loss = f"{st['loss']:.3f}" if st["loss"] is not None else ""
        f1 = f"{st['f1']:.3f}" if st["f1"] is not None else ""
        rows = st["fit"] or ""
        step = st.get("step") or ""
        elapsed = st.get("elapsed") or ""
        eta = st.get("eta") or ""
        rate = st.get("rate") or ""
        out.append(
            f"  {t:>5} {rows:>9} {bar:>9} {step:>14} "
            f"{elapsed:>9} {eta:>9} {rate:>10} {loss:>7} {f1:>6}  {st['state']}")
    return "\n".join(out), done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks", nargs="*", type=int)
    ap.add_argument("--follow", action="store_true")
    ap.add_argument("--every", type=int, default=60, help="seconds between refreshes")
    args = ap.parse_args()

    tasks = args.tasks
    if not tasks:
        tasks = sorted(int(m.group(1)) for p in log_dir().glob("train-task*.log")
                       if (m := re.search(r"train-task(\d+)\.log", p.name)))
    if not tasks:
        print(f"  No train-task*.log under {log_dir()} yet.")
        return

    t0 = time.time()
    while True:
        txt, done = table(tasks)
        mins = (time.time() - t0) / 60
        print(f"\n[{time.strftime('%H:%M:%S')}]  logs: {log_dir()}   "
              f"watching {mins:.0f}m")
        print(txt, flush=True)
        if done or not args.follow:
            if done:
                print("\n  All tasks reached a terminal state.")
            return
        time.sleep(args.every)


if __name__ == "__main__":
    main()
