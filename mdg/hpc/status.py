"""One screen of truth: what is trained, what is attacked, what is left.

    python mdg/hpc/status.py              # this machine's results + the Hub
    python mdg/hpc/status.py --no-hub     # skip the network, local checkpoints only

Run it on each VM. Attack results live on the machine that produced them until
they are rsync'd, so "done" here means done *here*.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "mdg/finetune/data"
RESULTS = ROOT / "mdg/adv_attack/results"
CKPT = ROOT / "mdg/finetune/checkpoints"

DATASETS = ["counterbench", "ac-reason", "bbh-causal-judgement", "causalbench_code",
            "causalbench_math", "causalbench_text", "corr2cause", "e-care",
            "fincausal-task1", "natquest", "Quriosity"]
MODELS = ["bert-base-uncased", "roberta-base", "t5-base"]


def recipes() -> list[str]:
    """Parse the recipe tuple out of attack.py — the one place it is declared."""
    src = (ROOT / "mdg/adv_attack/attack.py").read_text()
    src = src.split("MULTILINGUAL_RECIPES")[0]
    return re.findall(r'^\s*\("([^"]+)"', src, re.M)


def rows(stem: str) -> int:
    p = DATA / f"namesarnav_{stem}__test.jsonl"
    return sum(1 for _ in p.open()) if p.exists() else 0


def cell_state(stem: str, model: str, n_recipes: int) -> tuple[int, int]:
    """(finished recipes, recipes with shards still merging) for one cell."""
    d = RESULTS / f"namesarnav_{stem}__test" / model
    if not d.is_dir():
        return 0, 0
    done = {p.stem for p in d.glob("*.jsonl") if ".sh" not in p.name}
    part = {p.name.split(".sh")[0] for p in d.glob("*.sh*of*.jsonl")}
    return len(done), len(part - done)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-hub", action="store_true")
    args = ap.parse_args()

    RECIPES = recipes()
    nr = len(RECIPES)

    hub_state = lambda r: (False, "skipped")
    if not args.no_hub:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from preflight_check import hub_state  # noqa: F811

    print(f"\n  {len(DATASETS)} datasets x {len(MODELS)} models = "
          f"{len(DATASETS)*len(MODELS)} tasks, {nr} recipes each\n")
    print(f"  {'id':>3} {'dataset':<24}{'model':<19}{'test':>6} "
          f"{'weights':<9}{'attacked':>10}  state")
    print("  " + "-" * 86)

    ready, missing, attack_todo, full = [], [], [], []
    for mi, model in enumerate(MODELS):
        for di, stem in enumerate(DATASETS):
            task = mi * len(DATASETS) + di
            local = (CKPT / f"namesarnav_{stem}" / model / "config.json").exists()
            if local:
                w = "local"
                ok = True
            else:
                ok, _ = hub_state(f"namesarnav/{stem}-{model}")
                w = "hub" if ok else "--"
            done, part = cell_state(stem, model, nr)
            if ok:
                ready.append(task)
                if done < nr:
                    attack_todo.append(task)
                else:
                    full.append(task)
                state = "DONE" if done >= nr else (
                    f"{part} merging" if part else "attack pending" if not done
                    else "attacking")
            else:
                missing.append(task)
                state = "NEEDS TRAINING"
            print(f"  {task:>3} {stem:<24}{model:<19}{rows(stem):>6} "
                  f"{w:<9}{f'{done}/{nr}':>10}  {state}")

    print("  " + "-" * 86)
    print(f"  trained {len(ready)}/{len(DATASETS)*len(MODELS)}   "
          f"attacks complete {len(full)}/{len(ready)} of the trained cells")
    if missing:
        print(f'\n  TRAIN_TASKS="{" ".join(map(str, missing))}"')
    if attack_todo:
        print(f'  ATTACK_TASKS="{" ".join(map(str, attack_todo))}"')
    print()


if __name__ == "__main__":
    main()
