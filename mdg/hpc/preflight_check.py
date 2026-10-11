"""Resolve every task's weights, locally or on the Hub. Called by preflight.sh."""
from __future__ import annotations

import sys
from pathlib import Path

WEIGHTS = ("model.safetensors", "pytorch_model.bin")


def hub_state(repo_id: str) -> tuple[bool, str]:
    """(usable, note) for a Hub repo. An interrupted push leaves a repo that
    exists but holds no weights, which is the failure this is here to catch."""
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.utils import (
            GatedRepoError, RepositoryNotFoundError, HfHubHTTPError,
        )
    except ImportError:
        return False, "huggingface_hub not installed"
    try:
        files = set(HfApi().list_repo_files(repo_id))
    except RepositoryNotFoundError:
        return False, "no such Hub repo"
    except GatedRepoError:
        return False, "gated — needs an accepted licence"
    except HfHubHTTPError as e:
        return False, f"Hub error {e.response.status_code if e.response else '?'}"
    except Exception as e:                       # offline, DNS, proxy
        return False, f"unreachable ({type(e).__name__})"
    if "config.json" not in files:
        return False, "EMPTY REPO — push never completed"
    if not any(w in files for w in WEIGHTS):
        return False, "config but no weights"
    return True, "hub"


def main() -> None:
    rows = [l.rstrip("\n").split("|") for l in open(sys.argv[1]) if l.strip()]
    ready, missing = [], []

    print(f"\n{'':<4}{'DATASET':<24}{'MODEL':<18}{'SOURCE':<10}NOTE")
    print("-" * 82)
    for idx, dataset, model, ckpt, hub in rows:
        if (Path(ckpt) / "config.json").exists():
            src, note, ok = "local", ckpt, True
        else:
            ok, note = hub_state(hub)
            src = "hub" if ok else "MISSING"
            note = hub if ok else f"{hub}  <- {note}"
        (ready if ok else missing).append(idx)
        print(f"{idx:<4}{dataset:<24}{model:<18}{src:<10}{note}")

    print("-" * 82)
    print(f"  {len(ready)}/{len(rows)} tasks can load weights\n")
    if ready:
        print(f'  READY_TASKS="{" ".join(ready)}"')
    if missing:
        print(f'  MISSING_TASKS="{" ".join(missing)}"')
        print("\n  Attacking a MISSING task fails at model load. Either train and")
        print("  push it, copy its checkpoint onto this machine, or restrict the")
        print("  run with ONLY_TASKS=\"$READY_TASKS\".")
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
