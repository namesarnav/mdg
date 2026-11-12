"""
Attack ONE fine-tuned model on EVERY data point of its dataset with every
recipe in mdg/adv_attack/attack.py, streaming each example to Drive.

Built for Colab, where the session dies without warning and local disk is
small:

  - Each finished example is appended to a local part file and flushed. The
    part is copied to Drive every --sync-sec seconds and when it completes,
    so a dead session loses at most a couple of minutes of work.
  - Resume is per EXAMPLE, not per recipe: on restart the part files on Drive
    are read back and every example already in them is skipped.
    (attack.py cannot do this: Attacker.attack_dataset() returns only when the
    whole recipe is finished, so nothing reaches disk before then.)
  - Local disk only ever holds the part in progress. Finished parts are
    deleted locally once they are on Drive.
  - When every part of a recipe is done they are concatenated into ONE file,
    so the final layout is one file per (dataset, model, recipe):

        <drive>/results/namesarnav_<dataset>__all/<model>/<recipe>.jsonl
        <drive>/results/namesarnav_<dataset>__all/<model>/<recipe>_summary.json

    Same layout and row schema as attack.py, so merge_shards and
    consolidate_results read it unchanged.

    python colab/attack_all.py prepare --dataset e-care
    python colab/attack_all.py check   --dataset e-care --model namesarnav/e-care-bert-base-uncased
    python colab/attack_all.py run     --dataset e-care --model namesarnav/e-care-bert-base-uncased
    python colab/attack_all.py status  --dataset e-care --model namesarnav/e-care-bert-base-uncased

WORKER=i/n splits one (model, dataset) across n Colab sessions: session i
takes parts i, i+n, i+2n, ... and whichever session finishes a recipe's last
part writes the merged file.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# TensorFlow (Universal Sentence Encoder, used by several recipes' constraints)
# grabs the whole GPU on import unless told otherwise, and torch then OOMs on
# its first forward pass. Must be set before anything imports TF.
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

DATA_DIR = REPO / "mdg" / "finetune" / "data"

# Queries per WORD, same table as colab/run.sh. Recipes run cheapest first so a
# short session finishes whole recipes before reaching PSO / genetic ones.
QPW = {"CheckList2020": 0.35, "Pruthi2019": 1.0, "BadCharacters2021": 0.8,
       "DeepWordBugGao2018": 1.4, "TextBuggerLi2018": 2.5, "MorpheusTan2020": 2.5,
       "InputReductionFeng2018": 3.0, "A2TYoo2021": 5.0, "PWWSRen2019": 5.4,
       "Kuleshov2017": 8.0, "TextFoolerJin2019": 10.0, "BAEGarg2019": 12.0,
       "BERTAttackLi2020": 16.0, "CLARE2020": 25.0, "IGAWang2019": 30.0,
       "FasterGeneticAlgorithmJia2019": 40.0, "GeneticAlgorithmAlzantot2018": 60.0,
       "PSOZang2020": 133.0}

ERROR_TYPES = ("Error", "Timeout")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def recipes_in_cost_order(only: Optional[List[str]] = None) -> List[Tuple[str, str]]:
    from mdg.adv_attack.attack import RECIPES
    recipes = [(n, c) for n, c in RECIPES if not only or n in only]
    return sorted(recipes, key=lambda nc: QPW.get(nc[0], 10.0))


# ── data ──────────────────────────────────────────────────────────────────────

def all_file(dataset: str) -> Path:
    return DATA_DIR / f"namesarnav_{dataset}__all.jsonl"


def prepare(dataset: str) -> Path:
    """Build <stem>__all.jsonl from the train/validation/test splits.

    Unlike 02_build_all_splits.sh this drops exact duplicate rows. natquest and
    Quriosity ship train == test (byte-identical), so a plain concatenation
    attacks every row twice.
    """
    stem = f"namesarnav_{dataset}"
    parts = [DATA_DIR / f"{stem}__{s}.jsonl" for s in ("train", "validation", "test")]
    parts = [p for p in parts if p.exists()]
    if not parts:
        raise SystemExit(f"[FAIL] no split files for {stem} in {DATA_DIR}")
    for p in parts:
        if p.read_bytes()[:60].find(b"git-lfs.github.com/spec") >= 0:
            raise SystemExit(f"[FAIL] {p.name} is a Git LFS pointer -- run git lfs pull")
    seen, kept, total = set(), 0, 0
    out = all_file(dataset)
    with open(out, "w") as fout:
        for p in parts:
            for line in open(p):
                if not line.strip():
                    continue
                total += 1
                r = json.loads(line)
                key = hashlib.sha1(json.dumps([r["text"], str(r["label"])]).encode()).digest()
                if key in seen:
                    continue
                seen.add(key)
                fout.write(json.dumps({"text": r["text"], "label": r["label"]},
                                      ensure_ascii=False) + "\n")
                kept += 1
    log(f"{out.name}: {kept} rows ({total - kept} exact duplicates dropped) "
        f"from {', '.join(p.name for p in parts)}")
    return out


def load_rows(dataset: str) -> List[Tuple[str, str]]:
    path = all_file(dataset)
    if not path.exists():
        raise SystemExit(f"[FAIL] {path} missing -- run the data cell first")
    rows = []
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            rows.append((r["text"], str(r["label"]).strip().upper()))
    return rows


# ── model ─────────────────────────────────────────────────────────────────────

def load_model(model_id: str):
    """HuggingFace classifier → (TextAttack wrapper, label2id).

    attack.build_model_wrapper is not reused because it loads every model in
    fp16 on CUDA, and T5 overflows in fp16 (its activations exceed 65504), so
    the t5-base models would silently predict garbage. T5 runs in fp32.
    """
    import torch
    from textattack.models.wrappers import HuggingFaceModelWrapper
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cuda = torch.cuda.is_available()
    is_t5 = "t5" in model_id.lower()
    dtype = torch.float16 if cuda and not is_t5 else torch.float32
    model = AutoModelForSequenceClassification.from_pretrained(model_id, torch_dtype=dtype)
    model.eval()
    if cuda:
        model = model.cuda()
    log(f"model {model_id}  device={'cuda' if cuda else 'cpu'}  dtype={str(dtype).split('.')[-1]}")
    label2id = {str(k).strip().upper(): int(v) for k, v in model.config.label2id.items()}
    return HuggingFaceModelWrapper(model, tok), label2id


def resolve_labels(rows, model_label2id: Dict[str, int], fallback: List[str]) -> Dict[str, int]:
    """Use the model's own label map when it covers the data; it is what the
    model was trained with. Otherwise fall back to the --labels order, which is
    what attack.py and matrix.sh use."""
    data_labels = {lab for _, lab in rows}
    if data_labels <= set(model_label2id):
        return model_label2id
    log(f"[WARN] model labels {sorted(model_label2id)} do not cover data labels "
        f"{sorted(data_labels)} -- using --labels order {fallback}")
    return {lab.strip().upper(): i for i, lab in enumerate(fallback)}


def build_attack(recipe_class: str, wrapper, batch: Optional[int], budget: Optional[int]):
    import torch
    from mdg.adv_attack.attack import _import_recipe
    attack = _import_recipe(recipe_class).build(wrapper)
    gf = attack.goal_function
    if batch:
        gf.batch_size = batch
    if budget:
        gf.query_budget = budget
    if torch.cuda.is_available() and hasattr(attack, "cuda_"):
        attack.cuda_()
    return attack


# ── one example ───────────────────────────────────────────────────────────────

class ExampleTimeout(Exception):
    pass


def _on_alarm(signum, frame):
    raise ExampleTimeout()


def _on_term(signum, frame):
    # Treat a polite shutdown like Ctrl-C, so the finally block saves the part.
    raise KeyboardInterrupt()


def attack_one(attack, wrapper, text: str, gold: int, timeout_sec: int) -> dict:
    if timeout_sec:
        signal.alarm(timeout_sec)
    try:
        res = attack.attack(text, gold)
    except ExampleTimeout:
        return {"result_type": "Timeout", "original_label": clean_pred(wrapper, text),
                "error": f"exceeded {timeout_sec}s"}
    except Exception as e:
        return {"result_type": "Error", "original_label": clean_pred(wrapper, text),
                "error": f"{type(e).__name__}: {str(e)[:300]}"}
    finally:
        if timeout_sec:
            signal.alarm(0)
    pert = getattr(res, "perturbed_result", None)
    return {
        "result_type":     type(res).__name__,
        "original_label":  int(res.original_result.output),
        "perturbed_text":  pert.attacked_text.text if pert is not None else None,
        "perturbed_label": int(pert.output) if pert is not None else None,
        "num_queries":     getattr(res, "num_queries", None),
    }


def clean_pred(wrapper, text: str) -> Optional[int]:
    try:
        import numpy as np
        logits = wrapper([text])[0]
        if hasattr(logits, "detach"):           # the HF wrapper returns a torch tensor
            logits = logits.detach().float().cpu().numpy()
        return int(np.argmax(np.asarray(logits)))
    except Exception:
        return None


# ── files ─────────────────────────────────────────────────────────────────────

def read_valid_lines(path: Path) -> List[str]:
    """Lines of a part file that parse. The Drive copy can end mid-line if the
    session died during a copy; that line is simply attacked again."""
    if not path.exists():
        return []
    good = []
    for line in path.read_text().splitlines():
        try:
            json.loads(line)
            good.append(line)
        except json.JSONDecodeError:
            pass
    return good


def copy_to_drive(local: Path, remote: Path) -> None:
    # Copy then rename, so a reader never sees a half-written part. Drive's FUSE
    # mount only publishes a file once it is closed, which a copy guarantees.
    tmp = remote.with_suffix(remote.suffix + ".tmp")
    shutil.copyfile(local, tmp)
    os.replace(tmp, remote)


def summarize(rows: List[dict], recipe: str, budget: Optional[int]) -> dict:
    from sklearn.metrics import f1_score
    golds, clean, attacked = [], [], []
    n = {"S": 0, "F": 0, "K": 0, "E": 0}
    queries = 0
    for r in rows:
        rt = r["result_type"]
        if rt in ERROR_TYPES:
            n["E"] += 1
            continue
        n["S" if "Successful" in rt else "F" if "Failed" in rt else "K"] += 1
        queries += r.get("num_queries") or 0
        golds.append(r["ground_truth"])
        clean.append(r["original_label"])
        attacked.append(r["perturbed_label"] if r["perturbed_label"] is not None
                        else r["original_label"])

    def f1(pred, avg):
        return round(float(f1_score(golds, pred, average=avg, zero_division=0)), 4) if golds else 0.0

    tried = n["S"] + n["F"]
    return {
        "recipe": recipe, "row_shard": None, "num_examples": len(rows),
        "query_budget": budget,
        "clean_micro_f1": f1(clean, "micro"), "clean_macro_f1": f1(clean, "macro"),
        "attacked_micro_f1": f1(attacked, "micro"), "attacked_macro_f1": f1(attacked, "macro"),
        "n_successful": n["S"], "n_failed": n["F"], "n_skipped": n["K"],
        # Timeouts and exceptions: kept in the .jsonl, left out of the F1s.
        "n_errors": n["E"],
        "attack_success_rate": round(n["S"] / tried, 4) if tried else 0.0,
        "avg_queries": round(queries / max(1, len(rows) - n["E"]), 2),
    }


class Cell:
    """Paths for one (dataset, model) on Drive and on local disk."""

    def __init__(self, drive: Path, local: Path, dataset: str, model_id: str):
        self.dataset = dataset
        self.model_id = model_id
        self.model_name = model_id.split("/")[-1].replace(f"{dataset}-", "", 1)
        self.stem = f"namesarnav_{dataset}__all"
        self.out = drive / "results" / self.stem / self.model_name
        self.work = local / "work" / self.stem / self.model_name
        self.logs = drive / "logs"

    def parts_dir(self, recipe):  return self.out / "_parts" / recipe
    def final(self, recipe):      return self.out / f"{recipe}.jsonl"
    def summary(self, recipe):    return self.out / f"{recipe}_summary.json"
    def part(self, recipe, k):    return self.parts_dir(recipe) / f"part-{k:05d}.jsonl"
    def done(self, recipe, k):    return self.parts_dir(recipe) / f"part-{k:05d}.done"


def finalize(cell: Cell, recipe: str, n_parts: int, budget: Optional[int]) -> bool:
    """Merge a recipe's parts into one .jsonl + summary once every part is done."""
    if cell.summary(recipe).exists():
        return True
    if not all(cell.done(recipe, k).exists() for k in range(n_parts)):
        return False
    cell.work.mkdir(parents=True, exist_ok=True)
    tmp = cell.work / f"{recipe}.merged.jsonl"
    rows = []
    with open(tmp, "w") as fout:
        for k in range(n_parts):
            for line in read_valid_lines(cell.part(recipe, k)):
                fout.write(line + "\n")
                r = json.loads(line)
                rows.append({key: r.get(key) for key in
                             ("result_type", "ground_truth", "original_label",
                              "perturbed_label", "num_queries")})
    copy_to_drive(tmp, cell.final(recipe))
    tmp.unlink()
    summary = summarize(rows, recipe, budget)
    tmp_s = cell.work / f"{recipe}_summary.json"
    tmp_s.write_text(json.dumps(summary, indent=2))
    copy_to_drive(tmp_s, cell.summary(recipe))   # written last: the done-marker
    tmp_s.unlink()
    shutil.rmtree(cell.parts_dir(recipe), ignore_errors=True)
    log(f"[{recipe}] FINAL → {cell.final(recipe).name}  {len(rows)} rows  "
        f"ASR={summary['attack_success_rate']:.1%}  "
        f"macroF1 {summary['clean_macro_f1']:.3f}→{summary['attacked_macro_f1']:.3f}  "
        f"errors={summary['n_errors']}")
    write_all_summaries(cell)
    return True


def write_all_summaries(cell: Cell) -> None:
    sums = []
    for name, _ in recipes_in_cost_order():
        p = cell.summary(name)
        if p.exists():
            sums.append(json.loads(p.read_text()))
    tmp = cell.work / "all_summaries.json"
    tmp.write_text(json.dumps(sums, indent=2))
    copy_to_drive(tmp, cell.out / "all_summaries.json")
    tmp.unlink()


# ── commands ──────────────────────────────────────────────────────────────────

def cmd_check(args) -> None:
    """Build every recipe and attack 2 examples, writing nothing to Drive.

    This is also the download step: each recipe fetches its embeddings, language
    models and sentence encoders on first build. A recipe that fails here fails
    identically in the real run, so stop and fix it rather than burn a session.
    """
    rows = load_rows(args.dataset)
    wrapper, m_l2i = load_model(args.model)
    l2i = resolve_labels(rows, m_l2i, args.labels.split(","))
    sample = [r for r in rows if r[1] in l2i][:2]
    signal.signal(signal.SIGALRM, _on_alarm)
    failed = []
    for name, cls in recipes_in_cost_order(args.recipes):
        t0 = time.time()
        try:
            attack = build_attack(cls, wrapper, args.batch, args.query_budget)
            outs = [attack_one(attack, wrapper, t, l2i[lab], 600)["result_type"] for t, lab in sample]
            bad = [o for o in outs if o in ERROR_TYPES]
            status = f"FAILED ({', '.join(bad)})" if bad else "ok  " + " ".join(outs)
            if bad:
                failed.append(name)
        except Exception as e:
            status = f"FAILED to build: {type(e).__name__}: {str(e)[:200]}"
            failed.append(name)
            traceback.print_exc()
        print(f"  {name:32} {time.time() - t0:6.1f}s  {status}", flush=True)
        attack = None
        gc.collect()
    if failed:
        raise SystemExit(f"\n[FAIL] {len(failed)} recipe(s) failed: {' '.join(failed)}\n"
                         f"       They would fail the same way in the run. Fix before running.")
    print("\n  all recipes ready")


def cmd_run(args) -> None:
    import torch
    import textattack

    cell = Cell(Path(args.drive), Path(args.local), args.dataset, args.model)
    cell.out.mkdir(parents=True, exist_ok=True)
    cell.logs.mkdir(parents=True, exist_ok=True)
    cell.work.mkdir(parents=True, exist_ok=True)
    wi, wn = (int(x) for x in args.worker.split("/"))

    rows = load_rows(args.dataset)
    wrapper, m_l2i = load_model(args.model)
    l2i = resolve_labels(rows, m_l2i, args.labels.split(","))
    n_parts = (len(rows) + args.chunk - 1) // args.chunk
    mine = [k for k in range(n_parts) if k % wn == wi]
    recipes = recipes_in_cost_order(args.recipes)
    log(f"{cell.stem} × {cell.model_name}: {len(rows)} rows, {len(recipes)} recipes, "
        f"{n_parts} parts of {args.chunk} (worker {wi}/{wn} owns {len(mine)})")
    log(f"results → {cell.out}")
    log(f"query budget={args.query_budget or 'unlimited'}  "
        f"example timeout={args.example_timeout_min or 'none'} min  batch={args.batch or 32}")

    signal.signal(signal.SIGALRM, _on_alarm)
    signal.signal(signal.SIGTERM, _on_term)
    timeout_sec =int(args.example_timeout_min * 60) if args.example_timeout_min else 0

    for name, cls in recipes:
        if finalize(cell, name, n_parts, args.query_budget):
            log(f"[{name}] done already")
            continue
        todo = [k for k in mine if not cell.done(name, k).exists()]
        if not todo:
            log(f"[{name}] my parts are done; waiting on other workers to merge")
            continue
        try:
            attack = build_attack(cls, wrapper, args.batch, args.query_budget)
        except Exception as e:
            # No summary is written, so the next run retries this recipe.
            log(f"[{name}] BUILD FAILED: {type(e).__name__}: {e} -- skipping for now")
            with open(cell.logs / "build-errors.log", "a") as f:
                f.write(f"{time.ctime()} {cell.stem}/{cell.model_name}/{name}: {e}\n")
            continue
        textattack.shared.utils.set_seed(765)
        cell.parts_dir(name).mkdir(parents=True, exist_ok=True)
        log(f"[{name}] {len(todo)} part(s) to do")

        for k in todo:
            lo, hi = k * args.chunk, min(len(rows), (k + 1) * args.chunk)
            have = read_valid_lines(cell.part(name, k))
            done_idx = {json.loads(l)["idx"] for l in have}
            local = cell.work / f"{name}.part-{k:05d}.jsonl"
            with open(local, "w") as f:
                f.writelines(l + "\n" for l in have)
            t_part, t_sync, n_new = time.time(), time.time(), 0
            counts = {"S": 0, "F": 0, "K": 0, "E": 0}
            try:
                with open(local, "a") as fout:
                    for idx in range(lo, hi):
                        if idx in done_idx:
                            continue
                        text, lab = rows[idx]
                        if lab not in l2i:
                            continue
                        r = attack_one(attack, wrapper, text, l2i[lab], timeout_sec)
                        entry = {
                            "idx": idx, "row_shard": None, "dataset": args.dataset,
                            "model": cell.model_name, "recipe": name,
                            "result_type": r["result_type"], "ground_truth": l2i[lab],
                            "original_text": text, "original_label": r.get("original_label"),
                            "perturbed_text": r.get("perturbed_text"),
                            "perturbed_label": r.get("perturbed_label"),
                            "num_queries": r.get("num_queries"),
                        }
                        if "error" in r:
                            entry["error"] = r["error"]
                        fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
                        fout.flush()
                        n_new += 1
                        rt = r["result_type"]
                        counts["E" if rt in ERROR_TYPES else "S" if "Successful" in rt
                               else "F" if "Failed" in rt else "K"] += 1
                        if time.time() - t_sync >= args.sync_sec:
                            copy_to_drive(local, cell.part(name, k))
                            t_sync = time.time()
                            rate = (time.time() - t_part) / n_new
                            left = hi - lo - len(done_idx) - n_new
                            log(f"[{name}] part {k}/{n_parts - 1}  {len(done_idx) + n_new}/{hi - lo}  "
                                f"S{counts['S']} F{counts['F']} skip{counts['K']} err{counts['E']}  "
                                f"{rate:.1f}s/ex  part ETA {left * rate / 60:.0f}m  "
                                f"disk free {shutil.disk_usage(args.local).free // 2**30}GB")
            finally:
                # Also on Ctrl-C / the Colab stop button: what was attacked is kept.
                copy_to_drive(local, cell.part(name, k))
            cell.done(name, k).touch()
            local.unlink()
            log(f"[{name}] part {k} saved ({n_new} new, {time.time() - t_part:.0f}s)  "
                f"S{counts['S']} F{counts['F']} skip{counts['K']} err{counts['E']}")

        attack = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        finalize(cell, name, n_parts, args.query_budget)

    # Other workers may have finished recipes this one only contributed to.
    for name, _ in recipes:
        finalize(cell, name, n_parts, args.query_budget)
    cmd_status(args)


def cmd_status(args) -> None:
    cell = Cell(Path(args.drive), Path(args.local), args.dataset, args.model)
    if not all_file(args.dataset).exists():
        raise SystemExit(f"[FAIL] {all_file(args.dataset)} missing -- run the data cell first")
    n_rows = sum(1 for l in open(all_file(args.dataset)) if l.strip())
    n_parts = (n_rows + args.chunk - 1) // args.chunk
    print(f"\n  {cell.stem} × {cell.model_name}  ({n_rows} rows)  →  {cell.out}")
    finished = 0
    for name, _ in recipes_in_cost_order(args.recipes):
        if cell.summary(name).exists():
            s = json.loads(cell.summary(name).read_text())
            finished += 1
            print(f"  {name:32} DONE     ASR={s['attack_success_rate']:.1%}  "
                  f"macroF1 {s['clean_macro_f1']:.3f}→{s['attacked_macro_f1']:.3f}  "
                  f"errors={s.get('n_errors', 0)}")
            continue
        pd = cell.parts_dir(name)
        if not pd.exists():
            print(f"  {name:32} -")
            continue
        n_done = sum(len(read_valid_lines(p)) for p in pd.glob("part-*.jsonl"))
        parts_done = len(list(pd.glob("part-*.done")))
        print(f"  {name:32} {n_done:>7}/{n_rows} rows  ({parts_done}/{n_parts} parts)")
    print(f"\n  {finished}/{len(recipes_in_cost_order(args.recipes))} recipe files complete")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=["prepare", "check", "run", "status"])
    ap.add_argument("--dataset", required=True, help="e.g. e-care, corr2cause, Quriosity")
    ap.add_argument("--model", help="Hub id, e.g. namesarnav/e-care-bert-base-uncased")
    ap.add_argument("--labels", default="YES,NO",
                    help="label order if the model config does not cover the data")
    ap.add_argument("--drive", default="/content/drive/MyDrive/mdg")
    ap.add_argument("--local", default="/content/mdg-local")
    ap.add_argument("--chunk", type=int, default=200, help="rows per part file")
    ap.add_argument("--worker", default="0/1", help="i/n: split this cell across n sessions")
    ap.add_argument("--query-budget", type=int, default=None)
    ap.add_argument("--example-timeout-min", type=float, default=None)
    ap.add_argument("--batch", type=int, default=None, help="goal-function query batch")
    ap.add_argument("--sync-sec", type=int, default=120)
    ap.add_argument("--recipes", nargs="+", default=None)
    args = ap.parse_args()
    if args.command != "prepare" and not args.model:
        ap.error("--model is required")
    {"prepare": lambda a: prepare(a.dataset), "check": cmd_check,
     "run": cmd_run, "status": cmd_status}[args.command](args)


if __name__ == "__main__":
    main()
