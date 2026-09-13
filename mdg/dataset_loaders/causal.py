from typing import List, Optional, Dict, Any, Callable, Set
from datasets import load_dataset

from mdg.dataset_loaders import DatasetLoader
from mdg.registry import register, DATASET_LOADER


# Per-dataset column mapping. Keyed by hf_location.
#   id     : column name | callable(row)->id | None (auto index)
#   text   : column name | callable(row)->str   (the scenario+question shown to model)
#   label  : column name | callable(row)->str   (the gold answer)
#   labels : OPTIONAL. Fixed list of valid labels for this dataset, e.g.
#              ["YES", "NO"]                      -> binary
#              ["YES", "NO", "AMBIGUOUS"]          -> 3-way
#              ["A", "B", "C", "D", "E", "F", "G", "H"]  -> 8-way
#            Any number of classes is supported. Declare this whenever you
#            already know the dataset's label space (this also gets used to
#            filter out malformed rows automatically, see keep_labels below).
#            If omitted, the label space is auto-derived by scanning every
#            row's normalized label across all splits after loading.
# Add a row here whenever you upload a new causal dataset.

FIELD_MAPS: Dict[str, Dict[str, Any]] = {
    "namesarnav/counterbench": dict(
        id="question_id",
        text=lambda r: f"{r['given_info']}\nQuestion: {r['question']}",
        label="answer",
        labels=["YES", "NO"],
    ),
    "namesarnav/ac-reason": dict(
        id=None,
        text=lambda r: f"{r['story']}\nQuestion: {r['question']}",
        label="answer",
        labels=["YES", "NO"],
    ),
    "namesarnav/bbh-causal-judgement": dict(
        id=None,
        text="input",
        label="target",
        labels=["YES", "NO"],
    ),
    # Example of a 3-way dataset. Rename the hf_location key to match whatever
    # you actually upload BBH Extra Hard (Causal Understanding) as.
    "namesarnav/bbh-causal-understanding": dict(
        id=None,
        text="input",
        label="target",
        labels=["YES", "NO", "AMBIGUOUS"],
    ),
    "namesarnav/corr2cause": dict(
        id=None,
        text="input",
        label="label",
        labels=["0", "1"],
    ),
    # E-Care: choice-based cause/effect. label 0 = choice1, label 1 = choice2.
    "namesarnav/e-care": dict(
        id="idx",
        text=lambda r: (
            f"{r['premise']}\n"
            f"Question: What is the {r['question']}?\n"
            f"A: {r['choice1']}\n"
            f"B: {r['choice2']}"
        ),
        label="label",
        labels=["0", "1"],
    ),
    # FinCausal Task 1: binary causal sentence classification.
    "namesarnav/fincausal-task1": dict(
        id="index",
        text="text",
        label="gold",
        labels=["0", "1"],
    ),
    # CausalBench — three configs share the same label space (Yes/No).
    "namesarnav/causalbench:code": dict(
        id="Causal_Scenario_ID",
        text=lambda r: f"{r['Code']}\nQuestion: {r['Question']}",
        label="Ground Truth",
        labels=["YES", "NO"],
    ),
    "namesarnav/causalbench:math": dict(
        id="Scenario_ID",
        text=lambda r: f"{r['Mathematical Scenario']}\nQuestion: {r['Question']}",
        label="Ground Truth",
        labels=["YES", "NO"],
    ),
    "namesarnav/causalbench:text": dict(
        id="Scenario ID",
        text="Scenario and Question",
        label="Ground Truth",
        labels=["YES", "NO"],
    ),
}

# Datasets not listed above fall back to this. `labels=None` means "auto-derive
# from the data" -- so a brand-new dataset with an unknown/unlisted label space
# still works out of the box, it just costs one extra pass over the rows.
_DEFAULT_MAP = dict(id=None, text="text", label="label", labels=None)


def _norm_label(x) -> str:
    return "" if x is None else str(x).strip().upper().replace(" ", "_").replace("-", "_")


def _pretty_label(x: str) -> str:
    """Human-readable form of a normalized label, for showing choices in a
    prompt: NOT_ENTAILMENT -> 'Not Entailment'."""
    return x.replace("_", " ").title()


def _get(row: dict, spec) -> Any:
    if callable(spec):
        return spec(row)
    if spec is None:
        return None
    return row.get(spec)


def _route_splits(present: List[str]) -> Dict[str, Optional[str]]:
    """Map a dataset's real split names onto train/validation/test slots.
    Single-split datasets (e.g. counterbench=train-only, ac-reason=test-only)
    are exposed as the test set so evaluation always has data."""

    def pick(*names):
        return next((n for n in names if n in present), None)
    train = pick("train")
    val = pick("validation", "dev", "val")
    test = pick("test") or (present[0] if len(present) == 1 else None)
    return {"train": train, "validation": val, "test": test}


def _build_split(dataset_dict, real_split: Optional[str], fmap, keep_labels):
    if real_split is None or real_split not in dataset_dict:
        return []
    out = []
    for i, rec in enumerate(dataset_dict[real_split]):
        label = _norm_label(_get(rec, fmap["label"]))
        if keep_labels is not None and label not in keep_labels:
            continue
        rid = _get(rec, fmap["id"])
        rid = str(rid) if rid is not None else f"{real_split}-{i}"
        out.append({
            "id": rid,
            "doc_id": rid,
            "text": _get(rec, fmap["text"]),
            "label": label,
        })
    return out


def _derive_label_space(splits: Dict[str, Optional[List[dict]]]) -> List[str]:
    """Collect every distinct normalized label actually seen across all
    splits. Used as a fallback when a dataset's FIELD_MAP doesn't declare an
    explicit `labels` list -- lets a brand-new, uninspected dataset still work
    with any number of classes, not just binary."""
    seen: Set[str] = set()
    for recs in splits.values():
        if not recs:
            continue
        for rec in recs:
            if rec["label"]:
                seen.add(rec["label"])
    return sorted(seen)


class _BaseCausalLoaderHF(DatasetLoader):
    """Shared logic for causal classification loaders.
    Returns records with: id, doc_id, text, label, label_space, label_space_display.
    init_params: hf_location, keep_labels (optional set), field_map (optional override)."""

    DEFAULT_LOCATION = "mdg-nlp/causal-classification-sentence"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        cfg = self.config.get("config", self.config)
        self.dataset_location = cfg.get("hf_location", self.DEFAULT_LOCATION)
        # Optional HuggingFace config name (e.g. "code"/"math"/"text" for causalbench).
        # Can also be embedded in hf_location as "dataset:config".
        self.hf_config: Optional[str] = cfg.get("hf_config")
        if ":" in self.dataset_location and self.hf_config is None:
            self.dataset_location, self.hf_config = self.dataset_location.rsplit(":", 1)
        raw_keep = cfg.get("keep_labels")
        self.keep_labels = {_norm_label(l) for l in raw_keep} if raw_keep else None
        # allow an explicit field_map in the YAML; else look up by location+config; else default
        fmap_key = f"{self.dataset_location}:{self.hf_config}" if self.hf_config else self.dataset_location
        self.fmap = cfg.get("field_map") or FIELD_MAPS.get(fmap_key) or FIELD_MAPS.get(self.dataset_location, _DEFAULT_MAP)

        explicit_labels = self.fmap.get("labels")
        self.label_space: Optional[List[str]] = (
            [_norm_label(l) for l in explicit_labels] if explicit_labels else None
        )
        # If the dataset's valid label set is known and the caller didn't pass
        # their own keep_labels override, use it to silently drop rows with
        # unexpected/malformed label values instead of letting them corrupt
        # the eval set.
        if self.keep_labels is None and self.label_space is not None:
            self.keep_labels = set(self.label_space)

    def run(self, split: Optional[str] = None) -> Dict[str, Optional[List[dict]]]:
        if split is not None and split not in ("train", "validation", "test"):
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'validation', 'test', or None.")
        # Always load the full DatasetDict, then route real splits onto standard slots.
        dd = load_dataset(self.dataset_location, self.hf_config) if self.hf_config else load_dataset(self.dataset_location)
        routing = _route_splits(list(dd.keys()))
        result = {
            slot: _build_split(dd, routing[slot], self.fmap, self.keep_labels)
            for slot in ("train", "validation", "test")
        }

        # Resolve the label space once we've actually seen the data: use the
        # FIELD_MAP's explicit list if declared (any cardinality), otherwise
        # derive it empirically from whatever label values showed up.
        if self.label_space is None:
            self.label_space = _derive_label_space(result)
        display = [_pretty_label(l) for l in self.label_space]

        # Attach the resolved label space to every record so it travels with
        # the data straight into the prompt-building step -- the prompt
        # function can just read rec["label_space_display"] and list the
        # valid answer choices, whether there are 2 of them or 8.
        for recs in result.values():
            if recs:
                for rec in recs:
                    rec["label_space"] = self.label_space
                    rec["label_space_display"] = display

        if split is not None:
            return {split: result[split]}
        return result


@register(_type=DATASET_LOADER, _name="causal_hf")
class CausalDatasetLoaderHF(_BaseCausalLoaderHF):
    """Generic causal relation classification loader from HuggingFace."""
    DEFAULT_LOCATION = "namesarnav/counterbench"


@register(_type=DATASET_LOADER, _name="causal_hf_2")
class CausalDatasetLoaderHF2(_BaseCausalLoaderHF):
    """Second variant (e.g. perturbed / domain-shifted)."""
    DEFAULT_LOCATION = "mdg-nlp/causal-classification-sentence-perturbed"


@register(_type=DATASET_LOADER, _name="causal_hf_3")
class CausalDatasetLoaderHF3(_BaseCausalLoaderHF):
    """Document-level causal datasets."""
    DEFAULT_LOCATION = "mdg-nlp/causal-classification-document"