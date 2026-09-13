from typing import List, Optional, Dict


def stratified_sample(
    records: List[dict],
    label_space: List[str],
    min_seeds: int = 2,
) -> Dict[str, List[dict]]:
    """k = max(min_seeds, min(Lt) // 10) seeds per class."""
    by_label: Dict[str, List[dict]] = defaultdict(list)
    for r in records:
        if r["_norm_label"] in label_space:
            by_label[r["_norm_label"]].append(r)

    counts = [len(by_label.get(l, [])) for l in label_space]
    min_count = min(counts) if counts else 0
    k = max(min_seeds, min_count // 10)

    print(f"  Class counts: {dict(zip(label_space, counts))}")
    print(f"  min(Lt)={min_count}  →  k = max({min_seeds}, {min_count}//10) = {k} seeds per class")

    sampled: Dict[str, List[dict]] = {}
    for label in label_space:
        pool = by_label.get(label, [])
        sampled[label] = random.sample(pool, min(k, len(pool)))
    return sampled


def format_seed_list(sampled: Dict[str, List[dict]], schema: Dict) -> str:
    """Format seeds in the original field format (no _norm_label/_raw_label)."""
    fields = schema["fields"]
    lines = []
    for recs in sampled.values():
        for r in recs:
            entry = {f: r[f] for f in fields if f in r}
            lines.append(json.dumps(entry, ensure_ascii=False))
    return "\n".join(lines)
