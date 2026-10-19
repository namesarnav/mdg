# colab

Fallback for when the VMs don't work out. Attacks the **complete test split**
of every dataset — no example cap, no query budget.

## Use the notebook

Open `colab/mdg_colab.ipynb` in Colab (File → Open notebook → GitHub →
`namesarnav/mdg`). Run cells 1–5 once per session, then cell 6 repeatedly.

**Pick a GPU runtime.** Runtime → Change runtime type → T4. Attacks are
thousands of model forward passes per example; `attack.py` moves the model to
CUDA in fp16 automatically when a GPU is present.

## Or just the scripts

```bash
# in a cell, after mounting Drive and writing ~/.hf_token
!MDG_DRIVE=/content/drive/MyDrive/mdg bash colab/setup.sh
!MDG_DRIVE=/content/drive/MyDrive/mdg bash colab/run.sh 11
```

| Script | |
|---|---|
| `setup.sh` | deps, clone, `git lfs pull`, 33 models, attack cache. Once per session. |
| `run.sh N` | works N hours, stops cleanly between units. Re-run to resume. |

## Where output goes

```
/content/drive/MyDrive/mdg/
├── results/namesarnav_<dataset>__test/<model>/<recipe>.sh<i>of64.jsonl
├── results/.../<recipe>.sh<i>of64_summary.json
├── logs/colab-run.log
├── hf_cache/            33 models, so a new session re-downloads nothing
└── textattack_cache/
```

Results are written to **local disk** and rsynced to Drive **every 3 minutes**,
after every unit, and on exit.

**Every file is kept**, not just the summaries:

| File | What it is |
|---|---|
| `<recipe>.sh<i>of<n>.jsonl` | **one line per attacked example** — gold label, original and perturbed text, both predictions, query count. The experimental record. |
| `<recipe>.sh<i>of<n>_summary.json` | ASR, clean and attacked micro/macro F1 for that shard |
| `all_summaries*.json` | every recipe's summary for that cell |
| `attack_results.csv` | one appended row per finished cell |
| `logs/colab-run-shard<i>of<n>.log` | full output, per session |
| `logs/mirror-errors.log` | **check this** — empty means every copy succeeded |

`run.sh` prints a count every 10 units and at the end, so you can see it rather
than assume it:

```
saved: 412 jsonl / 406 summaries local  ->  412 jsonl / 406 summaries on Drive
```

If Drive is behind local it says so and points at `mirror-errors.log`. A
`.jsonl` with no matching `_summary.json` is a unit that was interrupted — its
partial records are still kept, and the unit re-runs next session.

They are deliberately *not* symlinked into Drive. `attack.py` holds the
per-example `.jsonl` open for the whole unit (`attack.py:267`), and Drive's
FUSE mount does not publish a file's contents until it is closed — so a
symlinked layout showed an empty Drive folder for hours, which looks exactly
like nothing happening. rsync of closed files works correctly.

Check at any time:

```python
!ls -laR /content/drive/MyDrive/mdg/results | head -40
!tail -40 /content/drive/MyDrive/mdg/logs/colab-run.log
```

## Built for a session that dies

Colab will disconnect, probably mid-run. That costs you at most one unit:

- **Results reach Drive within 3 minutes** of being written, and a new session
  restores them back from Drive so finished units are seen and skipped.
- **Model and asset caches live on Drive** too (`HF_HOME`, `MDG_CACHE_ROOT`),
  so a new session skips ~20GB of downloads.
- **Finished units are skipped.** `run_one.sh` checks its own done-marker, so
  re-running picks up exactly where it stopped.
- **`run.sh` stops between units**, never mid-attack, when its budget expires.

After a disconnect: re-run cells 2–5 (fast, everything is cached) then cell 6.

## Cheapest work first

`run.sh` orders units by estimated total queries — rows × mean words ×
queries-per-word — so a short session finishes whole cells instead of stalling
inside the most expensive one.

This matters because `attack.py`'s recipe list is *not* in cost order:
CheckList2020, the cheapest recipe there is, sits at index 10, and PSOZang2020
at 12. Ordering by list position would hand a 12-hour session the most
expensive work first. Four of the eighteen per-word costs are measured from
completed runs in this repo; the rest are estimates of the same shape.

## Several sessions at once

`SHARD=i/n` splits the grid into disjoint slices. In cell 6, set `SHARD` per
session:

| Session | `SHARD` | Units |
|---|---|---:|
| 1 | `0/4` | 9,504 |
| 2 | `1/4` | 9,504 |
| 3 | `2/4` | 9,504 |
| 4 | `3/4` | 9,504 |

Verified exhaustive and disjoint at `ROW_SHARDS=64`: the four slices sum to
exactly 38,016 units (33 × 18 × 64) with no duplicates, so four sessions never
repeat each other's work.

## GPU sizing

`run.sh` reads the GPU and sizes itself:

| GPU | Workers | Query batch |
|---|---:|---:|
| L4 / A100 (≥20GB) | 8 | 256 |
| T4 (≥14GB) | 4 | 128 |
| smaller GPU | 4 | 64 |
| CPU only | 4 | TextAttack default (32) |

The batch size matters more than the worker count. TextAttack's goal function
defaults to **32** queries per forward pass, which leaves most of an L4 idle —
a PSO-class recipe issues 3,500+ forward passes per example and they batch
perfectly. `attack.py` now reads `MDG_MODEL_BATCH` and sets it on the goal
function (the knob is there, not on `AttackArgs`), defensively, since the
attribute name has moved between TextAttack versions.

Override either:

```bash
!... JOBS=6 MDG_MODEL_BATCH=512 bash colab/run.sh 11
```

If you see CUDA OOM, lower `MDG_MODEL_BATCH` before lowering `JOBS`.

## Keep ROW_SHARDS at 64

The shard count is baked into output filenames (`<recipe>.sh3of64.jsonl`). A
cell attacked here at a different count **cannot be merged** with the same cell
attacked on a VM. 64 is the fleet default; leave it alone.

## Honest expectation

Colab gives you one GPU and 2–8 vCPU for 12 hours at a time. That is far less
than three 64-core VMs, and the uncapped grid is large. Use this to keep
progress alive when the VMs are down, or to get the cheap recipes finished — not
as the primary path for all 18 recipes on all 11 datasets.

The one thing Colab has that the VMs don't is a GPU, and PSO-class recipes are
the most batchable workload in the grid. If a GPU session turns out to be much
faster per example than a VM core, that is worth knowing — the figure to watch
is the `s/it` in `$MDG_DRIVE/logs/colab-run.log`, against the VMs' 111 s/example.

## When it's all done

```bash
!cd /content/mdg && python3 mdg/scripts/merge_shards.py
!cd /content/mdg && python3 -m mdg.scripts.consolidate_results
```

Final table: `mdg/results/consolidated.csv`, copied to Drive by cell 8.
