# mac-scripts

One script per dataset. Each runs that dataset's **full attack pipeline** on
this Mac: 3 models × 18 recipes, on the **complete test split**, with no
example cap and no query budget.

```bash
bash mac-scripts/setup.sh                          # once, before anything else
caffeinate -i bash mac-scripts/counterbench.sh     # one dataset
caffeinate -i bash mac-scripts/all.sh              # all 11, smallest first
bash mac-scripts/status.sh                         # progress, any time
```

`caffeinate -i` stops the Mac sleeping mid-run. Without it a closed lid ends
the run.

`setup.sh` builds `./.venv` with pyenv + Poetry, pulls the LFS data, and checks
that all 18 recipes import. The dataset scripts refuse to launch without it —
they verify the data and warm the asset cache first, so a missing dependency
surfaces in seconds rather than 20 minutes into an attack.

## What a script does

1. Checks the test split exists and is real data, not a Git LFS pointer
2. Warms the TextAttack asset cache once, serially (`.cache-warmed` marks it)
3. Attacks all 18 recipes × 3 models via `mdg/hpc/run_parallel.sh cells`
4. Merges row shards back into one result per cell
5. Prints how many recipes finished per model

Training is **not** a step. All 33 models are on the Hub, and `run_one.sh`
falls back to the Hub copy when no local checkpoint exists.

## Datasets

| Script | Test rows | Task ids |
|---|---:|---|
| `counterbench.sh` | 330 | 0, 11, 22 |
| `ac-reason.sh` | 1,000 | 1, 12, 23 |
| `bbh-causal-judgement.sh` | 1,000 | 2, 13, 24 |
| `corr2cause.sh` | 1,162 | 6, 17, 28 |
| `causalbench-code.sh` | 1,572 | 3, 14, 25 |
| `e-care.sh` | 2,122 | 7, 18, 29 |
| `causalbench-math.sh` | 2,580 | 4, 15, 26 |
| `fincausal-task1.sh` | 3,750 | 8, 19, 30 |
| `causalbench-text.sh` | 7,530 | 5, 16, 27 |
| `natquest.sh` | 13,500 | 9, 20, 31 |
| `quriosity.sh` | 13,500 | 10, 21, 32 |

Task ids are derived from `mdg/hpc/matrix.sh` at runtime, not hardcoded, so
appending a model to `MODELS` keeps them correct.

## Resuming

Every finished recipe writes a done-marker. Re-running a script skips what is
already done and continues — interrupt freely.

## Do not change ROW_SHARDS

`ROW_SHARDS` defaults to **6**, matching the VM fleet. The shard count is baked
into the output filenames (`<recipe>.sh3of6.jsonl`), so a cell attacked here
with a different count **cannot be merged** with the same cell attacked on a
VM. Only override it for a dataset no other machine is touching.

## Sizing

`run_parallel.sh` picks the worker count from this Mac's cores and RAM
(~3GB per worker, since each loads its own model). Override with `JOBS=N`.

On Apple Silicon each worker uses the GPU via MPS. If you see MPS errors or
thrashing, lower `JOBS` — the workers share one GPU, unlike the VMs where each
has its own core.

## A realistic expectation

Measured on a VM core, `PWWSRen2019` on natquest runs at **111 s/example**.
A Mac is faster per worker but has far fewer of them. `counterbench.sh`
(330 rows) is a sensible evening; `natquest.sh` (13,500 rows × 18 recipes ×
3 models) is not a laptop job — leave that one to the fleet.
