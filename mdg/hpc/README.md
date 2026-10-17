# Running the MDG attack grid

Fine-tunes **bert-base-uncased** and **roberta-base** on 11 causal
classification datasets, pushes each to the HuggingFace Hub, then runs **every
TextAttack recipe** against every model on **every data point** of every
dataset, and consolidates everything into one CSV.

Scale: 365,444 data points × 18 recipes × 2 models = **13.2M example-attacks**
across **594 cells** (11 datasets × 3 models × 18 recipes). Restrict the
attacks to bert and roberta with `ONLY_TASKS="$(seq -s' ' 0 21)"` if t5 is being
trained for its own sake rather than attacked.

Work is split two ways, and both are needed:

* **cells** — one (dataset, model, recipe) combination.
* **row shards** — `ROW_SHARDS=k` splits each dataset's rows k ways, so one cell
  can be worked by k machines at once. Cell-level splitting alone cannot
  finish: corr2cause is 208k rows, and a single recipe over it on one core is
  weeks, no matter how many machines you add.

## Your own cluster or VMs (no scheduler) — start here

Per machine:

```bash
# one command on a bare box: pyenv 3.11 + poetry + clone + LFS + deps
curl -fsSL https://raw.githubusercontent.com/namesarnav/mdg/main/mdg/hpc/bootstrap_vm.sh | bash
cd ~/mdg
echo 'hf_xxx' > ~/.hf_token && chmod 600 ~/.hf_token
bash mdg/hpc/warm_cache.sh                # download recipe assets once, serially
```

Training on CPU is memory-bandwidth bound, so threads buy far less than cores
suggest — measured on an 8-core laptop, 1 → 10 threads gave 1.3x, not 10x.
Measure your own box and get hours per remaining task:

```bash
poetry run python mdg/hpc/bench_train.py --epochs 2
```

Run several tasks at once rather than one task with every thread. A single task
cannot be split, so the largest one sets the floor however many machines you
have: corr2cause is 206k rows at ~249 tokens, which is 92% of the training cost
of the whole grid.

Train (several workers, a handful of threads each). Split
the 22 tasks across machines with `ONLY_TASKS`, and **wait for every machine to
finish pushing** before attacking, since the attack stage pulls models from the
Hub:

```bash
ONLY_TASKS="0 1 2 3" JOBS=4 THREADS_PER_JOB=16 bash mdg/hpc/run_parallel.sh train
```

Attack (many single-threaded workers — inference does not scale across
threads). Set `ROW_SHARDS` to the number of machines, and give each machine its
own `SHARD`:

```bash
bash mdg/hpc/preflight.sh      # every task must say local or hub, not MISSING
bash mdg/hpc/warm_cache.sh     # ~5GB of shared recipe assets, fetched once

# on every machine, identical except for SHARD — here, 4 machines
export ROW_SHARDS=8
DRY_RUN=1 SHARD=0/4 bash mdg/hpc/run_parallel.sh cells        # check sizing first
SHARD=0/4 nohup bash mdg/hpc/run_parallel.sh cells > ~/attack.log 2>&1 &
```

Each machine then takes an exactly equal quarter of every dataset, so no
machine inherits corr2cause while another gets counterbench. Eight row shards
across four machines rather than four is deliberate: the units are half the
size, so a machine killed mid-unit loses half as much work. Units are started biggest-first so
the run does not end with one huge unit holding up an otherwise idle box.
Workers are sized automatically from cores and RAM: 64 cores / 256GB gives 64.
Everything resumes — finished units are skipped, so re-run after any
interruption.

`ROW_SHARDS` and `SHARD` are independent: `ROW_SHARDS` is how finely the rows
are cut, `SHARD` is which cuts this machine takes. What must never change
mid-run is `ROW_SHARDS`, because the shard count is baked into the output
filenames — a half-finished grid with mixed shard counts cannot be merged.

Monitor with `.venv/bin/python mdg/hpc/status.py`.

### Finishing

Collect every machine's results onto one box, fold the shards back together,
then consolidate:

```bash
for h in vm2 vm3 vm4 vm5 vm6 vm7 vm8; do
  rsync -av "$h:~/mdg/mdg/adv_attack/results/" ~/mdg/mdg/adv_attack/results/
done
python -m mdg.scripts.merge_shards --backfill    # shards → one result per cell
python -m mdg.scripts.consolidate_results
```

`merge_shards` recomputes micro/macro F1 from the merged per-example records
rather than averaging the shards — macro F1 is not a weighted mean, so
averaging would be quietly wrong. A cell missing any shard is reported and left
alone rather than merged half-complete. `--backfill` also fills in F1 for older
runs whose summaries predate the metric, using the perturbed JSONL on disk.

### Train on a GPU, attack on CPU

Measured on a 64-core VM, corr2cause/bert projected **116 hours** of CPU
training. The same run on a free Colab T4 is 1–3 hours, because training is one
batched matmul stream. Do not reach for a GPU for the attacks: those are 13M
sequential adversarial searches, which one T4 would take months to finish while
a CPU fleet runs 192 of them at once.

One Colab cell:

```python
!git clone -q https://github.com/namesarnav/mdg.git /content/mdg
%env HF_TOKEN=hf_xxxxxxxxxxxxx
!cd /content/mdg && bash mdg/hpc/train_gpu.sh "6 17"
```

Parallel jobs do not help on one GPU — a single bert-base run already saturates
it, so a second only splits the same cores and may run out of VRAM. The script
therefore runs **one task per GPU** and works through the list in order. On
Kaggle's T4 x2 that means two at a time, automatically.

The way to get real parallelism from free tiers is to use both at once: Colab on
one half of the list, Kaggle on the other. Each task pushes to the Hub as it
finishes, so the two never collide and a disconnect costs only the task in
flight.

Runtime > Change runtime type > **T4 GPU** first, or the script refuses. It
installs the dependencies (never torch — the hosted build is matched to the
driver), fetches the LFS data, trains through the same `run_one.sh` path as the
cluster, and pushes each model. Kaggle works the same way with a 30h/week quota
and 12h sessions.

Keep the CPU machines on attacks the whole time; they need no local
checkpoints, only the Hub copies.

### Task ids

Tasks 0–10 are bert-base-uncased, 11–21 are roberta-base, 22–32 are t5-base,
in this dataset order (rows = every data point, what the attacks run on):

| | dataset | rows | bert | roberta | t5 |
|---|---|---|---|---|---|
| | counterbench | 1,320 | 0 | 11 | 22 |
| | ac-reason | 4,000 | 1 | 12 | 23 |
| | bbh-causal-judgement | 4,000 | 2 | 13 | 24 |
| | causalbench_code | 7,856 | 3 | 14 | 25 |
| | causalbench_math | 12,895 | 4 | 15 | 26 |
| | causalbench_text | 37,602 | 5 | 16 | 27 |
| | corr2cause | 207,972 | 6 | 17 | 28 |
| | e-care | 17,050 | 7 | 18 | 29 |
| | fincausal-task1 | 18,749 | 8 | 19 | 30 |
| | natquest | 27,000 | 9 | 20 | 31 |
| | Quriosity | 27,000 | 10 | 21 | 32 |

Split *training* by rows, not by task count — corr2cause is 57% of all rows and
deserves a machine to itself. For 4 machines:

| VM | `ONLY_TASKS` | `JOBS` | `THREADS_PER_JOB` | rows |
|---|---|---|---|---|
| 1 | `6` | 1 | 64 | 208k |
| 2 | `17` | 1 | 64 | 208k |
| 3 | `0 1 2 3 4 5 7 8 9 10` | 5 | 12 | 157k |
| 4 | `11 12 13 14 15 16 18 19 20 21` | 5 | 12 | 157k |

VMs 1 and 2 take corr2cause (bert and roberta); 3 and 4 take everything else,
split by model. For 8 machines, halve the lists on VMs 3 and 4 and give
corr2cause's two tasks a machine each with `TRAIN_ARGS="--epochs 2"`.

The *attacks* need no such care — `ROW_SHARDS` splits every dataset evenly.

### How many machines

One 64-core machine is ~64 workers, and the whole grid is roughly 36,000
core-hours at the measured ~10s per example-attack:

| Machines (64 cores each) | Workers | Compute | Realistic |
|---|---|---|---|
| 4 | 256 | ~6 days | ~1.5 weeks |
| 8 | 512 | ~3 days | ~1 week |
| 16 | 1024 | ~1.5 days | ~3 days |

`QUERY_BUDGET=200` is the one lever that changes this materially — it caps the
search per example and roughly halves the slowest recipes, at the cost of
understating their attack success rate.

## Scripts

| File | Purpose |
|---|---|
| `bootstrap_vm.sh` | Bare VM → working checkout: pyenv, Poetry, Git LFS, deps |
| `env.sh` | Paths, runtime detection, thread limits. Sourced by everything |
| `matrix.sh` | The grid: datasets × models. Edit here to change scope |
| `run_parallel.sh` | `train`, `cells` (attack), or `all` — the main runner |
| `run_one.sh` | One (model × dataset) task; shared by every path |
| `warm_cache.sh` | Download every recipe's assets once, serially, before the fleet starts |
| `status.py` | Progress: trained models, finished recipes, shards in flight |
| `preflight.sh` | Check every task's weights resolve (local or Hub) before launching |
| `preflight_check.py` | Hub/weight resolution used by `preflight.sh` and `status.py` |
| `train_gpu.sh` | Train on Colab/Kaggle/any CUDA box and push to the Hub |
| `watch_train.py` | Live table of training progress parsed out of the logs |
| `bench_train.py` | Measure this box's CPU training throughput; project hours per task |
| `01_prepare_data.sh` | Regenerate the `<stem>__{train,test}.jsonl` splits |
| `02_build_all_splits.sh` | Concatenate train+test+validation into `<stem>__all.jsonl` |
| `../scripts/merge_shards.py` | Fold row shards into one result per cell; backfill old F1 |
| `legacy/` | Slurm job arrays and superseded env builders — see its README |

## Knobs

| Variable | Default | Effect |
|---|---|---|
| `ATTACK_SPLIT` | `all` | `all` = every data point; `holdout` = the untrained split only |
| `NUM_EXAMPLES` | `-1` | Examples per recipe; `-1` is every one |
| `QUERY_BUDGET` | unset | Cap model queries per example |
| `JOBS` | auto | Concurrent workers |
| `THREADS_PER_JOB` | `1` | Threads per worker |
| `MEM_PER_WORKER_GB` | `3` | Planning figure for worker sizing |
| `ROW_SHARDS` | `1` | Split every dataset's rows this many ways |
| `SHARD` | unset | `i/n` — this machine's slice of the units |
| `ONLY_TASKS` | unset | Restrict to these task ids |
| `TRAIN_ARGS` | unset | Replace the training flags, e.g. `"--epochs 2 --batch 32 --lr 2e-5"` |
| `REQUIRE_LOCAL` | unset | Attack only pairs trained on this machine |
| `RECIPES` | unset | Restrict to these recipe names |
| `DRY_RUN` | unset | Print sizing and pending work, launch nothing |

## Outputs

| Path | Contents |
|---|---|
| `mdg/finetune/checkpoints/<stem>/<model>/` | Trained model |
| `mdg/finetune/train_results.csv` | Per model × dataset: micro + macro F1, accuracy |
| `mdg/adv_attack/results/<split>/<model>/<recipe>.sh<i>of<n>.jsonl` | One row shard's attacked examples, before merging |
| `mdg/adv_attack/results/<split>/<model>/<recipe>.jsonl` | **Every attacked example**: ground truth, original and perturbed text + predictions, query count |
| `mdg/adv_attack/results/<split>/<model>/<recipe>_summary.json` | ASR, clean and attacked micro + macro F1 |
| `mdg/results/consolidated.csv` | **Final CSV** — one row per cell, every metric |

