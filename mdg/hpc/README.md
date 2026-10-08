# Running the MDG attack grid

Fine-tunes **bert-base-uncased** and **roberta-base** on 11 causal
classification datasets, pushes each to the HuggingFace Hub, then runs **every
TextAttack recipe** against every model on **every data point** of every
dataset, and consolidates everything into one CSV.

Scale: 365,444 data points × 18 recipes × 2 models = **13.2M example-attacks**
across **396 cells** (dataset × model × recipe).

## Your own cluster or VMs (no scheduler) — start here

Per machine:

```bash
git clone https://github.com/namesarnav/mdg.git ~/mdg && cd ~/mdg
bash mdg/hpc/setup_vm.sh                  # conda env; no container needed
echo 'hf_xxx' > ~/.hf_token && chmod 600 ~/.hf_token
bash mdg/hpc/02_build_all_splits.sh       # build the <stem>__all.jsonl files
```

Train (few workers, many threads each — training scales across threads):

```bash
ONLY_TASKS="0 1 2 3 4 5 6" JOBS=7 THREADS_PER_JOB=16 bash mdg/hpc/run_parallel.sh train
```

Attack (many single-threaded workers — inference does not scale across threads):

```bash
DRY_RUN=1 bash mdg/hpc/run_parallel.sh cells      # check the sizing first
SHARD=0/3 nohup bash mdg/hpc/run_parallel.sh cells > ~/attack.log 2>&1 &
```

`SHARD=i/n` gives each machine a disjoint slice. Workers are sized
automatically from cores and RAM — on 144 cores / 512GB that is ~144 workers.
Everything resumes: finished cells are skipped, so just re-run after any
interruption.

Monitor with `bash mdg/hpc/status.sh`, merge at the end with
`python -m mdg.scripts.consolidate_results --extra /path/to/other/results`.

## Scripts

| File | Purpose |
|---|---|
| `setup_vm.sh` | Build the conda env on any Linux box (no Slurm, no Singularity) |
| `02_build_all_splits.sh` | Concatenate train+test+validation into `<stem>__all.jsonl` |
| `run_parallel.sh` | `train`, `cells` (attack), or `all` — the main runner |
| `run_one.sh` | One (model × dataset) task; shared by every path |
| `run_budget.sh` | Work cheapest-first for N hours then stop (capped sessions) |
| `status.sh` | Progress: trained pairs, recipes done, overall % |
| `migrate_results.sh` | Fold older result layouts into the canonical one |
| `matrix.sh` | The grid: datasets × models. Edit here to change scope |
| `check_access.sh` | Slurm pre-flight (NYU HPC only) |
| `00_setup_env.sh` | Singularity overlay env (NYU HPC only) |
| `*.sbatch`, `submit_all.sh` | Slurm job arrays (NYU HPC only) |

## Knobs

| Variable | Default | Effect |
|---|---|---|
| `ATTACK_SPLIT` | `all` | `all` = every data point; `holdout` = the untrained split only |
| `NUM_EXAMPLES` | `-1` | Examples per recipe; `-1` is every one |
| `QUERY_BUDGET` | unset | Cap model queries per example |
| `JOBS` | auto | Concurrent workers |
| `THREADS_PER_JOB` | `1` | Threads per worker |
| `MEM_PER_WORKER_GB` | `3` | Planning figure for worker sizing |
| `SHARD` | unset | `i/n` — this machine's slice of the cells |
| `ONLY_TASKS` | unset | Restrict to these task ids |
| `RECIPES` | unset | Restrict to these recipe names |
| `DRY_RUN` | unset | Print sizing and pending work, launch nothing |

## Outputs

| Path | Contents |
|---|---|
| `mdg/finetune/checkpoints/<stem>/<model>/` | Trained model |
| `mdg/finetune/train_results.csv` | Per model × dataset: micro + macro F1, accuracy |
| `mdg/adv_attack/results/<split>/<model>/<recipe>.jsonl` | **Every attacked example**: ground truth, original and perturbed text + predictions, query count |
| `mdg/adv_attack/results/<split>/<model>/<recipe>_summary.json` | ASR, clean and attacked micro + macro F1 |
| `mdg/results/consolidated.csv` | **Final CSV** — one row per cell, every metric |

---

# NYU HPC (Greene / Torch) — only if you use it

## Check you can get a GPU first

```bash
bash mdg/hpc/check_access.sh            # or: bash mdg/hpc/check_access.sh a100
```

Lists your Slurm accounts, shows how busy each candidate partition is, then asks
for one GPU for five minutes and runs `nvidia-smi`. If this fails, nothing else
will work — the error tells you whether it is the account, the partition, or a
full cluster. Safe to run from a login node.

### Which GPU to ask for

t5-base is 220M parameters and Llama-3.2-1B is 1B — in bf16 with LoRA they need
roughly **10–20 GB**, so a 24–48 GB card is ample. Asking for an H200 or B200
buys nothing here except a longer queue.

| Partition | Card | Nodes | Verdict |
|---|---|---|---|
| `a100` | A100 | 43 | **Default.** Ample memory and plenty of nodes |
| `l40s` | L40S 48GB | 68 | Most capacity on the cluster — good fallback |
| `rtx6000` | RTX 6000 24GB | 6 | Fits, but few nodes — expect queueing |
| `h100`/`h200`/`b200` | — | — | Overkill. Longer queue, no benefit |

Throughput here is decided by how many tasks run **concurrently**, not by card
speed — the attack stage is the long pole. `a100` (43 nodes) is the default; `l40s`
(68 nodes) is the fallback if the a100 queue is long.

### Slurm account (Torch)

Torch requires a **project account**; without one every `srun`/`sbatch` fails
with `Invalid Slurm account: users`. Your PI registers the project at
https://projects.hpc.nyu.edu. Once you have it:

```bash
sacctmgr -nP show assoc user=$USER format=account   # what you belong to
echo torch_pr_xxx_yyy > ~/.slurm_account            # picked up automatically
```

`env.sh` reads `~/.slurm_account` (or `$ACCOUNT`), and `submit_all.sh` passes it
to every stage. Interactive sessions need it explicitly: `srun -A torch_pr_xxx_yyy ...`

**Several projects?** An account is a charge code, not a lock — you can belong to
many, and any number of jobs (the 22-task array included) can run under one.
`~/.slurm_account` is just the default; override per run:

```bash
ACCOUNT=torch_pr_other bash mdg/hpc/submit_all.sh
ACCOUNT=torch_pr_other bash mdg/hpc/run_one.sh train 0
```

What caps concurrent jobs is the QOS/partition limits on the account
(`MaxJobs`/`MaxSubmit`), not this setting. Check yours with:
`sacctmgr -nP show assoc user=$USER format=account,qos,maxjobs`

### Partitions (Torch)

Torch has **no default GPU partition**, so every job must name one. `env.sh`
defaults to `PARTITION=a100`; `sinfo -s` lists what exists (`l40s`, `a100`,
`h100`, `h200`, `b200`, `rtx6000`, plus `*_public` / `*_plus` variants you may
or may not have access to). Override per run:

```bash
PARTITION=l40s bash mdg/hpc/submit_all.sh
PARTITION=a100 CPU_PARTITION=cpu_short bash mdg/hpc/submit_all.sh
```

An A100 is ample for t5-base and Llama-3.2-1B; there is no reason to queue for
an H200. Submitting an sbatch file by hand needs the flag too:
`sbatch -p a100 --array=0-21 mdg/hpc/train.sbatch`. On Greene, set `PARTITION=""`.

## Run everything

```bash
cd /scratch/$USER/mdg
bash mdg/hpc/submit_all.sh
```

That submits three dependent stages: train (22 GPU tasks) → attack (22 GPU
tasks, starts only if training succeeded) → consolidate (CPU).

Or submit stages by hand:

```bash
sbatch mdg/hpc/train.sbatch
sbatch mdg/hpc/attack.sbatch                     # 200 examples/recipe
NUM_EXAMPLES=500 sbatch mdg/hpc/attack.sbatch    # more thorough, much slower
sbatch mdg/hpc/consolidate.sbatch
```

Monitor:

```bash
squeue -u $USER                                  # queue state
tail -f /scratch/$USER/mdg-logs/train-*_0.out    # live log of one task
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS   # after the fact
```

Both stages are **resumable**: re-running skips any pair with an existing
checkpoint (train) or `all_summaries.json` (attack), so just resubmit after a
timeout or node failure.

## Running everything at once with GNU parallel

A Slurm **job array already runs all 22 tasks concurrently**, spread across as
many nodes as the scheduler gives you — that is the fastest option, and
`submit_all.sh` is what you want when Slurm is available.

Use `run_parallel.sh` when you hold one interactive multi-GPU node, or are on a
machine without a scheduler. It packs the grid onto the node you are on, one
task per GPU:

```bash
srun --gres=gpu:4 --cpus-per-task=32 --mem=200G --time=8:00:00 --pty /bin/bash
cd /scratch/$USER/mdg

bash mdg/hpc/run_parallel.sh train     # all 22 training runs, 4 at a time
bash mdg/hpc/run_parallel.sh attack    # all 22 attack runs
bash mdg/hpc/run_parallel.sh all       # train → attack → consolidate
JOBS=8 bash mdg/hpc/run_parallel.sh attack   # override the concurrency
```

Concurrency defaults to the GPU count, and slot *N* is pinned to GPU *N-1* via
`CUDA_VISIBLE_DEVICES`. **Running more tasks than GPUs makes them share VRAM and
usually ends in OOM** — only raise `JOBS` above the GPU count for attack recipes
whose search is CPU-bound.

One failing pair never kills the grid (`--halt never`). Every task writes
`/scratch/$USER/mdg-logs/<stage>-task<N>.log`, and a `--joblog` records exit
codes, so a re-run skips finished work and you can retry only the failures:

```bash
parallel --joblog /scratch/$USER/mdg-logs/parallel-attack.joblog --resume-failed ...
```

If `parallel` is missing, try `module load parallel`; the script prints an
`xargs -P` fallback if that fails.

## Outputs

| Path | Contents |
|---|---|
| `mdg/finetune/checkpoints/<stem>/<model>/` | Trained model (LoRA merged in for Llama) |
| `mdg/finetune/train_results.csv` | Per model × dataset: **micro F1, macro F1**, accuracy |
| `mdg/adv_attack/results/<stem>/<model>/<recipe>.jsonl` | **Every attacked example**: ground truth, original text + prediction, perturbed text + prediction, query count |
| `mdg/adv_attack/results/<stem>/<model>/<recipe>_summary.json` | ASR, clean/attacked micro + macro F1 |
| `mdg/adv_attack/attack_results.csv` | One row per model × dataset × recipe, appended live |
| `mdg/results/consolidated.csv` | **Final joined CSV** — train F1, clean F1, attacked F1, F1 drop, ASR |

`consolidated.csv` columns: `model, dataset, recipe, train_macro_f1,
train_micro_f1, train_accuracy, num_examples, clean_micro_f1, clean_macro_f1,
attacked_micro_f1, attacked_macro_f1, micro_f1_drop, macro_f1_drop,
attack_success_rate, avg_queries, …`

## Tuning the grid

Add or remove models and datasets in `matrix.sh` — both stages read it, so the
array size follows automatically (`submit_all.sh` computes the range; if you
`sbatch` by hand, update `#SBATCH --array=0-N`).

Attacks run on **every example of the attacked split with no query cap** by
default (`NUM_EXAMPLES=-1`, `QUERY_BUDGET` unset). That is 48,046 examples per
model across the 11 datasets, × 18 recipes × 2 models ≈ **1.7M example-attacks**,
each costing hundreds of model queries.

Expect this to run for **weeks**, not days, even with all 22 tasks in parallel.
To trade coverage for time, cap either axis:

```bash
NUM_EXAMPLES=500 sbatch mdg/hpc/attack.sbatch    # 500 examples per recipe
QUERY_BUDGET=2000 sbatch mdg/hpc/attack.sbatch   # cap the search per example
```

The two biggest splits dominate: `natquest` and `Quriosity` are 13,500 rows each
(56% of all attacked examples between them).

## Time and resource notes

- Training t5-base: minutes to ~2 hours per dataset. Llama-3.2-1B with LoRA:
  roughly 1–4 hours. The 12 h wall clock has headroom; `corr2cause` is the big one.
- Attacks are the expensive stage. On full splits a single (model, dataset) pair
  can take days, hence the 7-day limit. If a task hits the limit,
  completed recipes are already written and you can resubmit, but delete that
  pair's partial `<recipe>.jsonl` for whichever recipe was mid-run.
- Llama trains **unquantized** (`--no-4bit`) so the LoRA adapter can be merged
  into a plain classifier that TextAttack can load. 1B in bf16 fits a single
  GPU comfortably.
- If a GPU type is needed explicitly, add e.g. `#SBATCH --constraint=a100`.
