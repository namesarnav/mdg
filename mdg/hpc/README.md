# Running MDG on NYU HPC (Greene or Torch)

Fine-tunes **t5-base** and **meta-llama/Llama-3.2-1B** on all 11 causal
classification datasets, pushes each model to the HuggingFace Hub, then attacks
every model with every TextAttack recipe on every dataset and consolidates
everything into one CSV.

Grid: **2 models × 11 datasets = 22** training runs, then 22 attack runs ×
18 recipes = **396 model/dataset/recipe combinations**.

## Chipping away on free GPU hours (Kaggle / Colab)

No cluster and no budget? The grid is fully resumable at **recipe** granularity
(396 units), so you can work through it across many short sessions.

**Kaggle is the better free option**: 30 GPU-hours/week (P100 or 2×T4), 12-hour
sessions, versus Colab's tighter free tier.

```bash
bash mdg/hpc/status.sh          # what is done, what is left
bash mdg/hpc/run_budget.sh 11   # work for 11h, then stop cleanly
```

`run_budget.sh` goes **cheapest-first** — smallest dataset, fastest recipe — and
runs attacks one recipe at a time, so a short session completes whole units
rather than stalling halfway through `natquest`. Re-run it next session and it
picks up exactly where it stopped. `status.sh` prints a per-pair progress bar
and the overall percentage.

Persist results between sessions, since the VM is wiped: keep the repo on Drive
(Colab), save `/kaggle/working` as a Kaggle Dataset, or commit the small JSON/CSV
outputs back to git after each session.

## Many-core CPU servers (no GPU)

BERT-size inference runs fine on CPU, and attacks are embarrassingly parallel.
Core count and RAM decide throughput — the model itself needs ~1GB.

```bash
bash mdg/hpc/setup_vm.sh
DRY_RUN=1 bash mdg/hpc/run_parallel.sh cells   # check the sizing
bash mdg/hpc/run_parallel.sh cells             # run
```

The `cells` stage fans out over **(task, recipe) pairs — 396 units**, not 22
tasks, which is what keeps a big box busy. Workers are sized automatically as
`min(cores / THREADS_PER_JOB, RAM / MEM_PER_WORKER_GB)`; on a 128-core/256GB VM
that is ~85 workers, **limited by memory** — each worker is its own
python+torch+model copy at ~2-3GB. Override with `JOBS=`, `THREADS_PER_JOB=` or
`MEM_PER_WORKER_GB=`.

### Several machines

`SHARD="i/n"` splits the cell list round-robin, so each machine takes a disjoint
slice with a similar mix of big and small datasets:

```bash
SHARD=0/3 bash mdg/hpc/run_parallel.sh cells   # VM 1
SHARD=1/3 bash mdg/hpc/run_parallel.sh cells   # VM 2
SHARD=2/3 bash mdg/hpc/run_parallel.sh cells   # VM 3
```

Train with few workers and many threads each (training *does* scale across
threads, unlike inference):

```bash
ONLY_TASKS="0 1 2 3" JOBS=4 THREADS_PER_JOB=32 bash mdg/hpc/run_parallel.sh train
```

Training pushes each model to the Hub, and the attack stage falls back to the
Hub copy when there is no local checkpoint — so machines do not need to share a
filesystem or retrain each other's models.

Merge results at the end. The consolidation walks every source — summary JSONs,
the appended `attack_results.csv`, older per-recipe CSVs under
`adv_attack/legacy/`, and each checkpoint's `eval_results.json` — dedupes by
(model, dataset, recipe) and writes one row each:

```bash
python -m mdg.scripts.consolidate_results \
  --extra /path/to/vm2/results /path/to/vm3/results
```

`--extra` folds in results copied from other machines without moving them into
place first. Rows carry a `source` column and a `perturbed_file` path, so you
can see where each number came from and which have perturbed data on disk.

## Google Colab

Works, but Colab sessions are capped (~12h on Pro, less on free, and idle
disconnects), so the grid has to be chipped away at rather than run in one go.
Runs resume **per recipe**, so a dropped session loses at most the recipe that
was in flight.

Keep outputs on Drive so nothing is lost when the VM is recycled:

```python
from google.colab import drive; drive.mount('/content/drive')
%cd /content/drive/MyDrive
!git clone https://github.com/namesarnav/mdg.git || (cd mdg && git pull)
%cd /content/drive/MyDrive/mdg
```

Colab already ships torch built for its GPU — do **not** reinstall it:

```python
!pip install -q "transformers>=4.46,<5" "textattack>=0.3.10" peft accelerate \
    "datasets>=3.0,<4" sentencepiece scikit-learn
```

Then run tasks one at a time, re-running the cell after each disconnect:

```python
import os
os.environ["MDG_RUNTIME"] = "direct"      # no container on Colab
os.environ["PROJECT"]     = "/content/drive/MyDrive/mdg"
os.environ["MDG_VENV"]    = ""            # use Colab's own python

!bash mdg/hpc/run_one.sh train 0          # task 0..21
!bash mdg/hpc/run_one.sh attack 0
```

Finished pairs and finished recipes are skipped automatically, so re-running the
same cell always continues rather than restarting.

**Reality check:** a T4 is several times slower than an A100, and you get one
GPU instead of 22 in parallel. The full uncapped grid is not achievable on Colab
in any reasonable calendar time — use it to work through the small datasets
(`counterbench`, `ac-reason`, `bbh-causal-judgement`), or cap with
`NUM_EXAMPLES=200` per recipe.

## No cluster? Run on a plain GPU VM

The scripts detect their runtime: **container** (Singularity + overlay, on an
HPC cluster) or **direct** (a normal conda env, anywhere else). On any Linux box
with an NVIDIA driver — GCP, RunPod, Lambda, Vast, a lab machine:

```bash
git clone https://github.com/namesarnav/mdg.git ~/mdg && cd ~/mdg
bash mdg/hpc/setup_vm.sh                 # conda env, no container
echo 'hf_xxx' > ~/.hf_token && chmod 600 ~/.hf_token

bash mdg/hpc/run_one.sh train 0          # smoke test
bash mdg/hpc/run_parallel.sh all         # whole grid across this box's GPUs
```

No Slurm needed — `run_parallel.sh` is the scheduler, one task per GPU. Force a
runtime with `MDG_RUNTIME=direct` or `MDG_RUNTIME=container` if detection guesses
wrong.

## One-time setup (HPC cluster)

```bash
ssh <netid>@greene.hpc.nyu.edu        # or: <netid>@login.torch.hpc.nyu.edu

# 1. Clone to /scratch (NOT /home — its default quota is 30,000 inodes)
git clone git@github.com:namesarnav/mdg.git /scratch/$USER/mdg
cd /scratch/$USER/mdg

# 2. HuggingFace token — needed for gated Llama AND for pushing models
#    Get one at https://huggingface.co/settings/tokens (needs write access),
#    and accept the license at https://huggingface.co/meta-llama/Llama-3.2-1B
echo 'hf_xxxxxxxxxxxxxxxxx' > ~/.hf_token && chmod 600 ~/.hf_token

# 3. Build the container environment (~20 min, on a compute node)
srun --cpus-per-task=4 --mem=32G --time=2:00:00 --pty /bin/bash
bash mdg/hpc/00_setup_env.sh
exit
```

`env.sh` finds the container and overlay images automatically — Greene keeps
them under `/scratch/work/public/`, Torch under `/share/apps/`. If setup reports
that it found neither, locate them yourself and pass them in:

```bash
ls /share/apps/images/ /scratch/work/public/singularity/ 2>/dev/null | grep -i cuda
ls /share/apps/overlay-fs-ext3/ /scratch/work/public/overlay-fs-ext3/ 2>/dev/null

SIF=/path/to/cuda.sif OVERLAY_SRC=/path/to/overlay-15GB-500K.ext3.gz \
  bash mdg/hpc/00_setup_env.sh
```

Run setup from a login or compute node, **not** a data transfer node (`dtn*`).

### Check you can get a GPU first

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
