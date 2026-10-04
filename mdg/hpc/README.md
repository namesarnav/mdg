# Running MDG on NYU Greene

Fine-tunes **t5-base** and **meta-llama/Llama-3.2-1B** on all 11 causal
classification datasets, pushes each model to the HuggingFace Hub, then attacks
every model with every TextAttack recipe on every dataset and consolidates
everything into one CSV.

Grid: **2 models × 11 datasets = 22** training runs, then 22 attack runs ×
18 recipes = **396 model/dataset/recipe combinations**.

## One-time setup

```bash
ssh <netid>@greene.hpc.nyu.edu

# 1. Clone to /scratch (NOT /home — it has a strict inode quota)
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

Verify the cluster's image paths first — they change between clusters and over
time. If either command prints nothing, update `SIF` / the overlay path in
`env.sh`:

```bash
ls /scratch/work/public/singularity/ | grep cuda
ls /scratch/work/public/overlay-fs-ext3/
```

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

`NUM_EXAMPLES=200` per recipe is the default because each recipe queries the
model hundreds of times per example; the full `corr2cause` split (200k rows) ×
18 recipes would not finish. `QUERY_BUDGET=2000` likewise keeps the search
recipes (PSO, genetic) from stalling on long inputs.

## Time and resource notes

- Training t5-base: minutes to ~2 hours per dataset. Llama-3.2-1B with LoRA:
  roughly 1–4 hours. The 12 h wall clock has headroom; `corr2cause` is the big one.
- Attacks are the expensive stage — 18 recipes × 200 examples is typically
  4–12 hours per pair, hence the 24 h limit. If a task hits the limit,
  completed recipes are already written and you can resubmit, but delete that
  pair's partial `<recipe>.jsonl` for whichever recipe was mid-run.
- Llama trains **unquantized** (`--no-4bit`) so the LoRA adapter can be merged
  into a plain classifier that TextAttack can load. 1B in bf16 fits a single
  GPU comfortably.
- If a GPU type is needed explicitly, add e.g. `#SBATCH --constraint=a100`.
