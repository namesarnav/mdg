# Legacy orchestration scripts

Superseded, kept for reference. Nothing here is on the current path — the whole
grid now runs through `mdg/hpc/run_parallel.sh`, which shards work across
machines and resumes cleanly. These scripts were sequential and single-box.

| File | Was | Replaced by |
|---|---|---|
| `finetune_all.sh` | Sequential fine-tune of all 3 models × 11 datasets | `run_parallel.sh train` |
| `finetune_bert_roberta.sh` | Same, before T5 was added | `run_parallel.sh train` |
| `train_root.sh` | A copy of `finetune_bert_roberta.sh` that sat in the repo root | `run_parallel.sh train` |
| `attack_all.sh` | Sequential attack over every cell | `run_parallel.sh cells` |
| `run_pipeline.sh` | prepare data → fine-tune → attack, in one process | `run_parallel.sh all` |
| `run_all_causal.sh` | The earlier LLM phase: 5 datasets × 7 prompting strategies via `experiment_runner` | — (different experiment) |
| `machines/` | Per-machine runners (MacBook M4, GTX 1050 Ti, a Colab notebook) | `../../hpc/bootstrap_vm.sh`, `../../hpc/train_gpu.sh` |
