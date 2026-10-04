# Machine-specific run scripts

## Which machine does what

| Machine | Script | Role | Est. time |
|---|---|---|---|
| MacBook M4 Pro (24GB) | `run_macbook.sh` | Fine-tune all models + medium/slow attacks | ~1–2 weeks |
| GTX 1050 Ti (4GB) | `run_1050ti.sh` | Fast attack recipes only (in parallel with MacBook) | ~3–5 days |
| Colab A100 | `run_colab.ipynb` | Full pipeline (fastest option) | ~3–7 days |

## Recommended split (MacBook + 1050 Ti in parallel)

1. Start MacBook first — it fine-tunes all 33 models and pushes them to HuggingFace Hub
2. Once first models appear on Hub, start the 1050 Ti — it pulls from Hub and runs fast recipes
3. Both run simultaneously — MacBook handles medium/slow recipes, 1050 Ti handles fast ones
4. When both finish, run consolidation on MacBook

## MacBook usage

```bash
cd /Volumes/Github/mdg
caffeinate -i bash machines/run_macbook.sh 2>&1 | tee macbook.log
```

## 1050 Ti usage

```bash
# Clone repo first:
git clone https://github.com/YOUR_USERNAME/mdg.git && cd mdg
pip install poetry && poetry install

# Export datasets (needed before attacks):
poetry run python -m mdg.scripts.prepare_finetune_data \
  --datasets namesarnav/causalbench:code namesarnav/corr2cause namesarnav/e-care \
             namesarnav/fincausal-task1 namesarnav/natquest namesarnav/Quriosity \
  --local-files \
    "namesarnav_counterbench:mdg/synthetic/data/counterbench_task2_v2.jsonl" \
    "namesarnav_ac-reason:mdg/synthetic/data/ac_reason_task2.jsonl" \
    "namesarnav_bbh-causal-judgement:mdg/synthetic/data/bbh_causal_judgement_task2.jsonl" \
  --output-dir mdg/finetune/data

# Run fast attacks:
bash machines/run_1050ti.sh 2>&1 | tee 1050ti.log
```

## Colab usage

1. Open `run_colab.ipynb` in Google Colab
2. Runtime → Change runtime type → **A100 GPU** + **High RAM**
3. Add `HF_TOKEN` to Colab Secrets (key icon on left sidebar)
4. Run cells top to bottom
5. Run Cell 7 (save to Drive) before session ends
6. On reconnect: run Cell 8 (restore from Drive) before resuming

## Merging results from multiple machines

Copy all `*_results.csv` files into `mdg/adv_attack/` and `mdg/finetune/`, then:

```bash
poetry run python -m mdg.scripts.consolidate_results
```

Final output: `mdg/results/consolidated.csv`
