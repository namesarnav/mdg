#/bin/bash
set -e
# poetry run python -m mdg.synthetic.generate_causal --dataset namesarnav/counterbench --n 4000 --task 1 --seed 42 --output mdg/synthetic/data/counterbench_task1.jsonl
poetry run python -m mdg.synthetic.generate_causal --dataset namesarnav/counterbench --n 4000 --task 2 --seed 42 --output mdg/synthetic/data/counterbench_task2.jsonl
# poetry run python -m mdg.synthetic.generate_causal --dataset namesarnav/ac-reason --n 4000 --task 1 --seed 42 --output mdg/synthetic/data/ac_reason_task1.jsonl
poetry run python -m mdg.synthetic.generate_causal --dataset namesarnav/ac-reason --n 4000 --task 2 --seed 42 --output mdg/synthetic/data/ac_reason_task2.jsonl
#poetry run python -m mdg.synthetic.generate_causal --dataset namesarnav/bbh-causal-judgement --n 500 --task 1 --seed 42 --output mdg/synthetic/data/bbh_causal_judgement_task1.jsonl
poetry run python -m mdg.synthetic.generate_causal --dataset namesarnav/bbh-causal-judgement --n 4000 --task 2 --seed 42 --output mdg/synthetic/data/bbh_causal_judgement_task2.jsonl
echo "All done."
