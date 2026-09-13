#!/bin/bash
# 5 datasets × 7 strategies = 35 runs

# counterbench
poetry run python -m mdg.experiment_runner --config mdg/exp_configs/causal_counterbench_inductive.yml --output results/causal_counterbench_inductive.json
poetry run python -m mdg.experiment_runner --config mdg/exp_configs/causal_counterbench_deductive.yml --output results/causal_counterbench_deductive.json
poetry run python -m mdg.experiment_runner --config mdg/exp_configs/causal_counterbench_abductive.yml --output results/causal_counterbench_abductive.json
poetry run python -m mdg.experiment_runner --config mdg/exp_configs/causal_counterbench_inductive_deductive.yml --output results/causal_counterbench_inductive_deductive.json
poetry run python -m mdg.experiment_runner --config mdg/exp_configs/causal_counterbench_inductive_abductive.yml --output results/causal_counterbench_inductive_abductive.json
poetry run python -m mdg.experiment_runner --config mdg/exp_configs/causal_counterbench_deductive_abductive.yml --output results/causal_counterbench_deductive_abductive.json
poetry run python -m mdg.experiment_runner --config mdg/exp_configs/causal_counterbench_inductive_deductive_abductive.yml --output results/causal_counterbench_inductive_deductive_abductive.json

# ac_reason
poetry run python -m mdg.experiment_runner --config mdg/exp_configs/causal_ac_reason_inductive.yml --output results/causal_ac_reason_inductive.json
poetry run python -m mdg.experiment_runner --config mdg/exp_configs/causal_ac_reason_deductive.yml --output results/causal_ac_reason_deductive.json
poetry run python -m mdg.experiment_runner --config mdg/exp_configs/causal_ac_reason_abductive.yml --output results/causal_ac_reason_abductive.json
poetry run python -m mdg.experiment_runner --config mdg/exp_configs/causal_ac_reason_inductive_deductive.yml --output results/causal_ac_reason_inductive_deductive.json
poetry run python -m mdg.experiment_runner --config mdg/exp_configs/causal_ac_reason_inductive_abductive.yml --output results/causal_ac_reason_inductive_abductive.json
poetry run python -m mdg.experiment_runner --config mdg/exp_configs/causal_ac_reason_deductive_abductive.yml --output results/causal_ac_reason_deductive_abductive.json
poetry run python -m mdg.experiment_runner --config mdg/exp_configs/causal_ac_reason_inductive_deductive_abductive.yml --output results/causal_ac_reason_inductive_deductive_abductive.json

# bbh_judgement
poetry run python -m mdg.experiment_runner --config mdg/exp_configs/causal_bbh_judgement_inductive.yml --output results/causal_bbh_judgement_inductive.json
poetry run python -m mdg.experiment_runner --config mdg/exp_configs/causal_bbh_judgement_deductive.yml --output results/causal_bbh_judgement_deductive.json
poetry run python -m mdg.experiment_runner --config mdg/exp_configs/causal_bbh_judgement_abductive.yml --output results/causal_bbh_judgement_abductive.json
poetry run python -m mdg.experiment_runner --config mdg/exp_configs/causal_bbh_judgement_inductive_deductive.yml --output results/causal_bbh_judgement_inductive_deductive.json
poetry run python -m mdg.experiment_runner --config mdg/exp_configs/causal_bbh_judgement_inductive_abductive.yml --output results/causal_bbh_judgement_inductive_abductive.json
poetry run python -m mdg.experiment_runner --config mdg/exp_configs/causal_bbh_judgement_deductive_abductive.yml --output results/causal_bbh_judgement_deductive_abductive.json
poetry run python -m mdg.experiment_runner --config mdg/exp_configs/causal_bbh_judgement_inductive_deductive_abductive.yml --output results/causal_bbh_judgement_inductive_deductive_abductive.json

# bbh_understanding
poetry run python -m mdg.experiment_runner --config mdg/exp_configs/causal_bbh_understanding_inductive.yml --output results/causal_bbh_understanding_inductive.json
poetry run python -m mdg.experiment_runner --config mdg/exp_configs/causal_bbh_understanding_deductive.yml --output results/causal_bbh_understanding_deductive.json
poetry run python -m mdg.experiment_runner --config mdg/exp_configs/causal_bbh_understanding_abductive.yml --output results/causal_bbh_understanding_abductive.json
poetry run python -m mdg.experiment_runner --config mdg/exp_configs/causal_bbh_understanding_inductive_deductive.yml --output results/causal_bbh_understanding_inductive_deductive.json
poetry run python -m mdg.experiment_runner --config mdg/exp_configs/causal_bbh_understanding_inductive_abductive.yml --output results/causal_bbh_understanding_inductive_abductive.json
poetry run python -m mdg.experiment_runner --config mdg/exp_configs/causal_bbh_understanding_deductive_abductive.yml --output results/causal_bbh_understanding_deductive_abductive.json
poetry run python -m mdg.experiment_runner --config mdg/exp_configs/causal_bbh_understanding_inductive_deductive_abductive.yml --output results/causal_bbh_understanding_inductive_deductive_abductive.json

# corr2cause
poetry run python -m mdg.experiment_runner --config mdg/exp_configs/causal_corr2cause_inductive.yml --output results/causal_corr2cause_inductive.json
poetry run python -m mdg.experiment_runner --config mdg/exp_configs/causal_corr2cause_deductive.yml --output results/causal_corr2cause_deductive.json
poetry run python -m mdg.experiment_runner --config mdg/exp_configs/causal_corr2cause_abductive.yml --output results/causal_corr2cause_abductive.json
poetry run python -m mdg.experiment_runner --config mdg/exp_configs/causal_corr2cause_inductive_deductive.yml --output results/causal_corr2cause_inductive_deductive.json
poetry run python -m mdg.experiment_runner --config mdg/exp_configs/causal_corr2cause_inductive_abductive.yml --output results/causal_corr2cause_inductive_abductive.json
poetry run python -m mdg.experiment_runner --config mdg/exp_configs/causal_corr2cause_deductive_abductive.yml --output results/causal_corr2cause_deductive_abductive.json
poetry run python -m mdg.experiment_runner --config mdg/exp_configs/causal_corr2cause_inductive_deductive_abductive.yml --output results/causal_corr2cause_inductive_deductive_abductive.json

