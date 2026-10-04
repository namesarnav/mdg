#!/bin/bash


# Configuration
REMOTE_USER="arnav"
REMOTE_HOST="100.85.247.37"
REMOTE_DIR="/home/arnav/mdg/mdg/mdg/adv_attack/results"
LOCAL_DIR="/Volumes/Github/mdg/mdg/adv_attack"

# Array of specific files from your list
FILES=(
    "ac-reason-bert-base-uncased-CheckList2020.csv"
    "ac-reason-bert-base-uncased-DeepWordBugGao2018.csv"
    "ac-reason-roberta-base-CheckList2020.csv"
    "ac-reason-roberta-base-DeepWordBugGao2018.csv"
    "bbh-causal-judgement-bert-base-uncased-CheckList2020.csv"
    "bbh-causal-judgement-bert-base-uncased-DeepWordBugGao2018.csv"
    "bbh-causal-judgement-roberta-base-CheckList2020.csv"
    "causalbench_code-bert-base-uncased-CheckList2020.csv"
    "causalbench_math-bert-base-uncased-CheckList2020.csv"
    "consolidated.csv"
    "counterbench-bert-base-uncased-CheckList2020.csv"
    "counterbench-bert-base-uncased-DeepWordBugGao2018.csv"
    "counterbench-bert-base-uncased-PSOZang2020.csv"
    "counterbench-bert-base-uncased-PWWSRen2019.csv"
    "counterbench-roberta-base-CheckList2020.csv"
    "counterbench-roberta-base-DeepWordBugGao2018.csv"
    "counterbench-roberta-base-PSOZang2020.csv"
    "counterbench-roberta-base-PWWSRen2019.csv"
    "e-care-bert-base-uncased-CheckList2020.csv"
    "e-care-bert-base-uncased-DeepWordBugGao2018.csv"
)

# Loop through and copy each file
for FILE in "${FILES[@]}"; do
    echo "Downloading: $FILE..."
    scp "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/${FILE}" "$LOCAL_DIR"
done

echo "All transfers completed!"
