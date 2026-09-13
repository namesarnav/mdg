from datasets import load_dataset
from mdg.datamodels import TimeDocument as NewDocument, TimeSentence as NewSentence
from mdg.scripts.temporal_data_creation.datamodels import Document as OldDocument, Sentence as OldSentence
from collections import defaultdict
from tqdm import tqdm
import os
import json

hf_dataset_name_links = {
    "timex-recognition": "mdg-nlp/timex-recognition",
    "eventx-recognition": "mdg-nlp/eventx-recognition",
    "tlink-extraction": "mdg-nlp/tlink-extr-classification",
    "timex-grounding": "mdg-nlp/timex-grounding",
}

local_dataset_name_location = {
    "timex-recognition": "../../../data/new_timex_recognition_2",
    "eventx-recognition": "../../../data/new_eventx_recognition_2",
    "tlink-extraction": "../../../data/new_tlink_extraction_2",
    "timex-grounding": "../../../data/new_timex_grounding_2",
    "timex-extraction-sentence": "../../../data/new_timex_extraction_sentence_2",
}

for v in local_dataset_name_location.values():
    os.makedirs(v, exist_ok=True)


def convert_single_doc(old_doc_sentence: OldDocument | OldSentence, is_sentence: bool = False) -> NewDocument | NewSentence:
    if is_sentence:
        return NewSentence(**old_doc_sentence.model_dump())
    return NewDocument(**old_doc_sentence.model_dump())


def convert_to_new_format_ex(dataset_name: str):
    old_dataset = load_dataset(hf_dataset_name_links[dataset_name])
    new_dataset_loc = local_dataset_name_location[dataset_name]
    is_sentence = dataset_name == "timex-extraction-sentence"
    for k in ["train", "validation", "test"]:
        old_docs = [OldDocument(**doc) for doc in old_dataset[k]]
        new_docs = [
            convert_single_doc(old_doc, is_sentence=is_sentence)
            for old_doc in tqdm(old_docs, desc=f"Converting {dataset_name} {k} split")
        ]
        print(f"Converted {len(new_docs)} documents in split {k}")
        with open(f"{new_dataset_loc}/{k}.jsonl", "w", encoding="utf-8") as f:
            for doc in new_docs:
                f.write(json.dumps(doc.model_dump()) + "\n")
    return


if __name__ == "__main__":
    for dataset in hf_dataset_name_links.keys():
        if "grounding" in dataset:
            continue
        print(f"Converting dataset: {dataset}")
        convert_to_new_format_ex(dataset)
