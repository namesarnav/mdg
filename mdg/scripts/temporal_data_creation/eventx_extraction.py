"""
Create a dataset for event extraction, based on TempEval-3 and TimeBank.
"""

import os
import random
import json
from tqdm import tqdm
from mdg.scripts.temporal_data_creation.datamodels import EventExpression, Document
from typing import List
from dotenv import load_dotenv

load_dotenv()
TEMPEVAL_3_SILVER = "../../../data/TempEval-3/DATA-PUBLISHED/TE3-Silver-data-processed"
TEMPEVAL_3_PLATINUM = "../../../data/TempEval-3/DATA-PUBLISHED/te3-platinum-processed"
TIMEBANK = "../../../data/timebank_1_2/data/timeml_processed"


def get_eventx(_file: str, _file_data: dict) -> list:
    tags = _file_data["tags"]
    eid_2_eiid = {v: k for k, v in _file_data["eiid2eventid"].items()}
    _event_exprs = []
    for tag in tags.values():
        if tag["tag"] != "EVENT":
            continue
        try:
            tag_attrs = tag["attributes"]
            _event_exprs.append(
                EventExpression(
                    eid=tag_attrs["eid"],
                    eiid=eid_2_eiid.get(tag_attrs["eid"], tag_attrs["eid"]),
                    type=tag_attrs["class"],
                    text=_file_data["doc_text"][tag["start_char"] : tag["end_char"]],
                    start_char=tag["start_char"],
                    end_char=tag["end_char"],
                )
            )
        except KeyError as e:
            raise RuntimeError(f"KeyError: {e} in file {_file} for tag {tag}")
    event_tags = {x["eventID"]: x for x in _file_data["event_tags"]}
    event_exprs = []
    for _event_expr in _event_exprs:
        if _event_expr.eid not in event_tags:
            event_exprs.append(
                EventExpression(
                    eid=_event_expr.eid,
                    eiid=_event_expr.eiid,
                    type=_event_expr.type,
                    text=_event_expr.text,
                    start_char=_event_expr.start_char,
                    end_char=_event_expr.end_char,
                )
            )
            continue
        assert _event_expr.eiid == eid_2_eiid[_event_expr.eid], f"Mismatch in eiid for {tag_attrs['eid']}"
        event_exprs.append(
            EventExpression(
                eid=_event_expr.eid,
                eiid=_event_expr.eiid,
                type=_event_expr.type,
                text=_event_expr.text,
                start_char=_event_expr.start_char,
                end_char=_event_expr.end_char,
                pos=event_tags[_event_expr.eid].get("pos") if event_tags[_event_expr.eid].get("pos") != "NONE" else None,
                tense=event_tags[_event_expr.eid].get("tense") if event_tags[_event_expr.eid].get("tense") != "NONE" else None,
                aspect=(
                    event_tags[_event_expr.eid].get("aspect") if event_tags[_event_expr.eid].get("aspect") != "NONE" else None
                ),
                polarity=(
                    event_tags[_event_expr.eid].get("polarity")
                    if event_tags[_event_expr.eid].get("polarity") != "NONE"
                    else None
                ),
                modality=(
                    event_tags[_event_expr.eid].get("modality")
                    if event_tags[_event_expr.eid].get("modality") != "NONE"
                    else None
                ),
            )
        )
    return event_exprs


if __name__ == "__main__":
    all_files = {}
    assert os.path.exists(TEMPEVAL_3_SILVER)
    assert os.path.exists(TEMPEVAL_3_PLATINUM)
    assert os.path.exists(TIMEBANK)
    all_files["tempeval_3_silver"] = [
        os.path.join(TEMPEVAL_3_SILVER, f) for f in os.listdir(TEMPEVAL_3_SILVER) if f.endswith(".json")
    ]
    all_files["tempeval_3_platinum"] = [
        os.path.join(TEMPEVAL_3_PLATINUM, f) for f in os.listdir(TEMPEVAL_3_PLATINUM) if f.endswith(".json")
    ]
    all_files["timebank"] = [os.path.join(TIMEBANK, f) for f in os.listdir(TIMEBANK) if f.endswith(".json")]

    output_dir_eventx_recognition = os.path.join(os.environ["MDG_HOME"], "data", "eventx_recognition")
    os.makedirs(output_dir_eventx_recognition, exist_ok=True)

    all_data_eventx_recognition = []

    for dataset in all_files:
        for _file in tqdm(all_files[dataset], desc=dataset):
            _file_data = json.load(open(_file, "r", encoding="utf-8"))
            doc_text = _file_data["doc_text"]
            event_exprs: List[EventExpression] = []
            for event_expr in get_eventx(_file=_file, _file_data=_file_data):
                assert event_expr.text == doc_text[event_expr.start_char : event_expr.end_char]
                event_exprs.append(event_expr)
            doc = Document(
                doc_id=_file.split("/")[-1][:-5],
                text=doc_text,
                event_expressions=event_exprs,
                dataset=dataset,
            )
            all_data_eventx_recognition.append(doc)

    random.seed(42)
    random.shuffle(all_data_eventx_recognition)
    data_dict_eventx_recog = {"train": [], "validation": [], "test": []}

    data_dict_eventx_recog["train"] = all_data_eventx_recognition[: int(len(all_data_eventx_recognition) * 0.8)]
    data_dict_eventx_recog["validation"] = all_data_eventx_recognition[
        int(len(all_data_eventx_recognition) * 0.8) : int(len(all_data_eventx_recognition) * 0.9)
    ]
    data_dict_eventx_recog["test"] = all_data_eventx_recognition[int(len(all_data_eventx_recognition) * 0.9) :]
    print(f"Total documents: {len(all_data_eventx_recognition)}")
    print(f"Train documents: {len(data_dict_eventx_recog['train'])}")
    print(f"Validation documents: {len(data_dict_eventx_recog['validation'])}")
    print(f"Test documents: {len(data_dict_eventx_recog['test'])}")
    for k in ["train", "validation", "test"]:
        with open(f"{output_dir_eventx_recognition}/{k}.jsonl", "w", encoding="utf-8") as f:
            for _d in data_dict_eventx_recog[k]:
                json.dump(_d.model_dump(), f)
                f.write("\n")
    print(f"Data saved to {output_dir_eventx_recognition}")
