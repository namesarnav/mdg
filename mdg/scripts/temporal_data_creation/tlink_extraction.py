"""
Create a dataset for event extraction, based on TempEval-3 and TimeBank.
"""

import os
import random
import json
from tqdm import tqdm
from mdg.scripts.temporal_data_creation.datamodels import (
    Document,
    TimeLink,
    SignalExpression,
)
from typing import List
from dotenv import load_dotenv

load_dotenv()
TEMPEVAL_3_SILVER = "../../../data/TempEval-3/DATA-PUBLISHED/TE3-Silver-data-processed"
TEMPEVAL_3_PLATINUM = "../../../data/TempEval-3/DATA-PUBLISHED/te3-platinum-processed"
TIMEBANK = "../../../data/timebank_1_2/data/timeml_processed"
TIMEX_DOCS = "../../../data/timex_recognition"
EVENTX_DOCS = "../../../data/eventx_recognition"


def get_tlinks(_file_data: dict) -> List[TimeLink]:
    _tlinks = _file_data["tlinks"]
    tlinks = []
    for tlink in _tlinks:
        tlinks.append(
            TimeLink(
                link_id=tlink["lid"],
                time_id=tlink.get("timeID"),
                event_instance_id=tlink.get("eventInstanceID"),
                related_to_event_instance_id=tlink.get("relatedToEventInstance"),
                related_to_time_id=tlink.get("relatedToTime"),
                rel_type=tlink.get("relType"),
                signal_id=tlink.get("signalID"),
                origin=tlink.get("origin"),
                comment=tlink.get("comment"),
                syntax=tlink.get("syntax"),
            )
        )
    return tlinks


def get_signal_exprs(_file: str, _file_data: dict) -> List[SignalExpression]:
    tags = _file_data["tags"]
    _signal_exprs = []
    for tag in tags.values():
        if tag["tag"] != "SIGNAL":
            continue
        try:
            tag_attrs = tag["attributes"]
            _signal_exprs.append(
                SignalExpression(
                    signal_id=tag_attrs["sid"],
                    text=_file_data["doc_text"][tag["start_char"] : tag["end_char"]],
                    start_char=tag["start_char"],
                    end_char=tag["end_char"],
                )
            )
        except KeyError as e:
            raise RuntimeError(f"KeyError: {e} in file {_file} for tag {tag}")
    return _signal_exprs


if __name__ == "__main__":
    all_files = {}
    assert os.path.exists(TEMPEVAL_3_SILVER)
    assert os.path.exists(TEMPEVAL_3_PLATINUM)
    assert os.path.exists(TIMEBANK)
    assert os.path.exists(TIMEX_DOCS)
    assert os.path.exists(EVENTX_DOCS)

    all_files["tempeval_3_silver"] = [
        os.path.join(TEMPEVAL_3_SILVER, f) for f in os.listdir(TEMPEVAL_3_SILVER) if f.endswith(".json")
    ]
    all_files["tempeval_3_platinum"] = [
        os.path.join(TEMPEVAL_3_PLATINUM, f) for f in os.listdir(TEMPEVAL_3_PLATINUM) if f.endswith(".json")
    ]
    all_files["timebank"] = [os.path.join(TIMEBANK, f) for f in os.listdir(TIMEBANK) if f.endswith(".json")]

    output_dir_tlink_extraction = os.path.join(os.environ["MDG_HOME"], "data", "tlink_extraction")
    os.makedirs(output_dir_tlink_extraction, exist_ok=True)

    all_data_tlinks = []
    time_docs = (
        [Document(**json.loads(x)) for x in open(os.path.join(TIMEX_DOCS, "train.jsonl"), "r", encoding="utf-8")]
        + [Document(**json.loads(x)) for x in open(os.path.join(TIMEX_DOCS, "validation.jsonl"), "r", encoding="utf-8")]
        + [Document(**json.loads(x)) for x in open(os.path.join(TIMEX_DOCS, "test.jsonl"), "r", encoding="utf-8")]
    )
    event_docs = (
        [Document(**json.loads(x)) for x in open(os.path.join(EVENTX_DOCS, "train.jsonl"), "r", encoding="utf-8")]
        + [Document(**json.loads(x)) for x in open(os.path.join(EVENTX_DOCS, "validation.jsonl"), "r", encoding="utf-8")]
        + [Document(**json.loads(x)) for x in open(os.path.join(EVENTX_DOCS, "test.jsonl"), "r", encoding="utf-8")]
    )

    time_exprs = {t.doc_id: t.time_expressions for t in time_docs}
    event_exprs = {e.doc_id: e.event_expressions for e in event_docs}

    for dataset in all_files:
        for _file in tqdm(all_files[dataset], desc=dataset):
            _file_data = json.load(open(_file, "r", encoding="utf-8"))
            doc_text = _file_data["doc_text"]
            doc_id = _file.split("/")[-1][:-5]
            timex_this_doc = time_exprs.get(doc_id, [])
            event_this_doc = event_exprs.get(doc_id, [])
            signal_exprs = get_signal_exprs(_file=_file, _file_data=_file_data)
            try:
                assert timex_this_doc, f"No time expressions found for {doc_id}"
            except AssertionError as e:
                print(e)
                continue
            try:
                assert event_this_doc, f"No event expressions found for {doc_id}"
            except AssertionError as e:
                print(e)
                continue
            tlinks = get_tlinks(_file_data=_file_data)
            for tlink in tlinks:
                assert tlink.time_id or tlink.event_instance_id, f"Invalid TLink in {doc_id}: {tlink}"
                if tlink.time_id and not any(te.tid == tlink.time_id for te in timex_this_doc):
                    print(f"TLink time_id {tlink.time_id} not found in time expressions for {doc_id}")
                    continue
                if tlink.event_instance_id and not any(ee.eiid == tlink.event_instance_id for ee in event_this_doc):
                    print(f"TLink event_instance_id {tlink.event_instance_id} not found in event expressions for {doc_id}")
                    continue
                assert tlink.related_to_event_instance_id or tlink.related_to_time_id, f"Invalid TLink in {doc_id}: {tlink}"
                if tlink.related_to_event_instance_id and not any(
                    ee.eiid == tlink.related_to_event_instance_id for ee in event_this_doc
                ):
                    print(
                        f"TLink related_to_event_instance_id {tlink.related_to_event_instance_id} not found in event expressions for {doc_id}"
                    )
                    continue
                if tlink.related_to_time_id and not any(te.tid == tlink.related_to_time_id for te in timex_this_doc):
                    print(f"TLink related_to_time_id {tlink.related_to_time_id} not found in time expressions for {doc_id}")
                    continue
            if not tlinks:
                print(f"No TLinks found for {doc_id}, skipping this document.")
                continue
            tlink_doc = Document(
                doc_id=doc_id,
                text=doc_text,
                dataset=dataset,
                time_expressions=timex_this_doc,
                event_expressions=event_this_doc,
                signal_expressions=signal_exprs,
                tlinks=tlinks,
            )
            all_data_tlinks.append(tlink_doc)
    print(f"Total documents with TLinks: {len(all_data_tlinks)}")
    random.seed(42)
    random.shuffle(all_data_tlinks)
    data_dict_eventx_recog = {"train": [], "validation": [], "test": []}

    data_dict_eventx_recog["train"] = all_data_tlinks[: int(len(all_data_tlinks) * 0.8)]
    data_dict_eventx_recog["validation"] = all_data_tlinks[int(len(all_data_tlinks) * 0.8) : int(len(all_data_tlinks) * 0.9)]
    data_dict_eventx_recog["test"] = all_data_tlinks[int(len(all_data_tlinks) * 0.9) :]
    print(f"Total documents: {len(all_data_tlinks)}")
    print(f"Train documents: {len(data_dict_eventx_recog['train'])}")
    print(f"Validation documents: {len(data_dict_eventx_recog['validation'])}")
    print(f"Test documents: {len(data_dict_eventx_recog['test'])}")
    for k in ["train", "validation", "test"]:
        with open(f"{output_dir_tlink_extraction}/{k}.jsonl", "w", encoding="utf-8") as f:
            for _d in data_dict_eventx_recog[k]:
                json.dump(_d.model_dump(), f)
                f.write("\n")
    print(f"Data saved to {output_dir_tlink_extraction}")
