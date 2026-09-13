# we need to read the data from the old files and convert it to the new format
import os
import json
from tqdm import tqdm
from mdg.scripts.temporal_data_creation.datamodels import (
    TimeLink,
    SignalExpression,
    EventExpression,
    TimeExpression,
)
from mdg.scripts.temporal_data_creation.datamodels import Document as NewDocument

from typing import List
from pydantic import BaseModel, ValidationError


class Document(BaseModel):
    doc_id: str
    text: str
    dataset: str


class TimeDocument(Document):
    time_expressions: List[TimeExpression]


class EventDocument(Document):
    event_expressions: List[EventExpression]


class TLinkDocument(Document):
    time_expressions: List[TimeExpression]
    event_expressions: List[EventExpression]
    signal_expressions: List[SignalExpression] = []  # Optional, can be empty if no signals are present
    tlinks: List[TimeLink]


TIMEX_RECOGNITION_DIR = "../../../data/timex_recognition"
EVENTX_RECOGNITION_DIR = "../../../data/eventx_recognition"
TLINK_EXTRACTION_DIR = "../../../data/tlink_extraction"

NEW_TIMEX_RECOGNITION_DIR = "../../../data/new_timex_recognition"
NEW_EVENTX_RECOGNITION_DIR = "../../../data/new_eventx_recognition"
NEW_TLINK_EXTRACTION_DIR = "../../../data/new_tlink_extraction"

assert os.path.exists(TIMEX_RECOGNITION_DIR), "Timex recognition directory does not exist"
assert os.path.exists(EVENTX_RECOGNITION_DIR), "Eventx recognition directory does not exist"
assert os.path.exists(TLINK_EXTRACTION_DIR), "TLink extraction directory does not exist"

os.makedirs(NEW_TIMEX_RECOGNITION_DIR, exist_ok=True)
os.makedirs(NEW_EVENTX_RECOGNITION_DIR, exist_ok=True)
os.makedirs(NEW_TLINK_EXTRACTION_DIR, exist_ok=True)


def convert_timex_recognition():
    timex_files = [f for f in os.listdir(TIMEX_RECOGNITION_DIR) if f.endswith(".jsonl")]
    for file_name in tqdm(timex_files, desc="Converting Timex Recognition files"):
        input_file_path = os.path.join(TIMEX_RECOGNITION_DIR, file_name)
        output_file_path = os.path.join(NEW_TIMEX_RECOGNITION_DIR, file_name)
        with open(input_file_path, "r", encoding="utf-8") as f, open(output_file_path, "w", encoding="utf-8") as out_f:
            for line in f:
                data = json.loads(line)
                try:
                    doc = TimeDocument(**data)
                    new_doc = NewDocument(
                        doc_id=doc.doc_id, text=doc.text, dataset=doc.dataset, time_expressions=doc.time_expressions
                    )
                    out_f.write(json.dumps(new_doc.model_dump()) + "\n")
                except ValidationError as e:
                    print(f"Validation error in {file_name}: {e}")


def convert_eventx_recognition():
    eventx_files = [f for f in os.listdir(EVENTX_RECOGNITION_DIR) if f.endswith(".jsonl")]
    for file_name in tqdm(eventx_files, desc="Converting Eventx Recognition files"):
        input_file_path = os.path.join(EVENTX_RECOGNITION_DIR, file_name)
        output_file_path = os.path.join(NEW_EVENTX_RECOGNITION_DIR, file_name)
        with open(input_file_path, "r", encoding="utf-8") as f, open(output_file_path, "w", encoding="utf-8") as out_f:
            for line in f:
                data = json.loads(line)
                try:
                    doc = EventDocument(**data)
                    new_doc = NewDocument(
                        doc_id=doc.doc_id, text=doc.text, dataset=doc.dataset, event_expressions=doc.event_expressions
                    )
                    out_f.write(json.dumps(new_doc.model_dump()) + "\n")
                except ValidationError as e:
                    print(f"Validation error in {file_name}: {e}")


def convert_tlink_extraction():
    tlink_files = [f for f in os.listdir(TLINK_EXTRACTION_DIR) if f.endswith(".jsonl")]
    for file_name in tqdm(tlink_files, desc="Converting TLink Extraction files"):
        input_file_path = os.path.join(TLINK_EXTRACTION_DIR, file_name)
        output_file_path = os.path.join(NEW_TLINK_EXTRACTION_DIR, file_name)
        with open(input_file_path, "r", encoding="utf-8") as f, open(output_file_path, "w", encoding="utf-8") as out_f:
            for line in f:
                data = json.loads(line)
                try:
                    doc = TLinkDocument(**data)
                    new_doc = NewDocument(
                        doc_id=doc.doc_id,
                        text=doc.text,
                        dataset=doc.dataset,
                        time_expressions=doc.time_expressions,
                        event_expressions=doc.event_expressions,
                        signal_expressions=doc.signal_expressions,
                        tlinks=doc.tlinks,
                    )
                    out_f.write(json.dumps(new_doc.model_dump()) + "\n")
                except ValidationError as e:
                    print(f"Validation error in {file_name}: {e}")


if __name__ == "__main__":
    convert_timex_recognition()
    convert_eventx_recognition()
    convert_tlink_extraction()
    print("Conversion completed successfully.")
    print(
        f"Converted files saved to {NEW_TIMEX_RECOGNITION_DIR}, {NEW_EVENTX_RECOGNITION_DIR}, and {NEW_TLINK_EXTRACTION_DIR}."
    )
