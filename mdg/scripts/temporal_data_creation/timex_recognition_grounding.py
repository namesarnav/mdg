"""
Create a dataset for time expression recognition + grounding, based on TempEval-3 and TimeBank.
"""

import os
import random
import json
from tqdm import tqdm
from mdg.scripts.temporal_data_creation.datamodels import TimeExpression, TimeExpressionGrounding, Duration, Document
from typing import List, Optional, Dict
from dateutil import parser
from datetime import timezone
from mdg.scripts.openrouter import OpenRouter, JinjaTemplateProcessor
from jinja2 import Environment, FileSystemLoader
from dotenv import load_dotenv

load_dotenv()
TEMPEVAL_3_SILVER = "../../../data/TempEval-3/DATA-PUBLISHED/TE3-Silver-data-processed"
TEMPEVAL_3_PLATINUM = "../../../data/TempEval-3/DATA-PUBLISHED/te3-platinum-processed"
TIMEBANK = "../../../data/timebank_1_2/data/timeml_processed"
GROUNDING_ERRORS = 0


def get_timex(_file: str, _file_data: dict, _dataset: str) -> list:
    tags = _file_data["tags"]
    time_exprs = []
    for tag in tags.values():
        if tag["tag"] != "TIMEX3":
            continue
        try:
            tag_attrs = tag["attributes"]
            time_exprs.append(
                TimeExpression(
                    tid=tag_attrs["tid"],
                    type=tag_attrs["type"],
                    text=_file_data["doc_text"][tag["start_char"] : tag["end_char"]],
                    start_char=tag["start_char"],
                    end_char=tag["end_char"],
                    value=tag_attrs["value"],
                    temporal_function=tag_attrs.get("temporalFunction", False),
                    function_in_document=tag_attrs.get("functionInDocument"),
                    anchor_time=tag_attrs.get("anchorTimeID"),
                )
            )
        except KeyError as e:
            raise RuntimeError(f"KeyError: {e} in file {_file} for tag {tag}")
    return time_exprs


def get_grounded_timex_llm_anchor_time(
    anchor_time_expr: TimeExpression,
    llm: OpenRouter,
    templates: Dict[str, JinjaTemplateProcessor],
) -> Optional[str]:
    """
    This function grounds the anchor time expression based on its type and temporal function.
    For simplicity, we return the value as is, but in a real scenario, this would involve more complex logic.
    """
    anchor_time_expr = anchor_time_expr.text
    prompt = templates["timex_iso_format"]({"timex": anchor_time_expr})
    response = llm.get_response(prompt).strip('"')
    try:
        dt = parser.parse(response)
        return dt.astimezone(timezone.utc).isoformat()
    except (AttributeError, ValueError, TypeError) as e:
        print(f"Exception occurred while parsing response: {e}")
        return None


def get_grounded_query(
    query_time_expr: TimeExpression,
    anchor_timex: str,
    llm: OpenRouter,
    templates: Dict[str, JinjaTemplateProcessor],
) -> Optional[str | Duration]:
    """
    This function grounds the timex value based on its type and temporal function.
    For simplicity, we return the value as is, but in a real scenario, this would involve more complex logic.
    """
    timex_type = query_time_expr.type
    if timex_type in ["DATE", "TIME"]:
        return get_grounded_timex_llm_value(
            anchor_timex=anchor_timex,
            timex=query_time_expr.text,
            llm=llm,
            template=templates["date_time"],
            timex_source=query_time_expr,
        )
    elif timex_type == "DURATION":
        return get_grounded_timex_llm_duration(
            anchor_timex=anchor_timex,
            duration=query_time_expr.text,
            llm=llm,
            template=templates["duration"],
            timex_source=query_time_expr,
        )
    else:
        print(f"Unsupported timex type: {timex_type} for value: {query_time_expr.value}")
        return None


def get_grounded_timex_llm_duration(
    anchor_timex: str, duration: str, llm: OpenRouter, template: JinjaTemplateProcessor, timex_source: TimeExpression
) -> Optional[Duration]:
    """
    for a time expression like "the next year or so", we want to get the start and end of the duration.
    eg. get_grounded_timex_llm_duration(anchor_timex="1989-11-02", duration="fourth-quarter", llm=llm, template=t)
    {
        'duration_start': '1989-10-01T00:00:00+00:00',
        'duration_end': '1989-12-31T23:59:59+00:00'
    }
    t = JinjaTemplateProcessor("templates/timex-grounding-duration.j2")
    """
    global GROUNDING_ERRORS
    prompt = template({"anchor_timex": anchor_timex, "duration": duration})
    response = llm.get_response(prompt)
    try:
        response = json.loads(response)
        assert type(response) is dict and "duration_start" in response and "duration_end" in response
        duration_start = response["duration_start"]
        duration_end = response["duration_end"]
        duration_start_dt = parser.parse(duration_start).astimezone(timezone.utc).isoformat()
        duration_end_dt = parser.parse(duration_end).astimezone(timezone.utc).isoformat()
        response = {
            "duration_start": duration_start_dt,
            "duration_end": duration_end_dt,
        }
        return Duration(
            start_timex=response["duration_start"],
            end_timex=response["duration_end"],
        )
    except (
        json.JSONDecodeError,
        AssertionError,
        AttributeError,
        ValueError,
        TypeError,
    ) as e:
        GROUNDING_ERRORS += 1
        print(
            f"Failed to parse or validate JSON response: {response}, the following error occurred: {e}. timex: {timex_source}. Grounding error #{GROUNDING_ERRORS}"
        )
        return None


def get_grounded_timex_llm_value(
    anchor_timex: str, timex: str, llm: OpenRouter, template: JinjaTemplateProcessor, timex_source: TimeExpression
) -> str:
    """
    eg. usage: get_grounded_timex_llm_value(anchor_timex="1989-11-02", timex="the next year or so", llm=llm, template=t)
    # t = JinjaTemplateProcessor("templates/timex-grounding.j2")
    This function uses an LLM to ground a time expression based on an anchor time expression.
    """
    global GROUNDING_ERRORS
    prompt = template({"anchor_timex": anchor_timex, "timex": timex})
    response = llm.get_response(prompt)
    try:
        response = response.strip("\"'")  # Clean up the response to remove any surrounding quotes
        dt = parser.parse(response)
        dt_str = dt.astimezone(timezone.utc).isoformat()
        return dt_str
    except (AttributeError, ValueError, TypeError) as e:
        GROUNDING_ERRORS += 1
        print(f"Exception occurred while parsing response: {e} for timex: {timex_source}. Grounding error #{GROUNDING_ERRORS}")
        return None


def get_grounding_data(doc: Document, llm: OpenRouter, templates: Dict) -> List[TimeExpressionGrounding]:
    """ """
    grounding_data_doc = []
    # print(f"Processing document: {doc.doc_id} with {len(doc.time_expressions)} time expressions.")
    for time_expr in doc.event_expressions:
        anchor_time_id = time_expr.anchor_time
        if anchor_time_id is None:
            # print(f"Skipping time expression {time_expr.tid} in document {doc.doc_id} due to missing anchor time.")
            continue
        anchor_time_expr = [x for x in doc.event_expressions if x.tid == anchor_time_id]
        if not anchor_time_expr:
            # print(f"Skipping time expression {time_expr.tid} in document {doc.doc_id} due to missing anchor time expression.")
            continue
        grounded_anchor_timex = get_grounded_timex_llm_anchor_time(
            anchor_time_expr=anchor_time_expr[0], llm=llm, templates=templates
        )
        if grounded_anchor_timex is None:
            # print(f"Skipping time expression {time_expr.tid} in document {doc.doc_id} due to failed grounding.")
            continue
        grounded_query = get_grounded_query(
            query_time_expr=time_expr, anchor_timex=grounded_anchor_timex, llm=llm, templates=templates
        )
        if grounded_query is None:
            # print(f"Skipping time expression {time_expr.tid} in document {doc.doc_id} due to failed grounding of query.")
            continue
        grounding_data_doc.append(
            TimeExpressionGrounding(
                context_doc_id=doc.doc_id,
                anchor_time_id=anchor_time_id,
                anchor_timex=anchor_time_expr[0],
                grounded_anchor_timex=grounded_anchor_timex,
                query=time_expr,
                type=time_expr.type,
                grounded_query=grounded_query,
            )
        )
    return [x for x in grounding_data_doc if x is not None]


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

    output_dir_timex_recognition = os.path.join(os.environ["MDG_HOME"], "data", "timex_recognition")
    output_dir_timex_grounding = os.path.join(os.environ["MDG_HOME"], "data", "timex_grounding")
    os.makedirs(output_dir_timex_recognition, exist_ok=True)
    os.makedirs(output_dir_timex_grounding, exist_ok=True)

    all_data_timex_recognition = []

    for dataset in all_files:
        for _file in tqdm(all_files[dataset], desc=dataset):
            _file_data = json.load(open(_file, "r", encoding="utf-8"))
            doc_text = _file_data["doc_text"]
            time_exprs = get_timex(_file=_file, _file_data=_file_data, _dataset=dataset)
            for time_expr in time_exprs:
                assert time_expr.text == doc_text[time_expr.start_char : time_expr.end_char]
            doc = Document(
                doc_id=_file.split("/")[-1][:-5],
                text=doc_text,
                time_expressions=time_exprs,
                dataset=dataset,
            )

            all_data_timex_recognition.append(doc)

    random.seed(42)
    random.shuffle(all_data_timex_recognition)
    data_dict_timex_recog = {"train": [], "validation": [], "test": []}
    data_dict_timex_grounding = {"train": [], "validation": [], "test": []}

    data_dict_timex_recog["train"] = all_data_timex_recognition[: int(len(all_data_timex_recognition) * 0.8)]
    data_dict_timex_recog["validation"] = all_data_timex_recognition[
        int(len(all_data_timex_recognition) * 0.8) : int(len(all_data_timex_recognition) * 0.9)
    ]
    data_dict_timex_recog["test"] = all_data_timex_recognition[int(len(all_data_timex_recognition) * 0.9) :]
    print(f"Total documents: {len(all_data_timex_recognition)}")
    print(f"Train documents: {len(data_dict_timex_recog['train'])}")
    print(f"Validation documents: {len(data_dict_timex_recog['validation'])}")
    print(f"Test documents: {len(data_dict_timex_recog['test'])}")
    for k in ["train", "validation", "test"]:
        with open(f"{output_dir_timex_recognition}/{k}.jsonl", "w", encoding="utf-8") as f:
            for _d in data_dict_timex_recog[k]:
                json.dump(_d.model_dump(), f)
                f.write("\n")

    llm = OpenRouter(
        model_name="openai/gpt-4",
        api_key=os.environ["OPENROUTER_API_KEY"],
        api_url="https://openrouter.ai/api/v1/chat/completions",
    )
    templates = {
        "timex_iso_format": JinjaTemplateProcessor(
            env=Environment(loader=FileSystemLoader(".")), template_path="templates/timex-iso-format.j2"
        ),
        "date_time": JinjaTemplateProcessor(
            env=Environment(loader=FileSystemLoader(".")), template_path="templates/timex-grounding.j2"
        ),
        "duration": JinjaTemplateProcessor(
            env=Environment(loader=FileSystemLoader(".")), template_path="templates/timex-grounding-duration.j2"
        ),
    }

    for k in ["test", "validation", "train"]:
        with open(f"{output_dir_timex_grounding}/{k}.jsonl", "w", encoding="utf-8") as f:
            for doc in tqdm(data_dict_timex_recog[k], desc=f"grounding-{k}"):
                grounding_data = get_grounding_data(doc, llm=llm, templates=templates)
                for _d in grounding_data:
                    json.dump(_d.model_dump(), f)
                    f.write("\n")
            data_dict_timex_grounding[k].extend(grounding_data)

    print(
        f"Total grounding data: {len(data_dict_timex_grounding['train']) + len(data_dict_timex_grounding['validation']) + len(data_dict_timex_grounding['test'])}"
    )
    print(f"Train grounding data: {len(data_dict_timex_grounding['train'])}")
    print(f"Validation grounding data: {len(data_dict_timex_grounding['validation'])}")
    print(f"Test grounding data: {len(data_dict_timex_grounding['test'])}")
    print(f"Grounding errors: {GROUNDING_ERRORS}")
