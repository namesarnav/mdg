"""
currenly we have time expressions in the document level, but we need to have them at the sentence level as well.
"""

import os
import json
from mdg.scripts.temporal_data_creation.datamodels import Document, TimeExpression, Sentence
import nltk
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

nltk.download("punkt")

DOC_LEVEL_TIMEX_DIR = "../../../data/new_timex_recognition/"
SENTENCE_LEVEL_TIMEX_DIR = "../../../data/new_timex_recognition_sentence_level/"


def doc_timex_to_sentence(doc: Document) -> list[Sentence]:
    """
    Splits the document text into sentences and associates time expressions with their respective sentences.
    """
    doc_text = doc.text
    doc_timexes = doc.time_expressions
    assert doc_timexes, "Document must have time expressions to convert to sentences"
    assert [
        x for x in doc_timexes if x.start_char is not None and x.end_char is not None
    ], "All time expressions must have start and end characters"
    sentence_texts = sent_tokenize(doc_text)
    sentences = []
    sent_start_char = 0

    for _index, sentence_text in enumerate(sentence_texts):
        sent_start_char = doc_text.find(
            sentence_text, sent_start_char
        )  # Find the start character of the sentence, but from the last found position
        sent_end_char = sent_start_char + len(sentence_text)

        sentence_timexes = [
            timex for timex in doc_timexes if timex.start_char >= sent_start_char and timex.end_char <= sent_end_char
        ]
        adj_sentence_timexes = [
            TimeExpression(
                text=timex.text,
                start_char=timex.start_char - sent_start_char,
                end_char=timex.end_char - sent_start_char,
                tid=timex.tid,
                type=timex.type,
                value=timex.value,
                temporal_function=timex.temporal_function,
                anchor_time=timex.anchor_time,
                function_in_document=timex.function_in_document,
            )
            for timex in sentence_timexes
        ]
        skip_sentence = False
        for adj_timex in adj_sentence_timexes:
            assert (
                adj_timex.start_char is not None and adj_timex.end_char is not None
            ), "Adjusted timex must have start and end characters"
            assert adj_timex.start_char >= 0 and adj_timex.end_char <= len(sentence_text), "Adjusted timex out of bounds"
            try:
                assert sentence_text[adj_timex.start_char : adj_timex.end_char] == adj_timex.text
            except AssertionError as e:
                print(f"AssertionError: {e}")
                print(f"Sentence text: {sentence_text}")
                print(f"Adjusted timex: {adj_timex.model_dump()}")
                print(f"Sentence text: {sentence_text[adj_timex.start_char : adj_timex.end_char]}")
                print(f"doc_id: {doc.doc_id}")
                print("-" * 20)
                skip_sentence = True
                break

        if skip_sentence:
            continue

        sentence_obj = Sentence(
            sent_id=f"sent_{_index}_{doc.doc_id}",
            text=sentence_text,
            start_char=sent_start_char,
            end_char=sent_end_char,
            time_expressions=adj_sentence_timexes,
            doc_id=doc.doc_id,
            dataset=doc.dataset,
            event_expressions=[],
            signal_expressions=[],
            tlinks=[],
        )
        sentences.append(sentence_obj)
        sent_start_char = sent_end_char + 1

    return sentences


def main():
    if not os.path.exists(SENTENCE_LEVEL_TIMEX_DIR):
        os.makedirs(SENTENCE_LEVEL_TIMEX_DIR)

    timex_files = [f for f in os.listdir(DOC_LEVEL_TIMEX_DIR) if f.endswith(".jsonl")]
    for file_name in timex_files:
        input_file_path = os.path.join(DOC_LEVEL_TIMEX_DIR, file_name)
        output_file_path = os.path.join(SENTENCE_LEVEL_TIMEX_DIR, file_name)

        with open(input_file_path, "r", encoding="utf-8") as f, open(output_file_path, "w", encoding="utf-8") as out_f:
            all_sentences = []
            for line in tqdm(f, desc=f"Processing {file_name}"):
                doc = Document(**json.loads(line))
                sentences = doc_timex_to_sentence(doc)
                all_sentences.extend([x.model_dump() for x in sentences])
            for sentence in all_sentences:
                out_f.write(json.dumps(sentence) + "\n")


if __name__ == "__main__":
    main()
