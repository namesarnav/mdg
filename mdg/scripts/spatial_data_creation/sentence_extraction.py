import os
from bs4 import BeautifulSoup
import json
from datamodels import TrajectorExpression, LandmarkExpression, SpIndicatorExpression, Relation, Sentence


SEMEVAL_GOLD = "../../../data/mSpRL/data/newSprl2017_gold.xml"
SEMEVAL_TRAIN = "../../../data/mSpRL/data/newSprl2017_train.xml"
SEMEVAL_GOLD_JSONL = "../../../data/mSpRL/data/sentences_gold.jsonl"
SEMEVAL_TRAIN_JSONL = "../../../data/mSpRL/data/sentences_train.jsonl"



def extract_sentences(_file: str) -> list:
    with open(_file, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "xml")

    result = []

    for doc in soup.find_all("SCENE"):
        doc_tag = doc.find("DOCNO")
        doc_no = doc_tag.text.strip() if doc_tag else ""
        image_tag = doc.find("IMAGE")
        image = image_tag.text.strip() if image_tag else ""
        for sentence in doc.find_all("SENTENCE"):
            sentence_id = sentence.get("id", "")
            start = int(sentence.get("start", ""))
            end = int(sentence.get("end", ""))
            text_tag = sentence.find("TEXT")
            text = text_tag.text if text_tag else ""
            l_extra = len(text) - len(text.lstrip())
            r_extra = len(text) - len(text.rstrip())
            start += l_extra
            end -= r_extra
            text = text.strip()

            trajector_expressions = []
            trajector_map = {}
            for traj in sentence.find_all("TRAJECTOR"):
                t_id = traj.get("id")
                t_start = int(traj.get("start"))
                t_end = int(traj.get("end"))
                t_text = traj.get("text") or ""
                t_start -= l_extra
                t_end -= l_extra
                if t_text == "":
                    continue
                t_start += (len(t_text) - len(t_text.lstrip()))
                t_end -= (len(t_text) - len(t_text.rstrip()))
                t_text = t_text.strip()

                matched = False
                for trajector_exp in trajector_expressions:
                    if (t_text == trajector_exp.text) and (t_start == trajector_exp.start_char) and (t_end == trajector_exp.end_char):
                        trajector_map[t_id] = trajector_exp.t_id
                        matched = True
                        break

                if not matched:
                    trajector_map[t_id] = t_id
                    trajector_expressions.append(
                        TrajectorExpression(
                            t_id=t_id,
                            text=t_text,
                            start_char=t_start,
                            end_char=t_end
                        )
                    )

            landmark_expressions = []
            landmark_map = {}
            for landmark in sentence.find_all("LANDMARK"):
                l_id = landmark.get("id")
                l_start = int(landmark.get("start"))
                l_end = int(landmark.get("end"))
                l_text = landmark.get("text") or ""
                l_start -= l_extra
                l_end -= l_extra
                if l_text == "":
                    continue
                l_start += (len(l_text) - len(l_text.lstrip()))
                l_end -= (len(l_text) - len(l_text.rstrip()))
                l_text = l_text.strip()

                matched = False
                for landmark_exp in landmark_expressions:
                    if (l_text == landmark_exp.text) and (l_start == landmark_exp.start_char) and (l_end == landmark_exp.end_char):
                        landmark_map[l_id] = landmark_exp.l_id
                        matched = True
                        break

                if not matched:
                    landmark_map[l_id] = l_id
                    landmark_expressions.append(
                        LandmarkExpression(
                            l_id=l_id,
                            text=l_text,
                            start_char=l_start,
                            end_char=l_end
                        )
                    )

            sp_indicator_expressions = []
            sp_indicator_map = {}
            for sp_indicator in sentence.find_all("SPATIALINDICATOR"):
                sp_id = sp_indicator.get("id")
                sp_start = int(sp_indicator.get("start"))
                sp_end = int(sp_indicator.get("end"))
                sp_text = sp_indicator.get("text") or ""
                sp_start -= l_extra
                sp_end -= l_extra
                if sp_text == "":
                    continue
                sp_start += (len(sp_text) - len(sp_text.lstrip()))
                sp_end -= (len(sp_text) - len(sp_text.rstrip()))
                sp_text = sp_text.strip()

                matched = False
                for sp_indicator_exp in sp_indicator_expressions:
                    if (sp_text == sp_indicator_exp.text) and (sp_start == sp_indicator_exp.start_char) and (sp_end == sp_indicator_exp.end_char):
                        sp_indicator_map[sp_id] = sp_indicator_exp.sp_id
                        matched = True
                        break
                
                if not matched:
                    sp_indicator_map[sp_id] = sp_id
                    sp_indicator_expressions.append(
                        SpIndicatorExpression(
                            sp_id=sp_id,
                            text=sp_text,
                            start_char=sp_start,
                            end_char=sp_end
                        )
                    )

            relations = []
            for rel in sentence.find_all("RELATION"):
                r_id = rel.get("id")
                t_id = rel.get("trajector_id")
                l_id = rel.get("landmark_id")
                sp_id = rel.get("spatial_indicator_id")
                general_type = rel.get("general_type")
                specific_type = rel.get("specific_type")
                rcc8_val = rel.get("RCC8_value")
                frameOfRef = rel.get("FoR")
                if (t_id not in trajector_map) or (l_id not in landmark_map) or (sp_id not in sp_indicator_map):
                    continue

                relations.append(
                    Relation(
                        r_id=r_id,
                        t_id=trajector_map[t_id],
                        l_id=landmark_map[l_id],
                        sp_id=sp_indicator_map[sp_id],
                        general_type=general_type,
                        specific_type=specific_type,
                        rcc8_val=rcc8_val,
                        frameOfRef=frameOfRef
                    )
                )

            result.append(
                Sentence(
                    sentence_id=sentence_id,
                    doc_no=doc_no,
                    image=image,
                    text=text,
                    start=start,
                    end=end,
                    trajector_expressions=trajector_expressions,
                    landmark_expressions=landmark_expressions,
                    spatial_indicator_expressions=sp_indicator_expressions,
                    relations=relations
                )
            )
        
    return result

        


if __name__=="__main__":
    assert(os.path.exists(SEMEVAL_GOLD))
    assert(os.path.exists(SEMEVAL_TRAIN))

    data = extract_sentences(SEMEVAL_TRAIN)
    with open(SEMEVAL_TRAIN_JSONL, "w", encoding="utf-8") as f:
        for item in data:
            json.dump(item.model_dump(), f)
            f.write("\n")
    print(f"data saved to {SEMEVAL_TRAIN_JSONL}")

    data = extract_sentences(SEMEVAL_GOLD)
    with open(SEMEVAL_GOLD_JSONL, "w", encoding="utf-8") as f:
        for item in data:
            json.dump(item.model_dump(), f)
            f.write("\n")
    print(f"data saved to {SEMEVAL_GOLD_JSONL}")
