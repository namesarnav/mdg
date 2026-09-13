from pydantic import BaseModel
from typing import Optional, List


class Expression(BaseModel):
    text: str
    start_char: Optional[int] = None
    end_char: Optional[int] = None


class LandmarkExpression(Expression):
    l_id: str


class TrajectorExpression(Expression):
    t_id: str


class SpIndicatorExpression(Expression):
    sp_id: str


class Relation(BaseModel):
    r_id: str
    t_id: str
    l_id: str
    sp_id: str
    general_type: str
    specific_type: str
    rcc8_val: str
    frameOfRef: str


class Sentence(BaseModel):
    sentence_id: str
    doc_no: str
    image: str
    text: str
    start: Optional[int] = None
    end: Optional[int] = None
    trajector_expressions: List[TrajectorExpression]
    landmark_expressions: List[LandmarkExpression]
    spatial_indicator_expressions: List[SpIndicatorExpression]
    relations: List[Relation]


class Document(BaseModel):
    doc_no: str
    image: str
    text: str
    trajector_expressions: List[TrajectorExpression]
    landmark_expressions: List[LandmarkExpression]
    spatial_indicator_expressions: List[SpIndicatorExpression]
    relations: List[Relation]

