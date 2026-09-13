from pydantic import BaseModel, model_validator
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

class CausalExpression(Expression):
    c_id: str


class EventExpression(Expression):
    eid: str
    eiid: str  # event instance ID
    type: str
    pos: Optional[str] = None
    tense: Optional[str] = None
    aspect: Optional[str] = None
    polarity: Optional[str] = None
    modality: Optional[str] = None


class TimeExpression(Expression):
    tid: str
    type: str
    value: str = ""
    temporal_function: Optional[bool] = False
    function_in_document: Optional[str] = None
    anchor_time: Optional[str] = None


class SignalExpression(BaseModel):
    signal_id: str
    text: str
    start_char: int
    end_char: int


class Relation(BaseModel):
    r_id: str
    t_id: str
    l_id: str
    sp_id: str
    general_type: str
    specific_type: str
    rcc8_val: str
    frameOfRef: str


class TimeLink(BaseModel):
    link_id: str
    time_id: Optional[str] = None  # This can be None if not applicable
    event_instance_id: Optional[str] = None
    related_to_event_instance_id: Optional[str] = None
    related_to_time_id: Optional[str] = None
    rel_type: str
    signal_id: Optional[str] = None  # This can be None if not applicable
    origin: Optional[str]
    comment: Optional[str] = None
    syntax: Optional[str] = None  # This can be a string or None, depending on the context

    @model_validator(mode="after")
    def validate(cls, value):
        if value.time_id is None and value.event_instance_id is None:
            raise ValueError("At least one of time_id or event_instance_id must be provided.")
        if value.related_to_event_instance_id is None and value.related_to_time_id is None:
            raise ValueError("At least one of related_to_event_instance_id or related_to_time_id must be provided.")
        return value


class Duration(BaseModel):
    start_timex: str
    end_timex: str


class TimeExpressionGrounding(BaseModel):
    context_doc_id: str
    anchor_time_id: str
    anchor_timex: TimeExpression
    grounded_anchor_timex: str
    type: str  # either date_time or duration
    query: TimeExpression
    grounded_query: str | Duration


class TimeMixin(BaseModel):
    """
    A container for temporal expressions, events, signals, and time links."""

    time_expressions: Optional[List[TimeExpression]] = None
    event_expressions: Optional[List[EventExpression]] = None
    signal_expressions: Optional[List[SignalExpression]] = None
    tlinks: Optional[List[TimeLink]] = None


class SpatialMixin(BaseModel):
    """
    A container for spatial expressions and relations.
    """

    trajector_expressions: Optional[List[TrajectorExpression]] = None
    landmark_expressions: Optional[List[LandmarkExpression]] = None
    spatial_indicator_expressions: Optional[List[SpIndicatorExpression]] = None
    relations: Optional[List[Relation]] = None


class Sentence(BaseModel):
    sent_id: str
    text: str
    start_char: Optional[int] = None # start character offset in the document
    end_char: Optional[int] = None # end character offset in the document
    doc_id: str


class TimeSentence(TimeMixin, Sentence):
    pass


class SpatialSentence(SpatialMixin, Sentence):
    image: str


class Document(BaseModel):
    doc_id: str
    text: str


class TimeDocument(TimeMixin, Document):
    dataset: str
    
    
class CausalDocument(Document):
    dataset:str


class SpatialDocument(SpatialMixin, Document):
    dataset: str
    image: str


class Result(BaseModel):
    metric_name: str
    value: float
    details: Optional[dict] = None


class CompositeResult(BaseModel):
    dataset_name: str
    results: List[Result]


