### This is deprecated. Use mdg/datamodels.py instead.

from pydantic import BaseModel
from typing import Optional, List


class Expression(BaseModel):
    text: str
    start_char: Optional[int] = None
    end_char: Optional[int] = None


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


class TimeLink(BaseModel):
    link_id: str
    time_id: Optional[str] = None  # This can be None if not applicable
    event_instance_id: Optional[str] = None
    related_to_event_instance_id: Optional[str] = None
    related_to_time_id: Optional[str] = None
    rel_type: str
    signal_id: Optional[str] = None  # This can be None if not applicable
    origin: Optional[str] = None
    comment: Optional[str] = None
    syntax: Optional[str] = None  # This can be a string or None, depending on the context

    @classmethod
    def validate(cls, value):
        if value.time_id is None and value.event_instance_id is None:
            raise ValueError("At least one of time_id or event_instance_id must be provided.")
        if value.related_to_event_instance_id is None and value.related_to_time_id is None:
            raise ValueError("At least one of related_to_event_instance_id or related_to_time_id must be provided.")
        return value

    def __init__(self, **data):
        super().__init__(**data)
        self.validate(self)


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


class Container(BaseModel):
    text: str
    time_expressions: Optional[List[TimeExpression]] = None
    event_expressions: Optional[List[EventExpression]] = None
    signal_expressions: Optional[List[SignalExpression]] = None
    tlinks: Optional[List[TimeLink]] = None


class Document(Container):
    doc_id: str
    dataset: str


class Sentence(Container):
    sent_id: str
    start_char: int  # start character offset in the document
    end_char: int  # end character offset in the document
    doc_id: str
    dataset: str
