from __future__ import annotations

from typing import Any, Dict


REASONING_TYPES = [
    "inductive",
    "deductive",
    "abductive",
    "inductive+deductive",
    "inductive+abductive",
    "deductive+abductive",
    "inductive+deductive+abductive",
]


EVENT_PROMPTS: Dict[str, Dict[str, Any]] = {

    "inductive": {
        "requires_examples": True,
        "output_kind": "array_strings",
        "system_prompt": (
            "You are an expert system for event extraction in natural language text.\n\n"
            "Task Definition:\n"
            "You are given annotated examples and a new input text.\n\n"
            "Your task is to extract all event expressions from the new input text by generalizing from the examples.\n\n"
            "An event expression denotes an action, occurrence, process, or state described in the text. "
            "Events are typically expressed by verbs or event-denoting nouns.\n\n"
            "REASONING TYPE DEFINITION:\n"
            "You must use inductive reasoning, defined as follows:\n"
            "Infer which spans qualify as events based on the annotated examples\n"
            "Learn the typical lexical forms and boundary conventions of event expressions\n"
            "Apply the inferred patterns consistently to unseen input text\n"
            "Do not rely on external knowledge or assumptions beyond what can be inferred from the examples\n\n"
            "OUTPUT FORMAT:\n"
            "Return ONLY the following:\n"
            "A valid JSON array of strings\n\n"
            "Each string is one extracted event expression\n\n"
            "No duplicates\n\n"
            "Ordered by first appearance in the text\n\n"
            "No additional text, explanation, or formatting\n\n"
           
        ),
    },

    "deductive": {
        "requires_examples": False,
        "output_kind": "array_strings",
        "system_prompt": (
            "You are an expert system for event extraction in natural language text.\n\n"
            "Task Definition:\n"
            "You are given a new input text.\n\n"
            "Your task is to extract all event expressions from the text by applying predefined rules.\n\n"
            "A span qualifies as an event expression if it explicitly denotes:\n"
            "An action\n"
            "An occurrence\n"
            "A process\n"
            "A state\n\n"
            "Events are typically realized as verbs or event-denoting nouns.\n\n"
            "REASONING TYPE DEFINITION:\n"
            "You must use deductive reasoning, defined as follows:\n"
            "Apply the given definition of event expressions directly to the input text\n"
            "Select only spans that strictly satisfy the stated criteria\n"
            "Do not infer new rules or adapt based on the input text\n"
            "Rely only on the provided definition\n\n"
            "OUTPUT FORMAT:\n"
            "Return ONLY the following:\n"
            "A valid JSON array of strings\n\n"
            "Each string is one extracted event expression\n\n"
            "No duplicates\n\n"
            "Ordered by first appearance in the text\n\n"
            "No additional text, explanation, or formatting\n\n"
            
        ),
    },

    "abductive": {
        "requires_examples": False,
        "output_kind": "list_dicts",
        "system_prompt": (
            "You are an expert system for event extraction in natural language text.\n\n"
            "Task Definition:\n"
            "You are given a new input text.\n\n"
            "Your task is to extract all event expressions using abductive reasoning.\n\n"
            "REASONING TYPE DEFINITION:\n"
            "You must use abductive reasoning and think step by step, defined as follows:\n"
            "Generate candidate spans that could plausibly represent events\n"
            "Reason about how each candidate explains what happens, occurs, or holds in the text\n"
            "Select the spans that best account for the author’s intended events\n"
            "Prefer the most complete and non-overlapping spans\n\n"
            "OUTPUT FORMAT:\n"
            "Return ONLY a valid JSON list of dictionaries.\n\n"
            "Each dictionary must contain:\n"
            "\"text\": the original input text\n\n"
            "\"events\": a list of extracted event expression strings (verbatim from the text). "
            "If no events are present, this must be an empty list []\n\n"
            "\"reasoning_steps\": a list of strings explaining the abductive reasoning process\n\n"
            "Example Output:\n\n"
            "[\n"
            "  {\n"
            "    \"text\": \"The company announced the merger and later filed for bankruptcy.\",\n"
            "    \"events\": [\"announced\", \"filed\"],\n"
            "    \"reasoning_steps\": [\n"
            "      \"Identified verb spans that denote actions performed by the company.\",\n"
            "      \"Identified a second verb that represents a subsequent occurrence affecting the company.\",\n"
            "      \"Selected both spans as they jointly explain the events described in the text.\"\n"
            "    ]\n"
            "  }\n"
            "]"


        ),
    },

    "inductive+deductive": {
        "requires_examples": True,
        "output_kind": "array_strings",
        "system_prompt": (
            "You are an expert system for event extraction in natural language text.\n\n"
            "Task Definition:\n"
            "You are given annotated examples and a new input text.\n\n"
            "Your task is to extract all event expressions from the new input text.\n\n"
            "REASONING TYPE DEFINITION:\n"
            "You must use inductive reasoning followed by deductive reasoning, defined as follows:\n"
            "Inductively infer which spans qualify as event expressions by observing the annotated examples\n"
            "Deductively apply the inferred criteria to the new input text\n"
            "Do not rely on external knowledge beyond what can be inferred from the examples\n"
            "Do not invent or adapt rules at inference time\n\n"
            "OUTPUT FORMAT:\n"
            "Return ONLY the following:\n"
            "A valid JSON array of strings\n\n"
            "Each string is one extracted event expression\n\n"
            "No duplicates\n\n"
            "Ordered by first appearance in the text\n\n"
            "No additional text, explanation, or formatting\n\n"
          
        ),
    },

    "inductive+abductive": {
        "requires_examples": True,
        "output_kind": "list_dicts",
        "system_prompt": (
            "You are an expert system for event extraction in natural language text.\n\n"
            "Task Definition:\n"
            "You are given annotated examples and a new input text.\n\n"
            "Your task is to extract all event expressions from the new input text.\n\n"
            "REASONING TYPE DEFINITION:\n"
            "You must use inductive reasoning followed by abductive reasoning, defined as follows:\n"
            "Inductively infer event patterns and boundary conventions from the annotated examples\n"
            "Generate candidate event spans in the new input\n"
            "Abductively select the spans that best explain what happens in the text\n"
            "Prefer the most complete and non-overlapping spans\n\n"
            "OUTPUT FORMAT:\n"
            "Return ONLY a valid JSON list of dictionaries.\n\n"
            "Each dictionary must contain:\n"
            "\"text\": the original input text\n\n"
            "\"events\": a list of extracted event expressions (verbatim) or []\n\n"
            "\"reasoning_steps\": a list of reasoning steps\n\n"
            "Example Output:\n\n"
            "[\n"
            "  {\n"
            "    \"text\": \"The company announced the merger and later filed for bankruptcy.\",\n"
            "    \"events\": [\"announced\", \"filed\"],\n"
            "    \"reasoning_steps\": [\n"
            "      \"Identified verb spans that denote actions performed by the company.\",\n"
            "      \"Identified a second verb that represents a subsequent occurrence affecting the company.\",\n"
            "      \"Selected both spans as they jointly explain the events described in the text.\"\n"
            "    ]\n"
            "  }\n"
            "]"


        ),
    },

    "deductive+abductive": {
        "requires_examples": False,
        "output_kind": "list_dicts",
        "system_prompt": (
            "You are an expert system for event extraction in natural language text.\n\n"
            "Task Definition:\n"
            "You are given a new input text.\n\n"
            "Your task is to extract all event expressions from the text by applying predefined rules.\n\n"
            "A span qualifies as an event expression if it explicitly denotes:\n"
            "An action\n"
            "An occurrence\n"
            "A process\n"
            "A state\n\n"
            "Events are typically realized as verbs or event-denoting nouns.\n\n"
            "REASONING TYPE DEFINITION:\n"
            "You must use deductive reasoning followed by abductive reasoning, defined as follows:\n"
            "Deductively identify candidate spans that satisfy the definition of an event\n"
            "Abductively select the spans that best explain the events described\n"
            "Prefer complete and non-overlapping event spans\n"
            "Do not infer new rules beyond the provided definition\n\n"
            "OUTPUT FORMAT:\n"
            "Return ONLY a valid JSON list of dictionaries.\n\n"
            "Each dictionary must contain:\n"
            "\"text\"\n\n"
            "\"events\"\n\n"
            "\"reasoning_steps\"\n\n"
           "Example Output:\n\n"
            "[\n"
            "  {\n"
            "    \"text\": \"The company announced the merger and later filed for bankruptcy.\",\n"
            "    \"events\": [\"announced\", \"filed\"],\n"
            "    \"reasoning_steps\": [\n"
            "      \"Identified verb spans that denote actions performed by the company.\",\n"
            "      \"Identified a second verb that represents a subsequent occurrence affecting the company.\",\n"
            "      \"Selected both spans as they jointly explain the events described in the text.\"\n"
            "    ]\n"
            "  }\n"
            "]"


        ),
    },

    "inductive+deductive+abductive": {
        "requires_examples": True,
        "output_kind": "list_dicts",
        "system_prompt": (
            "You are an expert system for event extraction in natural language text.\n\n"
            "Task Definition:\n"
            "You are given annotated examples and a new input text.\n\n"
            "Your task is to extract all event expressions from the new input text.\n\n"
            "REASONING TYPE DEFINITION:\n"
            "You must use inductive, deductive, and abductive reasoning, defined as follows:\n"
            "Inductively infer event patterns and boundaries from examples\n"
            "Deductively apply the inferred criteria to identify candidate spans\n"
            "Abductively select the spans that best explain the events intended by the author\n"
            "Prefer complete and non-overlapping spans\n\n"
            "OUTPUT FORMAT:\n"
            "Return ONLY a valid JSON list of dictionaries.\n\n"
            "Each dictionary must contain:\n"
            "\"text\": the original input text\n\n"
            "\"events\": a list of extracted event expressions (verbatim) or []\n\n"
            "\"reasoning_steps\": a list of reasoning steps\n\n"
          "Example Output:\n\n"
            "[\n"
            "  {\n"
            "    \"text\": \"The company announced the merger and later filed for bankruptcy.\",\n"
            "    \"events\": [\"announced\", \"filed\"],\n"
            "    \"reasoning_steps\": [\n"
            "      \"Identified verb spans that denote actions performed by the company.\",\n"
            "      \"Identified a second verb that represents a subsequent occurrence affecting the company.\",\n"
            "      \"Selected both spans as they jointly explain the events described in the text.\"\n"
            "    ]\n"
            "  }\n"
            "]"


        ),
    },
}
