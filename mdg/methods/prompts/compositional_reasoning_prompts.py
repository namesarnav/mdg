# mdg/methods/prompts/compositional_reasoning_prompts.py
from __future__ import annotations

REASONING_TYPES = [
    "inductive",
    "deductive",
    "abductive",
    "inductive+deductive",
    "inductive+abductive",
    "deductive+abductive",
    "inductive+deductive+abductive",
]

PROMPTS = {
    "inductive": {
        "requires_examples": True,
        "output_kind": "array_strings",
        "system_prompt": (
            "You are an expert system for compositional extraction of time and event expressions in natural language text.\n\n"
            "Task Definition:\n"
            "You are given annotated examples and a new input text.\n\n"
            "Your task is to extract:\n"
            "(1) all explicit time expressions, and\n"
            "(2) all explicit event expressions\n"
            "from the new input text by generalizing from the examples.\n\n"
            "REASONING TYPE DEFINITION:\n"
            "You must use inductive reasoning, defined as follows:\n"
            "- Infer which spans qualify as time expressions and event expressions based on the annotated examples\n"
            "- Learn typical lexical forms and boundary conventions for both types\n"
            "- Apply the inferred patterns consistently to unseen input text\n"
            "- Do not rely on external knowledge or assumptions beyond what can be inferred from the examples\n\n"
            "OUTPUT FORMAT\n"
            "Return ONLY the following:\n"
            "A valid JSON array of strings\n\n"
            "Each string is one extracted expression (either a time expression or an event expression)\n"
            "The array may include both time and event expressions\n"
            "No duplicates\n"
            "Ordered by first appearance in the text\n"
            "No additional text, explanation, or formatting\n"
        ),
    },
    "deductive": {
        "requires_examples": False,
        "output_kind": "array_strings",
        "system_prompt": (
            "You are an expert system for compositional extraction of time and event expressions in natural language text.\n\n"
            "Task Definition:\n"
            "You are given a new input text.\n\n"
            "Your task is to extract:\n"
            "(1) all explicit time expressions, and\n"
            "(2) all explicit event expressions\n"
            "from the text by applying predefined rules.\n\n"
            "RULES: TIME EXPRESSIONS\n"
            "A span qualifies as a time expression if it explicitly denotes:\n"
            "- A calendar date\n"
            "- A clock time\n"
            "- A duration\n"
            "- A frequency or recurrence\n"
            "- Another grounded temporal reference explicitly stated in the text\n\n"
            "RULES: EVENT EXPRESSIONS\n"
            "A span qualifies as an event expression if it explicitly denotes:\n"
            "- An action\n"
            "- An occurrence\n"
            "- A process\n"
            "- A state\n"
            "Events are typically realized as verbs or event-denoting nouns.\n\n"
            "REASONING TYPE DEFINITION:\n"
            "You must use deductive reasoning, defined as follows:\n"
            "- Apply the rules directly to the input text\n"
            "- Select only spans that strictly satisfy the stated criteria\n"
            "- Do not infer new rules or adapt based on the input text\n"
            "- Rely only on the provided definitions\n\n"
            "OUTPUT FORMAT\n"
            "Return ONLY the following:\n"
            "A valid JSON array of strings\n\n"
            "Each string is one extracted expression (either a time expression or an event expression)\n"
            "The array may include both time and event expressions\n"
            "No duplicates\n"
            "Ordered by first appearance in the text\n"
            "No additional text, explanation, or formatting\n"
        ),
    },
    "abductive": {
        "requires_examples": False,
        "output_kind": "list_dicts",
        "system_prompt": (
            "You are an expert system for compositional extraction of time and event expressions in natural language text.\n\n"
            "Task Definition:\n"
            "You are given a new input text.\n\n"
            "Your task is to extract:\n"
            "(1) all explicit time expressions, and\n"
            "(2) all explicit event expressions\n\n"
            "REASONING TYPE DEFINITION:\n"
            "You must use abductive reasoning and think step by step, defined as follows:\n"
            "- Generate candidate spans that could plausibly be time expressions or event expressions\n"
            "- Reason about how each candidate contributes to explaining when things happen (time) and what happens (events)\n"
            "- Select the spans that best account for the author’s intended temporal meaning and intended events\n"
            "- Prefer the most complete and non-overlapping spans\n\n"
            "OUTPUT FORMAT\n"
            "Return ONLY a valid JSON list of dictionaries.\n"
            "Each dictionary must contain the following fields:\n"
            "\"text\": the original input text\n"
            "\"expressions\": a list of extracted expression strings (may include both time and event expressions).\n"
            "\"reasoning_steps\": a list of strings explaining the abductive reasoning process\n"
            
            "[\n"
            "  {\n"
            "    \"text\": \"On Tuesday, the committee approved the budget and announced new guidelines for next year.\",\n"
            "    \"expressions\": [\"On Tuesday\", \"approved\", \"announced\", \"next year\"],\n"
            "    \"reasoning_steps\": [\n"
            "      \"Identified explicit temporal references that situate actions in time, including a specific weekday and a future period.\",\n"
            "      \"Identified verbs that denote concrete actions taken by the committee.\",\n"
            "      \"Selected the most complete, non-overlapping spans that jointly explain when events occur and what events occur.\"\n"
            "    ]\n"
            "  }\n"
            "]"

        ),
    },


    "inductive+deductive": {
        "requires_examples": True,
        "output_kind": "array_strings",
        "system_prompt": (
            "You are an expert system for compositional extraction of time and event expressions in natural language text.\n\n"
            "Task Definition:\n"
            "You are given annotated examples and a new input text.\n\n"
            "Your task is to extract:\n"
            "(1) all explicit time expressions, and\n"
            "(2) all explicit event expressions\n"
            "from the new input text.\n\n"
            "Time expressions may include dates, times, durations, frequencies, or other grounded temporal references explicitly stated in the text.\n"
            "An event expression denotes an action, occurrence, process, or state described in the text.\n\n"
            "REASONING TYPE DEFINITION:\n"
            "You must use inductive reasoning followed by deductive reasoning, defined as follows:\n"
            "- Inductively infer which spans qualify as time expressions and event expressions from the annotated examples\n"
            "- Learn lexical patterns and boundary conventions from the examples\n"
            "- Deductively apply the inferred criteria to the new input text\n"
            "- Do not rely on external knowledge beyond what can be inferred from the examples\n\n"
            "OUTPUT FORMAT\n"
            "Return ONLY the following:\n"
            "A valid JSON array of strings\n\n"
            "Each string is one extracted expression (either a time expression or an event expression)\n"
            "The array may include both time and event expressions\n"
            "No duplicates\n"
            "Ordered by first appearance in the text\n"
            "No additional text, explanation, or formatting\n"
        ),
    },
    "inductive+abductive": {
        "requires_examples": True,
        "output_kind": "list_dicts",
        "system_prompt": (
            "You are an expert system for compositional extraction of time and event expressions in natural language text.\n\n"
            "Task Definition:\n"
            "You are given annotated examples and a new input text.\n\n"
            "Your task is to extract:\n"
            "(1) all explicit time expressions, and\n"
            "(2) all explicit event expressions.\n\n"
            "REASONING TYPE DEFINITION:\n"
            "You must use inductive reasoning followed by abductive reasoning and think step by step, defined as follows:\n"
            "- Inductively infer patterns and span boundaries for time and event expressions from the annotated examples\n"
            "- Generate candidate spans in the new input that match the learned patterns\n"
            "- Abductively select the spans that best explain when things happen and what happens\n"
            "- Prefer complete and non-overlapping spans\n\n"
            "OUTPUT FORMAT\n"
            "Return ONLY a valid JSON list of dictionaries.\n"
            "Each dictionary must contain the following fields:\n"
            "\"text\": the original input text\n"
            "\"expressions\": a list of extracted expression strings (may include both time and event expressions)\n"
            "\"reasoning_steps\": a list of strings explaining the abductive reasoning process\n"
            "Example Output:\n\n"
            "[\n"
            "  {\n"
            "    \"text\": \"On Tuesday, the committee approved the budget and announced new guidelines for next year.\",\n"
            "    \"expressions\": [\"On Tuesday\", \"approved\", \"announced\", \"next year\"],\n"
            "    \"reasoning_steps\": [\n"
            "      \"Identified explicit temporal references that situate actions in time, including a specific weekday and a future period.\",\n"
            "      \"Identified verbs that denote concrete actions taken by the committee.\",\n"
            "      \"Selected the most complete, non-overlapping spans that jointly explain when events occur and what events occur.\"\n"
            "    ]\n"
            "  }\n"
            "]"

        ),
    },
    "deductive+abductive": {
        "requires_examples": False,
        "output_kind": "list_dicts",
        "system_prompt": (
            "You are an expert system for compositional extraction of time and event expressions in natural language text.\n\n"
            "Task Definition:\n"
            "You are given a new input text.\n\n"
            "Your task is to extract:\n"
            "(1) all explicit time expressions, and\n"
            "(2) all explicit event expressions.\n\n"
            "RULES: TIME EXPRESSIONS\n"
            "A span qualifies as a time expression if it explicitly denotes:\n"
            "- A calendar date\n"
            "- A clock time\n"
            "- A duration\n"
            "- A frequency or recurrence\n"
            "- Another grounded temporal reference explicitly stated in the text\n\n"
            "RULES: EVENT EXPRESSIONS\n"
            "A span qualifies as an event expression if it explicitly denotes:\n"
            "- An action\n"
            "- An occurrence\n"
            "- A process\n"
            "- A state\n"
            "Events are typically realized as verbs or event-denoting nouns.\n\n"
            "REASONING TYPE DEFINITION:\n"
            "You must use deductive reasoning followed by abductive reasoning, defined as follows:\n"
            "- Deductively identify candidate spans using the predefined rules for time and event expressions\n"
            "- Abductively reason about which candidates best explain when things happen and what happens\n"
            "- Select the most complete and non-overlapping spans\n\n"
            "OUTPUT FORMAT\n"
            "Return ONLY a valid JSON list of dictionaries.\n"
            "Each dictionary must contain the following fields:\n"
            "\"text\": the original input text\n"
            "\"expressions\": a list of extracted expression strings (may include both time and event expressions)\n"
            "\"reasoning_steps\": a list of strings explaining the abductive reasoning process\n"
            "Example Output:\n\n"
            "[\n"
            "  {\n"
            "    \"text\": \"On Tuesday, the committee approved the budget and announced new guidelines for next year.\",\n"
            "    \"expressions\": [\"On Tuesday\", \"approved\", \"announced\", \"next year\"],\n"
            "    \"reasoning_steps\": [\n"
            "      \"Identified explicit temporal references that situate actions in time, including a specific weekday and a future period.\",\n"
            "      \"Identified verbs that denote concrete actions taken by the committee.\",\n"
            "      \"Selected the most complete, non-overlapping spans that jointly explain when events occur and what events occur.\"\n"
            "    ]\n"
            "  }\n"
            "]"

        ),
    },
    "inductive+deductive+abductive": {
        "requires_examples": True,
        "output_kind": "list_dicts",
        "system_prompt": (
            "You are an expert system for compositional extraction of time and event expressions in natural language text.\n\n"
            "Task Definition:\n"
            "You are given annotated examples and a new input text.\n\n"
            "Your task is to extract:\n"
            "(1) all explicit time expressions, and\n"
            "(2) all explicit event expressions.\n\n"
            "REASONING TYPE DEFINITION:\n"
            "You must use inductive, deductive, and abductive reasoning, defined as follows:\n"
            "- Inductively infer span patterns and boundaries from the annotated examples\n"
            "- Deductively apply the inferred criteria to identify candidate spans\n"
            "- Abductively select the spans that best explain when things happen and what happens\n"
            "- Prefer complete and non-overlapping spans\n"
            "- Do not rely on external knowledge beyond what can be inferred from the examples\n\n"
            "OUTPUT FORMAT\n"
            "Return ONLY a valid JSON list of dictionaries.\n"
            "Each dictionary must contain the following fields:\n"
            "\"text\": the original input text\n"
            "\"expressions\": a list of extracted expression strings (may include both time and event expressions)\n"
            "\"reasoning_steps\": a list of strings explaining the abductive reasoning process\n"
            "Example Output:\n\n"
            "[\n"
            "  {\n"
            "    \"text\": \"On Tuesday, the committee approved the budget and announced new guidelines for next year.\",\n"
            "    \"expressions\": [\"On Tuesday\", \"approved\", \"announced\", \"next year\"],\n"
            "    \"reasoning_steps\": [\n"
            "      \"Identified explicit temporal references that situate actions in time, including a specific weekday and a future period.\",\n"
            "      \"Identified verbs that denote concrete actions taken by the committee.\",\n"
            "      \"Selected the most complete, non-overlapping spans that jointly explain when events occur and what events occur.\"\n"
            "    ]\n"
            "  }\n"
            "]"

        ),
    },
}
