
from __future__ import annotations

from typing import Any, Dict


TLINK_REASONING_TYPES = [
    "inductive",
    "deductive",
    "abductive",
    "inductive+deductive",
    "inductive+abductive",
    "deductive+abductive",
    "inductive+deductive+abductive",
]

# NOTE on output_kind:
# - "label": model should output a single label string (BEFORE/AFTER/OTHER/NONE)
# - "list_dicts": model should output a JSON list of dicts with fields:
#     text, label, reasoning_steps
TLINK_PROMPTS: Dict[str, Dict[str, Any]] = {
    "inductive": {
        "requires_examples": True,
        "output_kind": "label",
        "system_prompt": (
            "You are an expert system for temporal relation classification in natural language text.\n\n"
            "Task Definition:\n"
            "You are given annotated examples and a new input sentence with marked spans.\n\n"
            "<e1> … </e1> and <e2> … </e2> mark EVENT spans.\n"
            "<t1> … </t1> and <t2> … </t2> mark TIME spans.\n\n"
            "Role mapping:\n"
            "The FIRST element is the span inside <e1>...</e1> or <t1>...</t1>.\n"
            "The SECOND element is the span inside <e2>...</e2> or <t2>...</t2>.\n\n"
            "Pair types: event–event, time–time, event–time, or time–event.\n\n"
            "Your task is to decide whether a temporal relation exists of the FIRST element "
            "WITH RESPECT TO the SECOND element by generalizing from the examples.\n"
            "Do not flip direction.\n\n"
            "REASONING TYPE DEFINITION:\n"
            "You must use inductive reasoning, defined as follows:\n"
            "Infer from the annotated examples when a temporal relation is labeled YES or NO\n"
            "Learn typical temporal interaction patterns between marked spans\n"
            "Apply the inferred patterns consistently to the new input\n"
            "Do not rely on external knowledge beyond what can be inferred from the examples\n\n"
            "OUTPUT FORMAT:\n"
            "Return ONLY one label from: YES or NO.\n"
            "No explanation or additional text."
),
    },
    "deductive": {
    "requires_examples": False,
    "output_kind": "label_only",
    "system_prompt": (
        "You are an expert system for temporal relation classification in natural language text.\n\n"
        "TASK DEFINITION:\n"
        "You are given a sentence with marked spans.\n\n"
        "<e1> … </e1> and <e2> … </e2> mark EVENT spans.\n"
        "<t1> … </t1> and <t2> … </t2> mark TIME spans.\n\n"
        "ROLE MAPPING:\n"
        "The FIRST element is the span inside <e1>...</e1> or <t1>...</t1>.\n"
        "The SECOND element is the span inside <e2>...</e2> or <t2>...</t2>.\n\n"
        "PAIR TYPES:\n"
        "event–event, time–time, event–time, or time–event.\n\n"
        "Your task is to decide whether a temporal relation exists of the FIRST element "
        "WITH RESPECT TO the SECOND element. Do not flip direction.\n\n"
        "LABEL DEFINITIONS:\n"
        "YES:\n"
        "The sentence explicitly states or clearly implies a temporal relation between the two elements, including:\n"
        "- before\n"
        "- after\n"
        "- overlap\n"
        "- inclusion\n"
        "- simultaneity\n"
        "- explicitly vague but temporal linking\n\n"
        "NO:\n"
        "The sentence does not state or imply any temporal relation between the two elements.\n\n"
        "CRITICAL CONSTRAINTS (VERY IMPORTANT):\n"
        "- Shared topic, participation in the same event, or discourse proximity does NOT count as a temporal relation.\n"
        "- A TIME expression that modifies or anchors a different event than the one marked does not create a temporal relation.\n"
        "- If the temporal connection must be inferred from world knowledge or assumptions, the answer is NO.\n"
        "- When uncertain, choose NO.\n\n"
        "REASONING TYPE DEFINITION:\n"
        "You must use deductive reasoning only.\n"
        "Apply the label definitions and critical constraints directly.\n\n"
        "OUTPUT FORMAT:\n"
        "Return ONLY one label from: YES or NO.\n"
        "No explanation or additional text."
    )
},


    
    "abductive": {
        "requires_examples": False,
        "output_kind": "list_dicts",
        "system_prompt": (
            "You are an expert system for temporal relation classification in natural language text.\n\n"
            "Task Definition:\n"
            "You are given a sentence with marked spans.\n\n"
            "<e1> … </e1> and <e2> … </e2> mark EVENT spans.\n"
            "<t1> … </t1> and <t2> … </t2> mark TIME spans.\n\n"
            "Role mapping:\n"
            "The FIRST element is the span inside <e1>...</e1> or <t1>...</t1>.\n"
            "The SECOND element is the span inside <e2>...</e2> or <t2>...</t2>.\n\n"
            "Pair types: event–event, time–time, event–time, or time–event.\n\n"
            "Decide whether a temporal relation exists of the FIRST element WITH RESPECT TO the SECOND element "
            "using explanatory reasoning. Do not flip direction.\n\n"
            "REASONING TYPE DEFINITION:\n"
            "You must use abductive reasoning and think step by step, defined as follows:\n"
            "Generate plausible temporal interpretations linking the spans\n"
            "Evaluate which interpretation best explains their timing\n"
            "Answer YES if a temporal link best explains the sentence\n"
            "Otherwise answer NO\n\n"
            "OUTPUT FORMAT:\n"
            "Return ONLY a valid JSON list of dictionaries.\n"
            "Each dictionary must contain the following fields:\n"
            "\"text\": the original input text\n"
            "\"label\": the classification label — must be either \"YES\" or \"NO\"\n"
            "\"reasoning_steps\": a list of strings explaining the reasoning process\n\n"
            "Example Output:\n"
            "[\n"
            "  {\n"
            "    \"text\": \"The US embassy in Manila <e1>filed</e1> a diplomatic note <e2>invoking</e2> the right...\",\n"
            "    \"label\": \"YES\",\n"
            "    \"reasoning_steps\": [\n"
            "      \"Both spans are events.\",\n"
            "      \"The invoking event is described as part of the filing action.\",\n"
            "      \"This implies a temporal connection between the two events.\"\n"
            "    ]\n"
            "  }\n"
            "]"
            "[\n"
            "  {\n"
            "    \"text\": \"Writethru: Italy to complete troop withdrawal from Iraq within days\\n\\nItalian Prime Minister Romano Prodi said on <t1>Monday</t1> that Italy will complete its troop withdrawal from Iraq by <t2>Dec.</t2>\",\n"
            "    \"label\": \"NO\",\n"
            "    \"reasoning_steps\": [\n"
            "      \"Both spans are time expressions.\",\n"
            "      \"The sentence mentions two calendar references without asserting a direct temporal ordering relation between them.\",\n"
            "      \"No explicit temporal link is expressed between the FIRST and SECOND spans.\"\n"
            "    ]\n"
            "  }\n"
            "]"
),

    },

    "inductive+deductive": {
    "requires_examples": True,
    "output_kind": "label",
    "system_prompt": (
        "You are an expert system for temporal relation classification in natural language text.\n\n"
        "Task Definition:\n"
        "You are given annotated examples and a new input sentence with marked spans.\n"
        "<e1> … </e1> and <e2> … </e2> mark EVENT spans.\n"
        "<t1> … </t1> and <t2> … </t2> mark TIME spans.\n\n"
        "Role mapping:\n"
        "The FIRST element is the span inside <e1>...</e1> or <t1>...</t1>.\n"
        "The SECOND element is the span inside <e2>...</e2> or <t2>...</t2>.\n\n"
        "Pair types can be: event–event, time–time, event–time, or time–event.\n\n"
        "Classify the temporal relation of the FIRST element WITH RESPECT TO the SECOND element using the labels: BEFORE, AFTER, OTHER, NONE\n"
        "Do not flip direction..\n\n"
        "REASONING TYPE DEFINITION:\n"
        "You must use inductive reasoning followed by deductive reasoning, defined as follows:\n\n"
        "Infer labeling patterns from the annotated examples\n\n"
        "Form criteria from examples\n\n"
        "Apply those criteria directly\n\n"
        "Select the supported label\n\n"
        "Do not invent new rules at inference time\n\n"
        "OUTPUT FORMAT\n\n"
        "Return ONLY one label from the following set:\n\n"
        "BEFORE, AFTER, OTHER, or NONE\n\n"
        "No explanation.\n\n"
        
    ),
},


"inductive+abductive": {
    "requires_examples": True,
    "output_kind": "list_dicts",
    "system_prompt": (
            "You are an expert system for temporal relation classification in natural language text.\n\n"
            "Task Definition:\n"
            "You are given annotated examples and a new input sentence with marked spans.\n\n"
            "<e1> … </e1> and <e2> … </e2> mark EVENT spans.\n"
            "<t1> … </t1> and <t2> … </t2> mark TIME spans.\n\n"
            "Role mapping:\n"
            "The FIRST element is the span inside <e1>...</e1> or <t1>...</t1>.\n"
            "The SECOND element is the span inside <e2>...</e2> or <t2>...</t2>.\n\n"
            "Pair types: event–event, time–time, event–time, or time–event.\n\n"
            "Your task is to decide whether a temporal relation exists of the FIRST element "
            "WITH RESPECT TO the SECOND element. Do not flip direction.\n\n"
            "REASONING TYPE DEFINITION:\n"
            "You must use inductive reasoning followed by deductive reasoning, defined as follows:\n"
            "Infer from the annotated examples when a temporal relation is labeled YES or NO\n"
            "Form criteria based on the examples\n"
            "Apply those criteria directly to the new input\n"
            "Answer YES only if supported by the sentence under the inferred criteria; otherwise answer NO\n"
            "Do not invent new rules at inference time\n\n"
            "OUTPUT FORMAT:\n"
            "Return ONLY one label from: YES or NO.\n"
            "No explanation or additional text."
            "Example Output:\n"
            "[\n"
            "  {\n"
            "    \"text\": \"The US embassy in Manila <e1>filed</e1> a diplomatic note <e2>invoking</e2> the right...\",\n"
            "    \"label\": \"YES\",\n"
            "    \"reasoning_steps\": [\n"
            "      \"Both spans are events.\",\n"
            "      \"The invoking event is described as part of the filing action.\",\n"
            "      \"This implies a temporal connection between the two events.\"\n"
            "    ]\n"
            "  }\n"
            "]"
            "[\n"
            "  {\n"
            "    \"text\": \"Writethru: Italy to complete troop withdrawal from Iraq within days\\n\\nItalian Prime Minister Romano Prodi said on <t1>Monday</t1> that Italy will complete its troop withdrawal from Iraq by <t2>Dec.</t2>\",\n"
            "    \"label\": \"NO\",\n"
            "    \"reasoning_steps\": [\n"
            "      \"Both spans are time expressions.\",\n"
            "      \"The sentence mentions two calendar references without asserting a direct temporal ordering relation between them.\",\n"
            "      \"No explicit temporal link is expressed between the FIRST and SECOND spans.\"\n"
            "    ]\n"
            "  }\n"
            "]"
),

},

"deductive+abductive": {
    "requires_examples": False,
    "output_kind": "list_dicts",
    "system_prompt": (
        "You are an expert system for temporal relation classification in natural language text.\n\n"
        "TASK DEFINITION:\n"
        "You are given a sentence with marked spans.\n\n"
        "<e1> … </e1> and <e2> … </e2> mark EVENT spans.\n"
        "<t1> … </t1> and <t2> … </t2> mark TIME spans.\n\n"
        "ROLE MAPPING:\n"
        "The FIRST element is the span inside <e1>...</e1> or <t1>...</t1>.\n"
        "The SECOND element is the span inside <e2>...</e2> or <t2>...</t2>.\n\n"
        "PAIR TYPES:\n"
        "event–event, time–time, event–time, or time–event.\n\n"
        "Your task is to decide whether a temporal relation exists of the FIRST element "
        "WITH RESPECT TO the SECOND element. Do not flip direction.\n\n"
        "LABEL DEFINITIONS:\n"
        "YES:\n"
        "The sentence explicitly states or clearly implies a temporal relation between the two elements, including:\n"
        "- before\n"
        "- after\n"
        "- overlap\n"
        "- inclusion\n"
        "- simultaneity\n"
        "- explicitly vague but temporal linking\n\n"
        "NO:\n"
        "The sentence does not state or imply any temporal relation between the two elements.\n\n"
        "CRITICAL CONSTRAINTS (VERY IMPORTANT):\n"
        "- Shared topic, participation in the same event, or discourse proximity does NOT count as a temporal relation.\n"
        "- A TIME expression that modifies or anchors a different event than the one marked does not create a temporal relation.\n"
        "- If the temporal connection must be inferred from world knowledge or assumptions, the answer is NO.\n"
        "- When uncertain, choose NO.\n\n"
        "REASONING TYPE DEFINITION:\n"
        "You must use deductive reasoning followed by abductive reasoning and think step by step, defined as follows:\n"
        "- Deductively check whether the sentence explicitly supports any temporal relation listed in the YES definition\n"
        "- Abductively generate possible temporal interpretations ONLY if grounded in the sentence text\n"
        "- Select the interpretation that best explains the sentence evidence\n"
        "- Answer YES only if the final interpretation is directly supported by the sentence\n"
        "- Do NOT invent new rules or rely on external or world knowledge\n\n"
        "OUTPUT FORMAT:\n"
        "Return ONLY a valid JSON list of dictionaries.\n"
        "Each dictionary must contain the following fields:\n"
        "\"text\": the original input text\n"
        "\"label\": the classification label — must be either \"YES\" or \"NO\"\n"
        "\"reasoning_steps\": a list of strings explaining the reasoning process\n\n"
        "Example Output:\n"
        "[\n"
        "  {\n"
        "    \"text\": \"The US embassy in Manila <e1>filed</e1> a diplomatic note <e2>invoking</e2> the right...\",\n"
        "    \"label\": \"YES\",\n"
        "    \"reasoning_steps\": [\n"
        "      \"Both spans are events.\",\n"
        "      \"The invoking event is described as part of the filing action.\",\n"
        "      \"This explicitly links the timing of the two events.\"\n"
        "    ]\n"
        "  }\n"
        "]\n"
        "[\n"
        "  {\n"
        "    \"text\": \"Writethru: Italy to complete troop withdrawal from Iraq within days\\n\\nItalian Prime Minister Romano Prodi said on <t1>Monday</t1> that Italy will complete its troop withdrawal from Iraq by <t2>Dec.</t2>\",\n"
        "    \"label\": \"NO\",\n"
        "    \"reasoning_steps\": [\n"
        "      \"Both spans are time expressions.\",\n"
        "      \"Each time expression anchors a different event.\",\n"
        "      \"No explicit temporal relation is stated between the two marked spans.\"\n"
        "    ]\n"
        "  }\n"
        "]"
    ),
},





"inductive+deductive+abductive": {
    "requires_examples": True,
    "output_kind": "list_dicts",
    "system_prompt": (
        "You are an expert system for temporal relation classification in natural language text.\n\n"
        "TASK DEFINITION:\n"
        "You are given annotated examples and a new input sentence with marked spans.\n\n"
        "<e1> … </e1> and <e2> … </e2> mark EVENT spans.\n"
        "<t1> … </t1> and <t2> … </t2> mark TIME spans.\n\n"
        "ROLE MAPPING:\n"
        "The FIRST element is the span inside <e1>...</e1> or <t1>...</t1>.\n"
        "The SECOND element is the span inside <e2>...</e2> or <t2>...</t2>.\n\n"
        "PAIR TYPES:\n"
        "event–event, time–time, event–time, or time–event.\n\n"
        "Your task is to decide whether a temporal relation exists of the FIRST element "
        "WITH RESPECT TO the SECOND element. Do not flip direction.\n\n"
        "LABEL DEFINITIONS:\n"
        "YES:\n"
        "The sentence explicitly states or clearly implies a temporal relation between the two elements, including:\n"
        "- before\n"
        "- after\n"
        "- overlap\n"
        "- inclusion\n"
        "- simultaneity\n"
        "- explicitly vague but temporal linking\n\n"
        "NO:\n"
        "The sentence does not state or imply any temporal relation between the two elements.\n\n"
        "CRITICAL CONSTRAINTS (VERY IMPORTANT):\n"
        "- Shared topic, participation in the same event, or discourse proximity does NOT count as a temporal relation.\n"
        "- A TIME expression that modifies or anchors a different event than the one marked does not create a temporal relation.\n"
        "- If the temporal connection must be inferred from world knowledge or assumptions, the answer is NO.\n"
        "- When uncertain, choose NO.\n\n"
        "REASONING TYPE DEFINITION:\n"
        "You must use inductive, deductive, and abductive reasoning and think step by step, defined as follows:\n"
        "- Inductively infer from the annotated examples when relations are labeled YES or NO\n"
        "- Deductively apply the inferred criteria to the new input using the label definitions and constraints\n"
        "- Abductively select the interpretation that best explains the timing ONLY if grounded in the sentence text\n"
        "- Answer YES only if supported by explicit sentence evidence; otherwise answer NO\n"
        "- Do NOT rely on external or world knowledge beyond what is provided in the examples and sentence\n\n"
        "OUTPUT FORMAT:\n"
        "Return ONLY a valid JSON list of dictionaries.\n"
        "Each dictionary must contain the following fields:\n"
        "\"text\": the original input text\n"
        "\"label\": the classification label — must be either \"YES\" or \"NO\"\n"
        "\"reasoning_steps\": a list of strings explaining the reasoning process\n\n"
        "Example Output:\n"
        "[\n"
        "  {\n"
        "    \"text\": \"The US embassy in Manila <e1>filed</e1> a diplomatic note <e2>invoking</e2> the right...\",\n"
        "    \"label\": \"YES\",\n"
        "    \"reasoning_steps\": [\n"
        "      \"Both spans are events.\",\n"
        "      \"The invoking event is described as part of the filing action.\",\n"
        "      \"This explicitly links the timing of the two events.\"\n"
        "    ]\n"
        "  }\n"
        "]\n"
        "[\n"
        "  {\n"
        "    \"text\": \"Writethru: Italy to complete troop withdrawal from Iraq within days\\n\\nItalian Prime Minister Romano Prodi said on <t1>Monday</t1> that Italy will complete its troop withdrawal from Iraq by <t2>Dec.</t2>\",\n"
        "    \"label\": \"NO\",\n"
        "    \"reasoning_steps\": [\n"
        "      \"Both spans are time expressions.\",\n"
        "      \"Each time expression anchors a different event.\",\n"
        "      \"No explicit temporal relation is stated between the two marked spans.\"\n"
        "    ]\n"
        "  }\n"
        "]"
    )
},



}
