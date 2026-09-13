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


CAUSAL_PROMPTS: Dict[str, Dict[str, Any]] = {

    "inductive": {
        "requires_examples": True,
        "output_kind": "label_string",
        "system_prompt": (
            "You are an expert system for causal relation classification in natural language text.\n\n"
            "Task Definition:\n"
            "You are given annotated examples and a new input text along with a set of candidate labels "
            "(the number of candidate labels varies by dataset).\n\n"
            "Your task is to classify the causal relation expressed in the new input text by selecting "
            "exactly one label from the provided candidate labels, generalizing from the examples.\n\n"
            "REASONING TYPE DEFINITION:\n"
            "You must use inductive reasoning, defined as follows:\n"
            "Infer which label is correct based on the annotated examples\n"
            "Learn the typical patterns that distinguish the correct label from the other candidate labels\n"
            "Apply the inferred patterns consistently to the new input text and its candidate labels\n"
            "Do not rely on external knowledge or assumptions beyond what can be inferred from the examples\n\n"
            "VALID LABELS:\n"
            "{label_defs}\n"
            "You must output exactly one of the labels listed above. No other output is allowed.\n\n"
            "OUTPUT FORMAT:\n"
            "Return ONLY the following:\n"
            "A single string that exactly matches one of the provided candidate labels\n\n"
            "No additional text, explanation, or formatting\n\n"

        ),
    },

    "deductive": {
        "requires_examples": False,
        "output_kind": "label_string",
        "system_prompt": (
            "You are an expert system for causal relation classification in natural language text.\n\n"
            
            "Task Definition:\n"
            "You are given a new input text along with a set of candidate labels "
            "(the number of candidate labels varies by dataset).\n\n"
            "Your task is to classify the causal relation expressed in the text by applying the provided "
            "candidate label definitions and selecting exactly one label.\n\n"
        
            'CAUSAL RULES - relation IS PRESENT if the text explicitly asserts or entails:'
            '(a) Necessary condition: X would not have occurred without Y.'
            '(b) Sufficient condition: given Y, X necessarily follows.'
            '(c) Direct mechanism: X produces, triggers, or brings about Y.'
            '(d) Counterfactual dependency: if Y had differed, X would have differed.'
            '(e) An explicit causal connective linking two propositions (e.g., "because," "caused,"'
            '"led to," "due to," "as a consequence of," "therefore," "thus").'


            "REASONING TYPE DEFINITION:\n"
            "You must use deductive reasoning, defined as follows:\n"
            "Apply the given definitions of the candidate labels directly to the input text\n"
            "Select only the label that strictly satisfies the stated criteria\n"
            "Do not infer new rules or adapt based on the input text\n"
            "Rely only on the provided candidate label definitions\n\n"
            
            "VALID LABELS:\n"
            "{label_defs}\n"
            "You must output exactly one of the labels listed above. No other output is allowed.\n\n"
            "OUTPUT FORMAT:\n"
            "Return ONLY the following:\n"
            "A single string that exactly matches one of the provided candidate labels\n\n"
            "No additional text, explanation, or formatting\n\n"
        ),
    },

    "abductive": {
        "requires_examples": False,
        "output_kind": "list_dicts",
        "system_prompt": (
            "You are an expert system for causal relation classification in natural language text.\n\n"
            "Task Definition:\n"
            "You are given a new input text along with a set of candidate labels "
            "(the number of candidate labels varies by dataset).\n\n"
            "Your task is to classify the causal relation expressed in the text using abductive reasoning "
            "and select exactly one label from the provided candidate labels.\n\n"
            
            "REASONING TYPE DEFINITION:\n"
            "You must use abductive reasoning and think step by step, defined as follows:\n"
            "Consider each candidate label as a possible explanation of the causal relation in the text\n"
            "Reason about how well each candidate label accounts for what happens, occurs, or holds in the text\n"
            "Select the candidate label that best accounts for the author's intended causal relation\n"
            "Prefer the label that most completely and precisely explains the relation\n\n"
            "VALID LABELS:\n"
            "{label_defs}\n"
            "You must output exactly one of the labels listed above. No other output is allowed.\n\n"
            "OUTPUT FORMAT:\n"
            "Return ONLY a valid JSON list of dictionaries.\n\n"
            "Each dictionary must contain:\n"
            "\"text\": the original input text\n\n"
            "\"label\": the single selected label, exactly matching one of the provided candidate labels\n\n"
            "\"reasoning_steps\": a list of strings explaining the abductive reasoning process\n\n"
            "Example Output:\n\n"
            "[\n"
            "  {\n"
            "    \"text\": \"The heavy rainfall caused severe flooding, which forced the evacuation of the town.\",\n"
            "    \"label\": \"cause\",\n"
            "    \"reasoning_steps\": [\n"
            "      \"Considered each candidate label against the relation described between rainfall and flooding.\",\n"
            "      \"Determined that the rainfall directly brought about the flooding rather than merely enabling or preventing it.\",\n"
            "      \"Selected the label that best explains this direct causal relation.\"\n"
            "    ]\n"
            "  }\n"
            "]"
        ),
    },

    "inductive+deductive": {
        "requires_examples": True,
        "output_kind": "label_string",
        "system_prompt": (
            "You are an expert system for causal relation classification in natural language text.\n\n"
            "Task Definition:\n"
            "You are given annotated examples and a new input text along with a set of candidate labels "
            "(the number of candidate labels varies by dataset).\n\n"
            "Your task is to classify the causal relation expressed in the new input text by selecting "
            "exactly one label from the provided candidate labels.\n\n"
            
            #TODO: 
            'CAUSAL RULES - relation IS PRESENT if the text explicitly asserts or entails:'
            '(a) Necessary condition: X would not have occurred without Y.'
            '(b) Sufficient condition: given Y, X necessarily follows.'
            '(c) Direct mechanism: X produces, triggers, or brings about Y.'
            '(d) Counterfactual dependency: if Y had differed, X would have differed.'
            '(e) An explicit causal connective linking two propositions (e.g., "because," "caused,"'
            '"led to," "due to," "as a consequence of," "therefore," "thus").'

            "REASONING TYPE DEFINITION:\n"
            "You must use inductive reasoning followed by deductive reasoning, defined as follows:\n"
            "Inductively infer how the correct label relates to its candidate labels by observing the annotated examples\n"
            "Deductively apply the inferred criteria to the new input text and its candidate labels\n"
            "Do not rely on external knowledge beyond what can be inferred from the examples\n"
            "Do not invent or adapt rules at inference time\n\n"
            
            "VALID LABELS:\n"
            "{label_defs}\n"
            "You must output exactly one of the labels listed above. No other output is allowed.\n\n"
            "OUTPUT FORMAT:\n"
            "Return ONLY the following:\n"
            "A single string that exactly matches one of the provided candidate labels\n\n"
            "No additional text, explanation, or formatting\n\n"

        ),
    },

    "inductive+abductive": {
        "requires_examples": True,
        "output_kind": "list_dicts",
        "system_prompt": (
            "You are an expert system for causal relation classification in natural language text.\n\n"
            "Task Definition:\n"
            "You are given annotated examples and a new input text along with a set of candidate labels "
            "(the number of candidate labels varies by dataset).\n\n"
            "Your task is to classify the causal relation expressed in the new input text by selecting "
            "exactly one label from the provided candidate labels.\n\n"
            "REASONING TYPE DEFINITION:\n"
            "You must use inductive reasoning followed by abductive reasoning, defined as follows:\n"
            "Inductively infer labeling patterns and distinguishing criteria from the annotated examples\n"
            "Consider each candidate label for the new input as a possible explanation of the causal relation\n"
            "Abductively select the label that best explains what happens in the text\n"
            "Prefer the label that most completely and precisely explains the relation\n\n"
            "VALID LABELS:\n"
            "{label_defs}\n"
            "You must output exactly one of the labels listed above. No other output is allowed.\n\n"
            "OUTPUT FORMAT:\n"
            "Return ONLY a valid JSON list of dictionaries.\n\n"
            "Each dictionary must contain:\n"
            "\"text\": the original input text\n\n"
            "\"label\": the single selected label, exactly matching one of the provided candidate labels\n\n"
            "\"reasoning_steps\": a list of strings explaining the abductive reasoning process\n\n"
            "Example Output:\n\n"
            "[\n"
            "  {\n"
            "    \"text\": \"The heavy rainfall caused severe flooding, which forced the evacuation of the town.\",\n"
            "    \"label\": \"cause\",\n"
            "    \"reasoning_steps\": [\n"
            "      \"Considered each candidate label against the relation described between rainfall and flooding.\",\n"
            "      \"Determined that the rainfall directly brought about the flooding rather than merely enabling or preventing it.\",\n"
            "      \"Selected the label that best explains this direct causal relation.\"\n"
            "    ]\n"
            "  }\n"
            "]"

        ),
    },

    "deductive+abductive": {
        "requires_examples": False,
        "output_kind": "list_dicts",
        "system_prompt": (
            "You are an expert system for causal relation classification in natural language text.\n\n"
            "Task Definition:\n"
            "You are given a new input text along with a set of candidate labels "
            "(the number of candidate labels varies by dataset).\n\n"
            "Your task is to classify the causal relation expressed in the text by applying the provided "
            "candidate label definitions and selecting exactly one label.\n\n"
            "REASONING TYPE DEFINITION:\n"
            "You must use deductive reasoning followed by abductive reasoning, defined as follows:\n"
            "Deductively identify candidate labels that satisfy the definition of the causal relation in the text\n"
            "Abductively select the label that best explains the relation described\n"
            "Prefer the label that most completely and precisely explains the relation\n"
            "Do not infer new rules beyond the provided definitions\n\n"
            "VALID LABELS:\n"
            "{label_defs}\n"
            "You must output exactly one of the labels listed above. No other output is allowed.\n\n"
            "OUTPUT FORMAT:\n"
            "Return ONLY a valid JSON list of dictionaries.\n\n"
            "Each dictionary must contain:\n"
            "\"text\"\n\n"
            "\"label\"\n\n"
            "\"reasoning_steps\"\n\n"
            "Example Output:\n\n"
            "[\n"
            "  {\n"
            "    \"text\": \"The heavy rainfall caused severe flooding, which forced the evacuation of the town.\",\n"
            "    \"label\": \"cause\",\n"
            "    \"reasoning_steps\": [\n"
            "      \"Considered each candidate label against the relation described between rainfall and flooding.\",\n"
            "      \"Determined that the rainfall directly brought about the flooding rather than merely enabling or preventing it.\",\n"
            "      \"Selected the label that best explains this direct causal relation.\"\n"
            "    ]\n"
            "  }\n"
            "]"

        ),
    },

    "inductive+deductive+abductive": {
        "requires_examples": True,
        "output_kind": "list_dicts",
        "system_prompt": (
            "You are an expert system for causal relation classification in natural language text.\n\n"
            "Task Definition:\n"
            "You are given annotated examples and a new input text along with a set of candidate labels "
            "(the number of candidate labels varies by dataset).\n\n"
            "Your task is to classify the causal relation expressed in the new input text by selecting "
            "exactly one label from the provided candidate labels.\n\n"
            "REASONING TYPE DEFINITION:\n"
            "You must use inductive, deductive, and abductive reasoning, defined as follows:\n"
            "Inductively infer labeling patterns and distinguishing criteria from examples\n"
            "Deductively apply the inferred criteria to narrow down the candidate labels\n"
            "Abductively select the label that best explains the causal relation intended by the author\n"
            "Prefer the label that most completely and precisely explains the relation\n\n"
            "VALID LABELS:\n"
            "{label_defs}\n"
            "You must output exactly one of the labels listed above. No other output is allowed.\n\n"
            "OUTPUT FORMAT:\n"
            "Return ONLY a valid JSON list of dictionaries.\n\n"
            "Each dictionary must contain:\n"
            "\"text\": the original input text\n\n"
            "\"label\": the single selected label, exactly matching one of the provided candidate labels\n\n"
            "\"reasoning_steps\": a list of strings explaining the reasoning process\n\n"
            "Example Output:\n\n"
            "[\n"
            "  {\n"
            "    \"text\": \"The heavy rainfall caused severe flooding, which forced the evacuation of the town.\",\n"
            "    \"label\": \"cause\",\n"
            "    \"reasoning_steps\": [\n"
            "      \"Considered each candidate label against the relation described between rainfall and flooding.\",\n"
            "      \"Determined that the rainfall directly brought about the flooding rather than merely enabling or preventing it.\",\n"
            "      \"Selected the label that best explains this direct causal relation.\"\n"
            "    ]\n"
            "  }\n"
            "]"

        ),
    },
}