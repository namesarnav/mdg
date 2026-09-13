from __future__ import annotations
import os, time, json, re
from typing import Any, Dict, List, Optional
import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

def _array_of_strings_schema() -> Dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "time_expressions",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["time_expressions"],
                "properties": {
                    "time_expressions": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "description": "Exact surface form of a time expression"
                        }
                    }
                }
            }
        },
    }

def _compositional_list_of_dicts_schema(
    *, name: str = "compositional_extraction_with_reasoning"
) -> Dict[str, Any]:
    """
    Enforce compositional abductive output:
    [
      {
        "text": "...",
        "expressions": ["..."],
        "reasoning_steps": ["..."]
      }
    ]
    """
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": {
                "type": "array",
                "minItems": 1,
                "maxItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["text", "expressions", "reasoning_steps"],
                    "properties": {
                        "text": {"type": "string"},
                        "expressions": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "reasoning_steps": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                },
            },
        },
    }
def _one_label_enum_schema(allowed_labels: List[str], *, name: str = "tlink_label") -> Dict[str, Any]:
    """
    Enforce: array of exactly 1 string, and that string must be one of allowed_labels.
    """
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": {
                "type": "array",
                "minItems": 1,
                "maxItems": 1,
                "items": {
                    "type": "string",
                    "enum": list(allowed_labels),
                    "description": "One temporal relation label"
                }
            }
        },
    }
def _timex_list_of_dicts_schema(*, name: str = "timex_extraction_with_reasoning") -> Dict[str, Any]:
    """
    Enforce abductive output:
    [
      {
        "text": "...",
        "time_expressions": ["..."],
        "reasoning_steps": ["..."]
      }
    ]
    """
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": {
                "type": "array",
                "minItems": 1,
                "maxItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["text", "time_expressions", "reasoning_steps"],
                    "properties": {
                        "text": {"type": "string"},
                        "time_expressions": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "reasoning_steps": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                },
            },
        },
    }


def _parse_json_array(content: Any) -> List[str]:
    if isinstance(content, list):
        return [str(x) for x in content]
    if isinstance(content, str):
        try:
            obj = json.loads(content)
            if isinstance(obj, list):
                return [str(x) for x in obj]
        except Exception:
            pass
        m = re.search(r"\[[\s\S]*\]", content)
        if m:
            try:
                obj = json.loads(m.group(0))
                if isinstance(obj, list):
                    return [str(x) for x in obj]
            except Exception:
                pass
    return []

def _parse_json_any(content: Any) -> Any:
    """
    Parse any JSON value (list/dict/str/number/bool/null) from:
    - already-parsed python object
    - string containing JSON
    - string containing a JSON substring
    """
    if isinstance(content, (dict, list)):
        return content
    if isinstance(content, str):
        s = content.strip()
        # Try direct parse
        try:
            return json.loads(s)
        except Exception:
            pass

        # Try to extract a JSON array or object substring
        # Prefer array first, then object
        m = re.search(r"\[[\s\S]*\]", s)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        m = re.search(r"\{[\s\S]*\}", s)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return None

def _extract_first_allowed_label(content: Any, allowed_labels: List[str]) -> Optional[str]:
    """
    Soft parser for non-structured outputs:
    - Tries JSON array first (e.g., ["AFTER"])
    - Falls back to scanning the raw string for the first allowed label
    """
    # Try to parse as JSON array first
    arr = _parse_json_array(content)
    if arr:
        s = str(arr[0]).strip().upper().replace(" ", "_").replace("-", "_")
        return s if s in allowed_labels else None

    # Fallback: scan raw text
    if isinstance(content, str):
        text = content.strip().upper().replace("-", "_")
        for lab in allowed_labels:
            if lab in text:
                return lab
        # Last resort: first ALLCAPS token
        m = re.search(r"\b[A-Z_]{3,}\b", text)
        if m:
            cand = m.group(0)
            if cand in allowed_labels:
                return cand
    return None
def call_openrouter_json_array(
    *,
    api_key: Optional[str],
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    retries: int = 3,
    timeout_sec: int = 60,
    sleep_ms_between_calls: int = 400,
    headers_extra: Optional[Dict[str, str]] = None,
) -> List[str]:
    if not api_key:
        raise RuntimeError("OpenRouter API key is missing. Set env and pass through config.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if headers_extra:
        headers.update(headers_extra)

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "response_format": _array_of_strings_schema(),
        "temperature": temperature,
    }
    if max_tokens is not None:
        payload["max_tokens"] = int(max_tokens)

    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=timeout_sec)
            if resp.status_code >= 400 and resp.status_code != 429:
                print("\n[OpenRouter DEBUG] status:", resp.status_code)
                print("[OpenRouter DEBUG] text:", (resp.text or "")[:2000])
                print("[OpenRouter DEBUG] model:", payload.get("model"))
                print("[OpenRouter DEBUG] payload_keys:", list(payload.keys()), "\n")
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            

            obj = _parse_json_any(content)

            if isinstance(obj, dict):
    # ✅ only return if the key exists (otherwise DON'T early-return [])
                for key in ("time_expressions", "expressions", "post_processed_expressions", "outputs", "spans"):
                    if key in obj and isinstance(obj[key], list):
                        return [str(x) for x in obj[key]]
            

                # fallback if schema is ignored and model returns a bare JSON array: [...]
            return _parse_json_array(content)
        except Exception as e:
            last_err = e
            time.sleep((sleep_ms_between_calls + 200 * attempt) / 1000.0)

    print(f"[OpenRouter] Failed after {retries} retries: {last_err}")
    return []

def call_openrouter_json(
    *,
    api_key: Optional[str],
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    retries: int = 3,
    timeout_sec: int = 60,
    sleep_ms_between_calls: int = 400,
    headers_extra: Optional[Dict[str, str]] = None,
    response_schema: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Generic JSON caller.
    Used for abductive TIMEX outputs (list of dicts), or any other structured JSON.

    If response_schema is None, it will still try to parse JSON from the model output.
    """
    if not api_key:
        raise RuntimeError("OpenRouter API key is missing. Set env and pass through config.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if headers_extra:
        headers.update(headers_extra)

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if response_schema is not None:
        payload["response_format"] = response_schema
    if max_tokens is not None:
        payload["max_tokens"] = int(max_tokens)

    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=timeout_sec)
            if resp.status_code in (429, 500, 502, 503, 504):
                time.sleep((sleep_ms_between_calls + 400 * attempt) / 1000.0)
                continue
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            '''
            print("\n[DEBUG RAW CONTENT]")
            print("model:", model)
            print("content type:", type(content))
            print(content)
            '''

            return _parse_json_any(content)
        except Exception as e:
            last_err = e
            time.sleep((sleep_ms_between_calls + 200 * attempt) / 1000.0)

    print(f"[OpenRouter] Failed after {retries} retries: {last_err}")
    return None


'''
def call_openrouter_label_array(
    *,
    api_key: Optional[str],
    model: str,
    messages: List[Dict[str, str]],
    allowed_labels: List[str],
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    retries: int = 3,
    timeout_sec: int = 60,
    sleep_ms_between_calls: int = 400,
    headers_extra: Optional[Dict[str, str]] = None,
    #return_content: bool = False,
) -> List[str]:
    """
    Requests a JSON array with exactly one item from the allowed enum.
    """
    if not api_key:
        raise RuntimeError("OpenRouter API key is missing. Set env and pass through config.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if headers_extra:
        headers.update(headers_extra)

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "response_format": _one_label_enum_schema(list(allowed_labels)),
        "temperature": temperature,
    }
    if max_tokens is not None:
        payload["max_tokens"] = int(max_tokens)

    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=timeout_sec)
            if resp.status_code in (429, 500, 502, 503):
                time.sleep((sleep_ms_between_calls + 400 * attempt) / 1000.0)
                continue
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            #arr = _parse_json_array(content)
            return _parse_json_array(content)
            #return (arr, content) if return_content else arr
        except Exception as e:
            last_err = e
            time.sleep((sleep_ms_between_calls + 200 * attempt) / 1000.0)

    print(f"[OpenRouter] Failed after {retries} retries: {last_err}")
    return []
'''

def call_openrouter_label_array(
    *,
    api_key: Optional[str],
    model: str,
    messages: List[Dict[str, str]],
    allowed_labels: List[str],
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    retries: int = 3,
    timeout_sec: int = 60,
    sleep_ms_between_calls: int = 400,
    headers_extra: Optional[Dict[str, str]] = None,
    schema_fallback: bool = True,   # NEW: fallback enabled by default
) -> List[str]:
    """
    Tries structured outputs (json_schema). If the route/model rejects it (400) or
    returns junk, falls back to a non-structured call and coerces the output client-side.
    """
    if not api_key:
        raise RuntimeError("OpenRouter API key is missing. Set env and pass through config.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if headers_extra:
        headers.update(headers_extra)

    allowed_labels = list(allowed_labels)  # ensure list, not set

    # ---------- Attempt 1: structured outputs (json_schema) ----------
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "response_format": _one_label_enum_schema(allowed_labels),
        "temperature": temperature,
    }
    if max_tokens is not None:
        payload["max_tokens"] = int(max_tokens)

    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=timeout_sec)
            if resp.status_code in (429, 500, 502, 503, 504):
                ra = resp.headers.get("Retry-After")
                delay = float(ra) if ra else (sleep_ms_between_calls + 400 * attempt) / 1000.0
                time.sleep(delay)
                continue
            if not resp.ok:
                # Likely 400 when schema unsupported or bad payload
                last_err = Exception(f"{resp.status_code} {resp.text[:300]}")
                break
            j = resp.json()
            content = (j.get("choices") or [{}])[0].get("message", {}).get("content", "")
            arr = _parse_json_array(content)
            if len(arr) == 1:
                s = str(arr[0]).strip().upper().replace(" ", "_").replace("-", "_")
                if s in allowed_labels:
                    return [s]
            # If schema ignored and a bare string came back
            if isinstance(content, str):
                s = content.strip().upper().replace(" ", "_").replace("-", "_")
                if s in allowed_labels:
                    return [s]
            last_err = Exception("Structured parse failed")
        except Exception as e:
            last_err = e
        time.sleep((sleep_ms_between_calls + 200 * attempt) / 1000.0)

    # ---------- Attempt 2: fallback (no schema) ----------
    if not schema_fallback:
        print(f"[OpenRouter] Structured outputs failed and fallback disabled: {last_err}")
        return []

    # Strengthen the final user message: append explicit allowed list and strict rule
    try:
        msgs2 = list(messages)
        if msgs2 and msgs2[-1].get("role") == "user":
            allowed_list = "[" + ",".join(f"\"{x}\"" for x in allowed_labels) + "]"
            suffix = (
                "\n\nRespond ONLY with exactly one label in UPPERCASE, chosen from: "
                f"{allowed_list}\nIf you output anything else, it will be discarded."
            )
            msgs2[-1] = dict(msgs2[-1])
            msgs2[-1]["content"] = (msgs2[-1]["content"] or "") + suffix
        else:
            msgs2.append({
                "role": "user",
                "content": "Respond ONLY with one of: " + ", ".join(allowed_labels)
            })
    except Exception:
        msgs2 = messages  # best effort

    payload_fallback: Dict[str, Any] = {
        "model": model,
        "messages": msgs2,
        "temperature": temperature,
    }
    if max_tokens is not None:
        payload_fallback["max_tokens"] = int(max_tokens)

    for attempt in range(max(1, retries)):
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload_fallback, timeout=timeout_sec)
            if resp.status_code in (429, 500, 502, 503, 504):
                ra = resp.headers.get("Retry-After")
                delay = float(ra) if ra else (sleep_ms_between_calls + 400 * attempt) / 1000.0
                time.sleep(delay); continue
            resp.raise_for_status()
            j = resp.json()
            content = (j.get("choices") or [{}])[0].get("message", {}).get("content", "")
            s = _extract_first_allowed_label(content, allowed_labels)
            if s:
                return [s]
        except Exception as e:
            last_err = e
        time.sleep((sleep_ms_between_calls + 200 * attempt) / 1000.0)

    print(f"[OpenRouter] Failed after fallback: {last_err}")
    return []

__all__ = [
    "call_openrouter_json_array",
    "call_openrouter_json",
    "call_openrouter_label_array",
    "_timex_list_of_dicts_schema",
    "_compositional_list_of_dicts_schema",
]

