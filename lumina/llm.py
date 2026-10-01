from __future__ import annotations

import math
import time

import requests
from openai import OpenAI


def single_chat(llm: dict, llm_settings: dict, message_input: list[dict], temperature=0.01):
    provider = llm_settings[llm["source"]]
    if not provider.get('key') or not provider.get('url'):
        raise ValueError('chat provider requires an explicit nonempty key and URL')
    client = OpenAI(
        api_key=provider["key"],
        base_url=provider["url"],
    )
    request = {
        "model": llm["model"],
        "messages": message_input,
        "temperature": temperature,
        "stream": False,
    }
    if provider.get("supports_json_mode", True):
        request["response_format"] = {"type": "json_object"}
    try:
        completion = client.chat.completions.create(**request)
        if not completion.choices:
            raise ValueError('chat response has no choices')
        content = completion.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise ValueError('chat response has empty/null content or refusal')
        tokens = completion.usage.total_tokens if completion.usage is not None else None
        return content, tokens
    finally:
        client.close()


def embedding_response(text: str, embedding_model: dict, llm_settings: dict):
    if not llm_settings[embedding_model['source']].get('key'):
        raise ValueError('embedding provider key missing')
    payload = {"model": embedding_model["model"], "input": str(text), "encoding_format": "float"}
    headers = {
        "Authorization": "Bearer " + llm_settings[embedding_model["source"]]["key"],
        "Content-Type": "application/json",
    }
    provider = llm_settings[embedding_model['source']]
    url = embedding_model.get('url') or provider.get('url')
    if not url:
        raise ValueError('embedding URL missing in EMBEDDING_MODEL or provider settings')
    for attempt in range(3):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=120)
            response.raise_for_status()
            payload_data = response.json()
            data = payload_data.get('data') if isinstance(payload_data, dict) else None
            if not isinstance(data, list) or len(data) != 1 or not isinstance(data[0], dict):
                raise ValueError('embedding response requires exactly one data item')
            vector = data[0].get('embedding')
            if not isinstance(vector, list) or not vector or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in vector):
                raise ValueError('embedding must be a nonempty finite numeric vector')
            if not any(vector):
                raise ValueError('embedding vector has zero norm')
            return vector
        except (requests.ConnectionError, requests.Timeout):
            if attempt == 2:
                raise
        except requests.HTTPError as exc:
            if attempt == 2 or exc.response is None or exc.response.status_code not in {429, 500, 502, 503, 504}:
                raise
        time.sleep(2 ** attempt)


def llm_requery(llm: dict, llm_settings: dict, system_settings: str, prompt_rag: str, temperature=0.01):
    messages = [
        {"role": "system", "content": system_settings},
        {"role": "user", "content": prompt_rag},
    ]
    return single_chat(llm, llm_settings, messages, temperature=temperature)
