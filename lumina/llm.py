from __future__ import annotations

import requests
from openai import OpenAI


def single_chat(llm: dict, llm_settings: dict, message_input: list[dict], temperature=0.01):
    provider = llm_settings[llm["source"]]
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
    completion = client.chat.completions.create(**request)
    return completion.choices[0].message.content, completion.usage.total_tokens


def embedding_response(text: str, embedding_model: dict, llm_settings: dict):
    payload = {"model": embedding_model["model"], "input": str(text), "encoding_format": "float"}
    headers = {
        "Authorization": "Bearer " + llm_settings[embedding_model["source"]]["key"],
        "Content-Type": "application/json",
    }
    response = requests.post(embedding_model["url"], json=payload, headers=headers, timeout=120)
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]


def llm_requery(llm: dict, llm_settings: dict, system_settings: str, prompt_rag: str, temperature=0.01):
    messages = [
        {"role": "system", "content": system_settings},
        {"role": "user", "content": prompt_rag},
    ]
    return single_chat(llm, llm_settings, messages, temperature=temperature)
