"""Small, dependency-free helpers for terminal preferences and redacted setup logs."""
from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def redact(value, secrets=()):
    if isinstance(value, dict):
        return {k: '[REDACTED]' if str(k).lower().replace('-', '_') in {
                    'key', 'api_key', 'password', 'authorization', 'token', 'access_token',
                    'refresh_token', 'client_secret', 'secret', 'secret_key'}
                else redact(v, secrets) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [redact(v, secrets) for v in value]
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, '[REDACTED]')
        value = re.sub(r'(https?://)[^/\s@]+@', r'\1[REDACTED]@', value)
        value = re.sub(r'(?i)(bearer\s+)[^\s\"\']+', r'\1[REDACTED]', value)
        value = re.sub(r'(?i)([?&](?:key|api[-_]?key|token|(?:access|refresh)[-_]?token|password|(?:client[-_]?)?secret|secret[-_]?key)=)[^&#\s]+', r'\1[REDACTED]', value)
        value = re.sub(r'(?i)((?:api[-_]?key|(?:access|refresh)[-_]?token|password|(?:client[-_]?)?secret|secret[-_]?key)\s*[:=]\s*)[^\s,;]+', r'\1[REDACTED]', value)
    return value


def atomic_text(path: Path, text: str, *, private=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as stream:
        temp = Path(stream.name)
        stream.write(text)
    try:
        if private:
            os.chmod(temp, 0o600)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


class SetupLog:
    def __init__(self, directory: Path):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.secrets: list[str] = []

    def record(self, event: str, **fields):
        data = {'at': datetime.now(timezone.utc).isoformat(), 'event': event, **fields}
        with (self.directory / 'events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(redact(data, self.secrets), ensure_ascii=False, default=str) + '\n')

    def protect(self, value: str):
        if value and value not in self.secrets:
            self.secrets.append(value)


def read_preferences(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_preferences(path: Path, language: str, config: Path):
    atomic_text(path, json.dumps(dict(language=language, last_config=str(config.resolve())), ensure_ascii=False), private=True)
