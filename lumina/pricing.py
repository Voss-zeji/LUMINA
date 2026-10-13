"""Import user-selected public quotes; never infer a proxy's price from a model name."""
from __future__ import annotations

import json
import math
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

MAX_QUOTE_BYTES = 1024 * 1024


def quote_endpoint(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError('quote base_url must be a URL')
    parsed = urlsplit(value)
    if parsed.scheme not in {'http', 'https'} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError('quote URL must be HTTP(S), without credentials, query or fragment')
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path.rstrip('/'), '', ''))


def read_quotes(source: str, expected: dict[str, str]) -> tuple[dict, list[str]]:
    """Return matched rates and diagnostics. Caller must preview and confirm before saving."""
    if source.startswith(('http://', 'https://')):
        quote_endpoint(source)
        with urlopen(Request(source, headers={'Accept': 'application/json'}), timeout=10) as response:
            # Do not log a redirected URL that might contain credentials.
            quote_endpoint(response.geturl())
            raw = response.read(MAX_QUOTE_BYTES + 1)
        origin = source
    else:
        file = Path(source).expanduser().resolve(strict=True)
        with file.open('rb') as stream:
            raw = stream.read(MAX_QUOTE_BYTES + 1)
        origin = str(file)
    if len(raw) > MAX_QUOTE_BYTES:
        raise ValueError('quote document exceeds 1 MiB')
    data = json.loads(raw.decode('utf-8-sig'))
    if not isinstance(data, dict) or not isinstance(data.get('quotes'), list):
        raise ValueError('quote document requires a quotes array')
    targets = {model: quote_endpoint(url) for model, url in expected.items()}
    matched, notes, seen = {}, [], set()
    fetched_at = datetime.now(timezone.utc).isoformat()
    for index, row in enumerate(data['quotes'], 1):
        try:
            if not isinstance(row, dict):
                raise ValueError('row must be an object')
            model = row.get('model')
            endpoint = quote_endpoint(row.get('base_url'))
            if not isinstance(model, str) or model not in targets or endpoint != targets[model]:
                notes.append(f'Row {index}: endpoint/model not selected; skipped')
                continue
            if model in seen:
                matched.pop(model, None)
                notes.append(f'Row {index}: duplicate quote for {model}; manual resolution required')
                continue
            seen.add(model)
            currency = row.get('currency')
            if not isinstance(currency, str) or len(currency) != 3 or not currency.isascii() or not currency.isalpha():
                raise ValueError('currency must be a three-letter code')
            as_of = row.get('as_of')
            if not isinstance(as_of, str):
                raise ValueError('as_of must be YYYY-MM-DD')
            date.fromisoformat(as_of)
            rates = {}
            for field in ('input_per_million', 'output_per_million'):
                rate = row.get(field)
                if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate < 0:
                    raise ValueError('rates must be finite nonnegative numbers')
                rates[field] = float(rate)
            matched[model] = dict(rates, base_url=endpoint, currency=currency.upper(), as_of=as_of,
                                  source=origin, fetched_at=fetched_at,
                                  basis=f'Imported quote: {endpoint}; effective {as_of}')
        except (ValueError, TypeError):
            notes.append(f'Row {index}: invalid quote fields; skipped')
    currencies = {row['currency'] for row in matched.values()}
    if len(currencies) > 1:
        raise ValueError('matched quotes use different currencies; import comparable quotes or leave prices unknown')
    return matched, notes
