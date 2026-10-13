from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from lumina.pricing import MAX_QUOTE_BYTES, read_quotes


class QuoteTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'rates.json'
        self.row = dict(base_url='https://provider.invalid/v1', model='full/model', currency='USD',
                        input_per_million=1.2, output_per_million=3.4, as_of='2026-10-13')
        self.expected = {'full/model': self.row['base_url']}

    def load(self, rows):
        self.path.write_text(json.dumps({'quotes': rows}), encoding='utf-8')
        return read_quotes(str(self.path), self.expected)

    def test_exact_provider_and_model_match_preserves_provenance(self):
        rates, notes = self.load([self.row, dict(self.row, base_url='https://proxy.invalid/v1')])
        self.assertEqual(rates['full/model']['input_per_million'], 1.2)
        self.assertEqual(rates['full/model']['source'], str(self.path.resolve()))
        self.assertTrue(rates['full/model']['fetched_at'])
        self.assertEqual(len(notes), 1)

    def test_bad_missing_and_duplicate_rates_never_become_zero(self):
        for invalid in (None, -1, True, float('inf')):
            with self.subTest(invalid=invalid):
                rates, notes = self.load([dict(self.row, input_per_million=invalid)])
                self.assertEqual(rates, {})
                self.assertTrue(notes)
        rates, notes = self.load([self.row, self.row, self.row])
        self.assertEqual(rates, {})
        self.assertEqual(len(notes), 2)

    def test_public_url_timeout_size_and_credentials(self):
        response = MagicMock()
        response.geturl.return_value = 'https://provider.invalid/prices.json'
        response.read.return_value = json.dumps({'quotes': [self.row]}).encode()
        response.__enter__.return_value = response
        with patch('lumina.pricing.urlopen', return_value=response) as fetch:
            rates, _ = read_quotes(response.geturl(), self.expected)
            self.assertEqual(rates['full/model']['currency'], 'USD')
            self.assertEqual(fetch.call_args.kwargs['timeout'], 10)
            response.read.assert_called_once_with(MAX_QUOTE_BYTES + 1)
            with self.assertRaises(ValueError):
                read_quotes('https://user:secret@provider.invalid/prices', self.expected)
            self.assertEqual(fetch.call_count, 1)
        self.path.write_bytes(b'x' * (MAX_QUOTE_BYTES + 1))
        with self.assertRaisesRegex(ValueError, '1 MiB'):
            read_quotes(str(self.path), self.expected)

    def test_mixed_currency_matching_quotes_are_not_added_together(self):
        self.expected['other'] = self.row['base_url']
        with self.assertRaisesRegex(ValueError, 'different currencies'):
            self.load([self.row, dict(self.row, model='other', currency='CNY')])
