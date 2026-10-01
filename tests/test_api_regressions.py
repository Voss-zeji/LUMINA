from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import requests

from lumina import llm, prompts


class APIRegressions(unittest.TestCase):
    model = {'model': 'mock-model', 'source': 'mock', 'url': 'https://example.test/embeddings'}
    settings = {'mock': {'key': 'synthetic', 'url': 'https://example.test', 'supports_json_mode': False}}

    def client(self, content='{}', usage=None, choices=True):
        completion = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))] if choices else [], usage=usage)
        client = Mock()
        client.chat.completions.create.return_value = completion
        return client

    def test_optional_usage_keeps_valid_content_and_unknown_usage(self):
        client = self.client()
        with patch.object(llm, 'OpenAI', return_value=client):
            self.assertEqual(llm.single_chat(self.model, self.settings, []), ('{}', None))
        client.close.assert_called_once()

    def test_blank_provider_url_never_falls_back_to_another_endpoint(self):
        with patch.object(llm, 'OpenAI') as sdk:
            with self.assertRaisesRegex(ValueError, 'explicit'):
                llm.single_chat(self.model, {'mock':{'key':'synthetic','url':''}}, [])
        sdk.assert_not_called()

    def test_null_content_and_empty_choices_are_explicit_errors(self):
        for client in [self.client(content=None), self.client(choices=False)]:
            with patch.object(llm, 'OpenAI', return_value=client):
                with self.assertRaises(ValueError):
                    llm.single_chat(self.model, self.settings, [])
            client.close.assert_called_once()

    def test_client_closes_when_sdk_raises(self):
        client = self.client(); client.chat.completions.create.side_effect = RuntimeError('failure')
        with patch.object(llm, 'OpenAI', return_value=client):
            with self.assertRaises(RuntimeError):
                llm.single_chat(self.model, self.settings, [])
        client.close.assert_called_once()

    def response(self, vector=None):
        response = Mock()
        response.json.return_value = {'data': [{'embedding': [1., 0.] if vector is None else vector}]}
        return response

    def test_transient_embeddings_retry_is_bounded(self):
        http = Mock(status_code=503)
        bad = self.response(); bad.raise_for_status.side_effect = requests.HTTPError('503', response=http)
        with patch.object(llm.requests, 'post', side_effect=[bad, bad, self.response()]) as post, patch.object(llm.time, 'sleep'):
            self.assertEqual(llm.embedding_response('text', self.model, self.settings), [1., 0.])
            self.assertEqual(post.call_count, 3)
        with patch.object(llm.requests, 'post', return_value=bad) as post, patch.object(llm.time, 'sleep'):
            with self.assertRaises(requests.HTTPError):
                llm.embedding_response('text', self.model, self.settings)
            self.assertEqual(post.call_count, 3)

    def test_nonretryable_error_is_not_retried(self):
        bad = self.response(); bad.raise_for_status.side_effect = requests.HTTPError('400', response=Mock(status_code=400))
        with patch.object(llm.requests, 'post', return_value=bad) as post:
            with self.assertRaises(requests.HTTPError):
                llm.embedding_response('text', self.model, self.settings)
            self.assertEqual(post.call_count, 1)

    def test_malformed_or_nonfinite_embedding_is_rejected(self):
        responses = [self.response([]), self.response([float('nan')]), self.response([0., 0.]), self.response(['number'])]
        empty = self.response(); empty.json.return_value = {'data': []}; responses.append(empty)
        for response in responses:
            with patch.object(llm.requests, 'post', return_value=response):
                with self.assertRaises(ValueError):
                    llm.embedding_response('text', self.model, self.settings)

    def test_embedding_provider_url_fallback(self):
        with patch.object(llm.requests, 'post', return_value=self.response()) as post:
            llm.embedding_response('text', {**self.model, 'url': ''}, self.settings)
        self.assertEqual(post.call_args.args[0], 'https://example.test')

    def test_active_prompt_examples_are_valid_json(self):
        import json
        for domain in ['aqua', 'wildfire']:
            for question in prompts.questions_for_domain(domain):
                example = question[question.index('{'):question.rfind('}')+1]
                self.assertIsInstance(json.loads(example), dict)
        checker = prompts.checker_requery.format(context='body', answer='evidence', key_topic='topic')
        self.assertEqual(json.loads(checker[checker.index('{'):checker.rfind('}')+1]),
                         {'existing_flag':0, 'direct_quote':None})


if __name__ == '__main__':
    unittest.main()
