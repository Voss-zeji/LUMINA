from __future__ import annotations

import io
import json
import tempfile
import tomllib
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from unittest.mock import patch

import tomlkit

from lumina import wizard
from lumina.ui_support import SetupLog, redact
from test_configuration import study_files
from test_pdf_input import write_pdf


class WizardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.path = study_files(self.root)
        self.setup = self.root / 'setup'
        self.log = SetupLog(self.setup)

    def test_new_study_wizard_saves_optional_prices_and_budgets_without_api(self):
        folder = self.root / 'new-study'
        pdfs = self.root / 'source papers'
        pdfs.mkdir()
        write_pdf(pdfs / 'source.PDF')
        answers = []
        def answer(prompt):
            answers.append(prompt)
            if prompt.startswith('Configuration folder'):
                return str(folder)
            if prompt.startswith('Results root'):
                return str(self.root / 'results')
            if prompt.startswith('Folder or manifest path'):
                return str(pdfs)
            if prompt.startswith('Extraction model IDs'):
                return 'mock/First,mock/Second'
            if prompt.startswith('chat endpoint') and '[required]' in prompt:
                return 'https://provider.invalid/v1'
            if prompt.startswith('Embedding model ID'):
                return 'mock/embedding'
            if prompt.startswith('Complete embedding request URL'):
                return 'https://provider.invalid/v1/embeddings'
            if prompt.startswith('1 save / 2 validate'):
                return '1'
            return ''
        # Keep shipped templates, but isolate UI preferences from the real checkout.
        with patch.object(wizard, 'save_preferences') as prefs, patch.object(wizard, 'read_preferences', return_value={}), \
             patch('builtins.input', side_effect=answer), patch('getpass.getpass', return_value='SECRET-WIZARD-1234'), \
             patch('sys.stdin.isatty', return_value=True), \
             patch('lumina.agent.runtime.OpenAI') as api, patch('lumina.agent.runtime.requests.post') as post, \
             redirect_stdout(io.StringIO()) as output, redirect_stderr(io.StringIO()):
            code = wizard.main(['--lang', 'en', '--setup-log', str(self.setup)])
        self.assertEqual(code, 0, output.getvalue())
        api.assert_not_called()
        post.assert_not_called()
        config = wizard.load_config(str(folder / 'project.toml'))
        self.assertEqual(config.RUN['min_consensus_models'], 2)
        self.assertEqual(config.RUN['min_cross_scores'], [1])
        self.assertTrue(all(v is None for v in config.REQUEST['budget'].values()))
        data = tomllib.loads((folder / 'project.toml').read_text(encoding='utf-8'))
        self.assertEqual(data['budget'], {})
        self.assertNotIn('input_per_million', data['pricing']['mock/First'])
        self.assertTrue(any('Total call cap' in question for question in answers))
        self.assertNotIn('SECRET-WIZARD-1234', output.getvalue())
        self.assertNotIn('SECRET-WIZARD-1234', (self.setup / 'events.jsonl').read_text(encoding='utf-8'))
        prefs.assert_called_once()

    def test_edit_retains_comments_existing_values_and_hidden_credentials(self):
        self.path.write_text('# USER NOTE retain me\n' + self.path.read_text(encoding='utf-8'), encoding='utf-8')
        before = tomllib.loads(self.path.read_text(encoding='utf-8'))
        def answer(prompt):
            if prompt.startswith('1 save / 2 validate'):
                return '1'
            if prompt.startswith('Confirm writing'):
                return 'y'
            return ''
        with patch.object(wizard, 'save_preferences'), patch('builtins.input', side_effect=answer), \
             patch('getpass.getpass', return_value=''), patch('sys.stdin.isatty', return_value=True), \
             redirect_stdout(io.StringIO()) as output:
            code = wizard.main(['--config', str(self.path), '--lang', 'en', '--setup-log', str(self.setup)])
        self.assertEqual(code, 0, output.getvalue())
        after = tomllib.loads(self.path.read_text(encoding='utf-8'))
        self.assertEqual(before['budget'], after['budget'])
        self.assertEqual(before['pricing'], after['pricing'])
        self.assertEqual(before['models'], after['models'])
        self.assertEqual(before['selected_models'], after['selected_models'])
        self.assertIn('# USER NOTE retain me', self.path.read_text(encoding='utf-8'))
        self.assertNotIn('SECRET-DO-NOT-PERSIST', output.getvalue())
        self.assertNotIn('SECRET-DO-NOT-PERSIST', (self.setup / 'events.jsonl').read_text(encoding='utf-8'))

    def test_paper_directory_recursion_exclusion_and_manifest(self):
        pdfs = self.root / 'pdfs'
        (pdfs / 'sub').mkdir(parents=True)
        write_pdf(pdfs / 'one.pdf')
        write_pdf(pdfs / 'sub' / 'two.PDF')
        (pdfs / 'ignore.txt').write_text('ignore', encoding='utf-8')
        ui = wizard.Console('en', self.log)
        with patch('builtins.input', side_effect=['1', str(pdfs), 'y', '1', 'y']), redirect_stdout(io.StringIO()):
            selected = wizard.select_papers(ui, [])
        self.assertEqual(selected, [str((pdfs / 'sub' / 'two.PDF').resolve())])
        manifest = self.root / 'papers.txt'
        manifest.write_text('pdfs/one.pdf\n', encoding='utf-8')
        with patch('builtins.input', side_effect=['2', str(manifest), '', 'y']), redirect_stdout(io.StringIO()):
            self.assertEqual(wizard.select_papers(ui, []), [str((pdfs / 'one.pdf').resolve())])

    def test_interrupt_saves_draft_without_overwriting_original(self):
        draft = wizard.StudyDraft.load(self.path)
        before = self.path.read_bytes()
        ui = wizard.Console('en', self.log)
        # Redirect preferences only; templates/real core stay unchanged.
        with patch.object(wizard, 'ROOT', self.root), patch('builtins.input', side_effect=EOFError), redirect_stdout(io.StringIO()):
            self.assertEqual(wizard.configure(ui, draft), 2)
        self.assertEqual(self.path.read_bytes(), before)
        prefs = json.loads((self.root / '.lumina-ui.json').read_text(encoding='utf-8'))
        self.assertTrue(Path(prefs['last_config']).is_file())
        self.assertEqual(prefs['draft_target'], str(self.path))

    def test_editor_failure_manual_save_and_invalid_buffer_preserved(self):
        ui = wizard.Console('en', self.log)
        draft = wizard.StudyDraft.load(self.path)
        original = draft.questions['questions'][0]['prompt']
        calls = 0
        def editor(command, **kwargs):
            nonlocal calls
            calls += 1
            path = Path(command[-1])
            if calls == 1:
                path.write_text('invalid = [', encoding='utf-8')
                raise FileNotFoundError('editor not installed')
            self.assertEqual(path.read_text(encoding='utf-8'), 'invalid = [')
            fixed = tomlkit.parse(tomlkit.dumps(draft.questions))
            fixed['questions'][0]['prompt'] = 'Edited valid prompt'
            path.write_text(tomlkit.dumps(fixed), encoding='utf-8')
        with patch.dict('os.environ', {'EDITOR': 'fake-editor', 'VISUAL': ''}), \
             patch('lumina.wizard.subprocess.run', side_effect=editor), \
             patch('builtins.input', side_effect=['3', '', '3', '', '1']), redirect_stdout(io.StringIO()):
            wizard.edit_questions(ui, draft)
        self.assertNotEqual(draft.questions['questions'][0]['prompt'], original)
        self.assertEqual(draft.questions['questions'][0]['prompt'], 'Edited valid prompt')

    def test_noninteractive_never_opens_wizard_or_runs(self):
        with patch('sys.stdin.isatty', return_value=False), patch('builtins.input') as ask, redirect_stderr(io.StringIO()):
            self.assertEqual(wizard.main([]), 2)
        ask.assert_not_called()

    def test_redaction_does_not_leak_credentials(self):
        value = dict(key='abc', message='Bearer xyz https://u:p@host/path?token=def SECRET')
        encoded = json.dumps(redact(value, ['SECRET']))
        for secret in ('abc', 'xyz', 'u:p', 'def', 'SECRET'):
            self.assertNotIn(secret, encoded)
        self.log.protect('PRIVATE_VALUE')
        with patch('builtins.input', return_value='') as ask:
            value = wizard.Console('en', self.log).ask('地址', 'URL', 'https://host/PRIVATE_VALUE?token=xyz')
        self.assertNotIn('PRIVATE_VALUE', ask.call_args.args[0])
        self.assertNotIn('xyz', ask.call_args.args[0])
        self.assertEqual(value, 'https://host/PRIVATE_VALUE?token=xyz')
        credentials = {name: 'sensitive-credential' for name in
                       ('access_token', 'refresh_token', 'client_secret', 'secret_key', 'api-key')}
        self.log.record('structured_auth', nested=credentials)
        self.assertNotIn('sensitive-credential', (self.setup / 'events.jsonl').read_text(encoding='utf-8'))

    def test_setup_logs_link_without_copying_private_draft(self):
        from lumina.study import _link_setup
        secret = 'SECRET-DO-NOT-PERSIST'
        self.log.record('test', note=secret)
        (self.setup / 'draft').mkdir()
        (self.setup / 'draft' / 'secrets.local.toml').write_text(secret, encoding='utf-8')
        config = wizard.load_config(str(self.path))
        destination = self.root / 'run'
        _link_setup(destination, self.setup, config)
        files = list(destination.rglob('*'))
        self.assertFalse(any(p.name == 'secrets.local.toml' for p in files))
        linked = destination / 'reports' / 'setup' / self.setup.name / 'events.jsonl'
        self.assertNotIn(secret, linked.read_text(encoding='utf-8'))
