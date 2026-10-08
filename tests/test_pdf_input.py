from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from lumina import preparation
from lumina.agent.contracts import ResearchSpecification
from lumina.agent.controller import approve, begin, execute
from lumina.agent.store import Store
from test_agent_contracts import config_for, request_for
import test_agent_controller as controller_fixtures


def write_pdf(path: Path, *, empty_page=False):
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject({NameObject('/Type'): NameObject('/Font'), NameObject('/Subtype'): NameObject('/Type1'),
                             NameObject('/BaseFont'): NameObject('/Helvetica')})
    page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'): DictionaryObject({NameObject('/F1'): font})})
    stream = DecodedStreamObject()
    stream.set_data(b'BT /F1 12 Tf 72 720 Td (China. Fish. Flux is 2. Table 2.) Tj ET')
    page[NameObject('/Contents')] = stream
    if empty_page:
        writer.add_blank_page(width=612, height=792)
    writer.write(path)


class PdfInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.pdfs, self.mds = self.root / 'pdfs', self.root / 'mds'
        self.pdfs.mkdir()
        self.paper = self.pdfs / '文章 01.pdf'
        write_pdf(self.paper)

    def test_missing_marker_uses_real_pypdf_and_records_fallback(self):
        records = {}
        with patch.object(preparation, '_marker_converter', side_effect=ImportError('not installed')), \
             self.assertWarnsRegex(RuntimeWarning, 'pypdf'):
            files = preparation.convert_pdfs_to_markdown(self.pdfs, self.mds, records=records)
        self.assertIn('Flux is 2', Path(files[0]).read_text(encoding='utf-8'))
        self.assertEqual(records[self.paper.stem]['backend'], 'pypdf')
        self.assertTrue(records[self.paper.stem]['fallback_used'])
        self.assertTrue(records[self.paper.stem]['version'])

    def test_marker_runtime_failure_and_empty_result_use_fallback(self):
        for result in (RuntimeError('model unavailable'), ''):
            with self.subTest(result=result):
                convert = (lambda _path: '') if result == '' else (lambda _path: (_ for _ in ()).throw(result))
                with patch.object(preparation, '_marker_converter', return_value=convert), self.assertWarns(RuntimeWarning):
                    files = preparation.convert_pdfs_to_markdown(self.pdfs, self.mds)
                self.assertIn('China', Path(files[0]).read_text(encoding='utf-8'))
                Path(files[0]).unlink()

    def test_marker_success_reuses_model_within_preparation_pass(self):
        cache, records = {}, {}
        with patch.object(preparation, '_marker_converter', return_value=lambda _p: '# Marker body') as factory, \
             patch.object(preparation, '_pypdf_text') as fallback:
            preparation.convert_pdfs_to_markdown(self.pdfs, self.mds, cache=cache, records=records)
            write_pdf(self.pdfs / 'second.pdf')
            preparation.convert_pdfs_to_markdown(self.pdfs, self.mds, cache=cache, records=records)
        factory.assert_called_once()
        fallback.assert_not_called()
        self.assertEqual(records[self.paper.stem]['backend'], 'marker')
        self.assertFalse(records[self.paper.stem]['fallback_used'])

    def test_fallback_refuses_to_silently_omit_a_textless_page(self):
        write_pdf(self.paper, empty_page=True)
        with self.assertRaisesRegex(RuntimeError, 'page 2.*OCR'):
            preparation.convert_pdfs_to_markdown(self.pdfs, self.mds, pdf_config={'backend': 'pypdf'})
        self.assertFalse((self.mds / (self.paper.stem + '.md')).exists())

    def test_pdf_config_is_validated_and_frozen(self):
        config = config_for()
        first = ResearchSpecification.from_config(request_for(self.paper), config)
        config.DOMAINS['aqua']['pdf'] = dict(backend='marker', fallback='none')
        second = ResearchSpecification.from_config(request_for(self.paper), config)
        self.assertNotEqual(first.scientific_hash, second.scientific_hash)
        for raw in ({'backend': 'shell'}, {'fallback': 'run.exe'}, {'backend': 'pypdf', 'fallback': 'pypdf'}, {'command': 'x'}):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                preparation.pdf_options(raw)

    def test_real_pdf_to_trial_batch_and_traceable_markdown(self):
        config = config_for()
        config.DOMAINS['aqua']['pdf'] = dict(backend='marker', fallback='pypdf')
        request = request_for(self.paper)
        request['allow_pdf_resources'] = True
        fixtures = controller_fixtures.ControllerTests()
        client = fixtures.client('aqua')
        with patch.object(preparation, '_marker_converter', side_effect=ImportError('not installed')), \
             patch('lumina.agent.runtime.OpenAI', return_value=client), \
             patch('lumina.agent.runtime.requests.post', return_value=fixtures.embedding()), \
             patch('lumina.agent.runtime.RunContext.sleep'):
            run = begin(request, config, self.root / 'runs', dry_run=True)
            with self.assertWarns(RuntimeWarning):
                result = execute(run, config)
            self.assertEqual(result['state'], 'HUMAN_GATE_SMOKE', result)
            report = json.loads((run / 'reports' / 'smoke_report.json').read_text(encoding='utf-8'))
            self.assertEqual(report['preparation'][0]['converter']['backend'], 'pypdf')
            with Store(run) as store:
                gate = next(g for g in store.gates() if g['kind'] == 'smoke')
            approve(run, gate['gate_id'], 'Reviewed synthetic PDF conversion and trial')
            calls = client.chat.completions.create.call_count
            self.assertEqual(execute(run, config)['state'], 'DONE')
            self.assertEqual(client.chat.completions.create.call_count, calls)
        self.assertEqual(len(list((run / 'inputs').glob('*.pdf'))), 1)
        self.assertEqual(len(list((run / 'outputs' / 'prepared').glob('*.md'))), 1)

    def test_pypdf_needs_no_marker_resource_permission(self):
        config = config_for()
        config.DOMAINS['aqua']['pdf'] = dict(backend='pypdf', fallback='none')
        # Prepare only: API credentials are unused and the OCR package is never imported.
        config.PROJECT = {'stage': 'prepare'}
        run = begin(request_for(self.paper), copy.deepcopy(config), self.root / 'runs', dry_run=True)
        with patch.object(preparation, '_marker_converter') as marker, self.assertWarns(RuntimeWarning):
            self.assertEqual(execute(run, config)['state'], 'PAUSED')
        marker.assert_not_called()
