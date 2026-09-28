import json
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from scripts import document_ingest
from scripts import infosucker


class FakeResponse:
    def __init__(self, text='', headers=None, status_code=200, history=None):
        self.text = text
        self.headers = headers or {}
        self.status_code = status_code
        self.history = history or []
        self.url = self.headers.get('X-Url', 'https://example.com/resource')

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size=1024):
        yield self.text.encode('utf-8')

    def close(self):
        return None


class FakeEntry:
    def __init__(self, namespace='C', url='home.html', full_url='Chome.html', title='Title', mimetype='text/html', body='<h1>Hello</h1>', is_redirect=False):
        self.namespace = namespace
        self.url = url
        self.full_url = full_url
        self.title = title
        self.mimetype = mimetype
        self._body = body
        self.is_redirect = is_redirect

    def read(self):
        return self._body.encode('utf-8')


class FakeArchive:
    def __init__(self, entries, metadata=None):
        self._entries = entries
        self._metadata = metadata or {'Language': 'eng'}

    def get_metadata_dict(self, as_unicode=True):
        return self._metadata

    def iter_articles(self):
        return iter(self._entries)

    def close(self):
        return None


class InfoSuckerTests(unittest.TestCase):
    def test_discover_direct_zim_url(self):
        limits = infosucker.BatchLimits(
            max_files_per_batch=2,
            max_total_bytes_per_batch=100,
            max_zim_files_per_batch=1,
            max_total_discovered_files=5,
            validation_failure_threshold=0.5,
            zim_max_records=None,
            zim_stop_after_records=None,
            progress_every_records=100,
        )
        with mock.patch.object(infosucker, 'validate_safe_url', side_effect=lambda url: url), \
             mock.patch.object(infosucker, '_head_metadata', return_value={'size_bytes': 12, 'last_modified': 'yesterday', 'content_type': 'application/octet-stream'}):
            result = infosucker.discover_source('https://example.com/wiki-mini.zim', limits)
        self.assertEqual(result['source_url'], 'https://example.com/wiki-mini.zim')
        self.assertEqual(len(result['discovered_items']), 1)
        item = result['discovered_items'][0]
        self.assertEqual(item['file_name'], 'wiki-mini.zim')
        self.assertEqual(item['file_type'], '.zim')
        self.assertTrue(item['directly_downloadable'])
        self.assertEqual(item['size_bytes_estimate'], 12)

    def test_discover_html_index_collects_multiple_candidates(self):
        limits = infosucker.BatchLimits(
            max_files_per_batch=2,
            max_total_bytes_per_batch=100,
            max_zim_files_per_batch=1,
            max_total_discovered_files=5,
            validation_failure_threshold=0.5,
            zim_max_records=None,
            zim_stop_after_records=None,
            progress_every_records=100,
        )
        html = '<html><body><a href="mini.zim">mini</a><a href="notes.txt">notes</a><a href="ignore.exe">ignore</a></body></html>'
        with mock.patch.object(infosucker, 'validate_safe_url', side_effect=lambda url: url), \
             mock.patch.object(infosucker, '_safe_request', return_value=FakeResponse(text=html, headers={'Content-Type': 'text/html'})), \
             mock.patch.object(infosucker, '_head_metadata', side_effect=lambda url: {'size_bytes': 7 if url.endswith('.txt') else 22, 'last_modified': None, 'content_type': 'text/plain'}):
            result = infosucker.discover_source('https://example.com/index/', limits)
        self.assertEqual(len(result['discovered_items']), 2)
        self.assertEqual([item['file_name'] for item in result['discovered_items']], ['mini.zim', 'notes.txt'])
        self.assertEqual(result['skipped_items'][0]['skip_reason'], 'unsupported_extension')

    def test_plan_batches_is_size_aware_and_prioritizes_small_zims(self):
        limits = infosucker.BatchLimits(
            max_files_per_batch=2,
            max_total_bytes_per_batch=30,
            max_zim_files_per_batch=1,
            max_total_discovered_files=10,
            validation_failure_threshold=0.5,
            zim_max_records=None,
            zim_stop_after_records=None,
            progress_every_records=100,
        )
        items = [
            {'file_id': '3', 'file_name': 'wiki-maxi.zim', 'source_url': 'https://e/maxi', 'file_type': '.zim', 'size_bytes_estimate': 20},
            {'file_id': '1', 'file_name': 'wiki-mini.zim', 'source_url': 'https://e/mini', 'file_type': '.zim', 'size_bytes_estimate': 5},
            {'file_id': '2', 'file_name': 'notes.txt', 'source_url': 'https://e/notes', 'file_type': '.txt', 'size_bytes_estimate': 4},
        ]
        batches = infosucker.plan_batches(items, limits)
        self.assertEqual(len(batches), 2)
        self.assertEqual([f['file_name'] for f in batches[0]['files']], ['notes.txt', 'wiki-mini.zim'])
        self.assertEqual([f['file_name'] for f in batches[1]['files']], ['wiki-maxi.zim'])

    def test_validation_failure_pauses_before_next_batch(self):
        limits = infosucker.BatchLimits(
            max_files_per_batch=1,
            max_total_bytes_per_batch=100,
            max_zim_files_per_batch=1,
            max_total_discovered_files=10,
            validation_failure_threshold=0.4,
            zim_max_records=None,
            zim_stop_after_records=None,
            progress_every_records=100,
        )
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            called = []

            def fake_download(item, destination_dir):
                called.append(('download', item['file_name']))
                destination = destination_dir / item['file_name']
                destination.write_text('content', encoding='utf-8')
                return {'stored_path': str(destination), 'size_bytes': destination.stat().st_size, 'mime_type': 'text/plain'}

            def fake_import(user, file_state, scope):
                called.append(('import', file_state['file_name']))
                return {
                    'import_attempted': True,
                    'extracted_record_count': 0,
                    'indexed_document_count': 0,
                    'chunk_count': 0,
                    'embedding_success': False,
                    'failure_reason': 'Import yielded zero extracted records.',
                }

            manager = infosucker.InfoSuckerJobManager(tmp_path / 'jobs', downloader=fake_download, importer=fake_import, limits=limits)
            state = manager._initial_state({'user_id': 7}, 'https://example.com/source', 'personal')
            manager._save_state(state)
            discovered = {
                'source_url': 'https://example.com/source',
                'discovered_items': [
                    {'file_id': 'a', 'file_name': 'first.txt', 'source_url': 'https://example.com/first.txt', 'file_type': '.txt', 'size_bytes_estimate': 10},
                    {'file_id': 'b', 'file_name': 'second.txt', 'source_url': 'https://example.com/second.txt', 'file_type': '.txt', 'size_bytes_estimate': 10},
                ],
                'skipped_items': [],
                'total_estimated_bytes': 20,
            }
            with mock.patch.object(infosucker, 'discover_source', return_value=discovered):
                manager._run_job(state['job_id'], {'user_id': 7}, 'personal', tmp_path / 'downloads', False)
            job = manager.get_job({'user_id': 7}, state['job_id'])
            self.assertEqual(job['status'], 'paused')
            self.assertEqual(called, [('download', 'first.txt'), ('import', 'first.txt')])
            self.assertEqual(job['current_batch_number'], 1)
            self.assertEqual(job['batch_count'], 2)
            self.assertEqual(job['completed_batches'], [1])
            self.assertEqual(job['per_file_status'][0]['failure_reason'], 'Import yielded zero extracted records.')

    def test_resume_continues_from_next_incomplete_batch(self):
        limits = infosucker.BatchLimits(
            max_files_per_batch=1,
            max_total_bytes_per_batch=100,
            max_zim_files_per_batch=1,
            max_total_discovered_files=10,
            validation_failure_threshold=0.4,
            zim_max_records=None,
            zim_stop_after_records=None,
            progress_every_records=100,
        )
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            importer_calls = []

            def fake_download(item, destination_dir):
                destination = destination_dir / item['file_name']
                destination.write_text('content', encoding='utf-8')
                return {'stored_path': str(destination), 'size_bytes': destination.stat().st_size, 'mime_type': 'text/plain'}

            importer_state = {'second_should_fail': True}

            def fake_import(user, file_state, scope):
                importer_calls.append(file_state['file_name'])
                if file_state['file_name'] == 'second.txt' and importer_state['second_should_fail']:
                    return {
                        'import_attempted': True,
                        'extracted_record_count': 0,
                        'indexed_document_count': 0,
                        'chunk_count': 0,
                        'embedding_success': False,
                        'failure_reason': 'Import yielded zero extracted records.',
                    }
                return {
                    'import_attempted': True,
                    'extracted_record_count': 1,
                    'indexed_document_count': 1,
                    'chunk_count': 1,
                    'embedding_success': True,
                    'failure_reason': None,
                }

            manager = infosucker.InfoSuckerJobManager(tmp_path / 'jobs', downloader=fake_download, importer=fake_import, limits=limits)
            state = manager._initial_state({'user_id': 9}, 'https://example.com/source', 'personal')
            manager._save_state(state)
            discovered = {
                'source_url': 'https://example.com/source',
                'discovered_items': [
                    {'file_id': 'a', 'file_name': 'first.txt', 'source_url': 'https://example.com/first.txt', 'file_type': '.txt', 'size_bytes_estimate': 10},
                    {'file_id': 'b', 'file_name': 'second.txt', 'source_url': 'https://example.com/second.txt', 'file_type': '.txt', 'size_bytes_estimate': 10},
                ],
                'skipped_items': [],
                'total_estimated_bytes': 20,
            }
            with mock.patch.object(infosucker, 'discover_source', return_value=discovered):
                manager._run_job(state['job_id'], {'user_id': 9}, 'personal', tmp_path / 'downloads', False)
            importer_state['second_should_fail'] = False
            manager._run_job(state['job_id'], {'user_id': 9}, 'personal', tmp_path / 'downloads', True)
            job = manager.get_job({'user_id': 9}, state['job_id'])
            self.assertEqual(job['status'], 'completed')
            self.assertEqual(importer_calls, ['first.txt', 'second.txt'])
            self.assertEqual(job['completed_batches'], [1, 2])
            self.assertEqual(job['summary']['validated'], 1)

    def test_small_zim_import_through_batch_flow_and_zero_record_detection(self):
        limits = infosucker.BatchLimits(
            max_files_per_batch=1,
            max_total_bytes_per_batch=100,
            max_zim_files_per_batch=1,
            max_total_discovered_files=10,
            validation_failure_threshold=1.0,
            zim_max_records=1,
            zim_stop_after_records=1,
            progress_every_records=1,
        )
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            def fake_download(item, destination_dir):
                destination = destination_dir / item['file_name']
                destination.write_bytes(b'fake-zim')
                return {'stored_path': str(destination), 'size_bytes': destination.stat().st_size, 'mime_type': 'application/octet-stream'}

            def fake_import(user, file_state, scope):
                records = list(document_ingest.iter_ingest_records(file_state['stored_path'], max_records=1, stop_after=1, progress_every=1))
                return {
                    'import_attempted': True,
                    'extracted_record_count': len(records),
                    'indexed_document_count': len(records),
                    'chunk_count': len(records),
                    'embedding_success': bool(records),
                    'failure_reason': None if records else 'Import yielded zero extracted records.',
                }

            archive_module = types.ModuleType('pyzim.archive')
            archive_module.Zim = type('FakeZim', (), {'open': staticmethod(lambda path, mode='r': FakeArchive([FakeEntry(title='Mini title')]))})
            pyzim_module = types.ModuleType('pyzim')
            pyzim_module.archive = archive_module
            manager = infosucker.InfoSuckerJobManager(tmp_path / 'jobs', downloader=fake_download, importer=fake_import, limits=limits)
            state = manager._initial_state({'user_id': 12}, 'https://example.com/source', 'personal')
            manager._save_state(state)
            discovered = {
                'source_url': 'https://example.com/source',
                'discovered_items': [
                    {'file_id': 'zim1', 'file_name': 'sample-mini.zim', 'source_url': 'https://example.com/sample-mini.zim', 'file_type': '.zim', 'size_bytes_estimate': 10},
                ],
                'skipped_items': [],
                'total_estimated_bytes': 10,
            }
            with mock.patch.dict(sys.modules, {'pyzim': pyzim_module, 'pyzim.archive': archive_module}), \
                 mock.patch.object(infosucker, 'discover_source', return_value=discovered):
                manager._run_job(state['job_id'], {'user_id': 12}, 'personal', tmp_path / 'downloads', False)
            job = manager.get_job({'user_id': 12}, state['job_id'])
            self.assertEqual(job['status'], 'completed')
            self.assertEqual(job['summary']['extracted_records'], 1)
            self.assertEqual(job['summary']['imported'], 1)
            self.assertEqual(job['per_file_status'][0]['status'], 'validated')

    def test_iter_ingest_records_accepts_non_a_canonical_zim_uri(self):
        archive_module = types.ModuleType('pyzim.archive')
        archive_module.Zim = type('FakeZim', (), {'open': staticmethod(lambda path, mode='r': FakeArchive([FakeEntry(full_url='Chome.html', body='<h1>Hello</h1><p>World</p>')]))})
        pyzim_module = types.ModuleType('pyzim')
        pyzim_module.archive = archive_module
        with mock.patch.dict(sys.modules, {'pyzim': pyzim_module, 'pyzim.archive': archive_module}):
            records = list(document_ingest.iter_ingest_records('/tmp/sample.zim', max_records=1))
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['source_metadata']['canonical_uri'], 'Chome.html')
        self.assertIn('Hello World', records[0]['text'])

    def test_iter_ingest_records_reports_zero_records_for_empty_zim(self):
        archive_module = types.ModuleType('pyzim.archive')
        archive_module.Zim = type('FakeZim', (), {'open': staticmethod(lambda path, mode='r': FakeArchive([]))})
        pyzim_module = types.ModuleType('pyzim')
        pyzim_module.archive = archive_module
        with mock.patch.dict(sys.modules, {'pyzim': pyzim_module, 'pyzim.archive': archive_module}):
            records = list(document_ingest.iter_ingest_records('/tmp/empty.zim'))
        self.assertEqual(records, [])


if __name__ == '__main__':
    unittest.main()
