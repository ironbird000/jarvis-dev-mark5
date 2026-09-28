import importlib.util
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts import document_ingest as real_document_ingest
from scripts import infosucker as real_infosucker

REPO_ROOT = Path(__file__).resolve().parents[1]
INTERFACE_PATH = REPO_ROOT / 'scripts' / 'interface.py'


def load_interface_module(temp_root: Path):
    config_module = types.ModuleType('config')
    config_module.HOST = '127.0.0.1'
    config_module.PORT = 5000
    config_module.WEB_DIR = REPO_ROOT / 'web'
    config_module.DATA_DIR = temp_root / 'data'
    config_module.UPLOADS_DIR = temp_root / 'uploads'
    config_module.DEFAULT_DISPLAY_NAME = 'Sir'
    config_module.render_greeting = lambda *args, **kwargs: 'Hello'
    config_module.ensure_directories = lambda: None
    config_module.BRAIN_SOCKET = temp_root / 'brain.sock'
    config_module.EARS_SOCKET = temp_root / 'ears.sock'
    config_module.EYES_SOCKET = temp_root / 'eyes.sock'
    config_module.MOUTH_SOCKET = temp_root / 'mouth.sock'
    config_module.ADMIN_EMAILS = {'user@example.com'}

    class StubAuthManager:
        def __init__(self, store=None):
            self.store = store

        def validate_session(self, token):
            if token == 'ok':
                return {'user_id': 1, 'email': 'user@example.com', 'display_name': 'User', 'session_token': token, 'settings': {}}
            return None

        def logout_session(self, token):
            return None

        def register_user(self, *args, **kwargs):
            return False, 'not implemented', None

        def login_user(self, *args, **kwargs):
            return False, 'not implemented', {}

        def update_display_name(self, *args, **kwargs):
            return None

        def update_password(self, *args, **kwargs):
            return None

        def create_reset_token(self, *args, **kwargs):
            return False, 'not implemented', None

        def use_reset_token(self, *args, **kwargs):
            return False, 'not implemented'

    auth_module = types.ModuleType('auth')
    auth_module.AuthManager = StubAuthManager

    class StubMemoryStore:
        def __init__(self, *args, **kwargs):
            pass

        def connect(self):
            raise AssertionError('connect should not be used in route payload tests')

    memory_module = types.ModuleType('memory')
    memory_module.MemoryStore = StubMemoryStore

    location_module = types.ModuleType('location_service')
    location_module.validate_home_location = lambda value: (True, value, None, None)

    flask_sock_module = types.ModuleType('flask_sock')

    class StubSock:
        def __init__(self, app):
            self.app = app

        def route(self, *args, **kwargs):
            def decorator(fn):
                return fn
            return decorator

    flask_sock_module.Sock = StubSock

    flask_module = types.ModuleType('flask')

    class StubRequest:
        def __init__(self):
            self.cookies = {}
            self.headers = {}
            self.args = {}
            self.form = {}
            self.files = {}
            self.host_url = 'http://localhost/'
            self._json = {}

        def get_json(self, force=False, silent=False):
            return self._json

    class StubFlask:
        def __init__(self, name):
            self.name = name

        def route(self, *args, **kwargs):
            def decorator(fn):
                return fn
            return decorator

    flask_module.Flask = StubFlask
    flask_module.request = StubRequest()
    flask_module.jsonify = lambda payload: payload
    flask_module.send_from_directory = lambda *args, **kwargs: {'send_from_directory': True}
    flask_module.make_response = lambda payload: payload

    werkzeug_module = types.ModuleType('werkzeug')
    werkzeug_utils_module = types.ModuleType('werkzeug.utils')
    werkzeug_utils_module.secure_filename = lambda value: value
    werkzeug_module.utils = werkzeug_utils_module

    module_name = f'test_interface_module_{temp_root.name}'
    spec = importlib.util.spec_from_file_location(module_name, INTERFACE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    with unittest.mock.patch.dict(sys.modules, {
        'config': config_module,
        'auth': auth_module,
        'memory': memory_module,
        'location_service': location_module,
        'flask_sock': flask_sock_module,
        'flask': flask_module,
        'werkzeug': werkzeug_module,
        'werkzeug.utils': werkzeug_utils_module,
        'document_ingest': real_document_ingest,
        'infosucker': real_infosucker,
    }):
        spec.loader.exec_module(module)
    return module


class InterfaceInfoSuckerRouteTests(unittest.TestCase):
    def test_infosucker_routes_expose_structured_job_payload(self):
        with TemporaryDirectory() as tmp:
            module = load_interface_module(Path(tmp))
            expected_job = {
                'job_id': 'job-123',
                'source_url': 'https://example.com/index/',
                'scope': 'personal',
                'phase': 'validating',
                'status': 'running',
                'created_at': '2026-09-28T00:00:00Z',
                'updated_at': '2026-09-28T00:00:10Z',
                'started_at': '2026-09-28T00:00:02Z',
                'completed_at': None,
                'current_batch_number': 2,
                'batch_count': 3,
                'discovered_items': [{'file_id': 'a'}],
                'batch_plan': [{'batch_number': 1}, {'batch_number': 2}, {'batch_number': 3}],
                'completed_batches': [1],
                'per_file_status': [{'file_id': 'a', 'failure_reason': None}],
                'summary': {'discovered': 3, 'downloaded': 2, 'imported': 1, 'validated': 1, 'skipped': 0, 'failed': 0},
                'failure_details': [],
                'validation': {'2': {'failure_rate': 0.0}},
                'total_estimated_bytes': 42,
            }

            class StubManager:
                def start_job(self, user, source_url, scope, download_root):
                    return expected_job

                def get_latest_job(self, user):
                    return expected_job

                def get_job(self, user, job_id):
                    return expected_job if job_id == 'job-123' else None

                def resume_job(self, user, job_id, download_root):
                    return expected_job

            module.infosucker_manager = StubManager()
            module.request.cookies = {'jarvis_mark3_session': 'ok'}
            module.request.headers = {'Origin': 'http://localhost'}
            module.request._json = {'source_url': 'https://example.com/index/'}

            start_payload, status_code = module.route_infosucker_start()
            self.assertEqual(status_code, 202)
            self.assertTrue(start_payload['success'])
            self.assertEqual(start_payload['job']['batch_count'], 3)
            self.assertIn('summary', start_payload['job'])
            self.assertIn('discovered_items', start_payload['job'])

            latest_payload = module.route_infosucker_latest()
            self.assertEqual(latest_payload['job']['current_batch_number'], 2)
            self.assertEqual(latest_payload['job']['total_estimated_bytes'], 42)

            job_payload = module.route_infosucker_job('job-123')
            self.assertEqual(job_payload['job']['validation']['2']['failure_rate'], 0.0)

            resume_payload = module.route_infosucker_resume('job-123')
            self.assertEqual(resume_payload['job']['status'], 'running')


if __name__ == '__main__':
    unittest.main()
