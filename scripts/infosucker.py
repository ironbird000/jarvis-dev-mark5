import json
import logging
import mimetypes
import os
import socket
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional
from urllib.parse import urljoin, urlparse
import ipaddress

import requests

LOGGER = logging.getLogger(__name__)
DEFAULT_CANDIDATE_EXTENSIONS = {
    '.txt', '.md', '.rtf', '.csv', '.tsv', '.json', '.xml', '.yaml', '.yml',
    '.html', '.htm', '.pdf', '.doc', '.docx', '.odt', '.xls', '.xlsx', '.ods',
    '.ppt', '.pptx', '.odp', '.zip', '.zim'
}
DEFAULT_TIMEOUT_SECONDS = 30


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def _env_int(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, '').strip())
        return value if value > 0 else default
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, '').strip())
        return value if value >= 0 else default
    except Exception:
        return default


@dataclass(frozen=True)
class BatchLimits:
    max_files_per_batch: int
    max_total_bytes_per_batch: int
    max_zim_files_per_batch: int
    max_total_discovered_files: int
    validation_failure_threshold: float
    zim_max_records: Optional[int]
    zim_stop_after_records: Optional[int]
    progress_every_records: int

    @classmethod
    def from_env(cls) -> 'BatchLimits':
        return cls(
            max_files_per_batch=_env_int('INFOSUCKER_MAX_FILES_PER_BATCH', 3),
            max_total_bytes_per_batch=_env_int('INFOSUCKER_MAX_TOTAL_BYTES_PER_BATCH', 200 * 1024 * 1024),
            max_zim_files_per_batch=_env_int('INFOSUCKER_MAX_ZIM_FILES_PER_BATCH', 1),
            max_total_discovered_files=_env_int('INFOSUCKER_MAX_TOTAL_DISCOVERED_FILES', 25),
            validation_failure_threshold=_env_float('INFOSUCKER_VALIDATION_FAILURE_THRESHOLD', 0.5),
            zim_max_records=_optional_env_int('INFOSUCKER_MAX_RECORDS_PER_ZIM'),
            zim_stop_after_records=_optional_env_int('INFOSUCKER_STOP_AFTER_N_RECORDS'),
            progress_every_records=_env_int('INFOSUCKER_PROGRESS_EVERY_RECORDS', 250),
        )


def _optional_env_int(name: str) -> Optional[int]:
    raw = (os.environ.get(name) or '').strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except Exception:
        return None
    return value if value > 0 else None


class LinkExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links: List[str] = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() != 'a':
            return
        for key, value in attrs:
            if key.lower() == 'href' and value:
                self.links.append(value)


def _is_private_host(hostname: str) -> bool:
    if not hostname:
        return True
    lowered = hostname.strip().lower()
    if lowered in {'localhost', '127.0.0.1', '::1'} or lowered.endswith('.local'):
        return True
    try:
        ip = ipaddress.ip_address(lowered)
    except ValueError:
        try:
            infos = socket.getaddrinfo(lowered, None)
        except Exception:
            return False
        for info in infos:
            try:
                ip = ipaddress.ip_address(info[4][0])
            except Exception:
                continue
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
                return True
        return False
    return ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast


def validate_safe_url(url: str) -> str:
    parsed = urlparse((url or '').strip())
    if parsed.scheme not in {'http', 'https'}:
        raise ValueError('Only http(s) URLs are allowed.')
    if not parsed.netloc:
        raise ValueError('URL host is required.')
    if _is_private_host(parsed.hostname or ''):
        raise ValueError('Private or local network URLs are not allowed.')
    return parsed.geturl()


def _looks_like_document_url(url: str) -> bool:
    path = urlparse(url).path or ''
    return Path(path).suffix.lower() in DEFAULT_CANDIDATE_EXTENSIONS


def _safe_request(method: str, url: str, timeout: int = DEFAULT_TIMEOUT_SECONDS, **kwargs):
    safe_url = validate_safe_url(url)
    response = requests.request(method, safe_url, allow_redirects=True, timeout=timeout, **kwargs)
    for hop in list(response.history) + [response]:
        validate_safe_url(hop.url)
    return response


def _head_metadata(url: str) -> Dict[str, object]:
    headers = {}
    try:
        response = _safe_request('HEAD', url)
        headers = dict(response.headers)
    except Exception:
        try:
            response = _safe_request('GET', url, stream=True)
            headers = dict(response.headers)
            response.close()
        except Exception:
            headers = {}
    size_raw = headers.get('Content-Length') or headers.get('content-length')
    try:
        size_value = int(size_raw) if size_raw else None
    except Exception:
        size_value = None
    return {
        'size_bytes': size_value,
        'last_modified': headers.get('Last-Modified') or headers.get('last-modified'),
        'content_type': headers.get('Content-Type') or headers.get('content-type') or None,
    }


def discover_source(source_url: str, limits: BatchLimits) -> Dict[str, object]:
    validated_source = validate_safe_url(source_url)
    discovered_items: List[Dict[str, object]] = []
    skipped_items: List[Dict[str, object]] = []
    seen_urls = set()

    def add_candidate(candidate_url: str, directly_downloadable: bool, parent_url: Optional[str] = None):
        normalized = validate_safe_url(candidate_url)
        if normalized in seen_urls:
            return
        seen_urls.add(normalized)
        ext = Path(urlparse(normalized).path).suffix.lower()
        file_name = Path(urlparse(normalized).path).name or normalized.rsplit('/', 1)[-1]
        meta = _head_metadata(normalized)
        item = {
            'file_id': str(uuid.uuid4()),
            'source_url': normalized,
            'parent_url': parent_url,
            'file_name': file_name,
            'file_type': ext or (meta.get('content_type') or 'unknown'),
            'size_bytes_estimate': meta.get('size_bytes'),
            'last_modified': meta.get('last_modified'),
            'directly_downloadable': directly_downloadable,
            'discovery_status': 'included',
            'skip_reason': None,
        }
        discovered_items.append(item)

    if _looks_like_document_url(validated_source):
        add_candidate(validated_source, True, parent_url=None)
    else:
        response = _safe_request('GET', validated_source)
        content_type = (response.headers.get('Content-Type') or '').lower()
        if 'html' not in content_type and '<html' not in response.text[:2048].lower():
            raise ValueError('Source URL is neither a supported file nor an HTML index page.')
        parser = LinkExtractor()
        parser.feed(response.text)
        for href in parser.links:
            if len(discovered_items) >= limits.max_total_discovered_files:
                skipped_items.append({
                    'source_url': href,
                    'discovery_status': 'skipped',
                    'skip_reason': 'discovery_limit_reached',
                })
                continue
            absolute = urljoin(validated_source, href)
            if _looks_like_document_url(absolute):
                add_candidate(absolute, True, parent_url=validated_source)
            else:
                skipped_items.append({
                    'source_url': absolute,
                    'discovery_status': 'skipped',
                    'skip_reason': 'unsupported_extension',
                })
    total_estimated_bytes = sum(item['size_bytes_estimate'] or 0 for item in discovered_items)
    return {
        'source_url': validated_source,
        'discovered_items': discovered_items,
        'skipped_items': skipped_items,
        'total_estimated_bytes': total_estimated_bytes,
    }


def _priority_tuple(item: Dict[str, object]):
    ext = (item.get('file_type') or '').lower()
    name = (item.get('file_name') or '').lower()
    size = item.get('size_bytes_estimate') or 0
    zim_rank = 0
    if ext == '.zim':
        if 'mini' in name:
            zim_rank = 0
        elif 'nopic' in name:
            zim_rank = 1
        elif 'maxi' in name:
            zim_rank = 3
        else:
            zim_rank = 2
    very_large_penalty = 1 if size and size >= 1024 * 1024 * 1024 else 0
    return (very_large_penalty, zim_rank, size, name, item.get('source_url') or '')


def plan_batches(discovered_items: Iterable[Dict[str, object]], limits: BatchLimits) -> List[Dict[str, object]]:
    items = sorted(list(discovered_items), key=_priority_tuple)
    batches: List[Dict[str, object]] = []
    current: List[Dict[str, object]] = []
    current_bytes = 0
    current_zim = 0

    def flush_batch():
        nonlocal current, current_bytes, current_zim
        if not current:
            return
        batch_number = len(batches) + 1
        batches.append({
            'batch_number': batch_number,
            'file_ids': [item['file_id'] for item in current],
            'files': [
                {
                    'file_id': item['file_id'],
                    'file_name': item['file_name'],
                    'source_url': item['source_url'],
                    'size_bytes_estimate': item.get('size_bytes_estimate'),
                    'file_type': item.get('file_type'),
                }
                for item in current
            ],
            'file_count': len(current),
            'estimated_total_bytes': current_bytes,
            'estimated_zim_files': current_zim,
            'status': 'pending',
        })
        current = []
        current_bytes = 0
        current_zim = 0

    for item in items:
        size = item.get('size_bytes_estimate') or 0
        is_zim = (item.get('file_type') or '').lower() == '.zim'
        next_zim = current_zim + (1 if is_zim else 0)
        exceeds_files = current and len(current) >= limits.max_files_per_batch
        exceeds_bytes = current and current_bytes and size and (current_bytes + size > limits.max_total_bytes_per_batch)
        exceeds_zim = current and is_zim and next_zim > limits.max_zim_files_per_batch
        if exceeds_files or exceeds_bytes or exceeds_zim:
            flush_batch()
        current.append(item)
        current_bytes += size
        if is_zim:
            current_zim += 1
    flush_batch()
    return batches


def _default_download_file(item: Dict[str, object], destination_dir: Path) -> Dict[str, object]:
    destination_dir.mkdir(parents=True, exist_ok=True)
    response = _safe_request('GET', item['source_url'], stream=True)
    response.raise_for_status()
    file_name = item.get('file_name') or Path(urlparse(item['source_url']).path).name or 'infosucker.bin'
    target_path = destination_dir / file_name
    suffix_counter = 1
    while target_path.exists():
        target_path = destination_dir / f"{Path(file_name).stem}_{suffix_counter}{Path(file_name).suffix}"
        suffix_counter += 1
    with target_path.open('wb') as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)
    mime_type = response.headers.get('Content-Type') or mimetypes.guess_type(str(target_path))[0] or 'application/octet-stream'
    return {
        'stored_path': str(target_path),
        'size_bytes': target_path.stat().st_size if target_path.exists() else 0,
        'mime_type': mime_type,
    }


def serialize_job_state(state: Dict[str, object]) -> Dict[str, object]:
    return {
        'job_id': state['job_id'],
        'source_url': state['source_url'],
        'scope': state['scope'],
        'phase': state['phase'],
        'status': state['status'],
        'created_at': state['created_at'],
        'updated_at': state['updated_at'],
        'started_at': state.get('started_at'),
        'completed_at': state.get('completed_at'),
        'current_batch_number': state.get('current_batch_number', 0),
        'batch_count': len(state.get('batch_plan') or []),
        'discovered_items': state.get('discovered_items') or [],
        'batch_plan': state.get('batch_plan') or [],
        'completed_batches': state.get('completed_batches') or [],
        'per_file_status': list((state.get('per_file_status') or {}).values()),
        'summary': state.get('summary') or {},
        'failure_details': state.get('failure_details') or [],
        'validation': state.get('validation') or {},
        'total_estimated_bytes': state.get('total_estimated_bytes') or 0,
    }


class InfoSuckerJobManager:
    def __init__(
        self,
        base_dir: Path,
        downloader: Optional[Callable[[Dict[str, object], Path], Dict[str, object]]] = None,
        importer: Optional[Callable[[Dict[str, object], Dict[str, object], str], Dict[str, object]]] = None,
        limits: Optional[BatchLimits] = None,
    ):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.downloader = downloader or _default_download_file
        self.importer = importer or self._noop_importer
        self.limits = limits or BatchLimits.from_env()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='infosucker')
        self._lock = threading.RLock()
        self._futures: Dict[str, object] = {}

    def _noop_importer(self, user: Dict[str, object], item: Dict[str, object], scope: str) -> Dict[str, object]:
        return {
            'import_attempted': True,
            'extracted_record_count': 0,
            'indexed_document_count': 0,
            'chunk_count': 0,
            'embedding_success': None,
            'failure_reason': 'No importer configured.',
        }

    def _job_dir(self, user_id: int) -> Path:
        path = self.base_dir / str(int(user_id))
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _normalized_job_id(self, job_id: str) -> str:
        return str(uuid.UUID(str(job_id)))

    def _job_path(self, user_id: int, job_id: str) -> Path:
        return self._job_dir(user_id) / f'{self._normalized_job_id(job_id)}.json'

    def _load_state(self, user_id: int, job_id: str) -> Optional[Dict[str, object]]:
        try:
            path = self._job_path(user_id, job_id)
        except (TypeError, ValueError, AttributeError):
            return None
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding='utf-8'))

    def _save_state(self, state: Dict[str, object]):
        self._refresh_failed_summary(state)
        state['updated_at'] = utcnow_iso()
        path = self._job_path(state['user_id'], state['job_id'])
        path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding='utf-8')

    def _refresh_failed_summary(self, state: Dict[str, object]):
        per_file_status = state.get('per_file_status') or {}
        state['summary']['failed'] = sum(
            1
            for file_state in per_file_status.values()
            if file_state.get('failure_reason') and not file_state.get('validated')
        )

    def _future_is_active(self, job_id: str) -> bool:
        with self._lock:
            future = self._futures.get(job_id)
            if future and not future.done():
                return True
            if future and future.done():
                self._futures.pop(job_id, None)
        return False

    def _get_active_job_for_user(self, user_id: int) -> Optional[Dict[str, object]]:
        job_dir = self._job_dir(user_id)
        active_states = []
        for path in job_dir.glob('*.json'):
            try:
                state = json.loads(path.read_text(encoding='utf-8'))
            except Exception:
                continue
            if state.get('status') in {'queued', 'running'}:
                active_states.append(state)
        if not active_states:
            return None
        return sorted(
            active_states,
            key=lambda item: (
                item.get('created_at') or '',
                item.get('started_at') or '',
                item.get('updated_at') or '',
            ),
            reverse=True,
        )[0]

    def _initial_state(self, user: Dict[str, object], source_url: str, scope: str) -> Dict[str, object]:
        now = utcnow_iso()
        return {
            'job_id': str(uuid.uuid4()),
            'user_id': int(user['user_id']),
            'source_url': source_url,
            'scope': scope,
            'phase': 'discovering',
            'status': 'queued',
            'created_at': now,
            'updated_at': now,
            'started_at': None,
            'completed_at': None,
            'current_batch_number': 0,
            'discovered_items': [],
            'batch_plan': [],
            'completed_batches': [],
            'per_file_status': {},
            'summary': {
                'discovered': 0,
                'downloaded': 0,
                'imported': 0,
                'validated': 0,
                'skipped': 0,
                'failed': 0,
                'extracted_records': 0,
                'indexed_documents': 0,
                'chunk_count': 0,
            },
            'failure_details': [],
            'validation': {},
            'total_estimated_bytes': 0,
        }

    def start_job(self, user: Dict[str, object], source_url: str, scope: str, download_root: Path) -> Dict[str, object]:
        with self._lock:
            active = self._get_active_job_for_user(int(user['user_id']))
            if active:
                return serialize_job_state(active)
            state = self._initial_state(user, source_url, scope)
            self._save_state(state)
            future = self._executor.submit(self._run_job, state['job_id'], user, scope, Path(download_root), False)
            self._futures[state['job_id']] = future
        return serialize_job_state(state)

    def resume_job(self, user: Dict[str, object], job_id: str, download_root: Path) -> Optional[Dict[str, object]]:
        with self._lock:
            state = self._load_state(int(user['user_id']), job_id)
            if not state:
                return None
            active = self._get_active_job_for_user(int(user['user_id']))
            if active and active.get('job_id') != state.get('job_id'):
                blocked = serialize_job_state(state)
                blocked['resume_rejected'] = True
                return blocked
            if state.get('status') in {'queued', 'running'} or self._future_is_active(job_id):
                blocked = serialize_job_state(state)
                blocked['resume_rejected'] = True
                return blocked
            if state.get('status') not in {'paused', 'failed'}:
                return serialize_job_state(state)
            state['status'] = 'queued'
            state['phase'] = 'planning' if state.get('batch_plan') else 'discovering'
            state['completed_at'] = None
            self._save_state(state)
            future = self._executor.submit(self._run_job, job_id, user, state.get('scope') or 'personal', Path(download_root), True)
            self._futures[job_id] = future
        return serialize_job_state(state)

    def get_job(self, user: Dict[str, object], job_id: str) -> Optional[Dict[str, object]]:
        state = self._load_state(int(user['user_id']), job_id)
        if not state:
            return None
        return serialize_job_state(state)

    def get_latest_job(self, user: Dict[str, object]) -> Optional[Dict[str, object]]:
        job_dir = self._job_dir(int(user['user_id']))
        states = []
        for path in job_dir.glob('*.json'):
            try:
                state = json.loads(path.read_text(encoding='utf-8'))
            except Exception:
                continue
            states.append(state)
        if not states:
            return None
        state = sorted(
            states,
            key=lambda item: (
                item.get('created_at') or '',
                item.get('started_at') or '',
                item.get('updated_at') or '',
            ),
            reverse=True,
        )[0]
        return serialize_job_state(state)

    def _run_job(self, job_id: str, user: Dict[str, object], scope: str, download_root: Path, resume: bool):
        state = self._load_state(int(user['user_id']), job_id)
        if not state:
            return
        state['status'] = 'running'
        state['started_at'] = state.get('started_at') or utcnow_iso()
        self._save_state(state)
        try:
            if not resume or not state.get('discovered_items'):
                state['phase'] = 'discovering'
                self._save_state(state)
                discovered = discover_source(state['source_url'], self.limits)
                state['discovered_items'] = discovered['discovered_items']
                state['summary']['discovered'] = len(discovered['discovered_items'])
                state['summary']['skipped'] = len(discovered['skipped_items'])
                state['failure_details'] = list(discovered['skipped_items'])
                state['total_estimated_bytes'] = discovered['total_estimated_bytes']
                self._save_state(state)
            if not state.get('batch_plan'):
                state['phase'] = 'planning'
                state['batch_plan'] = plan_batches(state.get('discovered_items') or [], self.limits)
                self._save_state(state)
            pending_batches = [batch for batch in state['batch_plan'] if batch['batch_number'] not in state.get('completed_batches', [])]
            for batch in pending_batches:
                batch_number = batch['batch_number']
                state['current_batch_number'] = batch_number
                batch['status'] = 'running'
                download_batch_dir = Path(download_root) / state['job_id'] / f'batch_{batch_number:03d}'
                download_batch_dir.mkdir(parents=True, exist_ok=True)
                state['phase'] = 'downloading'
                self._save_state(state)
                batch_file_statuses = []
                for item in state['discovered_items']:
                    if item['file_id'] not in batch['file_ids']:
                        continue
                    file_state = state['per_file_status'].setdefault(item['file_id'], {
                        'file_id': item['file_id'],
                        'file_name': item['file_name'],
                        'source_url': item['source_url'],
                        'file_type': item.get('file_type'),
                        'batch_number': batch_number,
                        'status': 'pending',
                        'downloaded': False,
                        'import_attempted': False,
                        'validated': False,
                        'extracted_record_count': 0,
                        'indexed_document_count': 0,
                        'chunk_count': 0,
                        'embedding_success': None,
                        'failure_reason': None,
                    })
                    if file_state.get('validated'):
                        batch_file_statuses.append(file_state)
                        continue
                    try:
                        existing_path = Path(file_state.get('stored_path') or "")
                        if file_state.get('downloaded') and existing_path.exists():
                            file_state['status'] = 'downloaded'
                            file_state['failure_reason'] = None
                        else:
                            download_result = self.downloader(item, download_batch_dir)
                            file_state.update(download_result)
                            file_state['downloaded'] = True
                            file_state['status'] = 'downloaded'
                            file_state['failure_reason'] = None
                            state['summary']['downloaded'] += 1
                    except Exception as exc:
                        file_state['status'] = 'failed'
                        file_state['failure_reason'] = str(exc)
                    batch_file_statuses.append(file_state)
                    self._save_state(state)
                state['phase'] = 'importing'
                self._save_state(state)
                for file_state in batch_file_statuses:
                    if file_state.get('validated') or not file_state.get('downloaded'):
                        continue
                    try:
                        previous_extracted = int(file_state.get('extracted_record_count') or 0)
                        previous_indexed = int(file_state.get('indexed_document_count') or 0)
                        previous_chunks = int(file_state.get('chunk_count') or 0)
                        if file_state.get('import_attempted'):
                            state['summary']['extracted_records'] = max(0, state['summary']['extracted_records'] - previous_extracted)
                            state['summary']['indexed_documents'] = max(0, state['summary']['indexed_documents'] - previous_indexed)
                            state['summary']['chunk_count'] = max(0, state['summary']['chunk_count'] - previous_chunks)
                            if previous_indexed > 0:
                                state['summary']['imported'] = max(0, state['summary']['imported'] - 1)
                        import_result = self.importer(user, file_state, scope)
                        file_state.update(import_result)
                        file_state['import_attempted'] = bool(import_result.get('import_attempted', True))
                        file_state['status'] = 'imported' if import_result.get('indexed_document_count', 0) > 0 else 'imported_empty'
                        if import_result.get('indexed_document_count', 0) > 0:
                            file_state['failure_reason'] = None
                        state['summary']['extracted_records'] += int(import_result.get('extracted_record_count') or 0)
                        state['summary']['indexed_documents'] += int(import_result.get('indexed_document_count') or 0)
                        state['summary']['chunk_count'] += int(import_result.get('chunk_count') or 0)
                        if import_result.get('indexed_document_count', 0) > 0:
                            state['summary']['imported'] += 1
                    except Exception as exc:
                        file_state['status'] = 'failed'
                        file_state['failure_reason'] = str(exc)
                    self._save_state(state)
                state['phase'] = 'validating'
                failures = 0
                validated = 0
                validation_results = []
                for file_state in batch_file_statuses:
                    if file_state.get('downloaded') and file_state.get('import_attempted') and file_state.get('indexed_document_count', 0) > 0 and file_state.get('extracted_record_count', 0) > 0:
                        already_validated = bool(file_state.get('validated'))
                        file_state['validated'] = True
                        file_state['status'] = 'validated'
                        validated += 1
                        if not already_validated:
                            state['summary']['validated'] += 1
                    else:
                        failures += 1
                        if not file_state.get('failure_reason'):
                            if not file_state.get('downloaded'):
                                file_state['failure_reason'] = 'Download did not complete.'
                            elif file_state.get('extracted_record_count', 0) <= 0:
                                file_state['failure_reason'] = 'Import yielded zero extracted records.'
                            else:
                                file_state['failure_reason'] = 'No documents were indexed.'
                    validation_results.append({
                        'file_id': file_state['file_id'],
                        'validated': file_state.get('validated', False),
                        'failure_reason': file_state.get('failure_reason'),
                    })
                failure_rate = failures / max(len(batch_file_statuses), 1)
                state['validation'][str(batch_number)] = {
                    'batch_number': batch_number,
                    'validated_files': validated,
                    'failed_files': failures,
                    'failure_rate': failure_rate,
                    'results': validation_results,
                }
                if failure_rate > self.limits.validation_failure_threshold:
                    batch['status'] = 'paused'
                    state['phase'] = 'paused'
                    state['status'] = 'paused'
                    state['failure_details'].append({
                        'batch_number': batch_number,
                        'reason': 'validation_failure_threshold_exceeded',
                        'failure_rate': failure_rate,
                    })
                    self._save_state(state)
                    return
                batch['status'] = 'completed'
                state['completed_batches'].append(batch_number)
                self._save_state(state)
            state['phase'] = 'completed'
            state['status'] = 'completed'
            state['completed_at'] = utcnow_iso()
            self._save_state(state)
        except Exception as exc:
            state['phase'] = 'failed'
            state['status'] = 'failed'
            state['completed_at'] = utcnow_iso()
            state['failure_details'].append({'reason': str(exc)})
            self._save_state(state)
            LOGGER.exception('Info-Sucker job %s failed', job_id)
