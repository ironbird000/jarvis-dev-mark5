import hashlib
import html
import logging
import mimetypes
import re
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional

LOGGER = logging.getLogger(__name__)
TEXT_EXTENSIONS = {
    '.txt', '.md', '.rtf', '.csv', '.tsv', '.json', '.xml', '.yaml', '.yml',
    '.html', '.htm', '.py', '.js', '.ts', '.css', '.sh', '.bash', '.zsh', '.ps1', '.bat'
}
TEXT_MIME_PREFIXES = ('text/',)
TEXT_MIME_EXACT = {
    'application/json',
    'application/xml',
    'application/xhtml+xml',
    'application/javascript',
}


def _decode_bytes(value) -> str:
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='ignore')
    return str(value or '')


def _strip_html_markup(text: str) -> str:
    clean = re.sub(r'(?is)<script\b[^>]*>.*?</script\b[^>]*>', ' ', text)
    clean = re.sub(r'(?is)<style\b[^>]*>.*?</style\b[^>]*>', ' ', clean)
    clean = re.sub(r'(?s)<[^>]+>', ' ', clean)
    clean = html.unescape(clean)
    clean = re.sub(r'\s+', ' ', clean)
    return clean.strip()


def _guess_mime_type(file_path: Path) -> str:
    return mimetypes.guess_type(str(file_path))[0] or 'application/octet-stream'


def _read_text_file(file_path: Path, mime_type: str) -> str:
    raw = file_path.read_text(encoding='utf-8', errors='ignore')
    if mime_type in {'text/html', 'application/xhtml+xml'} or file_path.suffix.lower() in {'.html', '.htm'}:
        return _strip_html_markup(raw)
    return raw.strip()


def _text_record(file_path: Path, mime_type: Optional[str] = None) -> Iterator[Dict[str, object]]:
    mime = mime_type or _guess_mime_type(file_path)
    text = _read_text_file(file_path, mime)
    if not text.strip():
        return
    payload = {
        'file_name': file_path.name,
        'stored_path': str(file_path),
        'mime_type': mime,
        'text': text,
        'method': 'text_file',
        'ocr_used': False,
        'searchable_pdf_path': None,
        'source_metadata': {
            'source_type': 'file',
            'source_path': str(file_path),
            'content_hash': hashlib.sha256(text.encode('utf-8')).hexdigest(),
        },
    }
    yield payload


def _looks_textual(file_path: Path, mime_type: Optional[str] = None) -> bool:
    mime = mime_type or _guess_mime_type(file_path)
    suffix = file_path.suffix.lower()
    return suffix in TEXT_EXTENSIONS or mime.startswith(TEXT_MIME_PREFIXES) or mime in TEXT_MIME_EXACT


def _iter_zim_records(
    file_path: Path,
    max_records: Optional[int] = None,
    stop_after: Optional[int] = None,
    progress_every: Optional[int] = None,
) -> Iterator[Dict[str, object]]:
    try:
        from pyzim.archive import Zim
    except Exception as exc:  # pragma: no cover - depends on optional dependency
        raise RuntimeError('pyzim is required for ZIM ingestion') from exc

    archive = Zim.open(str(file_path), mode='r')
    emitted = 0
    try:
        metadata = archive.get_metadata_dict(as_unicode=True) or {}
        archive_language = _decode_bytes(metadata.get('Language')).strip()
        for entry in archive.iter_articles():
            if stop_after is not None and emitted >= stop_after:
                break
            if getattr(entry, 'is_redirect', False):
                continue
            mimetype = _decode_bytes(getattr(entry, 'mimetype', '')).strip().lower()
            if not mimetype.startswith('text/') and mimetype not in {'application/xhtml+xml'}:
                continue
            namespace = _decode_bytes(getattr(entry, 'namespace', '')).strip()
            url = _decode_bytes(getattr(entry, 'url', '')).strip()
            title = _decode_bytes(getattr(entry, 'title', '')).strip() or url or file_path.name
            canonical_uri = _decode_bytes(getattr(entry, 'full_url', '')).strip()
            if not canonical_uri:
                if namespace and url:
                    canonical_uri = f'{namespace}/{url}'
                else:
                    canonical_uri = url or namespace
            canonical_uri = canonical_uri.strip('/')
            if not canonical_uri:
                continue
            raw = entry.read()
            text = _decode_bytes(raw)
            if mimetype in {'text/html', 'application/xhtml+xml'}:
                text = _strip_html_markup(text)
            else:
                text = re.sub(r'\s+', ' ', text).strip()
            if not text:
                continue
            emitted += 1
            if progress_every and emitted % progress_every == 0:
                LOGGER.info('ZIM ingest progress for %s: %s records extracted', file_path.name, emitted)
            yield {
                'file_name': f'{file_path.name} - {title}',
                'stored_path': f'{file_path}#{canonical_uri}',
                'mime_type': mimetype or 'text/plain',
                'text': text,
                'method': 'zim_python_zim',
                'ocr_used': False,
                'searchable_pdf_path': None,
                'source_metadata': {
                    'source_type': 'zim',
                    'source_path': str(file_path),
                    'archive_name': file_path.name,
                    'entry_id': canonical_uri,
                    'entry_title': title,
                    'canonical_uri': canonical_uri,
                    'language': archive_language,
                    'content_hash': hashlib.sha256(text.encode('utf-8')).hexdigest(),
                },
            }
            if max_records is not None and emitted >= max_records:
                break
    finally:
        archive.close()


def iter_ingest_records(
    file_path: str,
    mime_type: Optional[str] = None,
    max_records: Optional[int] = None,
    stop_after: Optional[int] = None,
    progress_every: Optional[int] = None,
) -> Iterable[Dict[str, object]]:
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix == '.zim':
        return _iter_zim_records(path, max_records=max_records, stop_after=stop_after, progress_every=progress_every)
    if _looks_textual(path, mime_type=mime_type):
        return _text_record(path, mime_type=mime_type)
    return iter(())
