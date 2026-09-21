#!/app/jarvis-dev/venv/bin/python3
import json
import sqlite3
from pathlib import Path
from typing import List, Optional

from config import DB_FILE, ensure_directories

ensure_directories()


class MemoryStore:
    def __init__(self, db_path: Path = DB_FILE):
        self.db_path = str(db_path)

    def connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def apply_migration_file(self, migration_path: Path):
        sql = migration_path.read_text(encoding="utf-8")
        with self.connect() as conn:
            conn.executescript(sql)
            conn.commit()

    def apply_all_migrations(self, migrations_dir: Path):
        for path in sorted(migrations_dir.glob("*.sql")):
            self.apply_migration_file(path)

    def get_user_by_email(self, email: str) -> Optional[sqlite3.Row]:
        with self.connect() as conn:
            cur = conn.execute(
                "SELECT id, email, display_name, created_at, updated_at, last_login_at, is_active "
                "FROM users WHERE lower(email) = lower(?) LIMIT 1",
                (email.strip(),)
            )
            return cur.fetchone()

    def get_user_by_id(self, user_id: int) -> Optional[sqlite3.Row]:
        with self.connect() as conn:
            cur = conn.execute(
                "SELECT id, email, display_name, created_at, updated_at, last_login_at, is_active "
                "FROM users WHERE id = ? LIMIT 1",
                (user_id,)
            )
            return cur.fetchone()

    def list_users(self):
        with self.connect() as conn:
            cur = conn.execute(
                """
                SELECT
                    u.id,
                    u.email,
                    u.display_name,
                    u.created_at,
                    u.updated_at,
                    u.last_login_at,
                    u.is_active,
                    us.preferred_name,
                    us.first_name,
                    us.last_name,
                    us.home_location,
                    us.home_location_label,
                    us.preferred_language
                FROM users u
                LEFT JOIN user_settings us ON us.user_id = u.id
                ORDER BY lower(u.email) ASC
                """
            )
            rows = cur.fetchall()
            return [dict(row) for row in rows]

    def get_user_admin_detail(self, user_id: int):
        with self.connect() as conn:
            cur = conn.execute(
                """
                SELECT
                    u.id,
                    u.email,
                    u.display_name,
                    u.created_at,
                    u.updated_at,
                    u.last_login_at,
                    u.is_active,
                    us.preferred_name,
                    us.first_name,
                    us.last_name,
                    us.home_location,
                    us.home_location_label,
                    us.preferred_language,
                    us.home_location_lat,
                    us.home_location_lon,
                    us.browser_audio_enabled,
                    us.browser_mic_enabled,
                    us.local_audio_enabled,
                    us.local_mic_enabled
                FROM users u
                LEFT JOIN user_settings us ON us.user_id = u.id
                WHERE u.id = ?
                """,
                (user_id,)
            )
            row = cur.fetchone()
            if not row:
                return None

            result = dict(row)
            result.setdefault("preferred_name", None)
            result.setdefault("first_name", "")
            result.setdefault("last_name", "")
            result.setdefault("preferred_language", "auto")
            result.setdefault("home_location", None)
            result.setdefault("home_location_label", None)
            result.setdefault("home_location_lat", None)
            result.setdefault("home_location_lon", None)
            result.setdefault("browser_audio_enabled", 1)
            result.setdefault("browser_mic_enabled", 1)
            result.setdefault("local_audio_enabled", 1)
            result.setdefault("local_mic_enabled", 1)
            result["browser_mic_gain"] = None
            result["browser_noise_gate"] = None
            result["browser_speech_threshold"] = None
            return result

    def create_user(
        self,
        email: str,
        password_hash: str,
        display_name: str = "Sir",
        first_name: str = "",
        last_name: str = "",
        home_location: str = "",
        preferred_language: str = "auto",
    ) -> int:
        with self.connect() as conn:
            cur = conn.execute(
                "INSERT INTO users (email, password_hash, display_name) VALUES (?, ?, ?)",
                (email.strip().lower(), password_hash, display_name.strip() or "Sir")
            )
            user_id = cur.lastrowid
            conn.execute(
                "INSERT INTO user_settings (user_id, preferred_name, first_name, last_name, home_location, home_location_label, preferred_language, home_location_lat, home_location_lon) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    user_id,
                    display_name.strip() or "Sir",
                    first_name.strip(),
                    last_name.strip(),
                    home_location.strip(),
                    home_location.strip(),
                    preferred_language.strip() or "auto",
                    None,
                    None,
                )
            )
            conn.commit()
            return user_id

    def update_user_display_name(self, user_id: int, display_name: str):
        display_name = display_name.strip() or "Sir"
        with self.connect() as conn:
            conn.execute(
                "UPDATE users SET display_name = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (display_name, user_id)
            )
            conn.execute(
                "UPDATE user_settings SET preferred_name = ? WHERE user_id = ?",
                (display_name, user_id)
            )
            conn.commit()

    def get_password_hash(self, email: str) -> Optional[str]:
        with self.connect() as conn:
            cur = conn.execute(
                "SELECT password_hash FROM users WHERE lower(email) = lower(?) LIMIT 1",
                (email.strip(),)
            )
            row = cur.fetchone()
            return row["password_hash"] if row else None

    def update_password_hash(self, user_id: int, password_hash: str):
        with self.connect() as conn:
            conn.execute(
                "UPDATE users SET password_hash = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (password_hash, user_id)
            )
            conn.commit()

    def deactivate_user(self, user_id: int):
        with self.connect() as conn:
            conn.execute(
                "UPDATE users SET is_active = 0, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (user_id,)
            )
            conn.execute(
                "UPDATE sessions SET is_active = 0 WHERE user_id = ?",
                (user_id,)
            )
            conn.commit()

    def activate_user(self, user_id: int):
        with self.connect() as conn:
            conn.execute(
                "UPDATE users SET is_active = 1, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (user_id,)
            )
            conn.commit()

    def delete_user(self, user_id: int):
        with self.connect() as conn:
            conn.execute("DELETE FROM document_chunks WHERE owner_user_id = ?", (user_id,))
            conn.execute("DELETE FROM document_index WHERE owner_user_id = ?", (user_id,))
            conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
            conn.execute("DELETE FROM reset_tokens WHERE user_id = ?", (user_id,))
            conn.execute("DELETE FROM uploads WHERE user_id = ?", (user_id,))
            conn.execute("DELETE FROM conversation_memory WHERE user_id = ?", (user_id,))
            conn.execute("DELETE FROM long_term_facts WHERE user_id = ?", (user_id,))
            conn.execute("DELETE FROM voiceprints WHERE user_id = ?", (user_id,))
            conn.execute("DELETE FROM faceprints WHERE user_id = ?", (user_id,))
            conn.execute("DELETE FROM user_settings WHERE user_id = ?", (user_id,))
            conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
            conn.commit()

    def get_user_settings(self, user_id: int) -> Optional[sqlite3.Row]:
        with self.connect() as conn:
            cur = conn.execute(
                "SELECT * FROM user_settings WHERE user_id = ? LIMIT 1",
                (user_id,)
            )
            return cur.fetchone()

    def update_user_settings(self, user_id: int, **kwargs):
        allowed = {
            "preferred_name",
            "first_name",
            "last_name",
            "preferred_language",
            "home_location",
            "home_location_label",
            "home_location_lat",
            "home_location_lon",
            "browser_audio_enabled",
            "browser_mic_enabled",
            "browser_mic_gain",
            "browser_noise_gate",
            "browser_speech_threshold",
            "local_audio_enabled",
            "local_mic_enabled",
        }
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return

        fields = ", ".join(f"{k} = ?" for k in updates.keys())
        values = list(updates.values()) + [user_id]

        with self.connect() as conn:
            conn.execute(
                f"UPDATE user_settings SET {fields} WHERE user_id = ?",
                values
            )
            conn.commit()

    def create_session(self, user_id: int, session_token: str, source: str = "browser", expires_at: Optional[str] = None):
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO sessions (user_id, session_token, source, expires_at, last_seen_at) "
                "VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)",
                (user_id, session_token, source, expires_at)
            )
            conn.commit()

    def get_session(self, session_token: str) -> Optional[sqlite3.Row]:
        with self.connect() as conn:
            cur = conn.execute(
                "SELECT * FROM sessions WHERE session_token = ? AND is_active = 1 LIMIT 1",
                (session_token,)
            )
            return cur.fetchone()

    def touch_session(self, session_token: str):
        with self.connect() as conn:
            conn.execute(
                "UPDATE sessions SET last_seen_at = CURRENT_TIMESTAMP WHERE session_token = ?",
                (session_token,)
            )
            conn.commit()

    def deactivate_session(self, session_token: str):
        with self.connect() as conn:
            conn.execute(
                "UPDATE sessions SET is_active = 0 WHERE session_token = ?",
                (session_token,)
            )
            conn.commit()

    def store_reset_token(self, user_id: int, token: str, expires_at: str):
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO reset_tokens (user_id, token, expires_at) VALUES (?, ?, ?)",
                (user_id, token, expires_at)
            )
            conn.commit()

    def get_reset_token(self, token: str) -> Optional[sqlite3.Row]:
        with self.connect() as conn:
            cur = conn.execute(
                "SELECT * FROM reset_tokens WHERE token = ? LIMIT 1",
                (token,)
            )
            return cur.fetchone()

    def mark_reset_token_used(self, token: str):
        with self.connect() as conn:
            conn.execute(
                "UPDATE reset_tokens SET used_at = CURRENT_TIMESTAMP WHERE token = ?",
                (token,)
            )
            conn.commit()

    def add_conversation_entry(self, user_id: int, role: str, content: str, source: str = "browser"):
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO conversation_memory (user_id, role, content, source) VALUES (?, ?, ?, ?)",
                (user_id, role, content, source)
            )
            conn.commit()

    def get_recent_conversation(self, user_id: int, limit: int = 10):
        with self.connect() as conn:
            cur = conn.execute(
                "SELECT id, user_id, role, content, source, created_at "
                "FROM conversation_memory WHERE user_id = ? "
                "ORDER BY id DESC LIMIT ?",
                (user_id, limit)
            )
            return list(reversed(cur.fetchall()))

    def add_long_term_fact(self, user_id: int, fact_text: str, confidence: float = 0.5):
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO long_term_facts (user_id, fact_text, confidence) VALUES (?, ?, ?)",
                (user_id, fact_text, confidence)
            )
            conn.commit()

    def get_long_term_facts(self, user_id: int, limit: int = 20):
        with self.connect() as conn:
            cur = conn.execute(
                "SELECT id, user_id, fact_text, confidence, created_at "
                "FROM long_term_facts WHERE user_id = ? "
                "ORDER BY id DESC LIMIT ?",
                (user_id, limit)
            )
            return cur.fetchall()

    def store_voiceprint(self, user_id: int, vector: List[float], sample_rate: Optional[int] = None, source: str = "browser"):
        with self.connect() as conn:
            conn.execute(
                "UPDATE voiceprints SET is_active = 0 WHERE user_id = ?",
                (user_id,)
            )
            conn.execute(
                "INSERT INTO voiceprints (user_id, vector_json, sample_rate, source, is_active) VALUES (?, ?, ?, ?, 1)",
                (user_id, json.dumps(vector), sample_rate, source)
            )
            conn.commit()

    def get_active_voiceprint(self, user_id: int) -> Optional[sqlite3.Row]:
        with self.connect() as conn:
            cur = conn.execute(
                "SELECT * FROM voiceprints WHERE user_id = ? AND is_active = 1 ORDER BY id DESC LIMIT 1",
                (user_id,)
            )
            return cur.fetchone()

    def store_faceprint(self, user_id: int, encoding: List[float], source_image_path: Optional[str] = None):
        with self.connect() as conn:
            conn.execute(
                "UPDATE faceprints SET is_active = 0 WHERE user_id = ?",
                (user_id,)
            )
            conn.execute(
                "INSERT INTO faceprints (user_id, encoding_json, source_image_path, is_active) VALUES (?, ?, ?, 1)",
                (user_id, json.dumps(encoding), source_image_path)
            )
            conn.commit()

    def get_active_faceprint(self, user_id: int) -> Optional[sqlite3.Row]:
        with self.connect() as conn:
            cur = conn.execute(
                "SELECT * FROM faceprints WHERE user_id = ? AND is_active = 1 ORDER BY id DESC LIMIT 1",
                (user_id,)
            )
            return cur.fetchone()

    def store_upload(self, user_id: int, file_name: str, stored_path: str, mime_type: Optional[str] = None):
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO uploads (user_id, file_name, stored_path, mime_type) VALUES (?, ?, ?, ?)",
                (user_id, file_name, stored_path, mime_type)
            )
            conn.commit()

    def get_uploads(self, user_id: int):
        with self.connect() as conn:
            cur = conn.execute(
                "SELECT * FROM uploads WHERE user_id = ? ORDER BY id DESC",
                (user_id,)
            )
            return cur.fetchall()

    def create_document_index(
        self,
        owner_user_id: int,
        file_name: str,
        stored_path: str,
        mime_type: Optional[str],
        scope: str = "personal",
        extracted_text: str = "",
        ocr_status: str = "complete",
        is_searchable: int = 1,
    ) -> int:
        with self.connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO document_index
                (owner_user_id, file_name, stored_path, mime_type, scope, extracted_text, ocr_status, is_searchable, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (
                    owner_user_id,
                    file_name,
                    stored_path,
                    mime_type,
                    scope,
                    extracted_text,
                    ocr_status,
                    is_searchable,
                )
            )
            conn.commit()
            return cur.lastrowid

    def replace_document_chunks(
        self,
        document_id: int,
        owner_user_id: int,
        scope: str,
        chunks: List[str],
    ):
        with self.connect() as conn:
            conn.execute("DELETE FROM document_chunks WHERE document_id = ?", (document_id,))
            for idx, chunk_text in enumerate(chunks):
                conn.execute(
                    """
                    INSERT INTO document_chunks
                    (document_id, owner_user_id, scope, chunk_index, chunk_text)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (document_id, owner_user_id, scope, idx, chunk_text)
                )
            conn.execute(
                "UPDATE document_index SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (document_id,)
            )
            conn.commit()

    def list_documents_for_user(self, user_id: int):
        with self.connect() as conn:
            cur = conn.execute(
                """
                SELECT *
                FROM document_index
                WHERE (scope = 'global') OR (scope = 'personal' AND owner_user_id = ?)
                ORDER BY created_at DESC, id DESC
                """,
                (user_id,)
            )
            return [dict(row) for row in cur.fetchall()]

    def list_documents_for_owner(self, owner_user_id: int):
        with self.connect() as conn:
            cur = conn.execute(
                """
                SELECT *
                FROM document_index
                WHERE owner_user_id = ?
                ORDER BY created_at DESC, id DESC
                """,
                (owner_user_id,)
            )
            return [dict(row) for row in cur.fetchall()]

    def search_document_chunks_for_user(self, user_id: int, query: str, limit: int = 12):
        like = f"%{(query or '').strip()}%"
        if not query.strip():
            return []

        with self.connect() as conn:
            cur = conn.execute(
                """
                SELECT
                    dc.id,
                    dc.document_id,
                    dc.owner_user_id,
                    dc.scope,
                    dc.chunk_index,
                    dc.chunk_text,
                    di.file_name,
                    di.stored_path,
                    di.mime_type
                FROM document_chunks dc
                JOIN document_index di ON di.id = dc.document_id
                WHERE
                    (
                        dc.scope = 'global'
                        OR (dc.scope = 'personal' AND dc.owner_user_id = ?)
                    )
                    AND dc.chunk_text LIKE ?
                ORDER BY
                    CASE WHEN dc.scope = 'personal' THEN 0 ELSE 1 END,
                    dc.id DESC
                LIMIT ?
                """,
                (user_id, like, limit)
            )
            return [dict(row) for row in cur.fetchall()]
