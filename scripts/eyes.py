#!/app/jarvis-dev/venv/bin/python3
import base64
import json
import logging
import re
import shutil
import socket
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import cv2
import face_recognition
import numpy as np
import requests

from config import (
    ADMIN_EMAILS,
    AMCREST_CAMERA_NAME,
    AMCREST_ENABLE_PTZ,
    AMCREST_HTTP_BASE,
    AMCREST_PASSWORD,
    AMCREST_RTSP_URL,
    AMCREST_USERNAME,
    DATA_DIR,
    EMEET_CAMERA_INDEX,
    EMEET_CAMERA_NAME,
    EMEET_ENABLE_PTZ,
    EMEET_V4L2_DEVICE,
    EYES_SOCKET,
    KNOWN_FACES_DIR,
    VISION_ENABLE_TESSERACT,
    VISION_ENABLE_YOLO,
    VISION_LOW_CONFIDENCE_THRESHOLD,
    VISION_SNAPSHOTS_DIR,
    VISION_WEB_FALLBACK,
    YOLO_MODEL_PATH,
    ensure_directories,
)
from memory import MemoryStore
from sensory_awareness import (
    AUDIO_FRESHNESS_SECONDS,
    AUDIO_RECENT_ACTIVITY_SECONDS,
    VISUAL_RECENT_ACTIVITY_MULTIPLIER,
    classify_freshness,
    load_audio_runtime_state,
    max_timestamp_text,
)
from vision_targeting import (
    coordinate_space_for_payload,
    infer_target_mode,
    normalize_coordinate,
    target_region_from_point,
)
from web_lookup import build_web_search_url

try:
    import pytesseract
except Exception:  # pragma: no cover - optional dependency in dev
    pytesseract = None

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - optional dependency in dev
    YOLO = None

ensure_directories()

LOG_FILE = DATA_DIR / "logs" / "eyes.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

DEFAULT_PTZ_INTERVAL_SECONDS = 0.35
DEFAULT_AMCREST_SPEED = 4
DEFAULT_EMEET_ZOOM_STEP = 10
AUTOFRAME_SMALL_THRESHOLD = 0.08
AUTOFRAME_LARGE_THRESHOLD = 0.55
FACE_MATCH_DISTANCE_THRESHOLD = 0.6
TARGET_REGION_CONTEXT_MARGIN = 0.12
VISION_PERCEPTION_HISTORY_LIMIT = 6
VISION_CONVERSATIONAL_FRESHNESS_SECONDS = 12.0
GENERIC_PERSON_LABELS = {"person", "face", "head", "human"}
KNOWN_OBJECT_OVERRIDE_CONFIDENCE = 0.93
TARGET_PRIORITY_MIN_CONFIDENCE = 0.2
TARGET_PRIORITY_MIN_OVERLAP = 0.12
FRAME_CAPTURE_MAX_RETRIES = 3
FRAME_CAPTURE_RETRY_DELAY_SECONDS = 0.15
FRAME_CAPTURE_MIN_DIMENSION = 48
VISION_INTENT_LABELS = {
    "inspect_target": "Inspect target",
    "what_is_this": "What is this?",
    "read_text": "Read text",
    "remember_this": "Remember this",
    "troubleshoot_this": "Help me troubleshoot this",
    "who_do_you_see": "Who do you see?",
}


@dataclass
class CameraDescriptor:
    camera_id: str
    label: str
    source_kind: str
    source_ref: str
    ptz: bool = False
    zoom: bool = False
    browser: bool = False
    available: bool = True
    owner_user_id: Optional[int] = None
    permission_state: str = "prompt"
    device_id: str = ""


class CameraAccessError(PermissionError):
    def __init__(self, message: str, camera_id: str = ""):
        super().__init__(message)
        self.camera_id = (camera_id or "").strip().lower()


class RateLimiter:
    def __init__(self, min_interval_seconds: float = DEFAULT_PTZ_INTERVAL_SECONDS):
        self.min_interval_seconds = float(min_interval_seconds)
        self._last_seen: Dict[str, float] = {}

    def check(self, key: str) -> Tuple[bool, float]:
        now = time.monotonic()
        last = self._last_seen.get(key, 0.0)
        delta = now - last
        if delta < self.min_interval_seconds:
            return False, max(self.min_interval_seconds - delta, 0.0)
        self._last_seen[key] = now
        return True, 0.0


class EyesService:
    def __init__(self):
        self.store = MemoryStore()
        self.known_encodings: List = []
        self.known_labels: List[str] = []
        self.ptz_limiter = RateLimiter()
        self.visual_perception_states: Dict[str, Dict] = {}
        self.yolo_model = self._load_yolo_model()
        self.reload_known_faces()

    def _load_yolo_model(self):
        if not VISION_ENABLE_YOLO or YOLO is None or YOLO_MODEL_PATH is None:
            return None
        try:
            if not YOLO_MODEL_PATH.exists():
                logging.warning("YOLO model path does not exist: %s", YOLO_MODEL_PATH)
                return None
            return YOLO(str(YOLO_MODEL_PATH))
        except Exception as exc:
            logging.exception("Failed loading YOLO model: %s", exc)
            return None

    def reload_known_faces(self):
        self.known_encodings = []
        self.known_labels = []
        if not KNOWN_FACES_DIR.exists():
            return

        for person_dir in KNOWN_FACES_DIR.iterdir():
            if not person_dir.is_dir():
                continue
            label = person_dir.name
            for file_path in person_dir.iterdir():
                if file_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                try:
                    image = face_recognition.load_image_file(str(file_path))
                    encodings = face_recognition.face_encodings(image)
                    if encodings:
                        self.known_encodings.append(encodings[0])
                        self.known_labels.append(label)
                except Exception as exc:
                    logging.exception("Failed loading face %s: %s", file_path, exc)

    def process(self, payload: dict) -> dict:
        action = (payload.get("action") or "status").strip().lower()
        if action == "status":
            return self._status()
        if action == "reload":
            self.reload_known_faces()
            return self._status()
        if action in {"list_cameras", "camera_capabilities"}:
            return self._list_cameras(payload)
        if action in {"inspect", "inspect_object"}:
            return self.inspect_object(payload)
        if action in {"faces", "who_do_you_see"}:
            return self.inspect_faces(payload)
        if action == "conversation_vision_state":
            return self.conversation_vision_state(payload)
        if action == "resolve_identity_label":
            return self.resolve_identity_label(payload)
        if action == "live_sensory_awareness":
            return self.live_sensory_awareness(payload)
        if action in {"preview_frame_heartbeat", "preview_visual_state"}:
            return self.preview_frame_heartbeat(payload)
        if action in {"ptz", "camera_control"}:
            return self.handle_ptz(payload)
        if action in {"emeet_zoom", "zoom_helper"}:
            return self.handle_emeet_zoom(payload)
        return {"ok": False, "error": "Unsupported action.", "action": action}

    def _status(self):
        return {
            "ok": True,
            "known_faces": len(self.known_labels),
            "labels": self.known_labels,
            "snapshots_dir": str(VISION_SNAPSHOTS_DIR),
            "capabilities": {
                "yolo": bool(self.yolo_model),
                "tesseract": bool(VISION_ENABLE_TESSERACT and pytesseract is not None),
                "amcrest_ptz": bool(AMCREST_ENABLE_PTZ and AMCREST_HTTP_BASE),
                "emeet_zoom": bool(EMEET_ENABLE_PTZ and shutil.which("v4l2-ctl")),
                "web_fallback": bool(VISION_WEB_FALLBACK),
            },
        }

    def _browser_profile(self, user_id: Optional[int]) -> Dict:
        if not user_id:
            return {
                "browser_camera_enabled": 0,
                "browser_camera_permission_state": "prompt",
                "browser_camera_device_id": "",
                "browser_camera_label": "",
            }
        getter = getattr(self.store, "get_browser_camera_profile", None)
        if callable(getter):
            try:
                profile = getter(int(user_id))
            except Exception:
                profile = None
            if isinstance(profile, dict):
                return {
                    "browser_camera_enabled": int(bool(profile.get("browser_camera_enabled"))),
                    "browser_camera_permission_state": profile.get("browser_camera_permission_state") or "prompt",
                    "browser_camera_device_id": profile.get("browser_camera_device_id") or "",
                    "browser_camera_label": profile.get("browser_camera_label") or "",
                }
        return {
            "browser_camera_enabled": 0,
            "browser_camera_permission_state": "prompt",
            "browser_camera_device_id": "",
            "browser_camera_label": "",
        }

    def _is_admin_user(self, user_id: Optional[int]) -> bool:
        if not user_id:
            return False
        try:
            user = self.store.get_user_by_id(int(user_id))
        except Exception:
            return False
        if not user:
            return False
        email = self._record_value(user, "email").lower()
        return email in {item.strip().lower() for item in ADMIN_EMAILS}

    def _record_value(self, record: Any, key: str, default: str = "") -> str:
        if record is None:
            return default
        if isinstance(record, dict):
            value = record.get(key)
        else:
            getter = getattr(record, "get", None)
            if callable(getter):
                value = getter(key)
            else:
                try:
                    value = record[key]
                except Exception:
                    value = None
        if value is None:
            return default
        return str(value).strip()

    def _coerce_user_record(self, user: Any) -> Optional[Dict]:
        if not user:
            return None
        if isinstance(user, dict):
            return dict(user)
        keys = getattr(user, "keys", None)
        if callable(keys):
            try:
                return {str(key): user[key] for key in user.keys()}
            except Exception:
                return None
        return None

    def _enrich_user_record(self, user: Optional[Dict]) -> Optional[Dict]:
        user = self._coerce_user_record(user)
        if not user:
            return None
        user_id = self._record_value(user, "id") or self._record_value(user, "user_id")
        if user_id:
            getter = getattr(self.store, "get_user_admin_detail", None)
            if callable(getter):
                try:
                    detailed = getter(int(user_id))
                    detailed_user = self._coerce_user_record(detailed)
                    if detailed_user:
                        merged = dict(user)
                        merged.update(detailed_user)
                        return merged
                except Exception:
                    pass
        return user

    def _payload_is_admin(self, payload: Optional[Dict]) -> bool:
        payload = payload or {}
        if payload.get("is_admin") is True:
            return True
        email = (payload.get("email") or "").strip().lower()
        if email and email in {item.strip().lower() for item in ADMIN_EMAILS}:
            return True
        return self._is_admin_user(payload.get("user_id"))

    def _build_browser_descriptor(self, user_id: Optional[int], profile: Optional[Dict] = None) -> Optional[CameraDescriptor]:
        if not user_id:
            return None
        profile = profile or self._browser_profile(user_id)
        if not (
            bool(profile.get("browser_camera_enabled"))
            and (profile.get("browser_camera_permission_state") or "prompt") == "granted"
        ):
            return None
        return CameraDescriptor(
            camera_id="browser",
            label=profile.get("browser_camera_label") or "Browser Camera",
            source_kind="browser",
            source_ref=profile.get("browser_camera_device_id") or "browser",
            browser=True,
            available=profile.get("browser_camera_permission_state") != "denied",
            owner_user_id=user_id,
            permission_state=profile.get("browser_camera_permission_state") or "prompt",
            device_id=profile.get("browser_camera_device_id") or "",
        )

    def _payload_browser_profile(self, payload: Optional[Dict], user_id: Optional[int]) -> Dict:
        profile = dict(self._browser_profile(user_id))
        payload = payload or {}
        permission_state = str(payload.get("browser_camera_permission_state") or "").strip().lower()
        device_id = str(payload.get("browser_camera_device_id") or "").strip()
        label = str(payload.get("browser_camera_label") or "").strip()
        if payload.get("image_base64") or payload.get("image_data_url"):
            profile["browser_camera_enabled"] = 1
            profile["browser_camera_permission_state"] = permission_state or "granted"
            if device_id:
                profile["browser_camera_device_id"] = device_id
            if label:
                profile["browser_camera_label"] = label
        else:
            if permission_state in {"granted", "prompt", "denied"}:
                profile["browser_camera_permission_state"] = permission_state
            if device_id:
                profile["browser_camera_enabled"] = 1
                profile["browser_camera_device_id"] = device_id
            if label:
                profile["browser_camera_enabled"] = 1
                profile["browser_camera_label"] = label
        return profile

    def _build_camera_descriptors(
        self,
        user_id: Optional[int],
        *,
        admin: bool = False,
        include_server_cameras: bool = True,
        profile: Optional[Dict] = None,
    ) -> List[CameraDescriptor]:
        profile = profile or self._browser_profile(user_id)
        cameras: List[CameraDescriptor] = []
        if admin and include_server_cameras:
            cameras.append(
                CameraDescriptor(
                    camera_id="emeet",
                    label=EMEET_CAMERA_NAME,
                    source_kind="usb",
                    source_ref=str(EMEET_CAMERA_INDEX),
                    zoom=bool(EMEET_ENABLE_PTZ),
                )
            )
            if AMCREST_RTSP_URL or AMCREST_HTTP_BASE:
                cameras.append(
                    CameraDescriptor(
                        camera_id="amcrest",
                        label=AMCREST_CAMERA_NAME,
                        source_kind="rtsp",
                        source_ref=AMCREST_RTSP_URL or AMCREST_HTTP_BASE,
                        ptz=bool(AMCREST_ENABLE_PTZ and AMCREST_HTTP_BASE),
                        zoom=bool(AMCREST_ENABLE_PTZ and AMCREST_HTTP_BASE),
                    )
                )
        browser_descriptor = self._build_browser_descriptor(user_id, profile=profile)
        if browser_descriptor is not None:
            cameras.append(browser_descriptor)
        return cameras

    def _default_camera_descriptor(self, cameras: List[CameraDescriptor], profile: Optional[Dict] = None) -> Optional[CameraDescriptor]:
        profile = profile or {}
        if profile.get("browser_camera_enabled"):
            browser = next((item for item in cameras if item.camera_id == "browser"), None)
            if browser is not None:
                return browser
        for preferred_id in ("emeet", "amcrest", "browser"):
            preferred = next((item for item in cameras if item.camera_id == preferred_id), None)
            if preferred is not None:
                return preferred
        return cameras[0] if cameras else None

    def _normalize_intent(self, payload: Dict, default: str = "what_is_this") -> str:
        requested = str(payload.get("intent") or "").strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "inspect_target": "inspect_target",
            "look_at": "what_is_this",
            "inspect": "what_is_this",
            "inspect_object": "what_is_this",
            "what_is_this": "what_is_this",
            "read_text": "read_text",
            "remember_this": "remember_this",
            "help_me_troubleshoot_this": "troubleshoot_this",
            "troubleshoot_this": "troubleshoot_this",
            "troubleshoot": "troubleshoot_this",
            "task_guidance": "troubleshoot_this",
            "coaching": "troubleshoot_this",
            "who_do_you_see": "who_do_you_see",
            "faces": "who_do_you_see",
        }
        return aliases.get(requested, default)

    def _intent_payload(self, intent_id: str) -> Dict:
        normalized = self._normalize_intent({"intent": intent_id}, default="what_is_this")
        return {"id": normalized, "label": VISION_INTENT_LABELS.get(normalized, VISION_INTENT_LABELS["what_is_this"])}

    def _status_payload(
        self,
        camera: Optional[CameraDescriptor],
        *,
        browser_profile: Optional[Dict] = None,
        activity: str = "idle",
        result_kind: str = "status",
        message: str = "",
        last_task_label: str = "",
        access_mode: str = "",
    ) -> Dict:
        profile = browser_profile or {}
        permission_state = profile.get("browser_camera_permission_state") or "prompt"
        enabled = bool(profile.get("browser_camera_enabled"))
        return {
            "selected_camera_id": camera.camera_id if camera else "",
            "selected_camera_label": camera.label if camera else "",
            "selected_source_kind": camera.source_kind if camera else "",
            "browser_camera_enabled": enabled,
            "browser_camera_permission_state": permission_state,
            "browser_camera_authorized": permission_state == "granted",
            "browser_camera_active": bool(camera and camera.camera_id == "browser" and enabled and permission_state == "granted"),
            "activity": activity,
            "result_kind": result_kind,
            "last_task_label": last_task_label,
            "camera_access_mode": access_mode,
            "message": message,
        }

    def _text_excerpt(self, text: str, limit: int = 160) -> str:
        collapsed = " ".join((text or "").split())
        if len(collapsed) <= limit:
            return collapsed
        return collapsed[: max(limit - 1, 0)].rstrip() + "…"

    def _parse_utc_timestamp(self, value: str) -> Optional[datetime]:
        raw = (value or "").strip()
        if not raw:
            return None
        try:
            if raw.endswith("Z"):
                raw = raw[:-1] + "+00:00"
            parsed = datetime.fromisoformat(raw)
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except ValueError:
            return None

    def _build_source_list(self, descriptors: List[CameraDescriptor]) -> List[Dict]:
        sources: List[Dict] = []
        for descriptor in descriptors:
            sources.append(
                {
                    "camera_id": descriptor.camera_id,
                    "camera_key": self._camera_key(descriptor),
                    "label": descriptor.label,
                    "source_kind": descriptor.source_kind,
                }
            )
        return sources

    def _latest_user_perception_state(self, user_id: Optional[int], allowed_camera_keys: set) -> Optional[Dict]:
        if not user_id:
            return None
        newest_state = None
        newest_seen_at = None
        prefix = f"{int(user_id)}:"
        for source_key, state in self.visual_perception_states.items():
            if not source_key.startswith(prefix):
                continue
            active_source = state.get("active_visual_source") or {}
            camera_key = active_source.get("camera_key") or active_source.get("camera_id") or ""
            if camera_key not in allowed_camera_keys:
                continue
            seen_at = self._parse_utc_timestamp(state.get("last_observation_at") or "")
            if seen_at is None:
                continue
            if newest_seen_at is None or seen_at > newest_seen_at:
                newest_seen_at = seen_at
                newest_state = state
        return newest_state

    def _user_audio_settings(self, user_id: Optional[int]) -> Dict:
        if not user_id or not hasattr(self.store, "get_user_settings"):
            return {}
        settings = self.store.get_user_settings(int(user_id)) or {}
        result = dict(settings)
        result.setdefault("browser_audio_enabled", 1)
        result.setdefault("browser_mic_enabled", 1)
        result.setdefault("local_audio_enabled", 1)
        result.setdefault("local_mic_enabled", 1)
        return result

    def _build_audio_source_list(self, user_id: Optional[int]) -> List[Dict]:
        if not user_id:
            return []
        settings = self._user_audio_settings(user_id)
        sources: List[Dict] = []
        if bool(settings.get("browser_audio_enabled", 1)) and bool(settings.get("browser_mic_enabled", 1)):
            sources.append(
                {
                    "source_id": f"browser:{int(user_id)}",
                    "label": "Browser microphone",
                    "source_kind": "browser",
                    "owner_user_id": int(user_id),
                }
            )
        if bool(settings.get("local_audio_enabled", 1)) and bool(settings.get("local_mic_enabled", 1)):
            sources.append(
                {
                    "source_id": f"local:{int(user_id)}",
                    "label": "Local microphone",
                    "source_kind": "local",
                    "owner_user_id": int(user_id),
                }
            )
        return sources

    def _build_visual_awareness(self, payload: Dict) -> Dict:
        user_id = payload.get("user_id")
        profile = self._browser_profile(user_id)
        descriptors = self._build_camera_descriptors(
            user_id,
            admin=self._payload_is_admin(payload),
            include_server_cameras=False,
        )
        available_sources = self._build_source_list(descriptors)
        allowed_camera_keys = {item["camera_key"] for item in available_sources}
        latest_state = self._latest_user_perception_state(user_id, allowed_camera_keys)

        active_source = dict((latest_state or {}).get("active_visual_source") or {})
        default_camera = self._default_camera_descriptor(descriptors, profile=profile)
        default_camera_key = self._camera_key(default_camera) if default_camera is not None else ""
        if not active_source and default_camera is not None:
            active_source = {
                "camera_id": default_camera.camera_id,
                "camera_key": self._camera_key(default_camera),
                "label": default_camera.label,
                "source_kind": default_camera.source_kind,
            }
        active_camera_key = (active_source.get("camera_key") or active_source.get("camera_id") or "").strip()
        if default_camera_key and active_camera_key:
            source_alignment = "aligned" if default_camera_key == active_camera_key else "mismatch"
        elif default_camera_key and not active_camera_key:
            source_alignment = "default_only"
        elif active_camera_key and not default_camera_key:
            source_alignment = "state_only"
        else:
            source_alignment = "none"

        seen_at_text = (latest_state or {}).get("last_observation_at") or ""
        frame_diagnostics = dict((latest_state or {}).get("frame_diagnostics") or {})
        fresh_threshold = float(payload.get("freshness_threshold_seconds") or VISION_CONVERSATIONAL_FRESHNESS_SECONDS)
        freshness = classify_freshness(
            source_available=bool(default_camera is not None),
            last_updated_at=seen_at_text,
            fresh_threshold_seconds=fresh_threshold,
            recent_window_seconds=max(fresh_threshold * VISUAL_RECENT_ACTIVITY_MULTIPLIER, fresh_threshold),
            unavailable_reason="no_authorized_visual_source",
            missing_reason="missing_recent_perception_state",
            fresh_reason="fresh_perception_state_available",
            recent_reason="recent_perception_state_available",
            stale_reason="perception_state_stale",
        )

        if freshness["reason"] == "no_authorized_visual_source":
            response_mode = "no_live_view"
        elif freshness["reason"] == "missing_recent_perception_state":
            response_mode = "no_recent_perception"
        elif freshness["freshness"] == "fresh":
            response_mode = "live_view"
        else:
            response_mode = "stale_view"
        if freshness["reason"] in {"missing_recent_perception_state", "perception_state_stale"} and default_camera is not None:
            logging.info(
                "Visual awareness not fresh user_id=%s reason=%s age=%s threshold=%s default_camera_key=%s active_camera_key=%s source_alignment=%s last_event=%s detection_ran=%s person_present=%s",
                user_id,
                freshness["reason"],
                freshness["age_seconds"],
                freshness["freshness_threshold_seconds"],
                default_camera_key,
                active_camera_key,
                source_alignment,
                frame_diagnostics.get("event_kind") or "",
                bool(frame_diagnostics.get("detection_ran")),
                bool(frame_diagnostics.get("person_present")),
            )

        scene = (latest_state or {}).get("scene") or {}
        attention = (latest_state or {}).get("attention") or {}
        attended_subject = attention.get("attended_subject") or {}
        memory = (latest_state or {}).get("memory") or {}
        subject_presentation = self._subject_presentation(
            attended_subject,
            target_region_active=bool(attention.get("target_region_active")),
        )
        data_origin = "live_perception_state" if latest_state else ("camera_profile_fallback" if active_source else "no_visual_source")

        return {
            "live_source_active": bool(default_camera is not None),
            "source_available": bool(default_camera is not None),
            "source_active": bool(default_camera is not None),
            "availability": freshness["availability"],
            "freshness": freshness["freshness"],
            "freshness_reason": freshness["reason"],
            "active_visual_source": active_source or None,
            "available_sources": available_sources,
            "last_updated_at": seen_at_text,
            "last_observation_at": seen_at_text,
            "last_observation_age_seconds": freshness["age_seconds"],
            "freshness_threshold_seconds": freshness["freshness_threshold_seconds"],
            "recent_window_seconds": freshness["recent_window_seconds"],
            "perception_is_fresh": bool(freshness["is_fresh"]),
            "has_recent_perception": latest_state is not None and freshness["age_seconds"] is not None,
            "person_present_in_frame": bool(scene.get("person_present")),
            "scene_summary": scene.get("scene_summary") or "",
            "attended_subject_label": (attended_subject.get("label") or ""),
            "attended_subject_presented_label": subject_presentation["label"],
            "attended_subject_label_source": (attended_subject.get("label_source") or ""),
            "attended_subject_label_downgraded": bool(subject_presentation["downgraded"]),
            "attended_subject_presentation_reason": subject_presentation["reason"],
            "attended_subject_entity_type": (attended_subject.get("entity_type") or ""),
            "attended_subject_present_in_frame": bool(attended_subject.get("present_in_frame")),
            "target_region_active": bool(attention.get("target_region_active")),
            "attention_anchor_source": attention.get("attention_anchor_source") or "",
            "continuity_state": attention.get("continuity_state") or "",
            "latest_event": memory.get("latest_event") or "",
            "response_mode": response_mode,
            "reason": freshness["reason"],
            "data_origin": data_origin,
            "freshness_diagnostics": {
                "selected_source_camera_key": default_camera_key,
                "active_source_camera_key": active_camera_key,
                "source_alignment": source_alignment,
                "last_frame_event_kind": (frame_diagnostics.get("event_kind") or ""),
                "detection_ran": bool(frame_diagnostics.get("detection_ran")),
                "face_count": frame_diagnostics.get("face_count"),
                "object_count": frame_diagnostics.get("object_count"),
                "person_present_from_detection": bool(frame_diagnostics.get("person_present")),
                "frame_state_origin": frame_diagnostics.get("source_path") or "",
            },
        }

    def _build_audio_awareness(self, payload: Dict) -> Dict:
        user_id = payload.get("user_id")
        available_sources = self._build_audio_source_list(user_id)
        runtime_state = load_audio_runtime_state(user_id)
        active_source = dict(runtime_state.get("active_audio_source") or {})
        if not active_source and available_sources:
            active_source = dict(available_sources[0])
        fresh_threshold = float(payload.get("audio_freshness_threshold_seconds") or AUDIO_FRESHNESS_SECONDS)
        recent_window = float(payload.get("audio_recent_window_seconds") or AUDIO_RECENT_ACTIVITY_SECONDS)
        last_audio_event_at = (runtime_state.get("last_audio_event_at") or "").strip()
        freshness = classify_freshness(
            source_available=bool(available_sources),
            last_updated_at=last_audio_event_at,
            fresh_threshold_seconds=fresh_threshold,
            recent_window_seconds=max(recent_window, fresh_threshold),
            unavailable_reason="no_authorized_audio_source",
            missing_reason="missing_recent_audio_state",
            fresh_reason="fresh_audio_state_available",
            recent_reason="recent_audio_state_available",
            stale_reason="audio_state_stale",
        )
        recent_speech_detected = bool(runtime_state.get("last_speech_at"))
        if freshness["freshness"] == "fresh" and runtime_state.get("audio_state") in {"voice", "wake_heard"}:
            summary = "recent speech detected"
        elif freshness["freshness"] == "fresh":
            summary = "listening"
        elif freshness["freshness"] == "recently_active":
            summary = "recent audio input"
        elif freshness["freshness"] == "stale":
            summary = "no recent audio input"
        elif freshness["freshness"] == "inactive":
            summary = "audio source available without recent input"
        else:
            summary = "hearing unavailable"
        data_origin = "live_audio_runtime" if last_audio_event_at else ("audio_source_profile_fallback" if available_sources else "no_audio_source")

        return {
            "source_available": bool(available_sources),
            "source_active": bool(available_sources),
            "availability": freshness["availability"],
            "freshness": freshness["freshness"],
            "freshness_reason": freshness["reason"],
            "active_audio_source": active_source or None,
            "available_sources": available_sources,
            "last_updated_at": last_audio_event_at,
            "last_audio_event_at": last_audio_event_at,
            "last_transcript_at": (runtime_state.get("last_transcript_at") or "").strip(),
            "last_speech_at": (runtime_state.get("last_speech_at") or "").strip(),
            "last_audio_event_age_seconds": freshness["age_seconds"],
            "freshness_threshold_seconds": freshness["freshness_threshold_seconds"],
            "recent_window_seconds": freshness["recent_window_seconds"],
            "audio_state": (runtime_state.get("audio_state") or "idle") if available_sources else "unavailable",
            "recent_audio_activity": bool(freshness["is_recently_active"]),
            "recent_speech_detected": recent_speech_detected and freshness["freshness"] in {"fresh", "recently_active"},
            "summary": summary,
            "last_event_kind": (runtime_state.get("last_event_kind") or "").strip(),
            "last_transcript_type": (runtime_state.get("last_transcript_type") or "").strip(),
            "wake_word_detected": bool(runtime_state.get("wake_word_detected", False)),
            "data_origin": data_origin,
        }

    def _build_live_sensory_awareness(self, payload: Dict) -> Dict:
        visual = self._build_visual_awareness(payload)
        audio = self._build_audio_awareness(payload)
        senses = {"visual": visual, "audio": audio}
        active_senses = [name for name, state in senses.items() if state.get("freshness") in {"fresh", "recently_active"}]
        available_senses = [name for name, state in senses.items() if state.get("source_available")]
        if available_senses and all(senses[name].get("freshness") == "fresh" for name in available_senses):
            overall_freshness = "fresh"
        elif active_senses:
            overall_freshness = "mixed"
        elif available_senses:
            overall_freshness = "stale"
        else:
            overall_freshness = "unavailable"

        return {
            "senses": senses,
            "active_senses": active_senses,
            "available_senses": available_senses,
            "last_updated_at": max_timestamp_text([visual.get("last_updated_at"), audio.get("last_updated_at")]),
            "overall_freshness": overall_freshness,
            "overall_status_summary": f"active senses: {', '.join(active_senses) if active_senses else 'none'}",
            "diagnostics": {
                "active_senses": active_senses,
                "available_senses": available_senses,
                "sense_freshness": {name: state.get("freshness") for name, state in senses.items()},
                "sense_reasons": {name: state.get("freshness_reason") for name, state in senses.items()},
                "sense_ages_seconds": {
                    "visual": visual.get("last_observation_age_seconds"),
                    "audio": audio.get("last_audio_event_age_seconds"),
                },
                "source_labels": {
                    "visual": ((visual.get("active_visual_source") or {}).get("label") or ""),
                    "audio": ((audio.get("active_audio_source") or {}).get("label") or ""),
                },
                "source_ids": {
                    "visual": ((visual.get("active_visual_source") or {}).get("camera_key") or ""),
                    "audio": ((audio.get("active_audio_source") or {}).get("source_id") or ""),
                },
                "data_origin": {name: state.get("data_origin") for name, state in senses.items()},
                "visual_freshness_diagnostics": visual.get("freshness_diagnostics") or {},
            },
        }

    def conversation_vision_state(self, payload: Dict) -> Dict:
        return {
            "ok": True,
            "action": "conversation_vision_state",
            "vision_awareness": self._build_visual_awareness(payload),
        }

    def live_sensory_awareness(self, payload: Dict) -> Dict:
        return {
            "ok": True,
            "action": "live_sensory_awareness",
            "live_sensory_awareness": self._build_live_sensory_awareness(payload),
        }

    def preview_frame_heartbeat(self, payload: Dict) -> Dict:
        action = "preview_frame_heartbeat"
        try:
            camera = self._resolve_camera(payload)
        except CameraAccessError as exc:
            return self._forbidden_camera_response(action, exc)
        except Exception as exc:
            return {
                "ok": False,
                "action": action,
                "error": str(exc),
                "message": str(exc),
            }

        try:
            frame = self._capture_frame(camera, payload)
        except Exception as exc:
            logging.warning(
                "Preview heartbeat frame capture failed user_id=%s camera_id=%s error=%s",
                payload.get("user_id"),
                camera.camera_id,
                exc,
            )
            return {
                "ok": False,
                "action": action,
                "camera_id": camera.camera_id,
                "error": str(exc),
                "message": str(exc),
            }

        run_detection = bool(payload.get("run_detection", True))
        faces: List[Dict] = []
        objects: List[Dict] = []
        if run_detection:
            try:
                faces = self._detect_faces(frame)
            except Exception as exc:
                logging.exception("Preview heartbeat face detection failed: %s", exc)
                faces = []
            try:
                objects = self._detect_objects(frame)
            except Exception as exc:
                logging.exception("Preview heartbeat object detection failed: %s", exc)
                objects = []
        primary = self._preview_primary_detection(faces, objects)
        scene_entities = self._visible_scene_entities(objects, faces, frame=frame, primary=primary)
        person_present = any(
            (entity.get("entity_type") or "").strip().lower() == "face"
            or (entity.get("label") or "").strip().lower() in GENERIC_PERSON_LABELS
            for entity in scene_entities
        )
        perception = self._build_visual_perception_state(
            payload,
            camera,
            intent_id="preview_frame",
            workflow_mode="preview",
            target_region=None,
            scene_entities=scene_entities,
            primary=primary,
            frame_diagnostics={
                "event_kind": (payload.get("event_kind") or "preview_frame").strip() or "preview_frame",
                "source_path": "hud_preview_stream",
                "detection_ran": run_detection,
                "face_count": len(faces),
                "object_count": len(objects),
                "person_present": person_present,
            },
        )
        awareness = self._build_visual_awareness(payload)
        freshness_details = awareness.get("freshness_diagnostics") or {}
        logging.info(
            "Preview heartbeat updated user_id=%s camera_key=%s freshness=%s reason=%s person_present=%s source_alignment=%s",
            payload.get("user_id"),
            (perception.get("active_visual_source") or {}).get("camera_key") or self._camera_key(camera),
            awareness.get("freshness"),
            awareness.get("freshness_reason"),
            bool(awareness.get("person_present_in_frame")),
            freshness_details.get("source_alignment") or "",
        )
        return {
            "ok": True,
            "action": action,
            "camera": {
                "camera_id": camera.camera_id,
                "label": camera.label,
                "source_kind": camera.source_kind,
            },
            "vision_awareness": awareness,
        }

    def _forbidden_camera_response(self, action: str, error: CameraAccessError) -> Dict:
        return {
            "ok": False,
            "action": action,
            "camera_id": error.camera_id or "",
            "error_code": "forbidden_camera",
            "status_code": 403,
            "error": str(error),
            "message": str(error),
        }

    def _resolve_camera(self, payload: Dict, for_ptz: bool = False) -> CameraDescriptor:
        user_id = payload.get("user_id")
        requested = (payload.get("camera_id") or "").strip().lower()
        profile = self._payload_browser_profile(payload, user_id)
        candidates = self._build_camera_descriptors(user_id, admin=self._payload_is_admin(payload), profile=profile)
        browser_descriptor = next((item for item in candidates if item.camera_id == "browser"), None)

        if requested:
            for descriptor in candidates:
                if descriptor.camera_id == requested:
                    return descriptor
            raise CameraAccessError(
                f"You are not authorized to access camera '{requested}'.",
                camera_id=requested,
            )

        if payload.get("image_base64") or payload.get("image_data_url"):
            if browser_descriptor is not None:
                return browser_descriptor
            return CameraDescriptor(
                camera_id="browser",
                label=profile.get("browser_camera_label") or "Browser Camera",
                source_kind="browser",
                source_ref=profile.get("browser_camera_device_id") or "browser",
                browser=True,
                available=True,
                owner_user_id=int(user_id) if user_id else None,
                permission_state=profile.get("browser_camera_permission_state") or "granted",
                device_id=profile.get("browser_camera_device_id") or "",
            )

        if for_ptz:
            for descriptor in candidates:
                if descriptor.ptz:
                    return descriptor
            raise CameraAccessError(
                "You are not authorized to use PTZ controls on any available camera.",
                camera_id=requested or "",
            )

        for preferred_id in ("emeet", "amcrest", "browser"):
            for descriptor in candidates:
                if descriptor.camera_id == preferred_id:
                    return descriptor
        if not candidates:
            raise CameraAccessError("No authorized cameras are available for this user.")
        return candidates[0]

    def _camera_key(self, descriptor: CameraDescriptor) -> str:
        if descriptor.camera_id == "browser" and descriptor.owner_user_id:
            return f"browser:{descriptor.owner_user_id}"
        return descriptor.camera_id

    def _persist_camera(self, descriptor: CameraDescriptor) -> int:
        upsert = getattr(self.store, "upsert_camera_device", None)
        if not callable(upsert):
            return 0
        return upsert(
            camera_key=self._camera_key(descriptor),
            label=descriptor.label,
            source_kind=descriptor.source_kind,
            owner_user_id=descriptor.owner_user_id,
            source_ref=descriptor.source_ref,
            capabilities={
                "ptz": descriptor.ptz,
                "zoom": descriptor.zoom,
                "browser": descriptor.browser,
                "permission_state": descriptor.permission_state,
                "device_id": descriptor.device_id,
            },
            is_available=descriptor.available,
        )

    def _store_vision_observation(self, **kwargs) -> int:
        writer = getattr(self.store, "add_vision_observation", None)
        if not callable(writer):
            return 0
        try:
            return int(writer(**kwargs) or 0)
        except Exception:
            logging.exception("Failed storing vision observation")
            return 0

    def _list_cameras(self, payload: Dict) -> Dict:
        user_id = payload.get("user_id")
        profile = self._payload_browser_profile(payload, user_id)
        is_admin = self._payload_is_admin(payload)
        descriptors = self._build_camera_descriptors(
            user_id,
            admin=is_admin,
            include_server_cameras=False,
            profile=profile,
        )
        default_camera = self._default_camera_descriptor(descriptors, profile=profile)
        cameras = []
        for descriptor in descriptors:
            self._persist_camera(descriptor)
            cameras.append(
                {
                    "camera_id": descriptor.camera_id,
                    "label": descriptor.label,
                    "source_kind": descriptor.source_kind,
                    "source_ref": descriptor.source_ref,
                    "available": descriptor.available,
                    "capabilities": {
                        "ptz": descriptor.ptz,
                        "zoom": descriptor.zoom,
                        "browser": descriptor.browser,
                        "permission_state": descriptor.permission_state,
                        "device_id": descriptor.device_id,
                    },
                }
            )
        access_mode = "admin" if is_admin else "personal"
        return {
            "ok": True,
            "cameras": cameras,
            "browser_profile": profile,
            "status": self._status_payload(
                default_camera,
                browser_profile=profile,
                activity="ready",
                result_kind="camera_list",
                message=(
                    f"{len(cameras)} camera source(s) available for {access_mode} access."
                    if cameras
                    else "No browser camera is available yet. Request browser camera permission to continue."
                ),
                last_task_label="Camera selection updated",
                access_mode=access_mode,
            ),
        }

    def _capture_from_source(self, source):
        for attempt in range(FRAME_CAPTURE_MAX_RETRIES):
            cap = cv2.VideoCapture(source)
            try:
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
                if not cap.isOpened():
                    frame = None
                else:
                    ok, frame = cap.read()
                    if not ok:
                        frame = None
                if self._frame_is_usable(frame):
                    return frame
            finally:
                cap.release()
            if attempt + 1 < FRAME_CAPTURE_MAX_RETRIES:
                time.sleep(FRAME_CAPTURE_RETRY_DELAY_SECONDS * (attempt + 1))
        return None

    def _frame_is_usable(self, frame) -> bool:
        shape = getattr(frame, "shape", None)
        if frame is None or not shape or len(shape) < 2:
            return False
        try:
            height = int(shape[0])
            width = int(shape[1])
        except (TypeError, ValueError):
            return False
        return height >= FRAME_CAPTURE_MIN_DIMENSION and width >= FRAME_CAPTURE_MIN_DIMENSION

    def _decode_browser_frame(self, payload: Dict):
        raw = payload.get("image_base64") or payload.get("image_data_url") or ""
        if not raw:
            raise ValueError("Browser camera image is required for browser inspection.")
        if "," in raw:
            raw = raw.split(",", 1)[1]
        last_error = None
        for attempt in range(FRAME_CAPTURE_MAX_RETRIES):
            try:
                buffer = np.frombuffer(base64.b64decode(raw), dtype=np.uint8)
                frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
            except Exception as exc:
                frame = None
                last_error = exc
            if self._frame_is_usable(frame):
                return frame
            if attempt + 1 < FRAME_CAPTURE_MAX_RETRIES:
                time.sleep(FRAME_CAPTURE_RETRY_DELAY_SECONDS * (attempt + 1))
        if last_error:
            raise ValueError(f"Unable to decode browser camera image: {last_error}")
        raise ValueError("Unable to decode browser camera image.")

    def _capture_frame(self, descriptor: CameraDescriptor, payload: Dict):
        if descriptor.camera_id == "browser":
            return self._decode_browser_frame(payload)
        if descriptor.camera_id == "amcrest":
            frame = self._capture_from_source(AMCREST_RTSP_URL)
            if frame is None:
                raise RuntimeError("Unable to capture frame from Amcrest camera.")
            return frame
        frame = self._capture_from_source(EMEET_CAMERA_INDEX)
        if frame is None and EMEET_V4L2_DEVICE:
            frame = self._capture_from_source(EMEET_V4L2_DEVICE)
        if frame is None:
            raise RuntimeError("Unable to capture frame from EMEET camera.")
        return frame

    def _save_snapshot(self, frame, camera_key: str) -> str:
        snapshot_path = VISION_SNAPSHOTS_DIR / f"{camera_key.replace(':', '_')}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        cv2.imwrite(str(snapshot_path), frame)
        return str(snapshot_path)

    def _profile_login_label(self, user: Dict) -> str:
        email = self._record_value(user, "email").lower()
        login = email.split("@", 1)[0].strip() if "@" in email else email
        if login:
            return login
        user_id = self._record_value(user, "id") or self._record_value(user, "user_id")
        return str(user_id).strip() if user_id is not None else ""

    def _profile_login_aliases(self, user: Dict) -> List[str]:
        login = self._profile_login_label(user)
        aliases: List[str] = []
        if login:
            aliases.append(login)
            without_numeric_suffix = re.sub(r"\d+$", "", login)
            if without_numeric_suffix and without_numeric_suffix != login:
                aliases.append(without_numeric_suffix)
        return aliases

    def _identity_label_for_user(self, user: Dict, fallback_label: str = "") -> Tuple[str, str]:
        preferred_name = self._record_value(user, "preferred_name")
        if preferred_name:
            return preferred_name, "profile_preferred_name"
        display_name = self._record_value(user, "display_name")
        if display_name:
            return display_name, "profile_display_name"
        first_name = self._record_value(user, "first_name")
        last_name = self._record_value(user, "last_name")
        full_name = " ".join(part for part in [first_name, last_name] if part).strip()
        if full_name:
            return full_name, "profile_full_name"
        login_aliases = self._profile_login_aliases(user)
        login_label = login_aliases[0] if login_aliases else ""
        if login_label:
            return login_label, "login"
        return (fallback_label or "").strip(), "known_face_label"

    def _identity_fields_for_user(self, user: Dict, fallback_label: str = "") -> Dict:
        first_name = self._record_value(user, "first_name")
        last_name = self._record_value(user, "last_name")
        full_name = " ".join(part for part in [first_name, last_name] if part).strip()
        preferred_display_name = self._record_value(user, "preferred_name") or self._record_value(user, "display_name")
        login_aliases = self._profile_login_aliases(user)
        login_label = login_aliases[0] if login_aliases else ""
        presented_label, presented_label_source = self._identity_label_for_user(user, fallback_label=fallback_label)
        return {
            "preferred_display_name": preferred_display_name,
            "full_name": full_name,
            "first_name": first_name,
            "last_name": last_name,
            "login_label": login_label,
            "presented_label": presented_label,
            "presented_label_source": presented_label_source,
        }

    def _matched_user_for_face_label(self, label: str) -> Optional[Dict]:
        clean_label = (label or "").strip()
        if not clean_label or clean_label.lower() in {"unknown", "unrecognized"}:
            return None
        list_users = getattr(self.store, "list_users", None)
        if not callable(list_users):
            return None
        try:
            users = list_users() or []
        except Exception:
            return None
        target_key = self._name_lookup_key(clean_label)
        if not target_key:
            return None
        for raw_user in users:
            user = self._enrich_user_record(raw_user)
            if not user:
                continue
            first_name = self._record_value(user, "first_name")
            last_name = self._record_value(user, "last_name")
            full_name = " ".join(part for part in [first_name, last_name] if part).strip()
            candidates = [
                self._record_value(user, "preferred_name"),
                self._record_value(user, "display_name"),
                *self._profile_login_aliases(user),
                full_name,
                self._record_value(user, "email").split("@", 1)[0] if self._record_value(user, "email") else "",
                self._record_value(user, "id"),
                self._record_value(user, "user_id"),
            ]
            if any(self._name_lookup_key(str(candidate)) == target_key for candidate in candidates if candidate not in {None, ""}):
                return user
        return None

    def _resolve_face_identity(self, label: str) -> Dict:
        clean_label = (label or "").strip()
        if not clean_label or clean_label.lower() in {"unknown", "unrecognized"}:
            return {"label": "unrecognized", "label_source": "unrecognized", "user_id": None, "identity_aliases": []}
        user = self._matched_user_for_face_label(clean_label)
        if not user:
            return {"label": clean_label, "label_source": "known_face_label", "user_id": None, "identity_aliases": [clean_label]}
        resolved_label, label_source = self._identity_label_for_user(user, fallback_label=clean_label)
        aliases = []
        for candidate in [
            clean_label,
            self._record_value(user, "preferred_name"),
            self._record_value(user, "display_name"),
            self._record_value(user, "first_name"),
            self._record_value(user, "last_name"),
            " ".join(
                part for part in [
                    self._record_value(user, "first_name"),
                    self._record_value(user, "last_name"),
                ] if part
            ).strip(),
            *self._profile_login_aliases(user),
        ]:
            candidate = (candidate or "").strip()
            if candidate and candidate not in aliases:
                aliases.append(candidate)
        return {
            "label": resolved_label or clean_label,
            "label_source": label_source,
            "user_id": user.get("id") or user.get("user_id"),
            "identity_aliases": aliases,
        }

    def _authorized_latest_scene_entities(self, payload: Dict) -> List[Dict]:
        user_id = payload.get("user_id")
        profile = self._browser_profile(user_id)
        descriptors = self._build_camera_descriptors(
            user_id,
            admin=self._payload_is_admin(payload),
            include_server_cameras=False,
            profile=profile,
        )
        available_sources = self._build_source_list(descriptors)
        allowed_camera_keys = {item["camera_key"] for item in available_sources}
        latest_state = self._latest_user_perception_state(user_id, allowed_camera_keys)
        return list((((latest_state or {}).get("scene") or {}).get("visible_entities") or []))

    def _get_user_record(self, user_id: Optional[int]) -> Optional[Dict]:
        if not user_id:
            return None
        getter = getattr(self.store, "get_user_by_id", None)
        if not callable(getter):
            return None
        try:
            user = getter(int(user_id))
        except Exception:
            return None
        if not user:
            return None
        return dict(user)

    def resolve_identity_label(self, payload: Dict) -> Dict:
        action = "resolve_identity_label"
        query_label = (payload.get("label") or "").strip()
        if not query_label:
            return {"ok": False, "action": action, "error": "Identity label is required."}

        target_key = self._name_lookup_key(query_label)
        if not target_key:
            return {"ok": False, "action": action, "error": "Identity label is required."}

        faces = [
            item
            for item in self._authorized_latest_scene_entities(payload)
            if (item.get("entity_type") or "").strip().lower() == "face"
        ]

        for face in faces:
            seen_label = (face.get("label") or "").strip()
            seen_label_key = self._name_lookup_key(seen_label)
            label_source = (face.get("label_source") or "").strip().lower()
            trusted_profile_sources = {"profile_preferred_name", "profile_display_name", "profile_full_name", "login"}
            trusted_face = label_source in trusted_profile_sources and bool(face.get("user_id"))
            face_user = self._get_user_record(face.get("user_id"))
            if not face_user and label_source in trusted_profile_sources:
                face_user = self._matched_user_for_face_label(seen_label)
            if face_user:
                face_user = self._enrich_user_record(face_user)
            lookup_keys = {key for key in [seen_label_key] if key}
            if face_user:
                identity = self._identity_fields_for_user(face_user, fallback_label=seen_label)
                lookup_keys.update(
                    {
                        self._name_lookup_key(identity.get("preferred_display_name") or ""),
                        self._name_lookup_key(identity.get("full_name") or ""),
                        self._name_lookup_key(identity.get("login_label") or ""),
                    }
                )
                for alias in self._profile_login_aliases(face_user):
                    lookup_keys.add(self._name_lookup_key(alias))
            if trusted_face:
                for alias in face.get("identity_aliases") or []:
                    lookup_keys.add(self._name_lookup_key(alias))
            if target_key not in {item for item in lookup_keys if item}:
                continue

            if not face_user:
                return {
                    "ok": True,
                    "action": action,
                    "query_label": query_label,
                    "seen_label": seen_label or query_label,
                    "seen_label_source": label_source,
                    "resolved": False,
                    "mapping_explicit": False,
                    "identity": {},
                }

            identity = self._identity_fields_for_user(face_user, fallback_label=seen_label or query_label)
            return {
                "ok": True,
                "action": action,
                "query_label": query_label,
                "seen_label": seen_label or query_label,
                "seen_label_source": label_source,
                "resolved": True,
                "mapping_explicit": True,
                "identity": identity,
            }

        return {
            "ok": True,
            "action": action,
            "query_label": query_label,
            "resolved": False,
            "mapping_explicit": False,
            "identity": {},
        }

    def _detect_faces(self, frame) -> List[Dict]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb, number_of_times_to_upsample=1, model="hog")
        encodings = face_recognition.face_encodings(rgb, locations)
        faces = []
        for (top, right, bottom, left), encoding in zip(locations, encodings):
            label = "unrecognized"
            confidence = 0.0
            label_source = "unrecognized"
            matched_user_id = None
            best_distance = None
            identity_aliases = []
            if self.known_encodings:
                distances = face_recognition.face_distance(self.known_encodings, encoding)
                if len(distances):
                    best_index = int(distances.argmin())
                    best_distance = float(distances[best_index])
                    confidence = max(0.0, min(1.0, 1.0 - (best_distance / FACE_MATCH_DISTANCE_THRESHOLD)))
                    if best_distance <= FACE_MATCH_DISTANCE_THRESHOLD:
                        identity = self._resolve_face_identity(self.known_labels[best_index])
                        label = identity["label"]
                        label_source = identity["label_source"]
                        matched_user_id = identity["user_id"]
                        identity_aliases = list(identity.get("identity_aliases") or [])
            faces.append(
                {
                    "label": label,
                    "entity_type": "face",
                    "confidence": round(confidence, 3),
                    "label_source": label_source,
                    "user_id": matched_user_id,
                    "identity_aliases": identity_aliases,
                    "match_distance": round(best_distance, 3) if best_distance is not None else None,
                    "box": {"top": top, "right": right, "bottom": bottom, "left": left},
                }
            )
        return faces

    def _name_lookup_key(self, value: str) -> str:
        return "".join(ch for ch in (value or "").strip().lower() if ch.isalnum())

    def _preferred_face_label(self, label: str) -> str:
        return self._resolve_face_identity(label).get("label") or (label or "").strip()

    def _detect_objects(self, frame) -> List[Dict]:
        if self.yolo_model is None:
            return []
        try:
            results = self.yolo_model(frame, verbose=False)
        except Exception as exc:
            logging.exception("YOLO inference failed: %s", exc)
            return []

        detections = []
        for result in results:
            names = getattr(result, "names", {}) or {}
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0].item()) if hasattr(box.cls[0], "item") else int(box.cls[0])
                conf = float(box.conf[0].item()) if hasattr(box.conf[0], "item") else float(box.conf[0])
                coords = box.xyxy[0].tolist() if hasattr(box.xyxy[0], "tolist") else list(box.xyxy[0])
                detections.append(
                    {
                        "label": names.get(cls_id, str(cls_id)),
                        "entity_type": "object",
                        "confidence": round(conf, 3),
                        "box": {
                            "left": int(coords[0]),
                            "top": int(coords[1]),
                            "right": int(coords[2]),
                            "bottom": int(coords[3]),
                        },
                    }
                )
        return detections

    def _run_ocr(self, frame) -> str:
        if not VISION_ENABLE_TESSERACT or pytesseract is None:
            return ""
        try:
            return pytesseract.image_to_string(frame).strip()
        except Exception as exc:
            logging.exception("OCR failed: %s", exc)
            return ""

    def _normalize_target_region(self, payload: Dict, frame) -> Optional[Dict]:
        region = payload.get("target_region")
        target_mode = infer_target_mode(
            target_mode=payload.get("target_mode"),
            target_point=payload.get("target_point"),
            target_region=region if isinstance(region, dict) else None,
        )
        if target_mode == "point" and not isinstance(region, dict) and isinstance(payload.get("target_point"), dict):
            region = target_region_from_point(payload.get("target_point") or {})
        if not isinstance(region, dict):
            return None
        coordinate_space = coordinate_space_for_payload(region)
        if not coordinate_space:
            return None
        try:
            x_raw = region.get("x")
            y_raw = region.get("y")
            width_raw = region.get("width")
            height_raw = region.get("height")
        except (TypeError, ValueError):
            return None
        frame_height = int(frame.shape[0]) if hasattr(frame, "shape") else 0
        frame_width = int(frame.shape[1]) if hasattr(frame, "shape") else 0
        if frame_width <= 0 or frame_height <= 0:
            return None
        x = normalize_coordinate(x_raw, max_dimension=frame_width, coordinate_space=coordinate_space)
        y = normalize_coordinate(y_raw, max_dimension=frame_height, coordinate_space=coordinate_space)
        width = normalize_coordinate(width_raw, max_dimension=frame_width, coordinate_space=coordinate_space)
        height = normalize_coordinate(height_raw, max_dimension=frame_height, coordinate_space=coordinate_space)
        if x is None or y is None or width is None or height is None:
            return None
        if width <= 0 or height <= 0:
            return None
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        if x >= 1.0 or y >= 1.0:
            return None
        width = max(0.01, min(1.0, width))
        height = max(0.01, min(1.0, height))
        if x + width > 1.0:
            width = 1.0 - x
        if y + height > 1.0:
            height = 1.0 - y
        if width <= 0.0 or height <= 0.0:
            return None
        left = int(round(x * frame_width))
        top = int(round(y * frame_height))
        right = int(round((x + width) * frame_width))
        bottom = int(round((y + height) * frame_height))
        right = max(left + 1, min(frame_width, right))
        bottom = max(top + 1, min(frame_height, bottom))
        margin_x = int(round((right - left) * TARGET_REGION_CONTEXT_MARGIN))
        margin_y = int(round((bottom - top) * TARGET_REGION_CONTEXT_MARGIN))
        analysis_left = max(0, left - margin_x)
        analysis_top = max(0, top - margin_y)
        analysis_right = min(frame_width, right + margin_x)
        analysis_bottom = min(frame_height, bottom + margin_y)
        normalized_region = {
            "x": round(x, 4),
            "y": round(y, 4),
            "width": round(width, 4),
            "height": round(height, 4),
            "coordinate_space": "normalized",
        }
        if str(region.get("mode") or "").strip().lower() == "crosshair":
            try:
                center_x = float(region.get("center_x"))
                center_y = float(region.get("center_y"))
            except (TypeError, ValueError):
                center_x = x + (width / 2.0)
                center_y = y + (height / 2.0)
            normalized_region["mode"] = "crosshair"
            normalized_region["center_x"] = round(max(0.0, min(1.0, center_x)), 4)
            normalized_region["center_y"] = round(max(0.0, min(1.0, center_y)), 4)
        return {
            "normalized": normalized_region,
            "pixels": {
                "left": left,
                "top": top,
                "right": right,
                "bottom": bottom,
            },
            "analysis_pixels": {
                "left": analysis_left,
                "top": analysis_top,
                "right": max(analysis_left + 1, analysis_right),
                "bottom": max(analysis_top + 1, analysis_bottom),
            },
        }

    def _crop_frame_to_target(self, frame, target_region: Optional[Dict]):
        if not target_region:
            return frame
        bounds = target_region.get("analysis_pixels") or target_region.get("pixels") or {}
        left = int(bounds.get("left", 0))
        top = int(bounds.get("top", 0))
        right = int(bounds.get("right", 0))
        bottom = int(bounds.get("bottom", 0))
        if right <= left or bottom <= top:
            return frame
        try:
            cropped = frame[top:bottom, left:right]
            if cropped is None:
                return frame
            cropped_shape = getattr(cropped, "shape", None)
            if not cropped_shape or int(cropped_shape[0]) <= 0 or int(cropped_shape[1]) <= 0:
                return frame
            return cropped
        except Exception:
            return frame

    def _offset_detections_to_frame(self, detections: List[Dict], target_region: Optional[Dict]) -> List[Dict]:
        if not target_region:
            return detections
        bounds = target_region.get("analysis_pixels") or target_region.get("pixels") or {}
        offset_left = int(bounds.get("left", 0))
        offset_top = int(bounds.get("top", 0))
        adjusted = []
        for item in detections or []:
            copy_item = dict(item)
            box = dict(copy_item.get("box") or {})
            if box:
                if "left" in box:
                    box["left"] = int(box.get("left", 0)) + offset_left
                if "right" in box:
                    box["right"] = int(box.get("right", 0)) + offset_left
                if "top" in box:
                    box["top"] = int(box.get("top", 0)) + offset_top
                if "bottom" in box:
                    box["bottom"] = int(box.get("bottom", 0)) + offset_top
            copy_item["box"] = box
            adjusted.append(copy_item)
        return adjusted

    def _compute_visual_signature(self, frame) -> Optional[List[float]]:
        if frame is None:
            return None
        try:
            thumb = cv2.resize(frame, (16, 16), interpolation=cv2.INTER_AREA)
            if len(getattr(thumb, "shape", ())) >= 3:
                thumb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
            vector = np.asarray(thumb, dtype=np.float32).reshape(-1)
            if vector.size <= 0:
                return None
            norm = float(np.linalg.norm(vector))
            if norm <= 0.0:
                return None
            normalized = vector / norm
            return [round(float(item), 6) for item in normalized.tolist()]
        except Exception:
            return None

    def _signature_similarity(self, left: Optional[List[float]], right: Optional[List[float]]) -> Optional[float]:
        if not left or not right:
            return None
        try:
            left_vec = np.asarray(left, dtype=np.float32).reshape(-1)
            right_vec = np.asarray(right, dtype=np.float32).reshape(-1)
            if left_vec.size == 0 or right_vec.size == 0:
                return None
            size = min(left_vec.size, right_vec.size)
            left_vec = left_vec[:size]
            right_vec = right_vec[:size]
            denom = float(np.linalg.norm(left_vec) * np.linalg.norm(right_vec))
            if denom <= 0.0:
                return None
            value = float(np.dot(left_vec, right_vec) / denom)
            return max(0.0, min(1.0, value))
        except Exception:
            return None

    def _match_known_object_by_signature(self, user_id: Optional[int], signature: Optional[List[float]]) -> Optional[Dict]:
        if not user_id or not signature:
            return None
        list_known_objects = getattr(self.store, "list_known_objects", None)
        if not callable(list_known_objects):
            return None
        try:
            known_objects = list_known_objects(int(user_id), limit=200) or []
        except Exception:
            return None
        best_match = None
        for known_object in known_objects:
            if not isinstance(known_object, dict):
                continue
            metadata = known_object.get("metadata") or {}
            stored_signature = metadata.get("visual_signature") if isinstance(metadata, dict) else None
            score = self._signature_similarity(signature, stored_signature if isinstance(stored_signature, list) else None)
            if score is None:
                continue
            if not best_match or score > float(best_match.get("confidence") or 0.0):
                best_match = {
                    "known_object_id": known_object.get("id"),
                    "label": (known_object.get("label") or "").strip(),
                    "confidence": round(score, 3),
                }
        if best_match and float(best_match.get("confidence") or 0.0) >= 0.9 and best_match.get("label"):
            return best_match
        return None

    def _box_area(self, box: Dict) -> int:
        left = int(box.get("left", 0))
        right = int(box.get("right", 0))
        top = int(box.get("top", 0))
        bottom = int(box.get("bottom", 0))
        return max(0, right - left) * max(0, bottom - top)

    def _target_intersection_area(self, detection_box: Dict, target_box: Dict) -> int:
        left = max(int(detection_box.get("left", 0)), int(target_box.get("left", 0)))
        top = max(int(detection_box.get("top", 0)), int(target_box.get("top", 0)))
        right = min(int(detection_box.get("right", 0)), int(target_box.get("right", 0)))
        bottom = min(int(detection_box.get("bottom", 0)), int(target_box.get("bottom", 0)))
        return max(0, right - left) * max(0, bottom - top)

    def _target_detection_metrics(self, detection: Dict, target_region: Optional[Dict], frame) -> Dict:
        metrics = {
            "in_target": False,
            "center_in_target": False,
            "overlap_ratio": 0.0,
            "target_coverage": 0.0,
            "center_proximity": 0.0,
            "inside_area_ratio": 0.0,
            "usable_target_evidence": False,
        }
        if not target_region:
            return metrics
        target_box = target_region.get("pixels") or {}
        box = detection.get("box") or {}
        if not target_box or not box:
            return metrics
        detection_area = self._box_area(box)
        target_area = self._box_area(target_box)
        if detection_area <= 0 or target_area <= 0:
            return metrics
        overlap_area = self._target_intersection_area(box, target_box)
        overlap_ratio = overlap_area / float(detection_area)
        target_coverage = overlap_area / float(target_area)
        center_x = (int(box.get("left", 0)) + int(box.get("right", 0))) / 2.0
        center_y = (int(box.get("top", 0)) + int(box.get("bottom", 0))) / 2.0
        center_in_target = (
            center_x >= int(target_box.get("left", 0))
            and center_x <= int(target_box.get("right", 0))
            and center_y >= int(target_box.get("top", 0))
            and center_y <= int(target_box.get("bottom", 0))
        )
        target_center_x = (int(target_box.get("left", 0)) + int(target_box.get("right", 0))) / 2.0
        target_center_y = (int(target_box.get("top", 0)) + int(target_box.get("bottom", 0))) / 2.0
        half_width = max((int(target_box.get("right", 0)) - int(target_box.get("left", 0))) / 2.0, 1.0)
        half_height = max((int(target_box.get("bottom", 0)) - int(target_box.get("top", 0))) / 2.0, 1.0)
        distance = (((center_x - target_center_x) / half_width) ** 2 + ((center_y - target_center_y) / half_height) ** 2) ** 0.5
        center_proximity = max(0.0, 1.0 - min(distance, 1.0))
        frame_shape = getattr(frame, "shape", None)
        frame_area = (
            max(int(frame_shape[0]), 1) * max(int(frame_shape[1]), 1)
            if frame_shape and len(frame_shape) >= 2
            else 1
        )
        inside_area_ratio = overlap_area / float(frame_area)
        confidence = float(detection.get("confidence") or 0.0)
        in_target = center_in_target or overlap_ratio >= TARGET_PRIORITY_MIN_OVERLAP or target_coverage >= TARGET_PRIORITY_MIN_OVERLAP
        usable_target_evidence = in_target and (
            confidence >= TARGET_PRIORITY_MIN_CONFIDENCE
            or target_coverage >= 0.25
            or overlap_ratio >= 0.4
            or center_proximity >= 0.65
        )
        metrics.update(
            {
                "in_target": bool(in_target),
                "center_in_target": bool(center_in_target),
                "overlap_ratio": float(overlap_ratio),
                "target_coverage": float(target_coverage),
                "center_proximity": float(center_proximity),
                "inside_area_ratio": float(inside_area_ratio),
                "usable_target_evidence": bool(usable_target_evidence),
            }
        )
        return metrics

    def _target_detection_score(self, detection: Dict, target_region: Optional[Dict], frame) -> float:
        confidence = float(detection.get("confidence") or 0.0)
        metrics = self._target_detection_metrics(detection, target_region, frame)
        face_label = (detection.get("label") or "").strip().lower()
        face_recognized = (
            (detection.get("entity_type") or "").strip().lower() == "face"
            and face_label not in {"", "unknown", "unrecognized"}
        )
        return (
            confidence * 1.2
            + (0.6 if metrics["center_in_target"] else 0.0)
            + (metrics["overlap_ratio"] * 0.5)
            + (metrics["target_coverage"] * 0.35)
            + (metrics["center_proximity"] * 0.25)
            + min(metrics["inside_area_ratio"] * 8.0, 0.2)
            + (0.05 if face_recognized else 0.0)
        )

    def _choose_primary_detection(
        self,
        objects: List[Dict],
        faces: List[Dict],
        target_region: Optional[Dict] = None,
        frame=None,
    ) -> Dict:
        best_object = max(objects, key=lambda item: float(item.get("confidence") or 0.0)) if objects else None
        best_face = max(faces, key=lambda item: float(item.get("confidence") or 0.0)) if faces else None
        if target_region:
            ranked = []
            for item in objects or []:
                metrics = self._target_detection_metrics(item, target_region, frame)
                ranked.append((item, metrics, self._target_detection_score(item, target_region, frame)))
            for item in faces or []:
                metrics = self._target_detection_metrics(item, target_region, frame)
                ranked.append((item, metrics, self._target_detection_score(item, target_region, frame)))
            if ranked:
                usable_target = [candidate for candidate in ranked if candidate[1]["usable_target_evidence"]]
                in_target = [candidate for candidate in ranked if candidate[1]["in_target"]]
                if usable_target or in_target:
                    pool = usable_target or in_target
                    winner, winner_metrics, _score = max(
                        pool,
                        key=lambda candidate: (
                            candidate[2],
                            float(candidate[0].get("confidence") or 0.0),
                        ),
                    )
                    winner_source = winner.get("source")
                    if not winner_source:
                        is_face = (winner.get("entity_type") or "").strip().lower() == "face"
                        label = (winner.get("label") or "").strip().lower()
                        winner_source = "face_match" if is_face and label not in {"", "unknown", "unrecognized"} else ("face_detection" if is_face else "local_detection")
                    return {
                        **winner,
                        "source": winner_source,
                        "target_metrics": {
                            "in_target": winner_metrics["in_target"],
                            "center_in_target": winner_metrics["center_in_target"],
                            "overlap_ratio": round(winner_metrics["overlap_ratio"], 3),
                            "target_coverage": round(winner_metrics["target_coverage"], 3),
                            "center_proximity": round(winner_metrics["center_proximity"], 3),
                            "inside_area_ratio": round(winner_metrics["inside_area_ratio"], 4),
                            "usable_target_evidence": winner_metrics["usable_target_evidence"],
                        },
                    }
                return {
                    "label": "",
                    "confidence": 0.0,
                    "box": dict((target_region or {}).get("pixels") or {}),
                    "source": "target_region",
                    "entity_type": "object",
                    "target_metrics": {
                        "in_target": False,
                        "center_in_target": False,
                        "overlap_ratio": 0.0,
                        "target_coverage": 0.0,
                        "center_proximity": 0.0,
                        "inside_area_ratio": 0.0,
                        "usable_target_evidence": False,
                    },
                }
        if best_face:
            face_label = (best_face.get("label") or "").strip().lower()
            face_recognized = face_label not in {"", "unknown", "unrecognized"}
            face_confidence = float(best_face.get("confidence") or 0.0)
            if best_object is None:
                return {**best_face, "source": "face_match" if face_recognized else "face_detection"}
            object_label = (best_object.get("label") or "").strip().lower()
            object_confidence = float(best_object.get("confidence") or 0.0)
            if face_recognized and (object_label in GENERIC_PERSON_LABELS or face_confidence >= object_confidence + 0.05):
                return {**best_face, "source": "face_match"}
        if best_object:
            return {**best_object, "source": best_object.get("source") or "local_detection"}
        if best_face:
            face_label = (best_face.get("label") or "").strip().lower()
            face_recognized = face_label not in {"", "unknown", "unrecognized"}
            return {**best_face, "source": "face_match" if face_recognized else "face_detection"}
        return {"label": "", "confidence": 0.0, "box": {}}

    def _detection_area_ratio(self, detection: Dict, frame) -> float:
        box = detection.get("box") or {}
        width = max(int(box.get("right", 0)) - int(box.get("left", 0)), 0)
        height = max(int(box.get("bottom", 0)) - int(box.get("top", 0)), 0)
        return (width * height) / max(frame.shape[0] * frame.shape[1], 1)

    def _suggest_fallback(self, payload: Dict, primary: Dict, ocr_text: str) -> Optional[Dict]:
        if not (VISION_WEB_FALLBACK or payload.get("force_web_fallback")):
            return None
        confidence = float(primary.get("confidence") or 0.0)
        if confidence >= VISION_LOW_CONFIDENCE_THRESHOLD and not payload.get("force_web_fallback"):
            return None
        query = (
            payload.get("label_hint")
            or primary.get("label")
            or " ".join((ocr_text or "").split()[:8])
            or "identify object from camera"
        )
        return {
            "triggered": True,
            "reason": "low_confidence_local_result" if confidence < VISION_LOW_CONFIDENCE_THRESHOLD else "user_requested",
            "query": query,
            "url": build_web_search_url(query),
            "confidence": confidence,
        }

    def _format_inspection_summary(
        self,
        camera: CameraDescriptor,
        faces: List[Dict],
        objects: List[Dict],
        ocr_text: str,
        fallback: Optional[Dict],
        intent_id: str = "what_is_this",
        primary: Optional[Dict] = None,
        target_region: Optional[Dict] = None,
        workflow_mode: str = "inspect",
        perception: Optional[Dict] = None,
    ) -> str:
        primary = primary or self._choose_primary_detection(objects, faces, target_region=target_region)
        primary_label = primary.get("label") or "unknown"
        primary_confidence = float(primary.get("confidence") or 0.0)
        primary_entity_type = (primary.get("entity_type") or "object").strip().lower()
        ocr_excerpt = self._text_excerpt(ocr_text, limit=180)
        embodied_summary = self._embodied_observation_summary(
            camera,
            perception,
            workflow_mode=workflow_mode,
            target_region_used=bool(target_region),
        )

        if intent_id == "read_text":
            if ocr_excerpt:
                summary = f"I read this from {camera.label}: {ocr_excerpt}."
            else:
                summary = f"I looked at {camera.label}, but I could not read any clear text yet."
            if objects:
                summary += f" I also noticed {primary_label}."
            if fallback and fallback.get("url"):
                summary += " I can also open a web fallback if you want a second opinion."
            return summary

        if intent_id == "remember_this":
            if objects:
                summary = f"I looked at {camera.label} and this seems to be {primary_label} ({primary_confidence:.2f})."
            else:
                summary = f"I captured a reference view from {camera.label}."
            if ocr_excerpt:
                summary += f" Visible text: {ocr_excerpt}."
            summary += " Add or confirm a label to save it as a known object."
            return summary

        if intent_id == "troubleshoot_this":
            parts = [f"I checked {camera.label} for troubleshooting."]
            if objects:
                parts.append(f"The clearest visual match is {primary_label} ({primary_confidence:.2f}).")
            else:
                parts.append("I did not get a confident object label.")
            if ocr_excerpt:
                parts.append(f"Readable text: {ocr_excerpt}.")
            if faces:
                parts.append("Faces in view: " + ", ".join(face.get("label") or "unrecognized" for face in faces) + ".")
            if fallback and fallback.get("url"):
                parts.append("A web fallback is available if you want broader troubleshooting context.")
            return " ".join(parts)

        if intent_id == "inspect_target":
            if embodied_summary:
                if ocr_excerpt:
                    return f"{embodied_summary} Visible text: {ocr_excerpt}."
                return embodied_summary
            if not primary_label or primary_confidence < VISION_LOW_CONFIDENCE_THRESHOLD:
                person_in_frame = any((face.get("label") or "").strip().lower() not in {"", "unknown", "unrecognized"} for face in faces)
                if target_region:
                    if person_in_frame:
                        return (
                            "I’m tracking the selected region, but I’m not getting a clear object there yet. "
                            "I can see a person elsewhere in the frame, but that is not the selected target."
                        )
                    return (
                        "I’m tracking the selected region, but I’m not getting a clear object there yet. "
                        "Please center the object in the selected region and move a little closer."
                    )
                return (
                    f"I do not know what this object is yet from {camera.label}. "
                    "Please tell me what it is so I can remember it for future recognition."
                )
            parts = [f"Target inspection complete for {camera.label}."]
            if primary_entity_type == "face":
                parts.append(f"Recognized person: {primary_label} ({primary_confidence:.2f}).")
            else:
                parts.append(f"Top object: {primary_label} ({primary_confidence:.2f}).")
            if ocr_excerpt:
                parts.append(f"Visible text: {ocr_excerpt}.")
            return " ".join(parts)

        if embodied_summary and workflow_mode in {"assist", "troubleshoot"}:
            if ocr_excerpt:
                return f"{embodied_summary} Visible text: {ocr_excerpt}."
            return embodied_summary
        parts = [f"Inspection complete for {camera.label}."]
        if objects:
            parts.append(f"Top object: {primary_label} ({primary_confidence:.2f}).")
        elif primary_entity_type == "face" and primary_label not in {"", "unknown", "unrecognized"}:
            parts.append(f"Recognized person: {primary_label} ({primary_confidence:.2f}).")
        else:
            parts.append("No confident local object label was available.")
        if faces:
            parts.append("Faces: " + ", ".join(face.get("label") or "unrecognized" for face in faces) + ".")
        if ocr_excerpt:
            parts.append(f"OCR excerpt: {ocr_excerpt}.")
        if fallback and fallback.get("url"):
            parts.append("Web fallback is available for a second opinion.")
        return " ".join(parts)

    def _assist_certainty_state(self, label: str, confidence: float, teaching_needed: bool) -> str:
        normalized_label = (label or "").strip().lower()
        if not teaching_needed and normalized_label and confidence >= VISION_LOW_CONFIDENCE_THRESHOLD:
            return "recognized"
        if normalized_label and normalized_label not in {"unknown", "unrecognized"} and confidence > 0:
            return "uncertain"
        return "unknown"

    def _detection_identity(self, detection: Optional[Dict]) -> Tuple[str, str, int, int, int, int]:
        detection = detection or {}
        box = detection.get("box") or {}
        return (
            (detection.get("entity_type") or "object").strip().lower(),
            (detection.get("label") or "").strip().lower(),
            int(box.get("left", 0) or 0),
            int(box.get("top", 0) or 0),
            int(box.get("right", 0) or 0),
            int(box.get("bottom", 0) or 0),
        )

    def _default_detection_source(self, detection: Dict) -> str:
        is_face = (detection.get("entity_type") or "").strip().lower() == "face"
        label = (detection.get("label") or "").strip().lower()
        if is_face and label not in {"", "unknown", "unrecognized"}:
            return "face_match"
        if is_face:
            return "face_detection"
        return "local_detection"

    def _rounded_target_metrics(self, detection: Dict, target_region: Optional[Dict], frame) -> Dict:
        metrics = self._target_detection_metrics(detection, target_region, frame)
        return {
            "in_target": bool(metrics["in_target"]),
            "center_in_target": bool(metrics["center_in_target"]),
            "overlap_ratio": round(float(metrics["overlap_ratio"]), 3),
            "target_coverage": round(float(metrics["target_coverage"]), 3),
            "center_proximity": round(float(metrics["center_proximity"]), 3),
            "inside_area_ratio": round(float(metrics["inside_area_ratio"]), 4),
            "usable_target_evidence": bool(metrics["usable_target_evidence"]),
        }

    def _visible_scene_entities(
        self,
        objects: List[Dict],
        faces: List[Dict],
        *,
        target_region: Optional[Dict] = None,
        frame=None,
        primary: Optional[Dict] = None,
    ) -> List[Dict]:
        primary_identity = self._detection_identity(primary)
        entities: List[Dict] = []
        for detection in list(objects or []) + list(faces or []):
            entity = dict(detection)
            entity["source"] = entity.get("source") or self._default_detection_source(entity)
            entity["target_metrics"] = self._rounded_target_metrics(entity, target_region, frame)
            entity["attended"] = bool(primary and self._detection_identity(entity) == primary_identity)
            entities.append(entity)
        entities.sort(key=lambda item: (not item.get("attended"), -float(item.get("confidence") or 0.0)))
        return entities

    def _scene_summary(self, scene_entities: List[Dict]) -> str:
        if not scene_entities:
            return ""
        labels: List[str] = []
        seen = set()
        for entity in scene_entities:
            entity_type = (entity.get("entity_type") or "").strip().lower()
            label = (entity.get("label") or "").strip()
            label_source = (entity.get("label_source") or "").strip().lower()
            if entity_type == "face" and (label_source == "known_face_label" or label.lower() in {"", "unknown", "unrecognized"}):
                label = "a person"
            if not label:
                label = "person" if entity_type == "face" else "object"
            key = label.lower()
            if key not in seen:
                labels.append(label)
                seen.add(key)
            if len(labels) >= 4:
                break
        return ", ".join(labels)

    def _source_perception_key(self, payload: Dict, camera: CameraDescriptor) -> str:
        return f"{int(payload.get('user_id') or 0)}:{self._camera_key(camera)}"

    def _build_visual_perception_state(
        self,
        payload: Dict,
        camera: CameraDescriptor,
        *,
        intent_id: str,
        workflow_mode: str,
        target_region: Optional[Dict],
        scene_entities: List[Dict],
        primary: Dict,
        frame_diagnostics: Optional[Dict] = None,
    ) -> Dict:
        source_key = self._source_perception_key(payload, camera)
        prior_state = self.visual_perception_states.get(source_key) or {}
        explicit_session_id = str(payload.get("assist_session_id") or "").strip()
        session_token = explicit_session_id or str(prior_state.get("session_token") or f"{workflow_mode}:{source_key}")
        normalized_action = str(payload.get("workflow_action") or "").strip().lower().replace("-", "_")
        session_changed = bool(prior_state and explicit_session_id and explicit_session_id != prior_state.get("session_token"))
        reset_requested = normalized_action == "start_over"
        if session_changed or reset_requested:
            prior_state = {}
        seen_at = datetime.now(timezone.utc).isoformat()
        prior_memory = prior_state.get("memory") or {}
        prior_attention = prior_state.get("attention") or {}
        prior_subject = prior_attention.get("attended_subject") or {}
        prior_identity = (
            prior_subject.get("entity_type", ""),
            prior_subject.get("label", "").strip().lower(),
            prior_subject.get("source", ""),
        )
        person_present = any(
            (entity.get("entity_type") or "").strip().lower() == "face"
            or (entity.get("label") or "").strip().lower() in GENERIC_PERSON_LABELS
            for entity in scene_entities
        )
        current_visible = bool((primary.get("label") or "").strip() or float(primary.get("confidence") or 0.0) > 0.0)
        current_subject = {}
        if current_visible:
            current_subject = {
                "label": (primary.get("label") or "").strip(),
                "confidence": round(float(primary.get("confidence") or 0.0), 3),
                "entity_type": (primary.get("entity_type") or "object").strip().lower(),
                "source": primary.get("source") or self._default_detection_source(primary),
                "label_source": primary.get("label_source") or "",
                "present_in_frame": True,
                "box": dict(primary.get("box") or {}),
            }
            if target_region:
                current_subject["target_metrics"] = dict(primary.get("target_metrics") or {})
        latest_event = "scene_initialized"
        continuity_state = "observing" if current_visible else "searching"
        if current_visible:
            current_identity = (
                current_subject.get("entity_type", ""),
                current_subject.get("label", "").strip().lower(),
                current_subject.get("source", ""),
            )
            if prior_identity == current_identity and current_identity[1]:
                if (prior_memory.get("latest_event") or "") == "target_lost":
                    latest_event = "target_reacquired"
                    continuity_state = "reacquired"
                else:
                    latest_event = "attention_stable"
                    continuity_state = "stable"
            elif prior_identity[1] and current_identity[1]:
                latest_event = "attention_shifted"
                continuity_state = "shifted"
            else:
                latest_event = "attention_acquired"
                continuity_state = "acquired"
        elif prior_subject and (target_region or workflow_mode in {"assist", "troubleshoot"}):
            current_subject = {
                **prior_subject,
                "present_in_frame": False,
            }
            latest_event = "target_lost"
            continuity_state = "lost"
        recent_observations = list(prior_memory.get("recent_observations") or [])
        recent_events = list(prior_memory.get("recent_events") or [])
        recent_attended_subjects = list(prior_memory.get("recent_attended_subjects") or [])
        recent_observations.append(
            {
                "seen_at": seen_at,
                "workflow_mode": workflow_mode,
                "intent_id": intent_id,
                "camera_id": camera.camera_id,
                "attended_label": current_subject.get("label") or "",
                "attended_entity_type": current_subject.get("entity_type") or "",
                "attended_visible": bool(current_subject.get("present_in_frame")),
                "person_present": bool(person_present),
                "event": latest_event,
            }
        )
        recent_events.append({"event": latest_event, "seen_at": seen_at})
        if current_subject.get("label"):
            current_label = current_subject["label"]
            if not recent_attended_subjects or recent_attended_subjects[-1] != current_label:
                recent_attended_subjects.append(current_label)
        visible_types = sorted(
            {
                (entity.get("entity_type") or "object").strip().lower()
                for entity in scene_entities
                if (entity.get("label") or "").strip() or float(entity.get("confidence") or 0.0) > 0.0
            }
        )
        state = {
            "active_visual_source": {
                "camera_id": camera.camera_id,
                "label": camera.label,
                "source_kind": camera.source_kind,
                "camera_key": self._camera_key(camera),
            },
            "visually_engaged": True,
            "has_live_input": True,
            "last_observation_at": seen_at,
            "workflow_mode": workflow_mode,
            "intent_id": intent_id,
            "session_token": session_token,
            "scene": {
                "visible_entities": scene_entities,
                "visible_subject_count": len(scene_entities),
                "visible_subject_types": visible_types,
                "person_present": bool(person_present),
                "scene_summary": self._scene_summary(scene_entities),
            },
            "attention": {
                "target_region_active": bool(target_region),
                "attention_anchor_source": "target_region" if target_region else "scene",
                "target_region": target_region["normalized"] if target_region else None,
                "attended_subject": current_subject,
                "continuity_state": continuity_state,
                "reason": (
                    "target_region_focus"
                    if target_region
                    else ("sustained_visual_attention" if workflow_mode in {"assist", "troubleshoot"} else "scene_observation")
                ),
            },
            "memory": {
                "recent_observations": recent_observations[-VISION_PERCEPTION_HISTORY_LIMIT:],
                "recent_events": recent_events[-VISION_PERCEPTION_HISTORY_LIMIT:],
                "recent_attended_subjects": recent_attended_subjects[-VISION_PERCEPTION_HISTORY_LIMIT:],
                "latest_event": latest_event,
                "reset_on_session_change": True,
            },
            "frame_diagnostics": dict(frame_diagnostics or {}),
        }
        self.visual_perception_states[source_key] = state
        return state

    def _preview_primary_detection(self, faces: List[Dict], objects: List[Dict]) -> Dict:
        if faces:
            return max(faces, key=lambda item: float(item.get("confidence") or 0.0))
        if objects:
            return max(objects, key=lambda item: float(item.get("confidence") or 0.0))
        return {}

    def _neutral_subject_label(self, subject: Optional[Dict], *, target_region_active: bool) -> str:
        entity_type = ((subject or {}).get("entity_type") or "object").strip().lower()
        if entity_type == "face":
            return "the person" if target_region_active else "a person"
        return "the object" if target_region_active else "something unclear"

    def _subject_presentation(self, subject: Optional[Dict], *, target_region_active: bool) -> Dict:
        subject = subject or {}
        label = (subject.get("label") or "").strip()
        label_key = label.lower()
        entity_type = (subject.get("entity_type") or "object").strip().lower()
        label_source = (subject.get("label_source") or "").strip().lower()
        confidence = float(subject.get("confidence") or 0.0)

        if entity_type == "face":
            if label_key in {"", "unknown", "unrecognized"}:
                return {
                    "label": self._neutral_subject_label(subject, target_region_active=target_region_active),
                    "downgraded": True,
                    "reason": "unrecognized_face",
                }
            if target_region_active and label_source in {"login", "known_face_label"}:
                return {
                    "label": self._neutral_subject_label(subject, target_region_active=target_region_active),
                    "downgraded": True,
                    "reason": "weak_face_identity_source",
                }
            return {
                "label": label,
                "downgraded": False,
                "reason": "recognized_face",
            }

        if label_source in {"profile_preferred_name", "profile_display_name", "profile_full_name", "login", "known_face_label"}:
            return {
                "label": self._neutral_subject_label(subject, target_region_active=target_region_active),
                "downgraded": True,
                "reason": "identity_label_not_object_label",
            }
        if label_key in {"", "unknown", "unrecognized"}:
            return {
                "label": self._neutral_subject_label(subject, target_region_active=target_region_active),
                "downgraded": True,
                "reason": "unknown_object_label",
            }
        if confidence < VISION_LOW_CONFIDENCE_THRESHOLD:
            return {
                "label": "an unclear object" if target_region_active else "something unclear",
                "downgraded": True,
                "reason": "low_confidence_object_label",
            }
        return {
            "label": label,
            "downgraded": False,
            "reason": "recognized_object",
        }

    def _embodied_focus_label(self, subject: Optional[Dict], *, target_region_active: bool = False) -> str:
        presentation = self._subject_presentation(subject, target_region_active=target_region_active)
        return (presentation.get("label") or "").strip() or "the target"

    def _embodied_observation_summary(
        self,
        camera: CameraDescriptor,
        perception: Optional[Dict],
        *,
        workflow_mode: str,
        target_region_used: bool,
    ) -> str:
        perception = perception or {}
        attention = perception.get("attention") or {}
        subject = attention.get("attended_subject") or {}
        latest_event = (perception.get("memory") or {}).get("latest_event") or ""
        person_present = bool((perception.get("scene") or {}).get("person_present"))
        focus_label = self._embodied_focus_label(subject, target_region_active=bool(target_region_used))
        subject_visible = bool(subject.get("present_in_frame", False))
        subject_entity_type = (subject.get("entity_type") or "object").strip().lower()

        if latest_event == "target_lost" and (target_region_used or workflow_mode in {"assist", "troubleshoot"}):
            if person_present:
                return "I lost sight of the selected region for a moment, but I can still see a person in frame."
            return "I lost sight of the selected region for a moment."
        if (
            target_region_used
            and person_present
            and subject_visible
            and subject_entity_type != "face"
            and focus_label != "the target"
        ):
            return f"I can see you in frame, but my focus is on {focus_label} inside the selected region."
        if latest_event == "target_reacquired" and subject_visible and focus_label != "the target":
            return f"I’m looking through {camera.label} right now. I found {focus_label} again."
        if latest_event == "attention_stable" and subject_visible and focus_label != "the target" and workflow_mode in {"assist", "troubleshoot"}:
            return f"I’m still watching {focus_label}."
        if target_region_used and subject_visible:
            return f"My focus is on {focus_label} inside the selected region."
        return ""

    def _assist_observation_payload(
        self,
        camera: CameraDescriptor,
        label: str,
        confidence: float,
        teaching_needed: bool,
        target_region_used: bool,
        summary: str,
        source: str,
        entity_type: str,
        reason: str,
        diagnostics: Optional[Dict] = None,
        workflow_mode: str = "assist",
        step_number: int = 0,
        ocr_text: str = "",
        workflow_context: Optional[Dict] = None,
        workflow_action: str = "",
        perception: Optional[Dict] = None,
    ) -> Dict:
        certainty = self._assist_certainty_state(label, confidence, teaching_needed)
        normalized_label = (label or "").strip() or "the target"
        embodied_summary = self._embodied_observation_summary(
            camera,
            perception,
            workflow_mode=workflow_mode,
            target_region_used=bool(target_region_used),
        )
        if embodied_summary:
            observation_summary = embodied_summary
        elif target_region_used and certainty == "uncertain":
            observation_summary = f"I’m tracking the selected region. It may be {normalized_label}, but I need a closer view to confirm."
        elif target_region_used and certainty == "unknown":
            observation_summary = "I’m tracking the selected region, but classification is still unclear."
        elif certainty == "recognized":
            observation_summary = f"{normalized_label} appears to be in view."
        elif certainty == "uncertain":
            observation_summary = f"I am not fully sure, but this may be {normalized_label}."
        else:
            observation_summary = "I cannot confidently identify the target yet."
        latest_event = ((perception or {}).get("memory") or {}).get("latest_event") or ""
        if latest_event == "target_lost":
            narration = f"I’m looking through {camera.label} right now. {observation_summary}"
        else:
            narration = (
                f"{observation_summary} Confidence {confidence:.2f}."
                if certainty == "recognized"
                else f"{observation_summary} Confidence is low at {confidence:.2f}."
            )
            if workflow_mode in {"assist", "troubleshoot"}:
                narration = f"I’m looking through {camera.label} right now. {narration}"
        needs_clarification = certainty != "recognized"
        guidance = self._build_workflow_guidance(
            workflow_mode=workflow_mode,
            step_number=step_number,
            label=label,
            confidence=confidence,
            certainty=certainty,
            ocr_text=ocr_text,
            summary=summary,
            workflow_context=workflow_context,
            workflow_action=workflow_action,
            target_region_used=target_region_used,
        )
        return {
            "summary": observation_summary,
            "narration": narration,
            "certainty": certainty,
            "confidence": round(confidence, 3),
            "target_region_used": bool(target_region_used),
            "needs_clarification": bool(needs_clarification or guidance["needs_clarification"]),
            "reason": reason,
            "source": source,
            "entity_type": entity_type,
            "clarification_prompt": (
                guidance["clarification_prompt"]
                if needs_clarification or guidance["needs_clarification"]
                else ""
            ),
            "camera_label": camera.label,
            "inspection_summary": summary,
            "diagnostics": diagnostics or {},
            "workflow_mode": guidance["workflow_mode"],
            "step_number": guidance["step_number"],
            "identified_target": guidance["identified_target"],
            "visible_text": guidance["visible_text"],
            "issue_summary": guidance["issue_summary"],
            "current_step": guidance["current_step"],
            "latest_recommendation": guidance["latest_recommendation"],
            "next_step": guidance["next_step"],
            "previously_recommended_steps": guidance["previously_recommended_steps"],
            "attempted_steps": guidance["attempted_steps"],
            "completed_steps": guidance["completed_steps"],
            "skipped_steps": guidance["skipped_steps"],
            "failed_steps": guidance["failed_steps"],
            "last_user_update": guidance["last_user_update"],
            "step_status": guidance["step_status"],
            "safety_note": guidance["safety_note"],
            "session_summary": guidance["session_summary"],
            "handoff_summary": guidance["handoff_summary"],
            "workflow_state": guidance["workflow_state"],
        }

    def _workflow_mode(self, payload: Dict, intent_id: str) -> str:
        raw_mode = str(payload.get("workflow_mode") or payload.get("mode") or "").strip().lower().replace("-", "_")
        if raw_mode in {"troubleshoot", "troubleshoot_this", "task_guidance", "coaching"}:
            return "troubleshoot"
        if intent_id == "troubleshoot_this":
            return "troubleshoot"
        if payload.get("assist_mode") or raw_mode == "assist":
            return "assist"
        return "inspect"

    def _workflow_step_number(self, payload: Dict, workflow_mode: str) -> int:
        raw_step = payload.get("step_number")
        if raw_step in (None, ""):
            workflow_context = payload.get("workflow_context")
            if isinstance(workflow_context, dict):
                raw_step = workflow_context.get("step_number")
        if raw_step in (None, ""):
            raw_step = payload.get("assist_tick")
        if raw_step in (None, ""):
            return 1 if workflow_mode == "troubleshoot" else 0
        try:
            return max(1 if workflow_mode == "troubleshoot" else 0, int(raw_step))
        except (TypeError, ValueError):
            return 1 if workflow_mode == "troubleshoot" else 0

    def _workflow_step_list(self, steps) -> List[str]:
        if not isinstance(steps, list):
            return []
        normalized: List[str] = []
        seen = set()
        for step in steps:
            text = self._text_excerpt(str(step or "").strip(), limit=220)
            key = text.lower()
            if text and key not in seen:
                normalized.append(text)
                seen.add(key)
        return normalized

    def _workflow_append_step(self, steps: List[str], step: str) -> List[str]:
        text = self._text_excerpt(step, limit=220)
        if not text:
            return list(steps or [])
        existing = self._workflow_step_list(steps)
        if text.lower() in {item.lower() for item in existing}:
            return existing
        return existing + [text]

    def _workflow_infer_category(self, identified_target: str, visible_text: str) -> str:
        combined = f"{identified_target} {visible_text}".lower()
        if re.search(r"\b(router|modem|network|ethernet|wifi|wi-fi|wan|lan|port|link)\b", combined):
            return "network"
        if re.search(r"\b(printer|toner|paper|tray|cartridge|print)\b", combined):
            return "printer"
        if re.search(r"\b(monitor|display|screen|hdmi|vga|input source|resolution)\b", combined):
            return "display"
        if re.search(r"\b(battery|power|charging|charger|voltage|outlet)\b", combined):
            return "power"
        if re.search(r"\b(camera|webcam|usb|bluetooth|device connection|adapter)\b", combined):
            return "connection"
        return "generic"

    def _workflow_template_steps(self, category: str, lowered_text: str) -> List[str]:
        templates = {
            "network": [
                "Check the power and link lights, then confirm the power cable is seated.",
                "Reseat the network cable on both ends or try another port, then retry once.",
                "Open the router or modem status page and confirm the WAN or link state.",
                "Restart the router or modem if the same network state remains after the cable checks.",
            ],
            "printer": [
                "Check paper, toner, and tray alignment, then retry the print job.",
                "Clear any visible jam or cover warning, then retry.",
                "Open the printer queue or status panel and confirm the printer is online.",
                "Power cycle the printer if the same printer status remains after the basic checks.",
            ],
            "display": [
                "Confirm the display power light and input source are correct.",
                "Reseat the video cable or adapter on both ends, then retry.",
                "Open the display menu and verify the selected input or resolution.",
                "Try another cable or source device if the display still shows the same problem.",
            ],
            "power": [
                "Confirm the charger or power cable is connected and the outlet is active.",
                "Inspect the battery or charging indicator, then retry after a short wait.",
                "Try another charger, cable, or power source if one is available.",
                "Power cycle the device once charging or power state is confirmed.",
            ],
            "connection": [
                "Confirm the device is powered and the connection cable is seated.",
                "Reconnect the device or try another port, then retry.",
                "Open the related settings page and confirm the device is detected.",
                "Restart the device or app if the connection state is unchanged.",
            ],
            "generic": [
                "Check the obvious connections, power state, and any visible controls, then retry once.",
                "Open the matching settings or error panel and confirm the reported status.",
                "Restart the device or task after the visible checks are complete.",
            ],
        }
        steps = list(templates.get(category, templates["generic"]))
        if re.search(r"\b(error|warning|fault|failed|failure|retry|code)\b", lowered_text):
            steps.insert(0, "Open the related warning or error control and confirm the exact status or code.")
        if re.search(r"\b(link down|offline|disconnected|unavailable)\b", lowered_text):
            steps.insert(1 if steps else 0, "Confirm the connection path is seated correctly and retry after checking the visible status.")
        return self._workflow_step_list(steps)

    def _build_workflow_guidance(
        self,
        *,
        workflow_mode: str,
        step_number: int,
        label: str,
        confidence: float,
        certainty: str,
        ocr_text: str,
        summary: str,
        workflow_context: Optional[Dict] = None,
        workflow_action: str = "",
        target_region_used: bool = False,
    ) -> Dict:
        resolved_mode = (workflow_mode or "assist").strip().lower()
        normalized_label = (label or "").strip()
        visible_text = self._text_excerpt(ocr_text, limit=220)
        lowered_text = visible_text.lower()
        action = str(workflow_action or "").strip().lower().replace("-", "_")
        prior_context = workflow_context if isinstance(workflow_context, dict) else {}
        base_step_number = max(1, int(step_number or 1))
        identified_target = normalized_label if normalized_label else self._text_excerpt(prior_context.get("identified_target") or "", limit=160)
        issue_summary = self._text_excerpt(summary, limit=180) if summary else self._text_excerpt(prior_context.get("issue_summary") or "", limit=180)
        latest_recommendation = self._text_excerpt(
            prior_context.get("latest_recommendation")
            or prior_context.get("current_step")
            or prior_context.get("next_step")
            or "",
            limit=220,
        )
        previous_recommended_steps = self._workflow_step_list(prior_context.get("previously_recommended_steps"))
        attempted_steps = self._workflow_step_list(prior_context.get("attempted_steps"))
        completed_steps = self._workflow_step_list(prior_context.get("completed_steps"))
        skipped_steps = self._workflow_step_list(prior_context.get("skipped_steps"))
        failed_steps = self._workflow_step_list(prior_context.get("failed_steps"))
        last_user_update = self._text_excerpt(prior_context.get("last_user_update") or "", limit=180)
        clarification_prompt = ""
        safety_note = ""
        needs_clarification = certainty != "recognized"
        step_status = str(prior_context.get("step_status") or "pending").strip().lower() or "pending"
        has_advance_action = action in {"mark_tried", "mark_failed", "skip_step", "mark_done", "next_step"}
        has_reset_action = action == "start_over"
        has_repeat_action = action == "repeat_step"
        default_next_step = "Run another quick inspection after centering the target."

        if resolved_mode != "troubleshoot":
            if needs_clarification:
                clarification_prompt = (
                    "I’m not getting a clear object inside the target box yet. Please center the object inside the box."
                    if target_region_used
                    else "Please adjust the target or tell me what object you want me to focus on."
                )
            return {
                "workflow_mode": resolved_mode,
                "step_number": step_number,
                "identified_target": identified_target,
                "visible_text": visible_text,
                "issue_summary": issue_summary,
                "current_step": latest_recommendation,
                "latest_recommendation": latest_recommendation,
                "next_step": default_next_step if needs_clarification else "",
                "previously_recommended_steps": previous_recommended_steps,
                "attempted_steps": attempted_steps,
                "completed_steps": completed_steps,
                "skipped_steps": skipped_steps,
                "failed_steps": failed_steps,
                "last_user_update": last_user_update,
                "step_status": step_status,
                "needs_clarification": needs_clarification,
                "clarification_prompt": clarification_prompt,
                "safety_note": "",
                "session_summary": "",
                "handoff_summary": "",
                "workflow_state": {
                    "workflow_mode": resolved_mode,
                    "step_number": step_number,
                    "identified_target": identified_target,
                    "visible_text": visible_text,
                    "issue_summary": issue_summary,
                    "latest_recommendation": latest_recommendation,
                    "previously_recommended_steps": previous_recommended_steps,
                    "attempted_steps": attempted_steps,
                    "completed_steps": completed_steps,
                    "skipped_steps": skipped_steps,
                    "failed_steps": failed_steps,
                    "step_status": step_status,
                    "needs_clarification": needs_clarification,
                },
            }

        has_error_signal = bool(re.search(r"\b(error|warning|fault|failed|failure|offline|disconnected|unavailable|retry)\b", lowered_text))
        has_power_signal = bool(re.search(r"\b(battery|power|charging|voltage)\b", lowered_text))
        has_heat_signal = bool(re.search(r"\b(overheat|hot|smoke|spark)\b", lowered_text))
        unchanged_visible_text = bool(visible_text and visible_text == self._text_excerpt(prior_context.get("visible_text") or "", limit=220))

        if has_reset_action:
            previous_recommended_steps = []
            attempted_steps = []
            completed_steps = []
            skipped_steps = []
            failed_steps = []
            latest_recommendation = ""
            last_user_update = "User restarted the troubleshooting session."
            step_status = "reset"

        if has_advance_action and latest_recommendation:
            previous_recommended_steps = self._workflow_append_step(previous_recommended_steps, latest_recommendation)
            if action == "mark_tried":
                attempted_steps = self._workflow_append_step(attempted_steps, latest_recommendation)
                last_user_update = "User already tried the current step."
                step_status = "attempted"
            elif action == "mark_failed":
                attempted_steps = self._workflow_append_step(attempted_steps, latest_recommendation)
                failed_steps = self._workflow_append_step(failed_steps, latest_recommendation)
                last_user_update = "User reported the current step did not work."
                step_status = "failed"
            elif action == "skip_step":
                skipped_steps = self._workflow_append_step(skipped_steps, latest_recommendation)
                last_user_update = "User skipped the current step."
                step_status = "skipped"
            elif action == "mark_done":
                completed_steps = self._workflow_append_step(completed_steps, latest_recommendation)
                last_user_update = "User marked the current step done."
                step_status = "completed"
            else:
                last_user_update = "User requested the next troubleshooting step."
                step_status = "advanced"
        elif has_repeat_action:
            last_user_update = "User asked to repeat the current troubleshooting step."
        elif action and not last_user_update:
            last_user_update = f"Workflow action: {action.replace('_', ' ')}."

        if has_error_signal:
            issue_summary = f"Visible status indicates a likely issue: {visible_text or 'warning/error text detected'}."
            needs_clarification = False
        elif visible_text:
            issue_summary = f"Visible status text: {visible_text}."
        elif normalized_label and certainty == "recognized":
            issue_summary = f"I likely see {normalized_label}, but clear status text is missing."
            needs_clarification = True
            clarification_prompt = "Please center the error panel or display and hold steady for another check."
        else:
            issue_summary = "I do not have enough visual evidence yet to recommend a precise fix."
            needs_clarification = True
            clarification_prompt = "What device, object, or error panel should I focus on?"

        if has_power_signal and not identified_target:
            identified_target = "device or power source"
        if has_heat_signal:
            safety_note = "If the device is overheating or emitting smoke/sparks, stop use and disconnect power if safe."

        if not clarification_prompt and needs_clarification:
            clarification_prompt = (
                "I’m not getting a clear object inside the target box yet. Please center the object inside the box."
                if target_region_used
                else "Please adjust the target or tell me what object you want me to focus on."
            )

        if needs_clarification:
            next_step = (
                "Please center the target, move closer, improve lighting, or tell me what object to focus on."
                if not visible_text and not identified_target
                else "Center the target panel and hold steady so I can confirm the next troubleshooting step."
            )
            current_step = latest_recommendation if has_repeat_action else next_step
            latest_recommendation = current_step
        else:
            category = self._workflow_infer_category(identified_target, visible_text)
            candidate_steps = self._workflow_template_steps(category, lowered_text)
            if has_power_signal and not any("power" in step.lower() for step in candidate_steps):
                candidate_steps.insert(0, "Confirm the device power and cable state before retrying.")
            if unchanged_visible_text and has_advance_action:
                candidate_steps = self._workflow_append_step(candidate_steps, "Escalate to a deeper device settings or hardware check because the visible status is unchanged.")

            if has_repeat_action and latest_recommendation:
                current_step = latest_recommendation
            else:
                seen_steps = {item.lower() for item in previous_recommended_steps}
                current_step = ""
                for candidate in candidate_steps:
                    if candidate.lower() not in seen_steps:
                        current_step = candidate
                        break
                if not current_step:
                    current_step = candidate_steps[-1] if candidate_steps else "Run another quick inspection after checking the visible controls."
                latest_recommendation = current_step
            next_step = latest_recommendation
            if has_power_signal and "power" not in next_step.lower():
                next_step = f"{next_step} Also confirm the device power and cable state."
                latest_recommendation = next_step
                current_step = next_step

        computed_step_number = max(1, len(previous_recommended_steps) + (1 if latest_recommendation else 0))
        step_number = computed_step_number if (has_advance_action or has_reset_action) else max(base_step_number, computed_step_number)
        attempted_text = ", ".join(attempted_steps[-3:])
        completed_text = ", ".join(completed_steps[-2:])
        skipped_text = ", ".join(skipped_steps[-2:])
        failed_text = ", ".join(failed_steps[-2:])
        summary_bits = []
        if identified_target:
            summary_bits.append(f"Target: {identified_target}.")
        if visible_text:
            summary_bits.append(f"Observed: {visible_text}.")
        elif issue_summary:
            summary_bits.append(issue_summary)
        if attempted_text:
            summary_bits.append(f"Tried: {attempted_text}.")
        if completed_text:
            summary_bits.append(f"Completed: {completed_text}.")
        if skipped_text:
            summary_bits.append(f"Skipped: {skipped_text}.")
        if failed_text:
            summary_bits.append(f"Failed: {failed_text}.")
        if latest_recommendation:
            summary_bits.append(f"Current step: {latest_recommendation}.")
        if needs_clarification and clarification_prompt:
            summary_bits.append(clarification_prompt)
        session_summary = self._text_excerpt(" ".join(summary_bits), limit=420)
        handoff_parts = []
        if identified_target:
            handoff_parts.append(f"Target in view: {identified_target}.")
        if visible_text:
            handoff_parts.append(f"Observed text/status: {visible_text}.")
        elif issue_summary:
            handoff_parts.append(f"Observed issue: {issue_summary}.")
        tried_for_handoff = attempted_steps or previous_recommended_steps
        if tried_for_handoff:
            handoff_parts.append(f"Suggested or tried: {', '.join(tried_for_handoff[-4:])}.")
        if latest_recommendation:
            handoff_parts.append(f"Recommended next: {latest_recommendation}.")
        if safety_note:
            handoff_parts.append(f"Safety: {safety_note}")
        handoff_summary = self._text_excerpt(" ".join(handoff_parts), limit=420)

        return {
            "workflow_mode": resolved_mode,
            "step_number": step_number if resolved_mode == "troubleshoot" else 0,
            "identified_target": identified_target,
            "visible_text": visible_text,
            "issue_summary": issue_summary,
            "current_step": current_step,
            "latest_recommendation": latest_recommendation,
            "next_step": latest_recommendation,
            "previously_recommended_steps": previous_recommended_steps,
            "attempted_steps": attempted_steps,
            "completed_steps": completed_steps,
            "skipped_steps": skipped_steps,
            "failed_steps": failed_steps,
            "last_user_update": last_user_update,
            "step_status": step_status,
            "needs_clarification": needs_clarification,
            "clarification_prompt": clarification_prompt,
            "safety_note": safety_note,
            "session_summary": session_summary,
            "handoff_summary": handoff_summary,
            "workflow_state": {
                "workflow_mode": resolved_mode,
                "step_number": step_number,
                "identified_target": identified_target,
                "visible_text": visible_text,
                "issue_summary": issue_summary,
                "latest_recommendation": latest_recommendation,
                "previously_recommended_steps": previous_recommended_steps,
                "attempted_steps": attempted_steps,
                "completed_steps": completed_steps,
                "skipped_steps": skipped_steps,
                "failed_steps": failed_steps,
                "last_user_update": last_user_update,
                "step_status": step_status,
                "needs_clarification": needs_clarification,
                "clarification_prompt": clarification_prompt,
                "session_summary": session_summary,
                "handoff_summary": handoff_summary,
            },
        }

    def _recognition_reason(self, primary: Dict, recognition_state: str, teaching_needed: bool) -> str:
        if recognition_state == "recognized":
            return "recognized"
        entity_type = (primary.get("entity_type") or "object").strip().lower()
        label = (primary.get("label") or "").strip().lower()
        confidence = float(primary.get("confidence") or 0.0)
        if entity_type == "face" and label in {"", "unknown", "unrecognized"}:
            return "face_unrecognized"
        if confidence <= 0.0:
            return "no_reliable_detection"
        if teaching_needed:
            return "needs_user_teaching"
        return "low_confidence_match"

    def inspect_object(self, payload: Dict) -> Dict:
        try:
            target_mode = infer_target_mode(
                target_mode=payload.get("target_mode"),
                target_point=payload.get("target_point"),
                target_region=payload.get("target_region") if isinstance(payload.get("target_region"), dict) else None,
            )
            camera = self._resolve_camera(payload)
            self._persist_camera(camera)
            intent = self._intent_payload(self._normalize_intent(payload))
            frame = self._capture_frame(camera, payload)
            target_region = self._normalize_target_region(payload, frame)
            analysis_frame = self._crop_frame_to_target(frame, target_region)
            snapshot_path = self._save_snapshot(frame, self._camera_key(camera))
            faces = self._offset_detections_to_frame(self._detect_faces(analysis_frame), target_region)
            objects = self._offset_detections_to_frame(self._detect_objects(analysis_frame), target_region)
            ocr_text = self._run_ocr(analysis_frame)
            visual_signature = self._compute_visual_signature(analysis_frame)
            local_primary = self._choose_primary_detection(objects, faces, target_region=target_region, frame=frame)
            full_frame_retry_used = False
            if target_region and (
                not (local_primary.get("label") or "").strip()
                or float(local_primary.get("confidence") or 0.0) < VISION_LOW_CONFIDENCE_THRESHOLD
                or not (local_primary.get("target_metrics") or {}).get("usable_target_evidence")
            ):
                full_frame_retry_used = True
                fallback_faces = self._detect_faces(frame)
                fallback_objects = self._detect_objects(frame)
                fallback_primary = self._choose_primary_detection(
                    fallback_objects,
                    fallback_faces,
                    target_region=target_region,
                    frame=frame,
                )
                fallback_target_usable = (fallback_primary.get("target_metrics") or {}).get("usable_target_evidence")
                local_target_usable = (local_primary.get("target_metrics") or {}).get("usable_target_evidence")
                if (
                    fallback_target_usable
                    or (
                        not local_target_usable
                        and float(fallback_primary.get("confidence") or 0.0) >= float(local_primary.get("confidence") or 0.0)
                    )
                ):
                    faces = fallback_faces
                    objects = fallback_objects
                    local_primary = fallback_primary
                if not (ocr_text or "").strip():
                    ocr_text = self._run_ocr(frame)
                if visual_signature is None and not target_region:
                    visual_signature = self._compute_visual_signature(frame)
            primary = local_primary
            known_object_match = self._match_known_object_by_signature(payload.get("user_id"), visual_signature)
            if known_object_match:
                top_confidence = float(primary.get("confidence") or 0.0)
                top_label = (primary.get("label") or "").strip().lower()
                primary_is_face = (primary.get("entity_type") or "").strip().lower() == "face" and top_label not in {"", "unknown", "unrecognized"}
                target_focused_intent = intent["id"] == "inspect_target" or bool(target_region)
                if (
                    not primary_is_face
                    and (
                        not top_label
                        or top_label in {"unknown", "unrecognized"}
                        or top_confidence < VISION_LOW_CONFIDENCE_THRESHOLD
                        or (
                            target_focused_intent
                            and float(known_object_match["confidence"]) >= max(top_confidence, KNOWN_OBJECT_OVERRIDE_CONFIDENCE)
                        )
                    )
                ):
                    primary = {
                        "label": known_object_match["label"],
                        "confidence": float(known_object_match["confidence"]),
                        "box": primary.get("box") or {},
                        "source": "known_object",
                        "entity_type": "object",
                    }
                    if not objects:
                        objects = [dict(primary)]
            primary_label = (primary.get("label") or "").strip().lower()
            primary_confidence = float(primary.get("confidence") or 0.0)
            primary_entity_type = (primary.get("entity_type") or "object").strip().lower()
            teaching_needed = (
                primary_entity_type != "face"
                and (not primary_label or primary_label in {"unknown", "unrecognized"} or primary_confidence < VISION_LOW_CONFIDENCE_THRESHOLD)
            )
            teaching_prompt = (
                "I do not know what that object is yet. Please tell me what it is."
                if teaching_needed
                else ""
            )
            recognition_state = self._assist_certainty_state(primary.get("label"), primary_confidence, teaching_needed)
            auto_frame = None
            if payload.get("auto_frame") and camera.camera_id == "emeet" and primary.get("label"):
                auto_frame = self._auto_frame_emeet(primary, frame)
            fallback = self._suggest_fallback(payload, primary, ocr_text)
            workflow_mode = self._workflow_mode(payload, intent["id"])
            scene_entities = self._visible_scene_entities(
                objects,
                faces,
                target_region=target_region,
                frame=frame,
                primary=primary,
            )
            perception = self._build_visual_perception_state(
                payload,
                camera,
                intent_id=intent["id"],
                workflow_mode=workflow_mode,
                target_region=target_region,
                scene_entities=scene_entities,
                primary=primary,
            )
            summary = self._format_inspection_summary(
                camera,
                faces,
                objects,
                ocr_text,
                fallback,
                intent["id"],
                primary=primary,
                target_region=target_region,
                workflow_mode=workflow_mode,
                perception=perception,
            )
            recognition_reason = self._recognition_reason(primary, recognition_state, teaching_needed)
            recognition_source = primary.get("source") or ("known_object" if known_object_match else "local_detection")
            subject_presentation = self._subject_presentation(
                perception["attention"].get("attended_subject") or {},
                target_region_active=bool(target_region),
            )
            recognition_diagnostics = {
                "source": recognition_source,
                "entity_type": primary_entity_type,
                "reason": recognition_reason,
                "target_region_applied": bool(target_region),
                "target_mode": target_mode,
                "target_region_expanded": bool(target_region and target_region.get("analysis_pixels") != target_region.get("pixels")),
                "target_primary_in_region": bool((primary.get("target_metrics") or {}).get("in_target")),
                "target_primary_centered": bool((primary.get("target_metrics") or {}).get("center_in_target")),
                "target_primary_overlap_ratio": (primary.get("target_metrics") or {}).get("overlap_ratio"),
                "target_primary_coverage": (primary.get("target_metrics") or {}).get("target_coverage"),
                "full_frame_retry_used": full_frame_retry_used,
                "known_object_consulted": bool(visual_signature),
                "known_object_match_confidence": round(float(known_object_match.get("confidence") or 0.0), 3) if known_object_match else None,
                "face_count": len(faces),
                "object_count": len(objects),
                "used_profile_full_name": primary.get("label_source") == "profile_full_name",
                "used_login_fallback": primary.get("label_source") == "login",
                "label_source": primary.get("label_source") or ("known_object" if recognition_source == "known_object" else "local_detection"),
                "active_visual_source": perception["active_visual_source"]["camera_id"],
                "visual_engaged": perception["visually_engaged"],
                "scene_visible_subject_count": perception["scene"]["visible_subject_count"],
                "scene_visible_subject_types": perception["scene"]["visible_subject_types"],
                "person_present_in_frame": perception["scene"]["person_present"],
                "attended_subject_label": (perception["attention"]["attended_subject"].get("label") or ""),
                "attended_subject_presented_label": subject_presentation["label"],
                "attended_subject_label_downgraded": bool(subject_presentation["downgraded"]),
                "attended_subject_presentation_reason": subject_presentation["reason"],
                "attended_subject_source": (perception["attention"]["attended_subject"].get("source") or ""),
                "attended_subject_label_source": (perception["attention"]["attended_subject"].get("label_source") or ""),
                "attention_anchor_source": perception["attention"]["attention_anchor_source"],
                "attention_continuity_state": perception["attention"]["continuity_state"],
                "recent_attention_event": perception["memory"]["latest_event"],
                "embodied_perception_used": bool(perception),
            }
            step_number = self._workflow_step_number(payload, workflow_mode)
            assist_observation = self._assist_observation_payload(
                camera=camera,
                label=primary.get("label") or "",
                confidence=primary_confidence,
                teaching_needed=teaching_needed,
                target_region_used=bool(target_region),
                summary=summary,
                source=recognition_source,
                entity_type=primary_entity_type,
                reason=recognition_reason,
                diagnostics=recognition_diagnostics,
                workflow_mode=workflow_mode,
                step_number=step_number,
                ocr_text=ocr_text,
                workflow_context=payload.get("workflow_context"),
                workflow_action=str(payload.get("workflow_action") or ""),
                perception=perception,
            )
            observation_id = self._store_vision_observation(
                user_id=payload.get("user_id"),
                camera_key=self._camera_key(camera),
                camera_label=camera.label,
                source_kind=camera.source_kind,
                snapshot_path=snapshot_path,
                detected_faces=faces,
                detected_objects=objects,
                ocr_excerpt=(ocr_text or "")[:400],
                top_label=primary.get("label") or "",
                top_confidence=float(primary.get("confidence") or 0.0),
                metadata={
                    "label_hint": payload.get("label_hint") or "",
                    "auto_frame": auto_frame,
                    "web_fallback": fallback,
                    "target_region": target_region["normalized"] if target_region else None,
                    "visual_signature": visual_signature,
                    "known_object_match": known_object_match,
                    "teaching_needed": teaching_needed,
                    "recognition_diagnostics": recognition_diagnostics,
                    "visual_perception": {
                        "active_visual_source": perception["active_visual_source"],
                        "attention": perception["attention"],
                        "memory": perception["memory"],
                    },
                },
            )
            return {
                "ok": True,
                "action": "inspect_object",
                "camera": {
                    "camera_id": camera.camera_id,
                    "label": camera.label,
                    "source_kind": camera.source_kind,
                },
                "intent": intent,
                "observation_id": observation_id,
                "snapshot_path": snapshot_path,
                "faces": faces,
                "objects": objects,
                "ocr_text": ocr_text[:2000],
                "primary": primary,
                "target_region": target_region["normalized"] if target_region else None,
                "target_region_applied": bool(target_region),
                "target_mode": target_mode,
                "perception": perception,
                "auto_frame": auto_frame,
                "web_fallback": fallback,
                "known_object_match": known_object_match,
                "recognition": {
                    "recognized": not teaching_needed,
                    "confidence": round(primary_confidence, 3),
                    "label": primary.get("label") if not teaching_needed else "",
                    "source": recognition_source,
                    "needs_user_teaching": teaching_needed,
                    "teaching_prompt": teaching_prompt,
                    "certainty": recognition_state,
                    "uncertainty": recognition_state != "recognized",
                    "entity_type": primary_entity_type,
                    "reason": recognition_reason,
                    "diagnostics": recognition_diagnostics,
                },
                "teaching": {
                    "needed": teaching_needed,
                    "prompt": teaching_prompt,
                },
                "assist_observation": assist_observation,
                "workflow_guidance": {
                    "workflow_mode": assist_observation.get("workflow_mode", workflow_mode),
                    "step_number": assist_observation.get("step_number", step_number),
                    "identified_target": assist_observation.get("identified_target", ""),
                    "visible_text": assist_observation.get("visible_text", ""),
                    "issue_summary": assist_observation.get("issue_summary", ""),
                    "current_step": assist_observation.get("current_step", ""),
                    "latest_recommendation": assist_observation.get("latest_recommendation", ""),
                    "next_step": assist_observation.get("next_step", ""),
                    "previously_recommended_steps": assist_observation.get("previously_recommended_steps", []),
                    "attempted_steps": assist_observation.get("attempted_steps", []),
                    "completed_steps": assist_observation.get("completed_steps", []),
                    "skipped_steps": assist_observation.get("skipped_steps", []),
                    "failed_steps": assist_observation.get("failed_steps", []),
                    "last_user_update": assist_observation.get("last_user_update", ""),
                    "step_status": assist_observation.get("step_status", ""),
                    "needs_clarification": assist_observation.get("needs_clarification", False),
                    "clarification_prompt": assist_observation.get("clarification_prompt", ""),
                    "safety_note": assist_observation.get("safety_note", ""),
                    "session_summary": assist_observation.get("session_summary", ""),
                    "handoff_summary": assist_observation.get("handoff_summary", ""),
                    "workflow_state": assist_observation.get("workflow_state", {}),
                },
                "summary": summary,
                "status": self._status_payload(
                    camera,
                    browser_profile=self._browser_profile(payload.get("user_id")),
                    activity="capture_complete",
                    result_kind="inspection",
                    message="Vision capture complete.",
                    last_task_label=intent["label"],
                    access_mode="admin" if self._payload_is_admin(payload) else "personal",
                ),
            }
        except CameraAccessError as exc:
            return self._forbidden_camera_response("inspect_object", exc)
        except Exception as exc:
            logging.exception("Object inspection failed: %s", exc)
            return {"ok": False, "error": str(exc), "action": "inspect_object"}

    def inspect_faces(self, payload: Dict) -> Dict:
        try:
            camera = self._resolve_camera(payload)
            self._persist_camera(camera)
            intent = self._intent_payload("who_do_you_see")
            frame = self._capture_frame(camera, payload)
            snapshot_path = self._save_snapshot(frame, self._camera_key(camera))
            faces = self._detect_faces(frame)
            recognized = [face for face in faces if face.get("label") not in {"", "unrecognized"}]
            summary = (
                f"I see {len(faces)} face(s). "
                + (
                    "Recognized: " + ", ".join(face["label"] for face in recognized) + "."
                    if recognized
                    else "No recognized faces yet."
                )
            )
            observation_id = self._store_vision_observation(
                user_id=payload.get("user_id"),
                camera_key=self._camera_key(camera),
                camera_label=camera.label,
                source_kind=camera.source_kind,
                snapshot_path=snapshot_path,
                detected_faces=faces,
                detected_objects=[],
                top_label=recognized[0]["label"] if recognized else (faces[0]["label"] if faces else ""),
                top_confidence=float(recognized[0]["confidence"] if recognized else (faces[0]["confidence"] if faces else 0.0)),
                metadata={
                    "face_count": len(faces),
                    "recognized_face_count": len(recognized),
                    "identity_sources": [face.get("label_source") for face in recognized],
                },
            )
            return {
                "ok": True,
                "action": "who_do_you_see",
                "camera": {
                    "camera_id": camera.camera_id,
                    "label": camera.label,
                    "source_kind": camera.source_kind,
                },
                "intent": intent,
                "observation_id": observation_id,
                "snapshot_path": snapshot_path,
                "faces": faces,
                "diagnostics": {
                    "source": "face_match" if recognized else "face_detection",
                    "face_count": len(faces),
                    "recognized_face_count": len(recognized),
                    "used_profile_full_name": any(face.get("label_source") == "profile_full_name" for face in recognized),
                    "used_login_fallback": any(face.get("label_source") == "login" for face in recognized),
                },
                "summary": summary,
                "status": self._status_payload(
                    camera,
                    browser_profile=self._browser_profile(payload.get("user_id")),
                    activity="capture_complete",
                    result_kind="faces",
                    message="Face inspection complete.",
                    last_task_label=intent["label"],
                    access_mode="admin" if self._payload_is_admin(payload) else "personal",
                ),
            }
        except CameraAccessError as exc:
            return self._forbidden_camera_response("who_do_you_see", exc)
        except Exception as exc:
            logging.exception("Face inspection failed: %s", exc)
            return {"ok": False, "error": str(exc), "action": "who_do_you_see"}

    def _amcrest_request(self, action: str, code: str, speed: int = DEFAULT_AMCREST_SPEED):
        if not (AMCREST_ENABLE_PTZ and AMCREST_HTTP_BASE):
            return {"ok": False, "error": "Amcrest PTZ is not enabled."}
        url = f"{AMCREST_HTTP_BASE}/cgi-bin/ptz.cgi"
        try:
            response = requests.get(
                url,
                params={
                    "action": action,
                    "channel": 0,
                    "code": code,
                    "arg1": 0,
                    "arg2": max(1, min(int(speed or DEFAULT_AMCREST_SPEED), 8)),
                    "arg3": 0,
                },
                auth=(AMCREST_USERNAME, AMCREST_PASSWORD),
                timeout=4,
            )
            text = (response.text or "").strip()
            ok = response.ok and not text.startswith("Error")
            return {"ok": ok, "status_code": response.status_code, "response_text": text[:500], "url": url}
        except Exception as exc:
            logging.exception("Amcrest PTZ request failed: %s", exc)
            return {"ok": False, "error": str(exc), "url": url}

    def handle_ptz(self, payload: Dict) -> Dict:
        try:
            camera = self._resolve_camera(payload, for_ptz=True)
            action = (payload.get("direction") or payload.get("ptz_action") or "").strip().lower()
            allowed, retry_after = self.ptz_limiter.check(f"{self._camera_key(camera)}:{action}")
            if not allowed:
                return {"ok": False, "action": "ptz", "camera_id": camera.camera_id, "error": "PTZ request rate limited.", "retry_after": round(retry_after, 3)}
            if camera.camera_id == "emeet" and action in {"zoom_in", "zoom_out"}:
                zoom_result = self.handle_emeet_zoom({"direction": action, "step": payload.get("step")})
                zoom_result.setdefault("camera_id", camera.camera_id)
                return zoom_result
            if camera.camera_id != "amcrest":
                return {"ok": False, "action": "ptz", "camera_id": camera.camera_id, "error": "Selected camera does not support HTTP PTZ control."}

            mapping = {
                "left": ("start", "Left"),
                "right": ("start", "Right"),
                "up": ("start", "Up"),
                "down": ("start", "Down"),
                "zoom_in": ("start", "ZoomTele"),
                "zoom_out": ("start", "ZoomWide"),
            }
            if action == "stop":
                results = [self._amcrest_request("stop", code, payload.get("speed") or DEFAULT_AMCREST_SPEED) for code in ("Left", "Right", "Up", "Down", "ZoomTele", "ZoomWide")]
                return {"ok": any(item.get("ok") for item in results), "action": "ptz", "camera_id": camera.camera_id, "ptz_action": action, "results": results}
            if action not in mapping:
                return {"ok": False, "action": "ptz", "error": "Unsupported PTZ action."}

            request_action, code = mapping[action]
            result = self._amcrest_request(request_action, code, payload.get("speed") or DEFAULT_AMCREST_SPEED)
            result.update({"action": "ptz", "camera_id": camera.camera_id, "ptz_action": action})
            return result
        except CameraAccessError as exc:
            return self._forbidden_camera_response("ptz", exc)

    def _v4l2_get_zoom_state(self) -> Dict:
        if not shutil.which("v4l2-ctl"):
            return {"ok": False, "error": "v4l2-ctl is not available."}
        try:
            result = subprocess.run(
                ["v4l2-ctl", "-d", EMEET_V4L2_DEVICE, "--list-ctrls"],
                capture_output=True,
                text=True,
                check=True,
            )
        except Exception as exc:
            logging.exception("Failed to inspect V4L2 controls: %s", exc)
            return {"ok": False, "error": str(exc)}

        zoom_line = next((line.strip() for line in result.stdout.splitlines() if "zoom_absolute" in line), "")
        if not zoom_line:
            return {"ok": False, "error": "zoom_absolute control is unavailable."}

        def _extract(token: str, default: int) -> int:
            marker = f"{token}="
            if marker not in zoom_line:
                return default
            try:
                return int(zoom_line.split(marker, 1)[1].split()[0])
            except Exception:
                return default

        return {
            "ok": True,
            "min": _extract("min", 100),
            "max": _extract("max", 400),
            "step": _extract("step", DEFAULT_EMEET_ZOOM_STEP),
            "value": _extract("value", 100),
            "raw": zoom_line,
        }

    def handle_emeet_zoom(self, payload: Dict) -> Dict:
        action = (payload.get("direction") or payload.get("zoom_action") or "").strip().lower()
        if action in {"in", "zoom_in", "in_step"}:
            delta = int(payload.get("step") or DEFAULT_EMEET_ZOOM_STEP)
        elif action in {"out", "zoom_out", "out_step"}:
            delta = -int(payload.get("step") or DEFAULT_EMEET_ZOOM_STEP)
        else:
            return {"ok": False, "action": "emeet_zoom", "error": "Unsupported EMEET zoom action."}

        state = self._v4l2_get_zoom_state()
        if not state.get("ok"):
            state["action"] = "emeet_zoom"
            return state

        target = max(state["min"], min(state["max"], state["value"] + delta))
        if target == state["value"]:
            return {"ok": True, "action": "emeet_zoom", "camera_id": "emeet", "zoom_value": state["value"], "message": "Zoom already at the requested limit."}

        try:
            subprocess.run(
                ["v4l2-ctl", "-d", EMEET_V4L2_DEVICE, f"--set-ctrl=zoom_absolute={target}"],
                capture_output=True,
                text=True,
                check=True,
            )
            return {"ok": True, "action": "emeet_zoom", "camera_id": "emeet", "zoom_value": target, "previous_zoom_value": state["value"]}
        except Exception as exc:
            logging.exception("Failed adjusting EMEET zoom: %s", exc)
            return {"ok": False, "action": "emeet_zoom", "error": str(exc)}

    def _auto_frame_emeet(self, primary: Dict, frame) -> Optional[Dict]:
        ratio = self._detection_area_ratio(primary, frame)
        box = primary.get("box") or {}
        cropped = (
            int(box.get("left", 1)) <= 1
            or int(box.get("top", 1)) <= 1
            or int(box.get("right", frame.shape[1] - 1)) >= frame.shape[1] - 1
            or int(box.get("bottom", frame.shape[0] - 1)) >= frame.shape[0] - 1
        )
        if ratio < AUTOFRAME_SMALL_THRESHOLD:
            result = self.handle_emeet_zoom({"direction": "zoom_in", "step": DEFAULT_EMEET_ZOOM_STEP})
            return {"decision": "zoom_in", "area_ratio": round(ratio, 3), "result": result}
        if ratio > AUTOFRAME_LARGE_THRESHOLD or cropped:
            result = self.handle_emeet_zoom({"direction": "zoom_out", "step": DEFAULT_EMEET_ZOOM_STEP})
            return {"decision": "zoom_out", "area_ratio": round(ratio, 3), "cropped": cropped, "result": result}
        return {"decision": "hold", "area_ratio": round(ratio, 3), "cropped": cropped}


def main():
    if EYES_SOCKET.exists():
        EYES_SOCKET.unlink()

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(EYES_SOCKET))
    EYES_SOCKET.chmod(0o660)
    server.listen(10)

    svc = EyesService()
    logging.info("Eyes service listening on %s", EYES_SOCKET)

    while True:
        conn, _ = server.accept()
        with conn:
            try:
                raw = conn.recv(10 * 1024 * 1024)
                if not raw:
                    continue
                payload = json.loads(raw.decode("utf-8", errors="ignore"))
                result = svc.process(payload)
                conn.sendall(json.dumps(result).encode("utf-8"))
            except Exception as exc:
                logging.exception("Eyes request failed: %s", exc)
                conn.sendall(json.dumps({"ok": False, "error": "Eyes request failed."}).encode("utf-8"))


if __name__ == "__main__":
    main()
