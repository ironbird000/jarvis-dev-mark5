#!/app/jarvis-dev/venv/bin/python3
import json
import logging
import re
import socket
from difflib import SequenceMatcher
from vosk import Model, KaldiRecognizer

from config import (
    VOSK_MODEL_PATH,
    EARS_SOCKET,
    DATA_DIR,
    WAKE_WORD,
    ensure_directories,
)

ensure_directories()

LOG_FILE = DATA_DIR / "logs" / "ears.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

class EarsService:
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.model = Model(str(VOSK_MODEL_PATH))
        self.recognizers = {}

    def _get_recognizer(self, stream_id: str):
        rec = self.recognizers.get(stream_id)
        if rec is None:
            rec = KaldiRecognizer(self.model, self.sample_rate)
            self.recognizers[stream_id] = rec
        return rec

    def _normalize(self, text: str) -> str:
        return re.sub(r"[^a-z0-9\s]+", "", (text or "").lower()).strip()

    def _wake_variants(self, wake_word: str):
        wake = self._normalize(wake_word or WAKE_WORD or "")
        variants = {wake} if wake else set()

        if wake == "jarvis":
            variants.update({
                "jarvis",
                "jervis",
                "jarviss",
                "jarv",
                "jarvus",
                "jarvez",
                "jarvice",
                "jarvies",
                "jarvish",
                "jarves",
                "jarvics",
                "jarvix",
            })

        return {v for v in variants if v}

    def _similar(self, a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()

    def _token_matches_wake(self, token: str, wake_variants) -> bool:
        token = self._normalize(token)
        if not token:
            return False

        if token in wake_variants:
            return True

        for variant in wake_variants:
            if token.startswith(variant) or variant.startswith(token):
                if abs(len(token) - len(variant)) <= 2:
                    return True

            if self._similar(token, variant) >= 0.82:
                return True

        return False

    def _wake_detected(self, text: str, wake_word: str) -> bool:
        wake_variants = self._wake_variants(wake_word)
        if not wake_variants:
            return True

        normalized = self._normalize(text)
        if not normalized:
            return False

        tokens = [t for t in normalized.split() if t]

        for token in tokens[:4]:
            if self._token_matches_wake(token, wake_variants):
                return True

        compact = normalized.replace(" ", "")
        for variant in wake_variants:
            if variant in normalized or variant in compact:
                return True

        return False

    def _trim_wake(self, text: str, wake_word: str) -> str:
        original = (text or "").strip()
        if not original:
            return ""

        wake_variants = self._wake_variants(wake_word)
        if not wake_variants:
            return original

        original_tokens = original.split()
        normalized_tokens = [self._normalize(t) for t in original_tokens]

        wake_index = None
        for idx, token in enumerate(normalized_tokens[:4]):
            if self._token_matches_wake(token, wake_variants):
                wake_index = idx
                break

        if wake_index is None:
            return original

        return " ".join(original_tokens[wake_index + 1:]).strip()

    def process(self, payload: dict) -> dict:
        action = payload.get("action", "process_audio")

        if action == "reset_stream":
            stream_id = payload.get("stream_id", "default")
            self.recognizers.pop(stream_id, None)
            return {"ok": True, "type": "reset", "stream_id": stream_id}

        if action != "process_audio":
            return {"ok": False, "error": "Unsupported action."}

        stream_id = payload.get("stream_id", "default")
        audio = payload.get("audio", "")
        wake_word = payload.get("wake_word") or WAKE_WORD
        wake_required = bool(payload.get("wake_required", True))

        if not audio:
            return {"ok": False, "error": "Missing audio payload."}

        audio_bytes = bytes.fromhex(audio)
        rec = self._get_recognizer(stream_id)

        if rec.AcceptWaveform(audio_bytes):
            result = json.loads(rec.Result())
            transcript = (result.get("text") or "").strip()

            wake_detected = self._wake_detected(transcript, wake_word)
            processed = self._trim_wake(transcript, wake_word) if wake_detected else transcript

            logging.info(
                "Final transcript [%s]: %s | wake_detected=%s | processed=%s | wake_required=%s",
                stream_id,
                transcript,
                wake_detected,
                processed,
                wake_required,
            )

            if wake_detected:
                audio_state = "wake_heard"
            elif transcript:
                audio_state = "voice"
            else:
                audio_state = "idle"

            # Reset recognizer state after finalized utterances to keep browser-stream
            # recognition stable across consecutive requests on the same stream.
            self.recognizers.pop(stream_id, None)

            return {
                "ok": True,
                "type": "final",
                "transcript": transcript,
                "wake_word_detected": wake_detected,
                "wake_required": wake_required,
                "processed_text": processed,
                "audio_state": audio_state,
            }

        partial = json.loads(rec.PartialResult()).get("partial", "").strip()
        if partial:
            logging.info("Partial transcript [%s]: %s", stream_id, partial)

        audio_state = "voice" if partial else "noise"

        return {
            "ok": True,
            "type": "partial",
            "transcript": partial,
            "audio_state": audio_state,
        }

def main():
    if EARS_SOCKET.exists():
        EARS_SOCKET.unlink()

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(EARS_SOCKET))
    EARS_SOCKET.chmod(0o660)
    server.listen(20)

    svc = EarsService()
    logging.info("Ears service listening on %s", EARS_SOCKET)

    while True:
        conn, _ = server.accept()
        with conn:
            try:
                raw = conn.recv(1024 * 1024)
                if not raw:
                    continue
                payload = json.loads(raw.decode("utf-8", errors="ignore"))
                result = svc.process(payload)
                conn.sendall(json.dumps(result).encode("utf-8"))
            except Exception as exc:
                logging.exception("Ears request failed: %s", exc)
                conn.sendall(json.dumps({
                    "ok": False,
                    "error": "Ears request failed."
                }).encode("utf-8"))

if __name__ == "__main__":
    main()
