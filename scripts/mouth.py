#!/app/jarvis-dev/venv/bin/python3
import json
import logging
import socket
import subprocess
import tempfile
import time
from pathlib import Path

from config import (
    DATA_DIR,
    MOUTH_SOCKET,
    ensure_directories,
)

ensure_directories()

LOG_FILE = DATA_DIR / "logs" / "mouth.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

PIPER_PATH = "/app/jarvis/piper/piper"
VOICE_MODEL = "/app/jarvis/voice/piper/jarvis-onnx/jarvis-high.onnx"
DEFAULT_SAMPLE_RATE = 22050
PCM_CACHE_DIR = DATA_DIR / "tts_pcm_cache"
PCM_CACHE_DIR.mkdir(parents=True, exist_ok=True)


class MouthService:
    def _cleanup_old_cache(self, max_age_seconds: int = 300):
        now = time.time()
        for file in PCM_CACHE_DIR.glob("*.pcm"):
            try:
                if now - file.stat().st_mtime > max_age_seconds:
                    file.unlink(missing_ok=True)
            except Exception:
                pass

    def synthesize_pcm(self, text: str) -> dict:
        text = (text or "").strip()
        if not text:
            return {"ok": False, "error": "Missing text for synthesis."}

        if not Path(PIPER_PATH).exists():
            logging.error("Piper binary missing: %s", PIPER_PATH)
            return {"ok": False, "error": "Piper binary missing."}

        if not Path(VOICE_MODEL).exists():
            logging.error("Voice model missing: %s", VOICE_MODEL)
            return {"ok": False, "error": "Voice model missing."}

        try:
            self._cleanup_old_cache()

            with tempfile.NamedTemporaryFile(
                dir=str(PCM_CACHE_DIR),
                prefix="jarvis_tts_",
                suffix=".pcm",
                delete=False,
            ) as tmp:
                tmp_path = Path(tmp.name)

            with open(tmp_path, "wb") as out_file:
                proc = subprocess.run(
                    [PIPER_PATH, "--model", VOICE_MODEL, "--output_raw"],
                    input=text.encode("utf-8"),
                    stdout=out_file,
                    stderr=subprocess.PIPE,
                    check=True,
                )

            if not tmp_path.exists() or tmp_path.stat().st_size == 0:
                stderr_text = (proc.stderr or b"").decode("utf-8", errors="ignore")
                logging.error("Piper returned empty PCM. stderr=%s", stderr_text)
                tmp_path.unlink(missing_ok=True)
                return {"ok": False, "error": "Synthesized PCM was empty."}


            logging.info(
                "Synthesized PCM file %s (%s bytes) for text=%r",
                tmp_path,
                tmp_path.stat().st_size,
                text[:120]
            )

            return {
                "ok": True,
                "pcm_file": str(tmp_path),
                "sample_rate": DEFAULT_SAMPLE_RATE,
                "text": text,
            }

        except subprocess.CalledProcessError as exc:
            stderr_text = (exc.stderr or b"").decode("utf-8", errors="ignore")
            logging.exception("Piper synthesis failed: %s", stderr_text)
            return {"ok": False, "error": "Piper synthesis failed."}
        except Exception as exc:
            logging.exception("Unexpected mouth synthesis failure: %s", exc)
            return {"ok": False, "error": "Unexpected synthesis failure."}

    def process(self, payload: dict) -> dict:
        action = payload.get("action")
        mode = payload.get("mode")
        text = payload.get("text", "")

        if action == "ping" or mode == "ping":
            return {"ok": True, "message": "mouth alive"}

        if mode == "speak":
            return self.synthesize_pcm(text)

        if action == "synthesize_pcm":
            return self.synthesize_pcm(text)

        return {"ok": False, "error": "Unsupported action."}


def main():
    if MOUTH_SOCKET.exists():
        MOUTH_SOCKET.unlink()

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(MOUTH_SOCKET))
    MOUTH_SOCKET.chmod(0o660)
    server.listen(20)

    svc = MouthService()
    logging.info("Mouth service listening on %s", MOUTH_SOCKET)

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
            except BrokenPipeError:
                logging.warning("Mouth client disconnected before response could be sent")
                continue
            except Exception as exc:
                logging.exception("Mouth request failed: %s", exc)
                try:
                    conn.sendall(json.dumps({
                        "ok": False,
                        "error": "Mouth request failed."
                    }).encode("utf-8"))
                except BrokenPipeError:
                    logging.warning("Mouth client disconnected while sending error response")
                except Exception:
                    logging.exception("Failed sending mouth error response")


if __name__ == "__main__":
    main()
