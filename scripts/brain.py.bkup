#!/app/jarvis-dev/venv/bin/python3
import json
import logging
import re
import socket
from typing import Dict, List

import requests

from config import (
    OLLAMA_URL,
    OLLAMA_MODEL,
    WAKE_WORD,
    BRAIN_SOCKET,
    build_system_prompt,
    ensure_directories,
    DATA_DIR,
    DEFAULT_DISPLAY_NAME,
)
from memory import MemoryStore
from live_router import route_live_query
from weather_service import get_weather
from sports_service import get_sports
from news_service import get_news

ensure_directories()

LOG_FILE = DATA_DIR / "logs" / "brain.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

store = MemoryStore()

DISALLOWED_OLD_HONORIFICS = {
    "my lord",
    "your excellency",
    "sirrah",
    "master",
    "boss",
    "captain",
}

DOCUMENT_HINT_PATTERNS = [
    r"\buploaded\b",
    r"\bdocument\b",
    r"\bdoc\b",
    r"\bfile\b",
    r"\bresume\b",
    r"\bmy resume\b",
    r"\bthe document i uploaded\b",
    r"\bthe file i uploaded\b",
]

OPEN_DOCUMENT_PATTERNS = [
    r"\bopen\b.*\b(document|file|resume)\b",
    r"\bshow\b.*\b(document|file|resume)\b",
    r"\bdisplay\b.*\b(document|file|resume)\b",
    r"\bview\b.*\b(document|file|resume)\b",
    r"\bhow you view\b.*\b(document|file|resume)\b",
]

ARMY_PATTERNS = [
    r"\barmy\b",
    r"\bmilitary\b",
    r"\bunited states army\b",
    r"\bterrain analyst\b",
]

def load_recent_history(user_id: int, limit: int = 8) -> str:
    rows = store.get_recent_conversation(user_id, limit=limit)
    if not rows:
        return ""
    lines = []
    for row in rows:
        role = row["role"].upper()
        content = (row["content"] or "").strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)

def load_long_term_memory(user_id: int, limit: int = 20) -> List[str]:
    rows = store.get_long_term_facts(user_id, limit=limit)
    if not rows:
        return []
    return [row["fact_text"] for row in rows if row["fact_text"]]

def filter_conflicting_memory(memory_lines: List[str], preferred_name: str) -> List[str]:
    if not memory_lines:
        return []

    preferred_name = (preferred_name or "").strip().lower()
    filtered_lines = []

    for line in memory_lines:
        lower = line.lower()

        if "preferred name" in lower and preferred_name and preferred_name not in lower:
            continue

        if any(title in lower for title in DISALLOWED_OLD_HONORIFICS):
            continue

        filtered_lines.append(line)

    return filtered_lines

def clean_spoken_query(text: str, wake_word: str) -> str:
    if not text:
        return ""
    text = text.strip()
    if wake_word:
        text = re.sub(rf"\b{re.escape(wake_word)}\b", "", text, flags=re.IGNORECASE).strip()
    return re.sub(r"\s+", " ", text).strip()

def text_matches_any(text: str, patterns: List[str]) -> bool:
    lowered = text.lower()
    return any(re.search(p, lowered, flags=re.IGNORECASE) for p in patterns)

def looks_like_uploaded_document_query(text: str) -> bool:
    return text_matches_any(text, DOCUMENT_HINT_PATTERNS)

def looks_like_open_document_request(text: str) -> bool:
    return text_matches_any(text, OPEN_DOCUMENT_PATTERNS)

def score_document_fact(query: str, fact: str) -> int:
    query_lower = query.lower()
    fact_lower = fact.lower()
    score = 0

    if "uploaded document:" in fact_lower:
        score += 4
    if "document contents excerpt:" in fact_lower:
        score += 5
    if "resume" in query_lower and "resume" in fact_lower:
        score += 6
    if "document" in query_lower and "document" in fact_lower:
        score += 3
    if "file" in query_lower and "uploaded document" in fact_lower:
        score += 2

    if text_matches_any(query, ARMY_PATTERNS):
        for token in ["army", "united states army", "terrain analyst", "sep 1990", "mar 1997", "1990", "1997"]:
            if token in fact_lower:
                score += 8

    query_words = [w for w in re.findall(r"[a-zA-Z0-9]+", query_lower) if len(w) > 2]
    for word in query_words:
        if word in fact_lower:
            score += 1

    return score

def get_relevant_document_memory(user_id: int, query: str, limit: int = 3) -> List[str]:
    facts = load_long_term_memory(user_id, limit=50)
    if not facts:
        return []

    doc_facts = [
        f for f in facts
        if "uploaded document:" in f.lower()
        or "document contents excerpt:" in f.lower()
        or "path:" in f.lower()
    ]
    if not doc_facts:
        return []

    ranked = sorted(
        ((score_document_fact(query, fact), fact) for fact in doc_facts),
        key=lambda x: x[0],
        reverse=True
    )

    ranked = [fact for score, fact in ranked if score > 0]
    return ranked[:limit]

def get_latest_uploaded_document_path(user_id: int):
    try:
        rows = store.get_uploads(user_id)
        if not rows:
            return None
        row = rows[0]
        return {
            "file_name": row["file_name"],
            "stored_path": row["stored_path"],
            "mime_type": row["mime_type"],
        }
    except Exception:
        logging.exception("Failed to get latest upload")
        return None

def build_prompt(context: Dict) -> str:
    system_prompt = build_system_prompt()
    preferred_name = (context.get("preferred_name") or context.get("display_name") or DEFAULT_DISPLAY_NAME).strip() or DEFAULT_DISPLAY_NAME
    recent_history = load_recent_history(context["user_id"])
    long_term_memory_lines = load_long_term_memory(context["user_id"], limit=20)
    long_term_memory_lines = filter_conflicting_memory(long_term_memory_lines, preferred_name)
    user_text = context["text"]

    document_memory = []
    if looks_like_uploaded_document_query(user_text):
        document_memory = get_relevant_document_memory(context["user_id"], user_text, limit=3)

    policy_lines = [
        "Critical personalization rules:",
        f"- The user's current preferred name is: {preferred_name}",
        f"- You must address the user as: {preferred_name}",
        "- Do not call the user by any previous nickname, title, or honorific.",
        "- Do not use alternatives like 'My Lord', 'Your Excellency', 'Sir', 'Master', or similar unless the user explicitly asks you to.",
        "- If older memory conflicts with the current preferred name, the current preferred name always wins.",
        "- Keep responses natural, concise, and conversational.",
    ]

    if document_memory:
        policy_lines.extend([
            "",
            "Critical document-grounding rules:",
            "- The user is asking about an uploaded document.",
            "- You must prioritize the uploaded document memory over general assumptions.",
            "- If the uploaded document contains the answer, answer from that document.",
            "- Do not deny the document contains information if the excerpt clearly shows it.",
            "- If a date range is present, use it directly.",
        ])

    parts = [
        system_prompt,
        "\n".join(policy_lines),
        f"Current interaction source: {context.get('source', 'unknown')}",
    ]

    if long_term_memory_lines:
        parts.append("Known long-term facts:\n" + "\n".join(f"- {line}" for line in long_term_memory_lines))

    if document_memory:
        parts.append("Relevant uploaded document memory:\n" + "\n\n".join(document_memory))

    if recent_history:
        parts.append("Recent conversation:\n" + recent_history)

    parts.append(f"User message:\n{user_text}")
    parts.append(f"Address the user as {preferred_name}.")
    parts.append("Jarvis response:")

    return "\n\n".join(p for p in parts if p).strip()

def run_llm(prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 260,
        },
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=60)
    response.raise_for_status()
    text = response.json().get("response", "").strip()
    return text or "I do not have a response at the moment."

def maybe_store_preferred_name_fact(user_id: int, preferred_name: str):
    preferred_name = (preferred_name or "").strip()
    if not preferred_name:
        return
    fact = f"User's current preferred name is {preferred_name}."
    try:
        store.add_long_term_fact(user_id, fact, confidence=1.0)
    except Exception:
        logging.exception("Failed to store preferred name fact")

def maybe_store_home_location_fact(user_id: int, home_location: str):
    home_location = (home_location or "").strip()
    if not home_location:
        return
    fact = f"User's home location is {home_location}."
    try:
        store.add_long_term_fact(user_id, fact, confidence=0.95)
    except Exception:
        logging.exception("Failed to store home location fact")

def extract_location_for_weather(text: str, home_location: str) -> str:
    lowered = text.lower()
    for prefix in ["weather in ", "forecast for ", "weather for ", "temperature in "]:
        idx = lowered.find(prefix)
        if idx != -1:
            return text[idx + len(prefix):].strip(" ?.")
    return home_location.strip() or text.strip()

def should_use_home_location(text: str) -> bool:
    lowered = text.lower()
    implicit_phrases = [
        "weather today",
        "what's the weather",
        "whats the weather",
        "forecast today",
        "local weather",
        "news today",
        "local news",
        "what's going on locally",
        "whats going on locally",
        "what's happening locally",
        "whats happening locally",
    ]
    return any(p in lowered for p in implicit_phrases)

def format_weather_result(data: Dict, preferred_name: str) -> str:
    if not data.get("ok"):
        return f"{preferred_name}, I couldn't retrieve live weather data right now. {data.get('error', '')}".strip()

    lines = [
        f"{preferred_name}, here is the latest weather for {data['resolved_location']}.",
        f"Source: {data['provider']}.",
    ]

    for p in data.get("forecast_periods", [])[:3]:
        lines.append(
            f"{p['name']}: {p['temperature']}°{p['temperatureUnit']}, "
            f"{p['shortForecast']}. Wind {p['windSpeed']} {p['windDirection']}."
        )

    return " ".join(lines)

def format_sports_result(data: Dict, preferred_name: str) -> str:
    if not data.get("ok"):
        return f"{preferred_name}, I couldn't retrieve live sports data right now. {data.get('error', '')}".strip()

    lines = [
        f"{preferred_name}, here are the latest {data['league'].upper()} updates.",
        f"Source: {data['provider']}.",
    ]

    for event in data.get("events", [])[:4]:
        teams = event.get("teams", [])
        if len(teams) >= 2:
            a = teams[0]
            b = teams[1]
            lines.append(
                f"{a['name']} {a['score']} versus {b['name']} {b['score']}. "
                f"Status: {event.get('detail') or event.get('status')}."
            )

    return " ".join(lines)

def format_news_result(data: Dict, preferred_name: str) -> str:
    if not data.get("ok"):
        return f"{preferred_name}, I couldn't retrieve live news right now. {data.get('error', '')}".strip()

    lines = [
        f"{preferred_name}, here are the latest news headlines.",
        f"Source: {data['provider']}.",
    ]

    for item in data.get("articles", [])[:5]:
        title = item.get("title") or "Untitled headline"
        source = item.get("source") or "Unknown source"
        pub = item.get("pubDate") or "Unknown time"
        lines.append(f"{title} — {source}, {pub}.")

    return " ".join(lines)

def handle_live_query(payload: Dict, preferred_name: str, home_location: str) -> Dict:
    text = (payload.get("text") or "").strip()
    route = route_live_query(text)
    domain = route.get("domain")

    if domain == "weather":
        location = extract_location_for_weather(text, home_location)
        result = get_weather(location)
        response_text = format_weather_result(result, preferred_name)
        return {
            "ok": True,
            "response_text": response_text,
            "live_domain": "weather",
            "live_result": result,
        }

    if domain == "sports":
        result = get_sports(text)
        response_text = format_sports_result(result, preferred_name)
        return {
            "ok": True,
            "response_text": response_text,
            "live_domain": "sports",
            "live_result": result,
        }

    if domain == "news":
        query_text = text
        if should_use_home_location(text) and home_location:
            query_text = f"{text} {home_location}"
        result = get_news(query_text, route.get("news_scope") or "general")
        response_text = format_news_result(result, preferred_name)
        return {
            "ok": True,
            "response_text": response_text,
            "live_domain": "news",
            "live_result": result,
        }

    return {
        "ok": False,
        "error": "No live domain matched."
    }

def handle_open_document_request(user_id: int, preferred_name: str) -> Dict:
    latest = get_latest_uploaded_document_path(user_id)
    if not latest:
        return {
            "ok": True,
            "response_text": f"{preferred_name}, I could not find a previously uploaded document to open.",
        }

    return {
        "ok": True,
        "response_text": f"{preferred_name}, opening your uploaded document now.",
        "action": "open_uploaded_document",
        "document": latest,
    }

def handle_request(payload: Dict) -> Dict:
    user_id = payload["user_id"]
    raw_text = (payload.get("text") or "").strip()
    source = payload.get("source", "unknown")
    display_name = (payload.get("display_name") or DEFAULT_DISPLAY_NAME).strip() or DEFAULT_DISPLAY_NAME
    preferred_name = (payload.get("preferred_name") or display_name).strip() or DEFAULT_DISPLAY_NAME
    home_location = (payload.get("home_location") or "").strip()

    if not raw_text:
        return {"ok": False, "error": "Empty input."}

    cleaned_text = raw_text
    if source == "voice":
        cleaned_text = clean_spoken_query(raw_text, payload.get("wake_word") or WAKE_WORD)

    maybe_store_preferred_name_fact(user_id, preferred_name)
    if home_location:
        maybe_store_home_location_fact(user_id, home_location)

    effective_text = cleaned_text or raw_text
    store.add_conversation_entry(user_id, "user", effective_text, source=source)

    if looks_like_open_document_request(effective_text):
        result = handle_open_document_request(user_id, preferred_name)
        if result.get("ok"):
            store.add_conversation_entry(user_id, "assistant", result["response_text"], source=source)
        return {
            "ok": True,
            "response_text": result["response_text"],
            "spoken_query": raw_text,
            "processed_query": effective_text,
            "display_name": preferred_name,
            "preferred_name": preferred_name,
            "home_location": home_location,
            "action": result.get("action"),
            "document": result.get("document"),
        }

    route = route_live_query(effective_text)
    if route.get("is_live_query"):
        live_result = handle_live_query(
            {
                **payload,
                "text": effective_text,
            },
            preferred_name,
            home_location,
        )
        if live_result.get("ok"):
            reply = live_result["response_text"]
            store.add_conversation_entry(user_id, "assistant", reply, source=source)
            return {
                "ok": True,
                "response_text": reply,
                "spoken_query": raw_text,
                "processed_query": effective_text,
                "display_name": preferred_name,
                "preferred_name": preferred_name,
                "home_location": home_location,
                "live_domain": live_result.get("live_domain"),
            }

    prompt = build_prompt({
        "user_id": user_id,
        "display_name": display_name,
        "preferred_name": preferred_name,
        "source": source,
        "text": effective_text,
    })

    logging.info("Brain prompt built for user_id=%s preferred_name=%s source=%s", user_id, preferred_name, source)

    reply = run_llm(prompt)
    store.add_conversation_entry(user_id, "assistant", reply, source=source)

    return {
        "ok": True,
        "response_text": reply,
        "spoken_query": raw_text,
        "processed_query": effective_text,
        "display_name": preferred_name,
        "preferred_name": preferred_name,
        "home_location": home_location,
    }

def main():
    if BRAIN_SOCKET.exists():
        BRAIN_SOCKET.unlink()

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(BRAIN_SOCKET))
    BRAIN_SOCKET.chmod(0o660)
    server.listen(20)

    logging.info("Brain listening on %s", BRAIN_SOCKET)

    while True:
        conn, _ = server.accept()
        with conn:
            try:
                raw = conn.recv(65535)
                if not raw:
                    continue
                payload = json.loads(raw.decode("utf-8", errors="ignore"))
                result = handle_request(payload)
                conn.sendall(json.dumps(result).encode("utf-8"))
            except Exception as exc:
                logging.exception("Brain request failed: %s", exc)
                conn.sendall(json.dumps({
                    "ok": False,
                    "error": "Brain request failed."
                }).encode("utf-8"))

if __name__ == "__main__":
    main()
