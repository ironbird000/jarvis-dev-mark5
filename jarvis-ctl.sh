#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="/app/jarvis-dev"
VENV_PY="/app/jarvis-dev/venv/bin/python3"
RUN_DIR="$BASE_DIR/run"
LOG_DIR="$BASE_DIR/data/logs"

# OpenAI / vector search environment
ENV_FILE="/app/jarvis-dev/.env.jarvis"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  source "$ENV_FILE"
  set +a
fi

export JARVIS_CHROMA_DIR="${JARVIS_CHROMA_DIR:-/app/jarvis-dev/data/chroma}"
export JARVIS_EMBED_MODEL="${JARVIS_EMBED_MODEL:-text-embedding-3-small}"

mkdir -p "$RUN_DIR" "$LOG_DIR" "$JARVIS_CHROMA_DIR"

service_script() {
  local svc="$1"
  case "$svc" in
    brain) echo "scripts/brain.py" ;;
    ears) echo "scripts/ears.py" ;;
    eyes) echo "scripts/eyes.py" ;;
    mouth) echo "scripts/mouth.py" ;;
    interface) echo "scripts/interface.py" ;;
    *) return 1 ;;
  esac
}

service_pidfile() {
  local svc="$1"
  case "$svc" in
    brain) echo "$RUN_DIR/jarvis_brain.pid" ;;
    ears) echo "$RUN_DIR/jarvis_ears.pid" ;;
    eyes) echo "$RUN_DIR/jarvis_eyes.pid" ;;
    mouth) echo "$RUN_DIR/jarvis_mouth.pid" ;;
    interface) echo "$RUN_DIR/jarvis_interface.pid" ;;
    *) return 1 ;;
  esac
}

service_logfile() {
  local svc="$1"
  echo "$LOG_DIR/${svc}.stdout.log"
}

cleanup_pidfile_if_stale() {
  local svc="$1"
  local pidfile
  pidfile="$(service_pidfile "$svc")"

  if [[ -f "$pidfile" ]]; then
    local pid
    pid="$(cat "$pidfile" 2>/dev/null || true)"
    if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
      rm -f "$pidfile"
    fi
  fi
}

is_running() {
  local svc="$1"
  local pidfile
  pidfile="$(service_pidfile "$svc")"

  if [[ -f "$pidfile" ]]; then
    local pid
    pid="$(cat "$pidfile" 2>/dev/null || true)"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      return 0
    fi
  fi
  return 1
}

start_service() {
  local svc="$1"
  cleanup_pidfile_if_stale "$svc"

  if is_running "$svc"; then
    echo "$svc is already running"
    return 0
  fi

  local script pidfile logfile
  script="$(service_script "$svc")"
  pidfile="$(service_pidfile "$svc")"
  logfile="$(service_logfile "$svc")"

  echo "Starting $svc..."
  nohup "$VENV_PY" "$BASE_DIR/$script" >>"$logfile" 2>&1 &
  local pid=$!
  echo "$pid" >"$pidfile"
  sleep 1

  if kill -0 "$pid" 2>/dev/null; then
    echo "$svc started (pid $pid)"
  else
    echo "Failed to start $svc"
    rm -f "$pidfile"
    return 1
  fi
}

stop_service() {
  local svc="$1"
  cleanup_pidfile_if_stale "$svc"

  local pidfile
  pidfile="$(service_pidfile "$svc")"

  if [[ ! -f "$pidfile" ]]; then
    echo "$svc is not running"
    return 0
  fi

  local pid
  pid="$(cat "$pidfile" 2>/dev/null || true)"

  if [[ -z "$pid" ]]; then
    rm -f "$pidfile"
    echo "$svc pid file was empty; cleaned up"
    return 0
  fi

  if ! kill -0 "$pid" 2>/dev/null; then
    rm -f "$pidfile"
    echo "$svc was not running; cleaned up stale pid file"
    return 0
  fi

  echo "Stopping $svc (pid $pid)..."
  kill "$pid" 2>/dev/null || true

  for _ in {1..20}; do
    if ! kill -0 "$pid" 2>/dev/null; then
      rm -f "$pidfile"
      echo "$svc stopped"
      return 0
    fi
    sleep 0.5
  done

  echo "Force killing $svc (pid $pid)..."
  kill -9 "$pid" 2>/dev/null || true
  rm -f "$pidfile"
  echo "$svc stopped"
}

status_service() {
  local svc="$1"
  cleanup_pidfile_if_stale "$svc"

  local pidfile
  pidfile="$(service_pidfile "$svc")"

  if is_running "$svc"; then
    local pid
    pid="$(cat "$pidfile")"
    echo "$svc: running (pid $pid)"
  else
    echo "$svc: stopped"
  fi
}

start_all() {
  for svc in brain ears eyes mouth interface; do
    start_service "$svc"
  done
}

stop_all() {
  for svc in interface mouth eyes ears brain; do
    stop_service "$svc"
  done
}

status_all() {
  for svc in brain ears eyes mouth interface; do
    status_service "$svc"
  done
}

restart_all() {
  stop_all
  start_all
}

usage() {
  cat <<EOF
Usage: $(basename "$0") {start|stop|restart|status} [service]

Services:
  all (default)
  brain
  ears
  eyes
  mouth
  interface
EOF
}

run_action() {
  local action="$1"
  local target="${2:-all}"

  case "$target" in
    all)
      case "$action" in
        start) start_all ;;
        stop) stop_all ;;
        restart) restart_all ;;
        status) status_all ;;
        *) usage; exit 1 ;;
      esac
      ;;
    brain|ears|eyes|mouth|interface)
      case "$action" in
        start) start_service "$target" ;;
        stop) stop_service "$target" ;;
        restart) stop_service "$target"; start_service "$target" ;;
        status) status_service "$target" ;;
        *) usage; exit 1 ;;
      esac
      ;;
    *)
      usage
      exit 1
      ;;
  esac
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

run_action "$1" "${2:-all}"
