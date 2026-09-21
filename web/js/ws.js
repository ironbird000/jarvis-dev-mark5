window.JarvisWS = (() => {
  let ws = null;

  function getSessionToken() {
    return localStorage.getItem("jarvis_mark3_session") || "";
  }

  function getBasePath() {
    const path = window.location.pathname || "";
    if (path.startsWith("/dev-mark3")) {
      return "/dev-mark3";
    }
    return "";
  }

  function connect({ onOpen, onMessage, onClose, onError }) {
    const proto = window.location.protocol === "https:" ? "wss://" : "ws://";
    const token = encodeURIComponent(getSessionToken());
    const basePath = getBasePath();
    const url = `${proto}${window.location.host}${basePath}/jarvis-websocket?session_token=${token}`;

    ws = new WebSocket(url);
    ws.binaryType = "arraybuffer";

    ws.onopen = () => onOpen && onOpen(ws);
    ws.onclose = (event) => onClose && onClose(event);
    ws.onerror = (event) => onError && onError(event);
    ws.onmessage = (event) => onMessage && onMessage(event);

    return ws;
  }

  function sendJson(payload) {
    if (!ws || ws.readyState !== WebSocket.OPEN) return false;
    ws.send(JSON.stringify(payload));
    return true;
  }

  function sendBinary(buffer) {
    if (!ws || ws.readyState !== WebSocket.OPEN) return false;
    ws.send(buffer);
    return true;
  }

  function getSocket() {
    return ws;
  }

  return {
    connect,
    sendJson,
    sendBinary,
    getSocket,
    getBasePath,
  };
})();
