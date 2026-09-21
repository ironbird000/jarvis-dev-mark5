(() => {
  const APP_BASE = "/dev-mark3";

  const statusLine = document.getElementById("status-line");
  const conversationLog = document.getElementById("conversation-log");
  const textInput = document.getElementById("text-input");
  const sendButton = document.getElementById("send-button");
  const logoutButton = document.getElementById("logout-button");
  const settingsButton = document.getElementById("settings-button");
  const adminButton = document.getElementById("admin-button");
  const micToggle = document.getElementById("mic-toggle");
  const micStatus = document.getElementById("mic-status");
  const sessionUser = document.getElementById("session-user");

  const state = {
    socket: null,
    user: null,
    mediaRecorder: null,
    mediaStream: null,
    micActive: false,
  };

  function url(path) {
    return `${APP_BASE}${path}`;
  }

  function wsUrl() {
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    return `${proto}//${window.location.host}${APP_BASE}/jarvis-websocket`;
  }

  function appendMessage(role, text) {
    const row = document.createElement("div");
    row.className = `message ${role}`;

    const label = document.createElement("div");
    label.className = "message-role";
    label.textContent = role === "assistant" ? "Jarvis" : "You";

    const body = document.createElement("div");
    body.className = "message-body";
    body.textContent = text;

    row.appendChild(label);
    row.appendChild(body);
    conversationLog.appendChild(row);
    conversationLog.scrollTop = conversationLog.scrollHeight;
  }

  function setStatus(text) {
    if (statusLine) {
      statusLine.textContent = text;
    }
  }

  function setSessionUser(user) {
    if (!sessionUser) return;

    if (!user) {
      sessionUser.textContent = "Not signed in.";
      adminButton?.classList.add("hidden");
      return;
    }

    const displayName = user.display_name || user.preferred_name || user.email || "User";
    sessionUser.textContent = `${displayName} (${user.email})`;

    if (user.is_admin) {
      adminButton?.classList.remove("hidden");
    } else {
      adminButton?.classList.add("hidden");
    }
  }

  async function fetchMe() {
    const response = await fetch(url("/api/me"));
    const data = await response.json().catch(() => ({}));

    if (!response.ok || !data.success) {
      window.location.href = url("/");
      return null;
    }

    state.user = data.user;
    setSessionUser(state.user);
    return data.user;
  }

  function connectSocket() {
    const socket = new WebSocket(wsUrl());
    socket.binaryType = "arraybuffer";

    socket.onopen = () => {
      setStatus("Connected.");
    };

    socket.onmessage = async (event) => {
      if (typeof event.data !== "string") return;

      let payload = {};
      try {
        payload = JSON.parse(event.data);
      } catch {
        return;
      }

      if (payload.type === "connected") {
        state.user = {
          ...(state.user || {}),
          ...(payload.user || {}),
          is_admin: payload.user?.is_admin ?? state.user?.is_admin ?? false,
        };
        setSessionUser(state.user);

        if (payload.message) {
          appendMessage("assistant", payload.message);
          setStatus(payload.message);
        }
        return;
      }

      if (payload.type === "partial_transcript") {
        setStatus(payload.text || "Listening...");
        return;
      }

      if (payload.type === "transcript") {
        if (payload.text) appendMessage("user", payload.text);
        return;
      }

      if (payload.type === "response") {
        if (payload.text) appendMessage("assistant", payload.text);
        setStatus("Ready.");
        return;
      }

      if (payload.type === "settings_updated") {
        state.user = {
          ...(state.user || {}),
          ...(payload.user || {}),
          is_admin: payload.user?.is_admin ?? state.user?.is_admin ?? false,
        };
        setSessionUser(state.user);
        setStatus("Settings updated.");
        return;
      }

      if (payload.type === "error") {
        const message = payload.message || "An error occurred.";
        setStatus(message);
        appendMessage("assistant", message);
        return;
      }

      if (payload.type === "info") {
        setStatus(payload.message || "Info.");
      }
    };

    socket.onclose = () => {
      setStatus("Disconnected. Reconnecting...");
      setTimeout(connectSocket, 1500);
    };

    socket.onerror = () => {
      setStatus("WebSocket error.");
    };

    state.socket = socket;
  }

  function sendText() {
    const text = textInput?.value.trim();
    if (!text || !state.socket || state.socket.readyState !== WebSocket.OPEN) {
      return;
    }

    state.socket.send(JSON.stringify({
      type: "text",
      text,
    }));

    textInput.value = "";
    setStatus("Sending...");
  }

  async function logout() {
    await fetch(url("/auth/logout"), { method: "POST" });
    window.location.href = url("/");
  }

  async function toggleMic() {
    if (state.micActive) {
      stopMic();
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      state.mediaStream = stream;

      const recorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
      state.mediaRecorder = recorder;

      recorder.ondataavailable = async (event) => {
        if (!event.data || event.data.size === 0) return;
        if (!state.socket || state.socket.readyState !== WebSocket.OPEN) return;

        const buffer = await event.data.arrayBuffer();
        state.socket.send(buffer);
      };

      recorder.start(600);
      state.micActive = true;

      if (micStatus) micStatus.textContent = "Browser mic active.";
      if (micToggle) micToggle.textContent = "Stop Mic";
      setStatus("Listening...");
    } catch {
      if (micStatus) micStatus.textContent = "Unable to access microphone.";
      setStatus("Microphone access denied or unavailable.");
    }
  }

  function stopMic() {
    if (state.mediaRecorder && state.mediaRecorder.state !== "inactive") {
      state.mediaRecorder.stop();
    }

    if (state.mediaStream) {
      for (const track of state.mediaStream.getTracks()) {
        track.stop();
      }
    }

    state.mediaRecorder = null;
    state.mediaStream = null;
    state.micActive = false;

    if (micStatus) micStatus.textContent = "Browser mic inactive.";
    if (micToggle) micToggle.textContent = "Start Mic";
    setStatus("Ready.");
  }

  sendButton?.addEventListener("click", sendText);

  textInput?.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      sendText();
    }
  });

  logoutButton?.addEventListener("click", logout);
  micToggle?.addEventListener("click", toggleMic);

  settingsButton?.addEventListener("click", () => {
    if (!state.user || !window.JarvisSettings) return;
    window.JarvisSettings.open(state.user);
  });

  adminButton?.addEventListener("click", () => {
    window.location.href = url("/admin");
  });

  if (window.JarvisSettings) {
    window.JarvisSettings.bind(async (payload) => {
      const response = await fetch(url("/auth/settings"), {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify(payload),
      });

      const data = await response.json().catch(() => ({}));
      if (!response.ok || !data.success) {
        alert(data.message || "Failed to save settings.");
        throw new Error(data.message || "Failed to save settings.");
      }

      state.user = {
        ...(state.user || {}),
        ...(data.user || {}),
        is_admin: data.user?.is_admin ?? state.user?.is_admin ?? false,
      };
      setSessionUser(state.user);

      if (state.socket && state.socket.readyState === WebSocket.OPEN) {
        state.socket.send(JSON.stringify({
          type: "update_settings",
          ...payload,
        }));
      }
    });
  }

  (async function init() {
    await fetchMe();
    connectSocket();
  })();
})();
