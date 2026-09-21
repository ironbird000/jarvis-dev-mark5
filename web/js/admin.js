(function () {
  function basePath() {
    const path = window.location.pathname || "";
    if (path === "/dev" || path.startsWith("/dev/")) return "/dev";
    return "";
  }

  function url(path) {
    const base = basePath();
    if (!path.startsWith("/")) path = "/" + path;
    return `${base}${path}`;
  }

  async function api(path, options = {}) {
    const response = await fetch(url(path), {
      credentials: "include",
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...(options.headers || {})
      }
    });

    let data = {};
    try {
      data = await response.json();
    } catch (e) {}

    if (!response.ok) {
      const message = data.message || data.error || `Request failed: ${response.status}`;
      throw new Error(message);
    }
    return data;
  }

  const tbody = document.getElementById("users-tbody");
  const searchInput = document.getElementById("search-input");
  const roleFilter = document.getElementById("role-filter");
  const refreshBtn = document.getElementById("refresh-btn");
  const statusMsg = document.getElementById("status-msg");
  const backToHudLink = document.getElementById("back-to-hud-link");

  if (backToHudLink) {
    backToHudLink.setAttribute("href", url("/hud"));
  }

  function esc(value) {
    return String(value ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function setStatus(message) {
    statusMsg.textContent = message || "";
  }

  function userMatches(user, search, roleMode) {
    const hay = [
      user.display_name,
      user.email,
      user.github_login,
      user.first_name,
      user.last_name,
      user.home_location,
      user.home_location_label
    ].join(" ").toLowerCase();

    if (search && !hay.includes(search)) return false;
    if (roleMode === "admin" && !user.is_admin) return false;
    if (roleMode === "non-admin" && user.is_admin) return false;
    return true;
  }

  function renderUsers(users) {
    const search = (searchInput.value || "").trim().toLowerCase();
    const roleMode = roleFilter.value || "";

    const filtered = users.filter(user => userMatches(user, search, roleMode));

    tbody.innerHTML = filtered.map(user => {
      const rolePill = user.is_admin
        ? `<span class="pill admin">admin</span>`
        : `<span class="pill user">user</span>`;

      const voicePill = user.voice_enabled
        ? `<span class="pill enabled">enabled</span>`
        : `<span class="pill disabled">disabled</span>`;

      const displayName = user.display_name || "";
      const email = user.email || "";
      const githubLogin = user.github_login || "";
      const preferredLanguage = user.preferred_language || "auto";
      const homeLocation = user.home_location_label || user.home_location || "";

      return `
        <tr data-user-id="${user.id}">
          <td>${esc(user.id)}</td>
          <td>
            <div><strong>${esc(displayName)}</strong></div>
            <div class="small muted">${esc(email)}</div>
            <div class="small muted">${esc(githubLogin)}</div>
          </td>
          <td>${rolePill}</td>
          <td>${voicePill}</td>
          <td>${esc(preferredLanguage)}</td>
          <td>${esc(homeLocation)}</td>
          <td class="right">
            <button class="btn primary btn-edit" type="button" data-user-id="${user.id}">Edit</button>
            <button class="btn danger btn-delete" type="button" data-user-id="${user.id}">Delete</button>
          </td>
        </tr>
      `;
    }).join("");

    if (!filtered.length) {
      tbody.innerHTML = `
        <tr>
          <td colspan="7" class="muted">No users found.</td>
        </tr>
      `;
    }

    bindRowActions(users);
  }

  async function loadCurrentUserAndGate() {
    const data = await api("/api/me");
    if (!data.success || !data.user) {
      throw new Error("Unable to load current user");
    }
    const isAdmin = !!data.is_admin || !!data.user.is_admin;
    if (!isAdmin) {
      window.location.href = url("/hud");
      return false;
    }
    return true;
  }

  async function loadUsers() {
    setStatus("Loading users...");
    const data = await api("/api/admin/users");
    const users = data.users || [];
    renderUsers(users);
    setStatus(`Loaded ${users.length} user(s).`);
    return users;
  }

  async function deleteUser(userId) {
    const ok = window.confirm(`Delete user ${userId}? This cannot be undone.`);
    if (!ok) return;

    await api(`/api/admin/users/${userId}/delete`, {
      method: "POST"
    });

    setStatus(`Deleted user ${userId}.`);
    await loadUsers();
  }

  async function updateUser(userId, payload) {
    await api(`/api/admin/users/${userId}/update`, {
      method: "POST",
      body: JSON.stringify(payload)
    });
  }

  function bindRowActions(users) {
    document.querySelectorAll(".btn-delete").forEach(btn => {
      btn.onclick = async () => {
        const userId = btn.getAttribute("data-user-id");
        try {
          await deleteUser(userId);
        } catch (err) {
          setStatus(err.message || String(err));
        }
      };
    });

    document.querySelectorAll(".btn-edit").forEach(btn => {
      btn.onclick = async () => {
        const userId = Number(btn.getAttribute("data-user-id"));
        const user = users.find(u => Number(u.id) === userId);
        if (!user) return;

        const display_name = window.prompt("Display name:", user.display_name || "") ?? user.display_name ?? "";
        const preferred_language = window.prompt("Preferred language:", user.preferred_language || "auto") ?? user.preferred_language ?? "auto";
        const home_location = window.prompt("Home location:", user.home_location || "") ?? user.home_location ?? "";
        const home_location_label = window.prompt("Home location label:", user.home_location_label || "") ?? user.home_location_label ?? "";
        const is_admin = window.confirm("Should this user be admin? OK = yes, Cancel = no");
        const voice_enabled = window.confirm("Should voice be enabled? OK = yes, Cancel = no");

        try {
          await updateUser(userId, {
            display_name,
            preferred_language,
            home_location,
            home_location_label,
            is_admin,
            voice_enabled
          });
          setStatus(`Updated user ${userId}.`);
          await loadUsers();
        } catch (err) {
          setStatus(err.message || String(err));
        }
      };
    });
  }

  async function init() {
    try {
      const allowed = await loadCurrentUserAndGate();
      if (!allowed) return;
      await loadUsers();
    } catch (err) {
      setStatus(err.message || String(err));
    }
  }

  refreshBtn.addEventListener("click", init);
  searchInput.addEventListener("input", () => init());
  roleFilter.addEventListener("change", () => init());

  init();
})();
