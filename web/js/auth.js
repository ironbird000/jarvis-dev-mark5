(() => {
  const els = {
    authForm: document.getElementById("auth-form"),
    email: document.getElementById("email"),
    nameRow: document.getElementById("name-row"),
    firstName: document.getElementById("first-name"),
    lastName: document.getElementById("last-name"),
    countryField: document.getElementById("country-field"),
    country: document.getElementById("country"),
    locationRow: document.getElementById("location-row"),
    city: document.getElementById("city"),
    region: document.getElementById("region"),
    regionLabel: document.getElementById("region-label"),
    postalRow: document.getElementById("postal-row"),
    postalCode: document.getElementById("postal-code"),
    postalLabel: document.getElementById("postal-label"),
    languageField: document.getElementById("language-field"),
    language: document.getElementById("preferred-language"),
    passwordField: document.getElementById("password-field"),
    password: document.getElementById("password"),
    confirmPasswordField: document.getElementById("confirm-password-field"),
    confirmPassword: document.getElementById("confirm-password"),
    displayNameField: document.getElementById("display-name-field"),
    displayName: document.getElementById("display-name"),
    recoverMessage: document.getElementById("recover-message"),
    loginButton: document.getElementById("login-button"),
    registerButton: document.getElementById("register-button"),
    forgotPasswordLink: document.getElementById("forgot-password-link"),
    backToLoginLink: document.getElementById("back-to-login-link"),
    message: document.getElementById("message"),
  };

  let mode = "login";

  function appBasePath() {
    const path = window.location.pathname || "/";
    if (path.startsWith("/dev/")) return "/dev";
    return "";
  }

  function hudUrl() {
    return `${window.location.origin}${appBasePath()}/hud`;
  }

  function apiUrl(path) {
    return `${appBasePath()}${path}`;
  }

  function setMessage(text, isError = false) {
    els.message.textContent = text || "";
    els.message.className = `message${isError ? " error" : ""}`;
  }

  function isUnitedStates(country) {
    return (country || "").trim().toLowerCase() === "united states";
  }

  function refreshCountryLabels() {
    const us = isUnitedStates(els.country.value);
    els.regionLabel.textContent = us ? "State" : "Region / Province / State";
    els.postalLabel.textContent = us ? "ZIP code" : "Postal code";
  }

  function buildHomeLocation() {
    const country = els.country.value.trim();
    const city = els.city.value.trim();
    const region = els.region.value.trim();
    const postal = els.postalCode.value.trim();
    return [city, region, postal, country].filter(Boolean).join(", ");
  }

  function setMode(nextMode) {
    mode = nextMode;

    const isLogin = mode === "login";
    const isRegister = mode === "register";
    const isRecover = mode === "recover";

    els.nameRow.classList.toggle("hidden", !isRegister);
    els.countryField.classList.toggle("hidden", !isRegister);
    els.locationRow.classList.toggle("hidden", !isRegister);
    els.postalRow.classList.toggle("hidden", !isRegister);
    els.languageField.classList.toggle("hidden", !isRegister);
    els.displayNameField.classList.toggle("hidden", !isRegister);
    els.confirmPasswordField.classList.toggle("hidden", !isRegister);
    els.recoverMessage.classList.toggle("hidden", !isRecover);
    els.passwordField.classList.toggle("hidden", isRecover);

    els.password.required = !isRecover;
    els.firstName.required = isRegister;
    els.lastName.required = isRegister;
    els.country.required = isRegister;
    els.city.required = isRegister;
    els.region.required = isRegister;
    els.confirmPassword.required = isRegister;
    els.displayName.required = false;

    if (isRecover) {
      els.password.value = "";
      els.confirmPassword.value = "";
    }

    if (isLogin) {
      els.loginButton.textContent = "Login";
      els.loginButton.hidden = false;
      els.registerButton.hidden = false;
      els.registerButton.textContent = "New Users";
      els.forgotPasswordLink.hidden = false;
      els.backToLoginLink.classList.add("hidden");
    } else if (isRegister) {
      els.loginButton.textContent = "Create Account";
      els.loginButton.hidden = false;
      els.registerButton.hidden = false;
      els.registerButton.textContent = "Back to Login";
      els.forgotPasswordLink.hidden = true;
      els.backToLoginLink.classList.add("hidden");
    } else {
      els.loginButton.textContent = "Send Recovery Email";
      els.loginButton.hidden = false;
      els.registerButton.hidden = true;
      els.forgotPasswordLink.hidden = true;
      els.backToLoginLink.classList.remove("hidden");
    }

    refreshCountryLabels();
    setMessage("");
  }

  async function submitLogin() {
    const payload = {
      email: els.email.value.trim(),
      password: els.password.value,
    };

    const response = await fetch(apiUrl("/auth/login"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "same-origin",
      body: JSON.stringify(payload),
    });

    const data = await response.json();
    if (!data.success) {
      setMessage(data.message || "Login failed.", true);
      return;
    }

    if (data.session_token) {
      localStorage.setItem("jarvis_mark3_session", data.session_token);
    }
    if (data.user?.email) {
      localStorage.setItem("jarvis_mark3_email", data.user.email);
    }
    if (data.user?.display_name) {
      localStorage.setItem("jarvis_mark3_display_name", data.user.display_name);
    }

    window.location.assign(hudUrl());
  }

  async function submitRegister() {
    const firstName = els.firstName.value.trim();
    const lastName = els.lastName.value.trim();
    const password = els.password.value;
    const confirmPassword = els.confirmPassword.value;
    const preferredName = els.displayName.value.trim();
    const city = els.city.value.trim();
    const region = els.region.value.trim();
    const country = els.country.value.trim();

    if (!firstName) {
      setMessage("First name is required.", true);
      return;
    }
    if (!lastName) {
      setMessage("Last name is required.", true);
      return;
    }
    if (!country) {
      setMessage("Country is required.", true);
      return;
    }
    if (!city) {
      setMessage("City/Town is required.", true);
      return;
    }
    if (!region) {
      setMessage(isUnitedStates(country) ? "State is required." : "Region / Province / State is required.", true);
      return;
    }
    if (!password) {
      setMessage("Password is required.", true);
      return;
    }
    if (password !== confirmPassword) {
      setMessage("Passwords do not match.", true);
      return;
    }

    const payload = {
      email: els.email.value.trim(),
      first_name: firstName,
      last_name: lastName,
      home_location: buildHomeLocation(),
      preferred_language: els.language.value || "auto",
      password,
      confirm_password: confirmPassword,
      preferred_name: preferredName,
    };

    const response = await fetch(apiUrl("/auth/register"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "same-origin",
      body: JSON.stringify(payload),
    });

    const data = await response.json();
    if (!data.success) {
      setMessage(data.message || "Registration failed.", true);
      return;
    }

    if (data.session_token) {
      localStorage.setItem("jarvis_mark3_session", data.session_token);
    }
    if (data.user?.email) {
      localStorage.setItem("jarvis_mark3_email", data.user.email);
    }
    if (data.user?.display_name) {
      localStorage.setItem("jarvis_mark3_display_name", data.user.display_name);
    }

    window.location.assign(hudUrl());
  }

  async function submitRecover() {
    const payload = {
      email: els.email.value.trim(),
    };

    const response = await fetch(apiUrl("/auth/recover"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "same-origin",
      body: JSON.stringify(payload),
    });

    const data = await response.json();
    if (!data.success) {
      setMessage(data.message || "Recovery failed.", true);
      return;
    }

    setMessage(data.message || "Recovery email sent.");
  }

  els.country.addEventListener("change", refreshCountryLabels);

  els.registerButton.addEventListener("click", () => {
    if (mode === "login") {
      setMode("register");
      return;
    }
    if (mode === "register") {
      setMode("login");
    }
  });

  els.forgotPasswordLink.addEventListener("click", () => {
    setMode("recover");
  });

  els.backToLoginLink.addEventListener("click", () => {
    setMode("login");
  });

  els.authForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    setMessage("");

    try {
      if (mode === "login") {
        await submitLogin();
      } else if (mode === "register") {
        await submitRegister();
      } else {
        await submitRecover();
      }
    } catch (err) {
      setMessage("Request failed. Please try again.", true);
    }
  });

  refreshCountryLabels();
  setMode("login");
})();
