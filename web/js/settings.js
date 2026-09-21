window.JarvisSettings = (() => {
  const modal = document.getElementById("settings-modal");
  const emailInput = document.getElementById("settings-email");
  const firstNameInput = document.getElementById("settings-first-name");
  const lastNameInput = document.getElementById("settings-last-name");
  const displayNameInput = document.getElementById("settings-display-name");
  const homeLocationInput = document.getElementById("settings-home-location");
  const preferredLanguageInput = document.getElementById("settings-preferred-language");
  const browserCameraEnabledInput = document.getElementById("settings-browser-camera-enabled");
  const browserCameraPermissionInput = document.getElementById("settings-browser-camera-permission");
  const browserCameraDeviceInput = document.getElementById("settings-browser-camera-device");
  const browserCameraLabelInput = document.getElementById("settings-browser-camera-label");
  const passwordInput = document.getElementById("settings-password");
  const passwordConfirmInput = document.getElementById("settings-password-confirm");
  const cancelButton = document.getElementById("settings-cancel");
  const saveButton = document.getElementById("settings-save");

  let currentUser = null;
  let saveHandler = null;

  function normalizedLanguage(value) {
    return (value || "auto").trim().toLowerCase() || "auto";
  }

  function open(user) {
    currentUser = user || null;
    if (!currentUser) return;

    emailInput.value = currentUser.email || "";
    firstNameInput.value = currentUser.settings?.first_name || "";
    lastNameInput.value = currentUser.settings?.last_name || "";
    displayNameInput.value = currentUser.display_name || "";
    homeLocationInput.value =
      currentUser.settings?.home_location_label ||
      currentUser.settings?.home_location ||
      "";
    preferredLanguageInput.value = normalizedLanguage(currentUser.settings?.preferred_language);
    if (browserCameraEnabledInput) {
      browserCameraEnabledInput.checked = !!currentUser.settings?.browser_camera_enabled;
    }
    if (browserCameraPermissionInput) {
      browserCameraPermissionInput.value = currentUser.settings?.browser_camera_permission_state || "prompt";
    }
    if (browserCameraDeviceInput) {
      browserCameraDeviceInput.value = currentUser.settings?.browser_camera_device_id || "";
    }
    if (browserCameraLabelInput) {
      browserCameraLabelInput.value = currentUser.settings?.browser_camera_label || "";
    }
    passwordInput.value = "";
    passwordConfirmInput.value = "";
    modal.classList.remove("hidden");
    modal.classList.add("open");
  }

  function close() {
    displayNameInput.value = currentUser?.display_name || "";
    homeLocationInput.value =
      currentUser?.settings?.home_location_label ||
      currentUser?.settings?.home_location ||
      "";
    preferredLanguageInput.value = normalizedLanguage(currentUser?.settings?.preferred_language);
    if (browserCameraEnabledInput) {
      browserCameraEnabledInput.checked = !!currentUser?.settings?.browser_camera_enabled;
    }
    if (browserCameraPermissionInput) {
      browserCameraPermissionInput.value = currentUser?.settings?.browser_camera_permission_state || "prompt";
    }
    if (browserCameraDeviceInput) {
      browserCameraDeviceInput.value = currentUser?.settings?.browser_camera_device_id || "";
    }
    if (browserCameraLabelInput) {
      browserCameraLabelInput.value = currentUser?.settings?.browser_camera_label || "";
    }
    passwordInput.value = "";
    passwordConfirmInput.value = "";
    modal.classList.add("hidden");
    modal.classList.remove("open");
  }

  function bind(onSave) {
    saveHandler = onSave;

    cancelButton.addEventListener("click", close);

    browserCameraDeviceInput?.addEventListener("change", () => {
      const option = browserCameraDeviceInput.options[browserCameraDeviceInput.selectedIndex];
      if (browserCameraLabelInput && option) {
        browserCameraLabelInput.value = option.dataset.label || option.textContent || "";
      }
    });

    saveButton.addEventListener("click", async () => {
      if (!saveHandler || !currentUser) return;

      await saveHandler({
        display_name: displayNameInput.value.trim(),
        preferred_name: displayNameInput.value.trim(),
        new_password: passwordInput.value,
        confirm_password: passwordConfirmInput.value,
        settings: {
          preferred_name: displayNameInput.value.trim(),
          first_name: firstNameInput.value.trim(),
          last_name: lastNameInput.value.trim(),
          home_location: homeLocationInput.value.trim(),
          preferred_language: normalizedLanguage(preferredLanguageInput.value),
          browser_camera_enabled: browserCameraEnabledInput?.checked ? 1 : 0,
          browser_camera_permission_state: browserCameraPermissionInput?.value || "prompt",
          browser_camera_device_id: browserCameraDeviceInput?.value || "",
          browser_camera_label: browserCameraLabelInput?.value || "",
        }
      });

      passwordInput.value = "";
      passwordConfirmInput.value = "";
      close();
    });

    modal.addEventListener("click", (event) => {
      if (event.target === modal) {
        close();
      }
    });
  }

  return {
    open,
    close,
    bind,
  };
})();
