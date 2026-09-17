/* OrcaRouter provider settings — credential entries + capability-filtered
 * model selector. Plain ES2020, no build step, matching the repository's
 * "vanilla JS, no build step" frontend convention.
 *
 * The browser never receives an OrcaRouter key: it posts one to the server,
 * and reads back a mask.
 */
(function () {
  "use strict";

  var API = "/api/providers";

  var el = function (id) { return document.getElementById(id); };

  var state = {
    endpoints: {},
    capability: "chat",
    multimodal: false,
    models: [],
    selected: null,
    catalogSource: "",
    degraded: false,
    // Login bookkeeping. `attempt` is discarded the moment the generation
    // moves on, so a late response can never repaint a newer login.
    generation: 0,
    attemptId: null,
    busy: false,
    pollTimer: null,
    // Catalogue requests race too (provider change + refresh + key save);
    // only the newest answer may paint the selector.
    catalogGeneration: 0
  };

  function jsonFetch(url, options) {
    return fetch(url, options).then(function (response) {
      return response.json().catch(function () { return {}; }).then(function (body) {
        if (!response.ok) {
          var detail = body && body.detail ? body.detail : "HTTP " + response.status;
          throw new Error(detail);
        }
        return body;
      });
    });
  }

  function showError(node, message) {
    if (!message) { node.hidden = true; node.textContent = ""; return; }
    node.hidden = false;
    node.textContent = message;
  }

  // ---------------------------------------------------------------- status

  function loadProviders() {
    return jsonFetch(API).then(function (data) {
      state.endpoints = data.endpoints || {};
      el("endpoints").textContent =
        "auth: " + state.endpoints.auth_base +
        "  ·  inference: " + state.endpoints.api_base;
      el("key-dashboard").href = data.key_dashboard_url || "#";

      (data.providers || []).forEach(function (provider) {
        if (provider.kind === "api_key") renderApiKey(provider.status);
        if (provider.kind === "pkce") renderPkce(provider.status);
      });
      return data;
    }).catch(function (err) {
      showError(el("global-error"), "Could not read provider state: " + err.message);
    });
  }

  function renderApiKey(status) {
    var node = el("api-key-status");
    if (status && status.configured) {
      node.textContent = "Stored key " + status.secret_masked +
        " (source: " + (status.grant_id || "store") + ")";
      node.dataset.masked = "true";
    } else {
      node.textContent = "No key stored";
      node.dataset.masked = "false";
    }
  }

  function renderPkce(status) {
    var node = el("pkce-state");
    if (status && status.configured) {
      node.textContent = "Connected as " + (status.grant_id || "your account") +
        " · key " + status.secret_masked +
        " · scope " + (status.scope || "api") +
        (status.needs_reauth ? " · needs reauthorization" : " · durable key, reused until revoked");
    } else {
      node.textContent = "Not connected";
    }
  }

  // ------------------------------------------------------------- api key

  el("api-key-save").addEventListener("click", function () {
    var value = el("api-key-input").value.trim();
    showError(el("global-error"), "");
    if (!value) { showError(el("global-error"), "Enter an sk-orca-… key first."); return; }
    jsonFetch(API + "/orcarouter/key", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ api_key: value })
    }).then(function (data) {
      el("api-key-input").value = "";
      renderApiKey(data.status);
      loadModels();
    }).catch(function (err) {
      showError(el("global-error"), err.message);
    });
  });

  el("api-key-clear").addEventListener("click", function () {
    jsonFetch(API + "/orcarouter/key", { method: "DELETE" }).then(function (data) {
      renderApiKey(data.status);
    }).catch(function (err) {
      showError(el("global-error"), err.message);
    });
  });

  // ---------------------------------------------------------------- pkce

  function setBusy(busy, hint) {
    state.busy = busy;
    el("pkce-connect").disabled = busy;
    el("pkce-cancel").disabled = !busy;
    el("pkce-connect").classList.toggle("busy", busy);
    if (typeof hint === "string") el("pkce-hint").textContent = hint;
  }

  function stopPolling() {
    if (state.pollTimer) { clearInterval(state.pollTimer); state.pollTimer = null; }
  }

  el("pkce-connect").addEventListener("click", function () {
    var generation = ++state.generation;
    stopPolling();
    showError(el("global-error"), "");
    el("pkce-url").hidden = true;
    el("pkce-code-row").hidden = true;
    setBusy(true, "Starting authorization…");

    jsonFetch(API + "/orcarouter/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ flow: el("pkce-flow").value, app_name: "AutoResearchClaw" })
    }).then(function (data) {
      if (generation !== state.generation) return;   // superseded
      applyAttempt(data.attempt);
      state.pollTimer = setInterval(function () { poll(generation); }, 1000);
    }).catch(function (err) {
      if (generation !== state.generation) return;
      // The server refuses a second login while one is pending. A cancelled
      // attempt is released asynchronously, so adopt whatever it still holds
      // instead of showing the user a dead end.
      return jsonFetch(API + "/orcarouter/auth").then(function (data) {
        if (generation !== state.generation) return;
        if (data.attempt && data.attempt.status === "pending") {
          applyAttempt(data.attempt);
          state.pollTimer = setInterval(function () { poll(generation); }, 1000);
          return;
        }
        setBusy(false, "");
        showError(el("global-error"), err.message);
      }).catch(function () {
        if (generation !== state.generation) return;
        setBusy(false, "");
        showError(el("global-error"), err.message);
      });
    });
  });

  function applyAttempt(attempt) {
    if (!attempt) return;
    state.attemptId = attempt.attempt_id;
    el("pkce-url").hidden = !attempt.authorize_url;
    if (attempt.authorize_url) el("pkce-url").textContent = attempt.authorize_url;
    el("pkce-code-row").hidden = !attempt.needs_code;

    if (attempt.status === "pending") {
      setBusy(true, attempt.hint || "Waiting for approval in the browser.");
      return;
    }
    stopPolling();
    setBusy(false, "");
    el("pkce-url").hidden = true;
    el("pkce-code-row").hidden = true;
    if (attempt.status === "connected") {
      renderPkce({
        configured: true,
        grant_id: attempt.account,
        secret_masked: attempt.secret_masked,
        scope: attempt.scope,
        needs_reauth: false
      });
      loadModels();
    } else if (attempt.status === "cancelled") {
      el("pkce-hint").textContent = "Cancelled.";
    } else {
      showError(el("global-error"), attempt.error ||
        ("Authorization " + attempt.status));
    }
  }

  function poll(generation) {
    if (generation !== state.generation || !state.attemptId) return;
    jsonFetch(API + "/orcarouter/auth/" + state.attemptId)
      .then(function (data) {
        if (generation !== state.generation) return;
        applyAttempt(data.attempt);
      })
      .catch(function () { /* transient; keep polling */ });
  }

  el("pkce-cancel").addEventListener("click", function () { cancelLogin("Cancelled."); });

  el("pkce-submit").addEventListener("click", function () {
    var code = el("pkce-code").value.trim();
    if (!code || !state.attemptId) return;
    var generation = state.generation;
    el("pkce-code").value = "";
    jsonFetch(API + "/orcarouter/auth/" + state.attemptId + "/code", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ code: code })
    }).then(function (data) {
      if (generation !== state.generation) return;
      applyAttempt(data.attempt);
    }).catch(function (err) {
      if (generation !== state.generation) return;
      showError(el("global-error"), err.message);
    });
  });

  function cancelLogin(message) {
    var attemptId = state.attemptId;
    state.generation += 1;           // any in-flight response is now stale
    stopPolling();
    state.attemptId = null;
    setBusy(false, message || "");
    el("pkce-url").hidden = true;
    el("pkce-code-row").hidden = true;
    if (attemptId) {
      fetch(API + "/orcarouter/auth/" + attemptId + "/cancel", { method: "POST" })
        .catch(function () { /* the server also expires idle attempts */ });
    }
  }

  // `pagehide` may put this page into the back-forward cache: the browser can
  // restore it *without* re-running our code, so the guarded cleanup paths
  // (which correctly refuse to touch a newer generation) must not be the only
  // thing clearing the UI. Clear busy state synchronously here, then ask the
  // server to drop the attempt with keepalive so it survives page teardown.
  window.addEventListener("pagehide", function () {
    var attemptId = state.attemptId;
    state.generation += 1;
    stopPolling();
    state.attemptId = null;
    // Synchronous, unguarded UI reset — this is the bfcache case.
    state.busy = false;
    el("pkce-connect").disabled = false;
    el("pkce-cancel").disabled = true;
    el("pkce-connect").classList.remove("busy");
    el("pkce-hint").textContent = "";
    el("pkce-url").hidden = true;
    el("pkce-code-row").hidden = true;
    if (attemptId) {
      try {
        fetch(API + "/orcarouter/auth/" + attemptId + "/cancel", {
          method: "POST",
          keepalive: true
        }).catch(function () {});
      } catch (err) { /* keepalive unsupported: the attempt still expires */ }
    }
  });

  // --------------------------------------------------------------- models

  el("model-capability").addEventListener("change", function () {
    state.capability = el("model-capability").value;
    loadModels();
  });
  el("model-multimodal").addEventListener("change", function () {
    state.multimodal = el("model-multimodal").checked;
    loadModels();
  });
  el("model-refresh").addEventListener("click", function () {
    loadModels(true);
  });

  function loadModels(refresh) {
    var modality = state.multimodal ? "image" : "";
    var query = "?capability=" + encodeURIComponent(state.capability) +
      "&modality=" + encodeURIComponent(modality) +
      (refresh ? "&refresh=true" : "");
    var generation = ++state.catalogGeneration;
    el("model-trigger-label").textContent = "Loading…";
    showError(el("model-error"), "");
    return jsonFetch(API + "/orcarouter/models" + query).then(function (data) {
      if (generation !== state.catalogGeneration) return;   // superseded
      state.models = data.models || [];
      state.catalogSource = data.catalog_source || "";
      state.degraded = !!data.degraded;
      renderModels(data);
    }).catch(function (err) {
      if (generation !== state.catalogGeneration) return;
      state.models = [];
      state.selected = null;
      renderModels({ models: [], error: err.message, degraded: false });
      showError(el("model-error"), err.message);
    });
  }

  function renderModels(data) {
    var list = el("model-list");
    list.innerHTML = "";
    var models = data.models || [];

    // A selection that is no longer compatible is cleared, never kept.
    if (state.selected && !models.some(function (m) { return m.id === state.selected; })) {
      state.selected = null;
      el("model-trigger-label").textContent = "Select a model…";
    }

    models.forEach(function (model) {
      var li = document.createElement("li");
      li.setAttribute("role", "option");
      li.setAttribute("data-testid", "model-option");
      li.setAttribute("data-model-id", model.id);
      li.setAttribute("aria-selected", String(state.selected === model.id));
      if (model.verified) li.classList.add("verified");

      var id = document.createElement("span");
      id.className = "id";
      id.textContent = model.id;
      li.appendChild(id);

      var meta = document.createElement("span");
      meta.className = "meta";
      var bits = [];
      if (model.context_length) bits.push("ctx " + model.context_length);
      if (model.input_modalities && model.input_modalities.length) {
        bits.push("in " + model.input_modalities.join("/"));
      }
      if (model.reasoning_efforts && model.reasoning_efforts.length) {
        bits.push("effort " + model.reasoning_efforts.join("/"));
      }
      meta.textContent = bits.join(" · ");
      li.appendChild(meta);

      li.addEventListener("click", function () { selectModel(model.id); });
      list.appendChild(li);
    });

    if (state.selected) {
      el("model-trigger-label").textContent = state.selected;
    } else if (models.length) {
      el("model-trigger-label").textContent = "Select a model…";
    } else {
      el("model-trigger-label").textContent = "No compatible models";
    }

    var empty = el("model-empty");
    if (!models.length) {
      empty.hidden = false;
      empty.textContent = data.error
        ? "No models available: " + data.error
        : "No models for this entry point on this account.";
    } else {
      empty.hidden = true;
      empty.textContent = "";
    }

    el("model-status").textContent = models.length + " model(s) · source: " +
      (data.source || state.catalogSource ? (data.source || "live") : "unknown") +
      (state.capability === "chat" && state.multimodal ? " · image input required" : "");
    el("model-status").dataset.count = String(models.length);
    el("model-status").dataset.source = data.source || "";

    var degraded = el("model-degraded");
    if (data.degraded) {
      degraded.hidden = false;
      degraded.textContent =
        "Live model discovery failed (" + (data.error || "unknown error") +
        "). Showing the verified " + (data.source === "cache" ? "last-known-good" : "cold-start") +
        " catalogue — it is not a live result.";
    } else {
      degraded.hidden = true;
      degraded.textContent = "";
    }
    filterOptions();
  }

  function selectModel(modelId) {
    state.selected = modelId;
    el("model-trigger-label").textContent = modelId;
    Array.prototype.forEach.call(
      el("model-list").querySelectorAll("li"),
      function (li) {
        li.setAttribute("aria-selected", String(li.dataset.modelId === modelId));
      }
    );
    closePanel();
  }

  function filterOptions() {
    var term = el("model-search").value.trim().toLowerCase();
    Array.prototype.forEach.call(
      el("model-list").querySelectorAll("li"),
      function (li) {
        var match = !term || li.dataset.modelId.toLowerCase().indexOf(term) !== -1;
        li.hidden = !match;
      }
    );
  }
  el("model-search").addEventListener("input", filterOptions);

  function openPanel() {
    el("model-panel").hidden = false;
    el("model-trigger").setAttribute("aria-expanded", "true");
  }
  function closePanel() {
    el("model-panel").hidden = true;
    el("model-trigger").setAttribute("aria-expanded", "false");
  }
  el("model-trigger").addEventListener("click", function () {
    if (el("model-panel").hidden) { openPanel(); } else { closePanel(); }
  });
  document.addEventListener("click", function (event) {
    if (!el("model-combo").contains(event.target)) closePanel();
  });
  document.addEventListener("keydown", function (event) {
    if (event.key === "Escape") closePanel();
  });

  // Expose a tiny surface for the UI test to read state without scraping DOM.
  window.rcOrcaProviders = {
    state: state,
    reload: loadProviders,
    loadModels: loadModels,
    openPanel: openPanel,
    closePanel: closePanel
  };

  loadProviders().then(loadModels);
})();
