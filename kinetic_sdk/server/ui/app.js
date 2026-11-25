/* Kinetic Agent Console — vanilla client for the stdlib AgentServer API.
 *
 * Talks only to: GET /health and POST /runs {message, session_id, stream:true}
 * (Server-Sent Events). No build step, no external JS deps.
 *
 * Security: every piece of model/tool output is rendered through
 * renderMarkdown()/textContent — never raw innerHTML of untrusted text.
 * Design tokens follow the ui-ux-pro-max "Dark Mode / OLED + run-green" system.
 */
"use strict";

/* ------------------------------------------------------------------ icons */
const I = {
  spark: '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"><path d="M12 3l1.9 5.6L19.5 10l-5.6 1.4L12 17l-1.9-5.6L4.5 10l5.6-1.4L12 3Z"/></svg>',
  user: '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="8" r="4"/><path d="M4 21c0-4 3.6-6 8-6s8 2 8 6"/></svg>',
  alert: '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M12 3 2 21h20L12 3Z"/><path d="M12 10v5M12 18v.5"/></svg>',
  copy: '<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h10"/></svg>',
  up: '<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M7 22V11M2 13v7a2 2 0 0 0 2 2h12.4a2 2 0 0 0 2-1.6l1.5-7A2 2 0 0 0 18 11h-5l1-4a2 2 0 0 0-2-2.5L7 11"/></svg>',
  down: '<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 2v11M22 11V4a2 2 0 0 0-2-2H7.6a2 2 0 0 0-2 1.6l-1.5 7A2 2 0 0 0 6 13h5l-1 4a2 2 0 0 0 2 2.5L17 13"/></svg>',
  retry: '<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12a9 9 0 1 1-2.6-6.4"/><path d="M21 3v6h-6"/></svg>',
  chev: '<svg class="chev" viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 6l6 6-6 6"/></svg>',
  terminal: '<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 17l6-5-6-5"/><path d="M12 19h8"/></svg>',
  route: '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="6" cy="19" r="2"/><circle cx="18" cy="5" r="2"/><path d="M8 19h6a4 4 0 0 0 0-8H10a4 4 0 0 1 0-8h6"/></svg>',
  check: '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>',
};

/* ------------------------------------------------------------------ state */
const SESSION_KEY = "kinetic.session_id";
const THEME_KEY = "kinetic.theme";

const $ = (id) => document.getElementById(id);
const messages = $("messages");
const emptyState = $("empty-state");
const input = $("input");
const form = $("form");
const sendBtn = $("send");
const statusPill = $("status-pill");
const statusText = $("status-text");

let sessionId = localStorage.getItem(SESSION_KEY) || makeId();
localStorage.setItem(SESSION_KEY, sessionId);

let running = false;
let streamedThisRun = false;
let current = null; // active streaming bubble
let lastUserMessage = null;
let lastAssistantWrap = null; // most recent assistant message (for run_id)

/* ------------------------------------------------------------------ utils */
function makeId() {
  if (window.crypto && crypto.randomUUID) return crypto.randomUUID();
  return "s-" + Date.now().toString(36) + "-" + Math.random().toString(36).slice(2, 8);
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

/* Minimal, safe markdown: fenced code, inline code, bold, italic, links. */
function renderMarkdown(raw) {
  const parts = String(raw).split("```");
  let html = "";
  for (let i = 0; i < parts.length; i++) {
    if (i % 2 === 1) {
      let code = parts[i];
      let lang = "";
      const nl = code.indexOf("\n");
      if (nl >= 0) {
        const first = code.slice(0, nl).trim();
        if (/^[a-z0-9+#._-]{1,20}$/i.test(first)) { lang = first; code = code.slice(nl + 1); }
      }
      code = code.replace(/\n$/, "");
      html +=
        '<div class="codeblock"><div class="codebar"><span>' + escapeHtml(lang || "code") +
        '</span><button class="copy-code" type="button">Copy</button></div><pre><code>' +
        escapeHtml(code) + "</code></pre></div>";
    } else {
      let t = escapeHtml(parts[i]);
      t = t.replace(/`([^`\n]+)`/g, "<code>$1</code>");
      t = t.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
      t = t.replace(/(^|[\s(])\*([^*\n]+)\*(?=[\s).,!?]|$)/g, "$1<em>$2</em>");
      t = t.replace(/(https?:\/\/[^\s<]+)/g, '<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>');
      t = t.replace(/\n/g, "<br>");
      html += t;
    }
  }
  return html;
}

async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch (_e) { /* fall through */ }
  try {
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.style.position = "fixed";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.select();
    document.execCommand("copy");
    ta.remove();
    return true;
  } catch (_e) {
    return false;
  }
}

let toastTimer = 0;
function toast(msg) {
  const el = $("toast");
  el.textContent = msg;
  el.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove("show"), 1800);
}

function setStatus(state, text) {
  statusPill.dataset.state = state;
  statusText.textContent = text;
}

function scrollBottom() {
  messages.scrollTop = messages.scrollHeight;
}

/* ------------------------------------------------------------------ DOM */
function clearEmpty() {
  if (emptyState && emptyState.isConnected) emptyState.remove();
}

function msgShell(role) {
  const wrap = document.createElement("div");
  wrap.className = "msg " + role;

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.innerHTML = role === "user" ? I.user : role === "err" ? I.alert : I.spark;

  const main = document.createElement("div");
  main.className = "msg-main";

  const head = document.createElement("div");
  head.className = "msg-head";
  const who = document.createElement("span");
  who.className = "who";
  who.textContent = role === "user" ? "You" : "Kinetic";
  const roleTag = document.createElement("span");
  roleTag.className = "role";
  roleTag.textContent = role === "user" ? "you" : "assistant";
  head.append(who, roleTag);

  main.appendChild(head);
  wrap.append(avatar, main);
  return { wrap, main };
}

function addActions(wrap, rawText) {
  const actions = document.createElement("div");
  actions.className = "msg-actions";
  const mk = (act, icon, label) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "icon-btn";
    b.dataset.act = act;
    b.setAttribute("aria-label", label);
    b.title = label;
    b.innerHTML = icon;
    actions.appendChild(b);
    return b;
  };
  mk("copy", I.copy, "Copy");
  mk("up", I.up, "Good response");
  mk("down", I.down, "Bad response");
  mk("retry", I.retry, "Retry");
  wrap.querySelector(".msg-main").appendChild(actions);
  wrap._raw = rawText;
}

function addUserMessage(text) {
  clearEmpty();
  const { wrap, main } = msgShell("user");
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = escapeHtml(text).replace(/\n/g, "<br>");
  main.appendChild(bubble);
  messages.appendChild(wrap);
  scrollBottom();
}

function addAssistantMessage(text) {
  clearEmpty();
  const { wrap, main } = msgShell("assistant");
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = renderMarkdown(text);
  main.appendChild(bubble);
  addActions(wrap, text);
  messages.appendChild(wrap);
  lastAssistantWrap = wrap;
  scrollBottom();
}

function typingEl() {
  const t = document.createElement("div");
  t.className = "typing";
  t.innerHTML = "<i></i><i></i><i></i>";
  return t;
}

function openAssistant() {
  if (current) return current;
  clearEmpty();
  const { wrap, main } = msgShell("assistant");
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  const typing = typingEl();
  bubble.appendChild(typing);
  main.appendChild(bubble);
  messages.appendChild(wrap);
  scrollBottom();
  current = { wrap, main, bubble, raw: "", typingShown: true, raf: 0, caret: null };
  return current;
}

function appendDelta(text) {
  const c = openAssistant();
  if (c.typingShown) { c.bubble.innerHTML = ""; c.typingShown = false; }
  c.raw += text;
  if (!c.raf) {
    c.raf = requestAnimationFrame(() => {
      c.raf = 0;
      c.bubble.innerHTML = renderMarkdown(c.raw);
      const caret = document.createElement("span");
      caret.className = "caret";
      c.bubble.appendChild(caret);
      scrollBottom();
    });
  }
}

function closeAssistant() {
  if (!current) return;
  const c = current;
  current = null;
  if (c.raf) { cancelAnimationFrame(c.raf); c.raf = 0; }
  if (!c.raw.trim()) { c.wrap.remove(); return; }
  c.bubble.innerHTML = renderMarkdown(c.raw);
  addActions(c.wrap, c.raw);
  lastAssistantWrap = c.wrap;
}

/* Insert before an empty (pre-opened) assistant bubble so status/tool lines
 * keep chronological order; otherwise append at the end. */
function insertNode(node) {
  if (current && !current.raw.trim() && current.wrap.isConnected) {
    messages.insertBefore(node, current.wrap);
  } else {
    messages.appendChild(node);
  }
  scrollBottom();
}

function addStatus(kind, text, icon) {
  clearEmpty();
  const el = document.createElement("div");
  el.className = "statusline" + (kind === "err" ? " err" : "");
  el.innerHTML = '<span class="s-ico">' + (icon || (kind === "err" ? I.alert : I.route)) + "</span>";
  const span = document.createElement("span");
  span.textContent = text;
  el.appendChild(span);
  insertNode(el);
}

function addToolCard(p) {
  clearEmpty();
  const card = document.createElement("div");
  card.className = "tool-card";
  card.dataset.error = p.is_error ? "1" : "0";

  const head = document.createElement("button");
  head.type = "button";
  head.className = "tool-head";
  head.innerHTML = '<span class="t-ico">' + I.terminal + "</span>";

  const name = document.createElement("span");
  name.className = "t-name";
  name.textContent = p.name || "tool";
  head.appendChild(name);

  const status = document.createElement("span");
  status.className = "t-status";
  status.textContent = p.is_error ? "failed" : "ok";
  head.appendChild(status);

  const meta = document.createElement("span");
  meta.className = "t-meta";
  const bits = [];
  if (typeof p.duration_seconds === "number") bits.push(p.duration_seconds.toFixed(2) + "s");
  if (typeof p.tokens_approx === "number") bits.push("~" + p.tokens_approx + " tok");
  meta.textContent = bits.join(" · ");
  head.appendChild(meta);
  head.insertAdjacentHTML("beforeend", I.chev);

  const body = document.createElement("div");
  body.className = "tool-body";
  const pre = document.createElement("pre");
  pre.textContent = p.output_preview || "(no output)";
  body.appendChild(pre);

  card.append(head, body);
  insertNode(card);
}

/* ------------------------------------------------------------------ SSE */
function handleBlock(block) {
  let type = "message";
  let data = "";
  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) type = line.slice(6).trim();
    else if (line.startsWith("data:")) data += line.slice(5).trim() + "\n";
  }
  data = data.trimEnd();
  let payload = {};
  if (data) { try { payload = JSON.parse(data); } catch (_e) { payload = { raw: data }; } }
  dispatch(type, payload);
}

function dispatch(type, p) {
  switch (type) {
    case "agent.run_started":
      streamedThisRun = false;
      setStatus("busy", "working…");
      break;
    case "agent.turn_started":
      if (current && current.raw.trim()) closeAssistant();
      break;
    case "agent.text_delta":
      streamedThisRun = true;
      appendDelta(String(p.delta || ""));
      break;
    case "agent.classified":
      addStatus("info", "routed · " + (p.complexity || "?") + " → " + (p.applied_mode || p.mode || "?"), I.route);
      break;
    case "agent.escalated":
      addStatus("info", "escalated · " + (p.from || "?") + " → " + (p.to || "?"), I.route);
      break;
    case "context.compacted":
      addStatus("info", "context compacted · −" + (p.removed ?? "?") + " messages", I.route);
      break;
    case "security.permission_denied":
      addStatus("err", "permission denied · " + (p.tool_name || "?") + (p.reason ? " — " + p.reason : ""), I.alert);
      break;
    case "agent.tool_call_finished":
      addToolCard(p);
      break;
    case "done":
      finalize(p);
      break;
    case "error":
      finalize({ status: "error", error: p.error || "unknown error" });
      break;
    default:
      break;
  }
}

function finalize(record) {
  running = false;
  closeAssistant();
  const finalText = record && record.final ? String(record.final) : "";
  if (!streamedThisRun && finalText) addAssistantMessage(finalText);
  else if (!streamedThisRun && record && record.structured_output) {
    addAssistantMessage(JSON.stringify(record.structured_output, null, 2));
  }
  // Attach the run id so the 👍/👎 actions can POST /runs/<id>/feedback.
  if (record && record.run_id && lastAssistantWrap) {
    lastAssistantWrap.dataset.runId = record.run_id;
  }
  if (record && record.status === "error") {
    addStatus("err", "run failed · " + (record.error || "unknown error"), I.alert);
    setStatus("err", "error");
    return;
  }
  const usage = record && record.usage;
  if (usage && (usage.calls || usage.total_tokens || usage.cost_usd)) {
    const bits = [];
    if (usage.calls) bits.push(usage.calls + " calls");
    if (usage.input_tokens || usage.output_tokens) bits.push((usage.input_tokens || 0) + " in / " + (usage.output_tokens || 0) + " out");
    if (usage.cost_usd) bits.push("$" + Number(usage.cost_usd).toFixed(4));
    if (bits.length) addStatus("info", "usage · " + bits.join(" · "), I.route);
  }
  setStatus("ok", "ready");
}

async function postRun(message) {
  const res = await fetch("/runs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, session_id: sessionId, stream: true }),
  });
  if (!res.ok) throw new Error("HTTP " + res.status + ": " + (await res.text()));
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let sep;
    while ((sep = buffer.indexOf("\n\n")) >= 0) {
      handleBlock(buffer.slice(0, sep));
      buffer = buffer.slice(sep + 2);
    }
  }
  if (buffer.trim()) handleBlock(buffer);
}

async function postFeedback(runId, rating) {
  try {
    const res = await fetch("/runs/" + encodeURIComponent(runId) + "/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ rating }),
    });
    return res.ok;
  } catch (_e) {
    return false;
  }
}

/* ------------------------------------------------------------------ actions */
async function sendMessage(text, echo) {
  if (running) return;
  lastUserMessage = text;
  if (echo) addUserMessage(text);
  openAssistant(); // shows the typing indicator immediately
  running = true;
  setStatus("busy", "working…");
  setComposerEnabled(false);
  try {
    await postRun(text);
  } catch (err) {
    finalize({ status: "error", error: err.message });
  } finally {
    running = false;
    setComposerEnabled(true);
    input.focus();
  }
}

function submitFromInput() {
  const text = input.value.trim();
  if (!text || running) return;
  input.value = "";
  autosize();
  sendMessage(text, true);
}

async function handleAction(btn) {
  const wrap = btn.closest(".msg");
  const act = btn.dataset.act;
  if (act === "copy") {
    const ok = await copyText(wrap._raw || "");
    toast(ok ? "Copied" : "Copy failed");
  } else if (act === "up" || act === "down") {
    const peer = wrap.querySelector('.icon-btn[data-act="' + (act === "up" ? "down" : "up") + '"]');
    if (peer) peer.classList.remove("active");
    btn.classList.toggle("active");
    if (!btn.classList.contains("active")) return;
    const runId = wrap.dataset.runId;
    if (runId) {
      btn.classList.add("active");
      postFeedback(runId, act).then((ok) =>
        toast(ok ? "Feedback saved" : "Could not save feedback")
      );
    } else {
      toast(act === "up" ? "Thanks for the feedback" : "Noted — thanks");
    }
  } else if (act === "retry") {
    if (running || !lastUserMessage) return;
    toast("Retrying last message…");
    sendMessage(lastUserMessage, false);
  }
}

function setComposerEnabled(on) {
  sendBtn.disabled = !on || !input.value.trim();
  input.readOnly = !on;
}

/* ------------------------------------------------------------------ composer */
function autosize() {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 200) + "px";
}

function onSendState() {
  sendBtn.disabled = running || !input.value.trim();
}

/* ------------------------------------------------------------------ theme */
function applyTheme(t) {
  document.documentElement.dataset.theme = t;
  localStorage.setItem(THEME_KEY, t);
  const label = $("theme-label");
  if (label) label.textContent = t === "dark" ? "Light" : "Dark";
}

/* ------------------------------------------------------------------ settings */
async function loadSettings() {
  const res = await fetch("/settings");
  if (!res.ok) return null;
  return res.json();
}

async function saveSettings(patch) {
  const res = await fetch("/settings", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patch),
  });
  if (!res.ok) throw new Error("HTTP " + res.status);
  return res.json();
}

async function openSettings() {
  const modal = $("settings-modal");
  modal.hidden = false;
  const hint = $("set-key-hint");
  hint.textContent = "Loading…";
  hint.className = "field-hint";
  try {
    const s = await loadSettings();
    if (!s) throw new Error("unavailable");
    $("set-model").value = s.model || "";
    $("set-base-url").value = s.base_url || "";
    $("set-api-key").value = "";
    const keySet = !!(s.secrets_set && s.secrets_set.api_key);
    $("set-api-key").placeholder = keySet ? "•••••••• (saved) — blank keeps it" : "sk-…";
    hint.textContent = keySet
      ? "An API key is saved on the server. Leave blank to keep it, or type a new one."
      : "No API key set yet — enter one so the agent can call a real model.";
    hint.className = "field-hint" + (keySet ? "" : " warn");
  } catch (_e) {
    hint.textContent = "Settings API is unavailable on this server.";
    hint.className = "field-hint warn";
  }
  setTimeout(() => $("set-model").focus(), 30);
}

function closeSettings() {
  $("settings-modal").hidden = true;
}

async function saveSettingsFromForm() {
  const patch = {
    model: $("set-model").value.trim(),
    base_url: $("set-base-url").value.trim(),
  };
  const apiKey = $("set-api-key").value.trim();
  if (apiKey) patch.api_key = apiKey; // blank => keep existing (server ignores missing)
  try {
    await saveSettings(patch);
    toast("Settings saved");
    closeSettings();
  } catch (_e) {
    toast("Could not save settings");
  }
}

/* ------------------------------------------------------------------ boot */
function wire() {
  // suggestions
  const sugg = $("suggestions");
  if (sugg) sugg.addEventListener("click", (e) => {
    const chip = e.target.closest(".chip");
    if (!chip || running) return;
    input.value = chip.textContent;
    autosize();
    submitFromInput();
  });

  form.addEventListener("submit", (e) => { e.preventDefault(); submitFromInput(); });

  input.addEventListener("input", () => { autosize(); onSendState(); });
  // IME-safe Enter: never submit mid-composition (VI/KO/JA/ZH).
  input.addEventListener("keydown", (e) => {
    if (e.key !== "Enter" || e.shiftKey || e.ctrlKey || e.metaKey || e.altKey) return;
    e.preventDefault();
    if (e.isComposing || e.keyCode === 229) return; // composing — let IME confirm
    if (e.repeat) return;
    submitFromInput();
  });

  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape") return;
    if (!$("settings-modal").hidden) { closeSettings(); return; }
    input.focus();
  });

  messages.addEventListener("click", (e) => {
    const copy = e.target.closest(".copy-code");
    if (copy) {
      const pre = copy.closest(".codeblock").querySelector("pre");
      copyText(pre ? pre.textContent : "").then((ok) => {
        copy.textContent = ok ? "Copied" : "Failed";
        setTimeout(() => { copy.textContent = "Copy"; }, 1200);
      });
      return;
    }
    const head = e.target.closest(".tool-head");
    if (head) { head.closest(".tool-card").classList.toggle("open"); return; }
    const action = e.target.closest(".icon-btn");
    if (action) handleAction(action);
  });

  $("new-chat").addEventListener("click", () => {
    if (running) return;
    sessionId = makeId();
    localStorage.setItem(SESSION_KEY, sessionId);
    $("session-id").textContent = shortId(sessionId);
    messages.replaceChildren();
    lastUserMessage = null;
    rebuildEmptyState();
    setStatus("ok", "ready");
    if (window.__closeMenu) window.__closeMenu();
    input.focus();
  });

  $("session-chip").addEventListener("click", async () => {
    const ok = await copyText(sessionId);
    toast(ok ? "Session id copied" : "Copy failed");
  });

  $("theme-toggle").addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    applyTheme(next);
  });

  // settings (Settings → LLM: model / base URL / API key)
  $("settings-btn").addEventListener("click", openSettings);
  $("settings-close").addEventListener("click", closeSettings);
  $("settings-save").addEventListener("click", saveSettingsFromForm);
  $("settings-modal").addEventListener("click", (e) => {
    if (e.target === $("settings-modal")) closeSettings();
  });
  $("set-api-key").addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); saveSettingsFromForm(); }
  });

  // mobile drawer
  const sidebar = document.querySelector(".sidebar");
  const scrim = $("scrim");
  const closeMenu = () => { sidebar.classList.remove("open"); scrim.hidden = true; };
  $("menu-btn").addEventListener("click", () => { sidebar.classList.add("open"); scrim.hidden = false; });
  scrim.addEventListener("click", closeMenu);
  window.__closeMenu = closeMenu;
}

function shortId(id) {
  return id.length > 8 ? id.slice(0, 8) : id;
}

function rebuildEmptyState() {
  const div = document.createElement("div");
  div.className = "empty";
  div.id = "empty-state";
  div.innerHTML =
    '<div class="empty-logo" aria-hidden="true">' +
    '<svg viewBox="0 0 24 24" width="40" height="40" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"><path d="M12 2 3 7v10l9 5 9-5V7l-9-5Z"/><path d="M12 22V12M3 7l9 5 9-5"/></svg></div>' +
    "<h2>How can I help you ship today?</h2>" +
    "<p>Ask me to explore the code, run tests, fix a build, or write a patch.</p>" +
    '<div class="suggestions" id="suggestions">' +
    '<button type="button" class="chip">Explain this repository</button>' +
    '<button type="button" class="chip">Run the test suite</button>' +
    '<button type="button" class="chip">Find and fix a bug</button>' +
    '<button type="button" class="chip">Write a small feature</button></div>';
  messages.appendChild(div);
  div.querySelector("#suggestions").addEventListener("click", (e) => {
    const chip = e.target.closest(".chip");
    if (!chip || running) return;
    input.value = chip.textContent;
    autosize();
    submitFromInput();
  });
}

async function boot() {
  applyTheme(localStorage.getItem(THEME_KEY) || "dark");
  $("session-id").textContent = shortId(sessionId);
  wire();
  input.focus();
  try {
    const res = await fetch("/health");
    if (!res.ok) throw new Error("HTTP " + res.status);
    await res.json();
    setStatus("ok", "ready");
  } catch (_e) {
    setStatus("err", "offline");
    addStatus("err", "cannot reach the server API — is `kinetic serve` running?", I.alert);
  }
  onSendState();
}

boot();
