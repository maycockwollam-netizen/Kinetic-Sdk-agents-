/* Chat UI for the Kinetic AgentServer.
 *
 * Talks only to the stdlib HTTP API: POST /runs with `stream: true` returns
 * a Server-Sent Events stream of the agent's bus events, ending in a `done`
 * event with the final RunRecord. No external dependencies, no CDN.
 *
 * All text rendered via `textContent` (never innerHTML) so untrusted model
 * or tool output cannot inject markup.
 */
"use strict";

const SESSION_KEY = "kinetic.session_id";

const conversation = document.getElementById("conversation");
const input = document.getElementById("input");
const sendBtn = document.getElementById("send");
const newChatBtn = document.getElementById("new-chat");
const statusPill = document.getElementById("status-pill");

let sessionId = localStorage.getItem(SESSION_KEY) || makeId();
localStorage.setItem(SESSION_KEY, sessionId);

let running = false;
let currentAssistant = null; // {body: HTMLElement, node: Text}
let streamedThisRun = false; // any agent.text_delta seen for the current run

function makeId() {
  if (window.crypto && crypto.randomUUID) return crypto.randomUUID();
  return "s-" + Date.now().toString(36) + "-" + Math.random().toString(36).slice(2, 8);
}

function setStatus(text, klass) {
  statusPill.textContent = text;
  statusPill.className = "pill" + (klass ? " " + klass : "");
}

function scrollBottom() {
  conversation.scrollTop = conversation.scrollHeight;
}

function addMessage(role, text) {
  const wrap = document.createElement("div");
  wrap.className = "msg " + role;

  const who = document.createElement("div");
  who.className = "who";
  who.textContent = role === "user" ? "you" : "kinetic";
  wrap.appendChild(who);

  const body = document.createElement("div");
  body.className = "body";
  body.textContent = text;
  wrap.appendChild(body);

  conversation.appendChild(wrap);
  scrollBottom();
  return wrap;
}

function addLine(cls, text) {
  const el = document.createElement("div");
  el.className = cls;
  el.textContent = text;
  conversation.appendChild(el);
  scrollBottom();
  return el;
}

function openAssistant() {
  if (currentAssistant) return currentAssistant;
  const wrap = document.createElement("div");
  wrap.className = "msg assistant";

  const who = document.createElement("div");
  who.className = "who";
  who.textContent = "kinetic";
  wrap.appendChild(who);

  const body = document.createElement("div");
  body.className = "body";
  const node = document.createTextNode("");
  body.appendChild(node);
  const cursor = document.createElement("span");
  cursor.className = "cursor";
  body.appendChild(cursor);

  wrap.appendChild(body);
  conversation.appendChild(wrap);
  currentAssistant = { body, node, cursor, open: true };
  scrollBottom();
  return currentAssistant;
}

function closeAssistant() {
  if (currentAssistant && currentAssistant.cursor) {
    currentAssistant.cursor.remove();
    currentAssistant.cursor = null;
  }
  currentAssistant = null;
}

function handleBlock(block) {
  let eventType = "message";
  let data = "";
  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) {
      eventType = line.slice(6).trim();
    } else if (line.startsWith("data:")) {
      data += line.slice(5).trim() + "\n";
    }
  }
  data = data.trimEnd();
  let payload = {};
  if (data) {
    try {
      payload = JSON.parse(data);
    } catch (_e) {
      payload = { raw: data };
    }
  }
  dispatch(eventType, payload);
}

/* Ensure an assistant bubble exists and appends *text* to it. */
function appendAssistant(text) {
  const a = openAssistant();
  a.node.appendData(text);
  scrollBottom();
}

function dispatch(type, p) {
  switch (type) {
    case "agent.run_started":
    case "agent.turn_started":
      setStatus("running…", "busy");
      // Reset the streaming bubble for this latest reply.
      closeAssistant();
      streamedThisRun = false;
      break;
    case "agent.text_delta":
      streamedThisRun = true;
      appendAssistant(String(p.delta || ""));
      break;
    case "agent.classified":
      addLine("statusline", `classified ${p.complexity || "?"} → route ${p.applied_mode || p.mode || "?"}`);
      break;
    case "agent.escalated":
      addLine("statusline", `escalated ${p.from || "?"} → ${p.to || "?"}`);
      break;
    case "context.compacted":
      addLine("statusline", `context compacted (−${p.removed ?? "?"} messages)`);
      break;
    case "context.summarization_failed":
      addLine("statusline err", "context summarization failed, fell back to truncation");
      break;
    case "security.permission_denied":
      addLine("toolline", `permission denied: ${p.tool_name || "?"} — ${p.reason || ""}`.trim());
      break;
    case "agent.tool_call_finished": {
      const ok = p.is_error ? "failed" : "ok";
      const preview = String(p.output_preview || "").replace(/\s+/g, " ").slice(0, 200);
      addLine("toolline", `tool ${p.name || "?"} [${ok}]${preview ? " — " + preview : ""}`);
      break;
    }
    case "done":
      finalize(p);
      break;
    case "error":
      setStatus("error", "err");
      addLine("statusline err", "run failed: " + (p.error || "unknown error"));
      running = false;
      closeAssistant();
      break;
    default:
      // Ignore events we do not surface (llm.usage, agent.budget_exceeded, etc.)
      break;
  }
}

function finalize(record) {
  running = false;
  closeAssistant();
  if (record && record.final && !streamedThisRun) {
    // Deltas already rendered the streamed reply; only materialise a bubble
    // when the run produced text without deltas (e.g. non-streaming turn).
    addMessage("assistant", record.final);
    setStatus("done", "ok");
  } else if (record && record.structured_output) {
    addMessage("assistant", JSON.stringify(record.structured_output, null, 2));
    setStatus("done", "ok");
  } else if (record && record.status === "error") {
    setStatus("error", "err");
    addLine("statusline err", "run failed: " + (record.error || "unknown error"));
    return;
  } else {
    setStatus("done", "ok");
  }
  if (record && record.usage) {
    addLine("statusline", `usage: ${JSON.stringify(record.usage)}`);
  }
}

async function postRun(message) {
  const res = await fetch("/runs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, session_id: sessionId, stream: true }),
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`HTTP ${res.status}: ${body}`);
  }
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
  // Any trailing block without a blank line terminator.
  if (buffer.trim()) handleBlock(buffer);
}

async function send() {
  if (running) return;
  const message = input.value.trim();
  if (!message) return;
  input.value = "";
  autosize();
  addMessage("user", message);
  setStatus("running…", "busy");
  running = true;
  try {
    await postRun(message);
  } catch (err) {
    setStatus("error", "err");
    addMessage("assistant", "Request failed: " + err.message);
  } finally {
    running = false;
    closeAssistant();
    input.focus();
  }
}

function autosize() {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 180) + "px";
}

input.addEventListener("input", autosize);
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    send();
  }
});
sendBtn.addEventListener("click", send);
newChatBtn.addEventListener("click", () => {
  sessionId = makeId();
  localStorage.setItem(SESSION_KEY, sessionId);
  conversation.replaceChildren();
  setStatus("connected", "ok");
  const note = addLine("statusline", "New chat — this conversation is no longer in the server's session cache.");
  note.setAttribute("data-empty", "1");
  input.focus();
});

async function boot() {
  try {
    const res = await fetch("/health");
    if (!res.ok) throw new Error("HTTP " + res.status);
    const health = await res.json();
    setStatus("connected", "ok");
    addLine("statusline", `server ok${health.active_sessions ? ` · ${health.active_sessions} active session(s)` : ""}`);
  } catch (_e) {
    setStatus("offline", "err");
    addLine("statusline err", "cannot reach the server API — is `kinetic serve` running?");
  }
  if (!conversation.children.length) {
    const note = document.createElement("div");
    note.className = "empty-note";
    note.textContent = "Send a message to get started.";
    conversation.appendChild(note);
  }
  sendBtn.disabled = false;
}

boot();