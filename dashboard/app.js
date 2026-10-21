// PrintGuard AI dashboard - plain ES module, no build step.
const DEFECTS = [
  { key: "stringing", color: "#f59e0b" },
  { key: "layer_shift", color: "#a78bfa" },
  { key: "extrusion_error", color: "#ef4444" },
  { key: "spaghetti", color: "#38bdf8", reserved: true },
  { key: "warping", color: "#22c55e", reserved: true },
];
const COLOR = Object.fromEntries(DEFECTS.map((d) => [d.key, d.color]));
const HISTORY = 240;

const $ = (id) => document.getElementById(id);
const state = {
  history: [], // [{t, scores}]
  alerts: new Map(),
  printer: { pause_enabled: false, dry_run: true },
  img: new Image(),
  lastEvent: null,
  ws: null,
};

const fmtPct = (v) => `${Math.round((v ?? 0) * 100)}%`;
const fmtTime = (ts) => new Date(ts * 1000).toLocaleTimeString();
const label = (k) => k.replace(/_/g, " ");
const token = () => localStorage.getItem("printguard_token");

async function api(path, opts = {}) {
  const headers = { "Content-Type": "application/json", ...(opts.headers || {}) };
  if (token()) headers.Authorization = `Bearer ${token()}`;
  const res = await fetch(path, { ...opts, headers });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.detail || `HTTP ${res.status}`);
  return body;
}

// ---------------------------------------------------------------- scores
function buildScoreRows() {
  $("scores").innerHTML = DEFECTS.map(
    (d) => `<div class="score-row ${d.reserved ? "reserved" : ""}" id="row-${d.key}" title="${
      d.reserved ? "Reserved: no public training data yet" : ""
    }">
      <span class="name">${label(d.key)}</span>
      <div class="bar"><i style="background:${d.color}"></i></div>
      <span class="val">${d.reserved ? "n/a" : "–"}</span></div>`,
  ).join("");
  $("legend").innerHTML = DEFECTS.filter((d) => !d.reserved)
    .map((d) => `<span><i style="background:${d.color}"></i>${label(d.key)}</span>`)
    .join("");
}

function scoresFrom(ev) {
  const s = {};
  for (const d of ev.detections) s[d.label] = Math.max(s[d.label] || 0, d.confidence);
  for (const [k, v] of Object.entries(ev.class_probs)) if (k !== "no_error") s[k] = Math.max(s[k] || 0, v);
  for (const [k, v] of Object.entries(ev.temporal_scores)) s[k] = v; // temporal model wins when present
  return s;
}

function renderScores(ev, scores) {
  $("score-src").textContent = Object.keys(ev.temporal_scores).length ? "temporal model (B)" : "frame model (A)";
  for (const d of DEFECTS) {
    if (d.reserved) continue;
    const row = $(`row-${d.key}`);
    const v = scores[d.key] ?? 0;
    const tr = ev.trends[d.key] ?? 0;
    row.querySelector("i").style.width = fmtPct(v);
    const arrow = tr > 0.01 ? "▲" : tr < -0.01 ? "▼" : "•";
    const cls = tr > 0.01 ? "trend-up" : tr < -0.01 ? "trend-down" : "trend-flat";
    row.querySelector(".val").innerHTML = `${fmtPct(v)} <span class="${cls}">${arrow}</span>`;
  }
}

// ---------------------------------------------------------------- video
function drawFrame(ev) {
  const canvas = $("video");
  const ctx = canvas.getContext("2d");
  const img = state.img;
  if (canvas.width !== img.naturalWidth) {
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
  }
  ctx.drawImage(img, 0, 0);
  const sx = canvas.width / ev.width;
  const sy = canvas.height / ev.height;
  ctx.lineWidth = Math.max(2, canvas.width / 320);
  ctx.font = `600 ${Math.max(12, canvas.width / 60)}px Inter, system-ui, sans-serif`;
  for (const d of ev.detections) {
    const [x1, y1, x2, y2] = d.xyxy;
    const c = COLOR[d.label] || "#14b8a6";
    ctx.strokeStyle = c;
    ctx.strokeRect(x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy);
    const text = `${label(d.label)} ${fmtPct(d.confidence)}`;
    const w = ctx.measureText(text).width + 10;
    const h = parseInt(ctx.font.match(/\d+/)[0], 10) + 8;
    ctx.fillStyle = c;
    ctx.fillRect(x1 * sx, Math.max(0, y1 * sy - h), w, h);
    ctx.fillStyle = "#0b1016";
    ctx.fillText(text, x1 * sx + 5, Math.max(h - 6, y1 * sy - 6));
  }
  $("video-empty").style.display = "none";
  const badges = Object.entries(ev.class_probs)
    .filter(([k]) => k !== "no_error")
    .sort((a, b) => b[1] - a[1])
    .map(([k, v]) => `<span class="badge ${v >= 0.5 ? "hot" : ""}">${label(k)} ${fmtPct(v)}</span>`);
  $("frame-badges").innerHTML = badges.join("") || '<span class="badge">frame classifier off</span>';
}

// ---------------------------------------------------------------- chart
function drawChart() {
  const canvas = $("chart");
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  if (canvas.width !== w * dpr) {
    canvas.width = w * dpr;
    canvas.height = h * dpr;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = "rgba(139,155,176,.15)";
  ctx.lineWidth = 1;
  for (const y of [0.25, 0.5, 0.75]) {
    ctx.beginPath();
    ctx.moveTo(0, h - y * h);
    ctx.lineTo(w, h - y * h);
    ctx.stroke();
  }
  ctx.setLineDash([4, 4]);
  ctx.strokeStyle = "rgba(239,68,68,.45)";
  ctx.beginPath();
  ctx.moveTo(0, h / 2);
  ctx.lineTo(w, h / 2);
  ctx.stroke();
  ctx.setLineDash([]);
  const pts = state.history;
  if (pts.length < 2) return;
  const step = w / (HISTORY - 1);
  const off = HISTORY - pts.length;
  for (const d of DEFECTS) {
    if (d.reserved) continue;
    ctx.strokeStyle = d.color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    pts.forEach((p, i) => {
      const x = (off + i) * step;
      const y = h - (p.scores[d.key] ?? 0) * (h - 6) - 3;
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.stroke();
  }
  ctx.fillStyle = "#ef4444";
  for (let i = 0; i < pts.length; i++) {
    if (pts[i].alert) ctx.fillRect((off + i) * step - 1, 0, 2, h);
  }
}

// ---------------------------------------------------------------- alerts
function renderSafety() {
  const p = state.printer;
  $("safety").innerHTML = p.pause_enabled
    ? `Pause requests <b style="color:var(--amber)">enabled</b> (${p.backend}${p.dry_run ? ", dry-run" : ""}). A pause is only sent after you approve a specific alert.`
    : `Pause requests are <b>disabled</b> in <code>configs/server.yaml</code>. Alerts are advisory only.`;
}

function renderAlerts() {
  const list = [...state.alerts.values()].sort((a, b) => b.id - a.id).slice(0, 30);
  const open = list.filter((a) => a.status === "open").length;
  const pill = $("alert-count");
  pill.textContent = `${open} open`;
  pill.className = `pill ${open ? "pill-bad" : "pill-off"}`;
  if (!list.length) {
    $("alerts").innerHTML = '<li class="empty">No alerts yet — the printer looks healthy.</li>';
    return;
  }
  $("alerts").innerHTML = list
    .map((a) => {
      const isOpen = a.status === "open";
      return `<li class="alert ${isOpen ? "" : "closed"}" data-id="${a.id}">
        <header><span>${label(a.label)}</span><span>${fmtPct(a.score)}</span></header>
        <p>#${a.id} · frame ${a.frame_idx} · ${fmtTime(a.ts)} · ${a.status.replace("_", " ")}${a.decided_by ? ` by ${a.decided_by}` : ""}</p>
        ${
          isOpen
            ? `<div class="actions">
          <button class="btn btn-sm" data-act="dismiss">Dismiss</button>
          <button class="btn btn-sm btn-danger" data-act="pause" ${state.printer.pause_enabled ? "" : 'disabled title="Pause requests disabled"'}>Approve pause</button>
        </div>`
            : ""
        }</li>`;
    })
    .join("");
}

async function onAlertClick(e) {
  const btn = e.target.closest("button[data-act]");
  if (!btn) return;
  const id = Number(btn.closest(".alert").dataset.id);
  const operator = $("operator").value.trim();
  if (!operator) {
    $("operator").focus();
    $("operator").placeholder = "required to decide";
    return;
  }
  localStorage.setItem("printguard_operator", operator);
  try {
    if (btn.dataset.act === "dismiss") {
      const a = await api(`/api/alerts/${id}/dismiss`, { method: "POST", body: JSON.stringify({ operator }) });
      state.alerts.set(id, a);
    } else {
      const a = state.alerts.get(id);
      $("confirm-text").textContent = `Alert #${id}: ${label(a.label)} at ${fmtPct(a.score)} (frame ${a.frame_idx}). Approving as "${operator}".`;
      $("confirm").showModal();
      const ok = await new Promise((r) => $("confirm").addEventListener("close", () => r($("confirm").returnValue === "ok"), { once: true }));
      if (!ok) return;
      const res = await api(`/api/alerts/${id}/approve-pause`, { method: "POST", body: JSON.stringify({ operator }) });
      state.alerts.set(id, res.alert);
      setMsg(res.dry_run ? "Dry-run: pause recorded, not sent." : res.ok ? "Pause sent to printer." : `Pause failed: ${res.detail}`, !res.ok);
    }
  } catch (err) {
    setMsg(err.message, true);
  }
  renderAlerts();
}

// ---------------------------------------------------------------- status / events
function renderStatus(st) {
  if (!st) return;
  $("fps").textContent = `${st.fps ?? 0} fps`;
  $("latency").textContent = st.latency_ms != null ? `${st.latency_ms} ms` : "– ms";
  $("frames").textContent = st.frames_processed ?? 0;
  const m = st.models || {};
  $("models").innerHTML = [
    ["Detector", m.detector],
    ["Classifier", m.classifier],
    ["Temporal", m.temporal],
  ]
    .map(([n, on]) => `<span class="tag ${on ? "on" : ""}">${n}</span>`)
    .join("");
  if (!st.running) $("video-empty").style.display = state.lastEvent ? "none" : "grid";
  if (st.last_error) setMsg(`Worker error: ${st.last_error}`, true);
}

async function refreshEvents() {
  try {
    const rows = await api("/api/events?limit=12");
    $("events").innerHTML =
      rows
        .map(
          (r) => `<tr><td>${fmtTime(r.ts)}</td><td>${r.frame_idx}</td>
      <td><span style="color:${COLOR[r.label] || "inherit"}">●</span> ${label(r.label)}</td>
      <td>${r.source}</td><td>${fmtPct(r.confidence)}</td>
      <td>${r.trend == null ? "–" : r.trend > 0.01 ? "▲ growing" : r.trend < -0.01 ? "▼ fading" : "steady"}</td></tr>`,
        )
        .join("") || '<tr><td colspan="6" class="empty">No events recorded</td></tr>';
  } catch {
    /* server restarting */
  }
}

function setMsg(text, isErr = false) {
  $("source-msg").textContent = text;
  $("source-msg").className = `hint ${isErr ? "err" : ""}`;
}

// ---------------------------------------------------------------- websocket
function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  state.ws = ws;
  ws.onopen = () => {
    $("conn").textContent = "live";
    $("conn").className = "pill pill-on";
  };
  ws.onclose = () => {
    $("conn").textContent = "reconnecting";
    $("conn").className = "pill pill-warn";
    setTimeout(connect, 1500);
  };
  ws.onmessage = (msg) => {
    const data = JSON.parse(msg.data);
    if (data.printer) {
      state.printer = data.printer;
      renderSafety();
      renderAlerts();
    }
    if (data.type === "status") renderStatus(data.status);
    if (data.type === "error") setMsg(data.message, true);
    if (data.type !== "frame") return;
    const ev = data.event;
    state.lastEvent = ev;
    const scores = scoresFrom(ev);
    state.history.push({ t: ev.timestamp, scores, alert: data.alerts.length > 0 });
    if (state.history.length > HISTORY) state.history.shift();
    for (const a of data.alerts) state.alerts.set(a.id, a);
    if (data.alerts.length) renderAlerts();
    renderScores(ev, scores);
    renderStatus(data.status);
    state.img.onload = () => drawFrame(ev);
    state.img.src = `data:image/jpeg;base64,${data.jpeg}`;
    drawChart();
  };
}

// ---------------------------------------------------------------- init
async function init() {
  buildScoreRows();
  $("operator").value = localStorage.getItem("printguard_operator") || "";
  $("alerts").addEventListener("click", onAlertClick);
  $("source-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    try {
      const body = { kind: $("src-kind").value, uri: $("src-uri").value.trim(), target_fps: Number($("src-fps").value), loop: true };
      const res = await api("/api/source", { method: "POST", body: JSON.stringify(body) });
      state.history = [];
      setMsg(`Session #${res.session_id} started.`);
    } catch (err) {
      setMsg(err.message, true);
    }
  });
  $("src-stop").addEventListener("click", async () => {
    await api("/api/source/stop", { method: "POST" }).catch((err) => setMsg(err.message, true));
    setMsg("Stopped.");
  });
  try {
    const st = await api("/api/status");
    state.printer = st.printer;
    renderStatus(st.worker);
    if (st.worker.source) {
      $("src-kind").value = st.worker.source.kind;
      $("src-uri").value = st.worker.source.uri;
    }
    for (const a of await api("/api/alerts?limit=30")) state.alerts.set(a.id, a);
  } catch (err) {
    setMsg(err.message, true);
  }
  renderSafety();
  renderAlerts();
  refreshEvents();
  setInterval(refreshEvents, 2000);
  window.addEventListener("resize", drawChart);
  drawChart();
  connect();
}

init();
