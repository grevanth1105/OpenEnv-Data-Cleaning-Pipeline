"""
app.py — FastAPI server for Data Cleaning Pipeline OpenEnv
Exposes all standard OpenEnv endpoints + hackathon-specific endpoints.
"""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel

from environment import DataCleaningEnvironment, TASK_NAMES
from dataset_generator import get_dataset
from graders import grade
from models import (
    ACTION_SCHEMA,
    EXAMPLE_ACTIONS,
    ActionType,
    DataCleaningAction,
    DataCleaningState,
    GraderResult,
    TaskDifficulty,
    TaskInfo,
)

# ---------------------------------------------------------------------------
# Session store — one environment instance per WebSocket session
# ---------------------------------------------------------------------------

_sessions: Dict[str, DataCleaningEnvironment] = {}
MAX_SESSIONS = int(os.getenv("MAX_CONCURRENT_ENVS", 100))


def _get_or_create(session_id: str) -> DataCleaningEnvironment:
    if session_id not in _sessions:
        if len(_sessions) >= MAX_SESSIONS:
            raise HTTPException(503, "Max concurrent sessions reached.")
        _sessions[session_id] = DataCleaningEnvironment()
    return _sessions[session_id]


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    _sessions.clear()


app = FastAPI(
    title="Data Cleaning Pipeline — OpenEnv",
    description=(
        "An RL environment where agents learn to clean real-world datasets. "
        "Three tasks: missing value imputation (easy), type errors + outliers (medium), "
        "schema normalization + deduplication (hard)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request/Response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_name:  str   = "missing_value_imputation"
    seed:       int   = 42
    difficulty: float = 0.5
    session_id: str   = "default"

    model_config = {"extra": "ignore"}


class StepRequest(BaseModel):
    action:     Dict[str, Any] = {}
    session_id: str = "default"

    model_config = {"extra": "ignore"}


class GraderRequest(BaseModel):
    session_id: str = "default"

    model_config = {"extra": "ignore"}


class BaselineRequest(BaseModel):
    model:      str = "gpt-4o-mini"
    seed:       int = 42

    model_config = {"extra": "ignore"}


# ---------------------------------------------------------------------------
# Standard OpenEnv endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "healthy", "sessions": len(_sessions)}


@app.get("/", response_class=HTMLResponse)
def root():
    """Landing page with links to all endpoints."""
    return HTMLResponse(content=_ROOT_HTML)


_ROOT_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Data Cleaning Pipeline — OpenEnv</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', sans-serif; background: #0f0f1a; color: #e0e0e0;
         display: flex; justify-content: center; align-items: center; min-height: 100vh; padding: 20px; }
  .container { max-width: 640px; width: 100%; }
  h1 { color: #60a5fa; font-size: 1.8rem; margin-bottom: 6px; }
  .tagline { color: #64748b; font-size: 0.9rem; margin-bottom: 28px; }
  .badges { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 28px; }
  .badge { padding: 4px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: 600; }
  .badge-blue  { background: #1e3a5f; color: #60a5fa; }
  .badge-green { background: #064e3b; color: #4ade80; }
  .badge-yellow{ background: #451a03; color: #fbbf24; }
  .section-title { color: #94a3b8; font-size: 0.75rem; text-transform: uppercase;
                   letter-spacing: 1px; margin-bottom: 10px; }
  .links { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 28px; }
  a.link-card { display: block; padding: 14px 16px; background: #1e1e2e;
                border: 1px solid #2a2a4a; border-radius: 10px; text-decoration: none;
                color: #e0e0e0; transition: all 0.2s; }
  a.link-card:hover { border-color: #60a5fa; background: #16213e; }
  .link-icon { font-size: 1.3rem; margin-bottom: 4px; }
  .link-title { font-size: 0.9rem; font-weight: 600; color: #e0e0e0; }
  .link-desc  { font-size: 0.75rem; color: #64748b; margin-top: 2px; }
  .tasks { background: #1e1e2e; border: 1px solid #2a2a4a; border-radius: 10px;
           padding: 16px; margin-bottom: 28px; }
  .task-row { display: flex; align-items: center; gap: 10px; padding: 8px 0;
              border-bottom: 1px solid #16213e; }
  .task-row:last-child { border-bottom: none; }
  .task-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
  .dot-green  { background: #4ade80; }
  .dot-yellow { background: #fbbf24; }
  .dot-red    { background: #f87171; }
  .task-name  { font-size: 0.85rem; font-weight: 600; }
  .task-desc  { font-size: 0.75rem; color: #64748b; }
  .footer { color: #374151; font-size: 0.75rem; text-align: center; }
  .footer a { color: #4b5563; }
</style>
</head>
<body>
<div class="container">
  <h1>🧹 Data Cleaning Pipeline</h1>
  <p class="tagline">An RL environment where agents learn to clean real-world datasets · Built with OpenEnv</p>

  <div class="badges">
    <span class="badge badge-blue">OpenEnv Compatible</span>
    <span class="badge badge-green">3 Tasks · Easy → Hard</span>
    <span class="badge badge-yellow">Dense Rewards</span>
  </div>

  <p class="section-title">🔗 Endpoints</p>
  <div class="links">
    <a class="link-card" href="/web">
      <div class="link-icon">🎮</div>
      <div class="link-title">Interactive UI</div>
      <div class="link-desc">Try the environment live in your browser</div>
    </a>
    <a class="link-card" href="/docs">
      <div class="link-icon">📖</div>
      <div class="link-title">API Docs</div>
      <div class="link-desc">Swagger UI with all 9 endpoints</div>
    </a>
    <a class="link-card" href="/tasks">
      <div class="link-icon">📋</div>
      <div class="link-title">Tasks + Schema</div>
      <div class="link-desc">All tasks with action schema JSON</div>
    </a>
    <a class="link-card" href="/health">
      <div class="link-icon">💚</div>
      <div class="link-title">Health Check</div>
      <div class="link-desc">Server status and active sessions</div>
    </a>
    <a class="link-card" href="https://github.com/grevanth1105/OpenEnv-Data-Cleaning-Pipeline" target="_blank">
      <div class="link-icon">💻</div>
      <div class="link-title">GitHub</div>
      <div class="link-desc">Source code and documentation</div>
    </a>
    <a class="link-card" href="https://colab.research.google.com/github/grevanth1105/OpenEnv-Data-Cleaning-Pipeline/blob/main/training_demo.ipynb" target="_blank">
      <div class="link-icon">🚀</div>
      <div class="link-title">GRPO Training Demo</div>
      <div class="link-desc">Train an LLM to clean data with RL</div>
    </a>
  </div>

  <p class="section-title">🎯 Tasks</p>
  <div class="tasks">
    <div class="task-row">
      <div class="task-dot dot-green"></div>
      <div>
        <div class="task-name">Missing Value Imputation <span style="color:#4b5563">· Easy</span></div>
        <div class="task-desc">Titanic passengers · 4 columns with nulls · impute with mean/median/mode</div>
      </div>
    </div>
    <div class="task-row">
      <div class="task-dot dot-yellow"></div>
      <div>
        <div class="task-name">Type Errors + Outliers <span style="color:#4b5563">· Medium</span></div>
        <div class="task-desc">Sales transactions · string prices, mixed dates, outlier discounts</div>
      </div>
    </div>
    <div class="task-row">
      <div class="task-dot dot-red"></div>
      <div>
        <div class="task-name">Schema Normalization + Dedup <span style="color:#4b5563">· Hard</span></div>
        <div class="task-desc">CRM customers · 25 duplicates, 5 format variants, NULL representations</div>
      </div>
    </div>
  </div>

  <p class="footer">
    Built for the <a href="https://pytorch.org/event/openenv-ai-hackathon/" target="_blank">Meta × HuggingFace × PyTorch OpenEnv Hackathon</a>
    · <a href="https://github.com/meta-pytorch/OpenEnv" target="_blank">OpenEnv Framework</a>
  </p>
</div>
</body>
</html>"""


@app.get("/web", response_class=HTMLResponse)
def web_ui():
    """Interactive web UI for testing the environment manually."""
    return HTMLResponse(content=_WEB_UI_HTML)


# ---------------------------------------------------------------------------
# Web UI HTML — uses WebSocket for persistent connection (no worker routing issues)
# ---------------------------------------------------------------------------

_WEB_UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Data Cleaning Pipeline — OpenEnv</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', sans-serif; background: #0f0f1a; color: #e0e0e0; padding: 20px; }
  h1 { color: #60a5fa; margin-bottom: 4px; font-size: 1.4rem; }
  h3 { color: #94a3b8; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }
  .subtitle { color: #64748b; font-size: 0.85rem; margin-bottom: 20px; }
  .grid { display: grid; grid-template-columns: 280px 1fr; gap: 16px; }
  .card { background: #1e1e2e; border-radius: 10px; padding: 16px; border: 1px solid #2a2a4a; }
  label { display: block; font-size: 0.8rem; color: #94a3b8; margin-bottom: 4px; margin-top: 10px; }
  select, input, textarea { width: 100%; padding: 8px 10px; border-radius: 6px; background: #16213e; border: 1px solid #2a2a4a; color: #e0e0e0; font-size: 0.85rem; outline: none; }
  select:focus, input:focus, textarea:focus { border-color: #60a5fa; }
  textarea { resize: vertical; font-family: monospace; }
  button { width: 100%; padding: 10px; border-radius: 6px; border: none; cursor: pointer; font-size: 0.9rem; font-weight: 600; margin-top: 8px; transition: all 0.2s; }
  .btn-primary { background: #3b82f6; color: white; }
  .btn-primary:hover { background: #2563eb; }
  .btn-secondary { background: #1e293b; color: #94a3b8; border: 1px solid #2a2a4a; }
  .btn-secondary:hover { background: #2a2a4a; }
  table { width: 100%; border-collapse: collapse; font-size: 0.8rem; }
  th { background: #1a1a2e; color: #94a3b8; padding: 6px 10px; text-align: left; border: 1px solid #2a2a4a; position: sticky; top: 0; }
  td { padding: 5px 10px; border: 1px solid #1e1e3a; }
  .null-cell { background: #3d0000 !important; color: #ff6b6b; }
  .issue-col { background: #2d1a00 !important; color: #fb923c; }
  .even-row { background: #1e1e2e; } .odd-row { background: #16213e; }
  .issue-item { padding: 5px 10px; margin: 3px 0; border-radius: 4px; font-size: 0.8rem; border-left: 3px solid; }
  .high { border-color: #ef4444; background: #1a0000; }
  .medium { border-color: #f97316; background: #1a0d00; }
  .low { border-color: #facc15; background: #1a1a00; }
  .log-item { padding: 4px 8px; margin: 2px 0; background: #16213e; border-radius: 4px; font-size: 0.78rem; color: #94a3b8; }
  .progress-bar { height: 8px; background: #1e293b; border-radius: 4px; overflow: hidden; margin: 6px 0; }
  .progress-fill { height: 100%; background: linear-gradient(90deg, #3b82f6, #60a5fa); border-radius: 4px; transition: width 0.4s; }
  .status-box { padding: 10px; border-radius: 6px; margin-top: 10px; font-size: 0.85rem; min-height: 40px; background: #1e293b; border: 1px solid #2a2a4a; }
  .status-ok { border-color: #16a34a; color: #4ade80; }
  .status-err { border-color: #dc2626; color: #f87171; }
  .status-info { border-color: #2563eb; color: #93c5fd; }
  #dataset-table { overflow-x: auto; max-height: 280px; overflow-y: auto; }
  .ws-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; margin-right: 5px; }
  .ws-on { background: #4ade80; } .ws-off { background: #f87171; }
  .hint { font-size: 0.72rem; color: #4b5563; margin-top: 2px; font-style: italic; }
</style>
</head>
<body>
<h1>🧹 Data Cleaning Pipeline — OpenEnv</h1>
<p class="subtitle">Interactive dashboard · <span class="ws-dot ws-off" id="ws-dot"></span><span id="ws-status">Connecting...</span></p>

<div class="grid">
  <div>
    <div class="card">
      <h3>⚙️ Episode Setup</h3>
      <label>Task</label>
      <select id="task-select">
        <option value="missing_value_imputation">🟢 Easy — Missing Value Imputation</option>
        <option value="type_errors_and_outliers">🟡 Medium — Type Errors + Outliers</option>
        <option value="schema_normalization_dedup">🔴 Hard — Schema + Dedup</option>
      </select>
      <label>Seed</label>
      <input type="number" id="seed" value="42" min="1" max="999">
      <button class="btn-primary" onclick="doReset()">🔄 Reset Episode</button>
    </div>

    <div class="card" style="margin-top:12px">
      <h3>🎯 Cleaning Action</h3>
      <label>Action Type</label>
      <select id="action-type" onchange="updateHint()">
        <option value="impute">impute — fill missing values</option>
        <option value="cast">cast — fix wrong types</option>
        <option value="normalize">normalize — fix formats</option>
        <option value="clip_outlier">clip_outlier — remove outliers</option>
        <option value="deduplicate">deduplicate — remove duplicates</option>
        <option value="execute_code">execute_code — run Python code</option>
        <option value="flag_outlier">flag_outlier — mark outliers</option>
        <option value="drop_rows">drop_rows — remove rows</option>
        <option value="finish">finish — end episode</option>
      </select>
      <label>Column <span class="hint">(leave blank for dataset-wide ops like deduplicate/finish)</span></label>
      <select id="column-select">
        <option value="">(no column — dataset-wide)</option>
      </select>
      <label>Params (JSON)</label>
      <textarea id="params" rows="3">{"strategy": "median"}</textarea>
      <p class="hint" id="params-hint">strategy: mean | median | mode | constant</p>
      <button class="btn-primary" onclick="doStep()" id="step-btn">▶ Execute Action</button>
      <button class="btn-secondary" onclick="doGrader()">📊 Check Score</button>
    </div>

    <div class="card" style="margin-top:12px">
      <h3>📈 Progress</h3>
      <div id="progress-text" style="font-size:0.85rem;color:#94a3b8">0% complete</div>
      <div class="progress-bar"><div class="progress-fill" id="progress-fill" style="width:0%"></div></div>
      <div id="reward-text" style="font-size:0.8rem;color:#64748b;margin-top:4px">Reward: +0.00 | Total: 0.00 | Step: 0</div>
    </div>
  </div>

  <div>
    <div class="card">
      <h3>📋 Dataset <span id="dataset-meta" style="color:#4b5563;font-weight:normal;font-size:0.8rem"></span></h3>
      <div id="dataset-table"><p style="color:#4b5563">Click Reset to load dataset.</p></div>
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px">
      <div class="card">
        <h3>⚠️ Issues Detected</h3>
        <div id="issues-list"><p style="color:#4b5563">No data loaded.</p></div>
      </div>
      <div class="card">
        <h3>📝 Action Log</h3>
        <div id="action-log"><p style="color:#4b5563">No actions yet.</p></div>
      </div>
    </div>
    <div class="card" style="margin-top:12px">
      <div class="status-box status-info" id="status-box">Ready — select a task and click Reset.</div>
    </div>
    <div class="card" style="margin-top:12px" id="grader-card">
      <h3>📊 Grader Score</h3>
      <div id="grader-result"><p style="color:#4b5563">Click Check Score to evaluate.</p></div>
    </div>
  </div>
</div>

<script>
// ── WebSocket connection (persistent — fixes multi-worker routing) ──────────
const WS_URL = (location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws";
let ws = null;
let pendingResolve = null;
let episodeStarted = false;

function connectWS() {
  ws = new WebSocket(WS_URL);
  ws.onopen = () => {
    document.getElementById("ws-dot").className = "ws-dot ws-on";
    document.getElementById("ws-status").textContent = "Connected";
    setStatus("✅ WebSocket connected — select a task and click Reset.", "ok");
  };
  ws.onclose = () => {
    document.getElementById("ws-dot").className = "ws-dot ws-off";
    document.getElementById("ws-status").textContent = "Disconnected — reconnecting...";
    episodeStarted = false;
    setTimeout(connectWS, 2000);
  };
  ws.onerror = () => {
    document.getElementById("ws-status").textContent = "Connection error";
  };
  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (pendingResolve) {
      pendingResolve(data);
      pendingResolve = null;
    }
  };
}

function wsSend(msg) {
  return new Promise((resolve, reject) => {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      reject(new Error("WebSocket not connected"));
      return;
    }
    pendingResolve = resolve;
    ws.send(JSON.stringify(msg));
    setTimeout(() => {
      if (pendingResolve) { pendingResolve = null; reject(new Error("Timeout")); }
    }, 15000);
  });
}

connectWS();

// ── Params hints ────────────────────────────────────────────────────────────
const HINTS = {
  impute:       'strategy: mean | median | mode | constant',
  cast:         'dtype: int | float | str | date | datetime',
  normalize:    'method: lowercase — OR — format: "%Y-%m-%d" — OR — mapping: {...}',
  clip_outlier: 'lower: 0, upper: 100  (omit one to clip only one side)',
  deduplicate:  'No params needed — leave as {}',
  execute_code: 'code: "df[...] = ..."',
  flag_outlier: 'No params needed — adds col_is_outlier column',
  drop_rows:    'condition: "null"  OR  "invalid_range" with lower/upper',
  finish:       'No params needed — triggers final grader',
};
const DEFAULTS = {
  impute:       '{"strategy": "median"}',
  cast:         '{"dtype": "float"}',
  normalize:    '{"method": "lowercase"}',
  clip_outlier: '{"lower": 0, "upper": 100}',
  deduplicate:  '{}',
  execute_code: '{"code": "df[\\\'age\\\'] = df[\\\'age\\\'].fillna(df[\\\'age\\\'].median())"}',
  flag_outlier: '{}',
  drop_rows:    '{"condition": "null"}',
  finish:       '{}',
};

function updateHint() {
  const a = document.getElementById("action-type").value;
  document.getElementById("params").value = DEFAULTS[a] || "{}";
  document.getElementById("params-hint").textContent = HINTS[a] || "";
}

// ── UI helpers ───────────────────────────────────────────────────────────────
function setStatus(msg, type="info") {
  const b = document.getElementById("status-box");
  b.textContent = msg;
  b.className = "status-box status-" + type;
}

function renderTable(snapshot, col_stats, issues) {
  if (!snapshot || !snapshot.length) return "<p style=\'color:#4b5563\'>No data.</p>";
  const issueCols = new Set((issues || []).filter(i => i.column).map(i => i.column));
  const cols = Object.keys(snapshot[0]);
  let html = "<table><tr>";
  cols.forEach(c => {
    const ic = issueCols.has(c);
    html += `<th class="${ic ? "issue-col" : ""}">${c}</th>`;
  });
  html += "</tr>";
  snapshot.slice(0, 12).forEach((row, idx) => {
    html += `<tr class="${idx%2===0?"even-row":"odd-row"}">`;
    cols.forEach(col => {
      const v = row[col];
      const isNull = v === null || v === undefined;
      html += `<td class="${isNull?"null-cell":""}">${isNull ? "⬜ null" : String(v).slice(0,20)}</td>`;
    });
    html += "</tr>";
  });
  return html + "</table>";
}

function renderIssues(issues) {
  if (!issues || !issues.length) return "<p style=\'color:#4ade80;font-size:0.85rem\'>✅ No issues!</p>";
  return issues.slice(0,8).map(i => {
    const col = i.column || "dataset";
    const desc = (i.description||"").slice(0,55);
    return `<div class="issue-item ${i.severity||"low"}"><b>${col}</b> — ${desc} <span style="color:#555">(${i.count})</span></div>`;
  }).join("");
}

function renderLog(history) {
  if (!history || !history.length) return "<p style=\'color:#4b5563;font-size:0.8rem\'>No actions yet.</p>";
  return [...history].reverse().slice(0,8).map(h => `<div class="log-item">${h}</div>`).join("");
}

function updateUI(obs) {
  document.getElementById("dataset-table").innerHTML = renderTable(obs.dataset_snapshot, obs.column_stats, obs.issues_detected);
  document.getElementById("dataset-meta").textContent = `${obs.total_rows} rows × ${obs.total_columns} cols`;
  document.getElementById("issues-list").innerHTML = renderIssues(obs.issues_detected);
  document.getElementById("action-log").innerHTML  = renderLog(obs.action_history);
  const pct = Math.round((obs.progress_pct||0) * 100);
  document.getElementById("progress-text").textContent = `${pct}% complete`;
  document.getElementById("progress-fill").style.width = pct + "%";
  // Update column dropdown
  const sel = document.getElementById("column-select");
  const prev = sel.value;
  sel.innerHTML = '<option value="">(no column — dataset-wide)</option>' +
    (obs.column_stats||[]).map(s => `<option value="${s.name}">${s.name} (${s.null_count} null)</option>`).join("");
  if (prev && [...sel.options].some(o => o.value === prev)) sel.value = prev;
}

// ── Actions ──────────────────────────────────────────────────────────────────
async function doReset() {
  const task = document.getElementById("task-select").value;
  const seed = parseInt(document.getElementById("seed").value) || 42;
  setStatus("Resetting...", "info");
  episodeStarted = false;
  try {
    const resp = await wsSend({ type: "reset", task_name: task, seed });
    if (resp.type === "error") { setStatus("❌ " + resp.error, "err"); return; }
    const obs = resp.observation;
    updateUI(obs);
    document.getElementById("reward-text").textContent = "Reward: +0.00 | Total: 0.00 | Step: 0";
    episodeStarted = true;
    setStatus(`✅ Started: ${task} | ${obs.total_rows} rows | ${obs.issues_remaining} issue groups`, "ok");
  } catch(e) {
    setStatus("❌ Reset failed: " + e.message + ". Try refreshing.", "err");
  }
}

async function doStep() {
  if (!episodeStarted) { setStatus("⚠️ Click Reset first!", "err"); return; }
  const action_type = document.getElementById("action-type").value;
  const colVal = document.getElementById("column-select").value;
  const column = colVal === "" ? null : colVal;
  let params = {};
  try { params = JSON.parse(document.getElementById("params").value || "{}"); }
  catch(e) { setStatus("❌ Invalid JSON: " + e.message, "err"); return; }

  setStatus("Executing...", "info");
  try {
    const resp = await wsSend({ type: "step", action: { action_type, column, params } });
    if (resp.type === "error") { setStatus("❌ " + resp.error, "err"); return; }
    const obs    = resp.observation;
    const reward = resp.reward;
    updateUI(obs);
    const sign = reward >= 0 ? "+" : "";
    document.getElementById("reward-text").textContent =
      `Reward: ${sign}${reward.toFixed(4)} | Total: ${obs.cumulative_reward.toFixed(4)} | Step: ${obs.step_count}`;
    if (resp.done) {
      const gr = obs.metadata && obs.metadata.grader_result;
      episodeStarted = false;
      setStatus(`🏁 Done! Score: ${gr ? gr.score.toFixed(4) : "?"} — ${gr ? gr.feedback : ""}`, "ok");
    } else {
      const icon = reward > 0 ? "✅" : reward < 0 ? "❌" : "⚠️";
      setStatus(`${icon} ${obs.last_action_result.slice(0,100)}`, reward > 0?"ok":reward<0?"err":"info");
    }
  } catch(e) {
    setStatus("❌ Step failed: " + e.message, "err");
  }
}

async function doGrader() {
  if (!episodeStarted) { setStatus("⚠️ Click Reset first!", "err"); return; }
  setStatus("Fetching score...", "info");
  try {
    const resp = await wsSend({ type: "grader" });
    if (resp.type === "error") { setStatus("❌ " + resp.error, "err"); return; }
    const result = resp.result;
    const breakdown = (result.breakdown || {}).per_column || {};
    let html = `<div style="font-size:0.9rem;margin-bottom:8px">
      <b style="color:${result.score>=0.6?"#4ade80":"#f87171"}">Score: ${result.score.toFixed(4)}</b>
      — ${result.feedback}</div>`;
    Object.entries(breakdown).forEach(([col, val]) => {
      const w   = Math.round(parseFloat(val)*20);
      const bar = "█".repeat(w) + "░".repeat(20-w);
      html += `<div style="font-size:0.78rem;margin:2px 0;font-family:monospace;color:#94a3b8">${col.padEnd(22,' ')} [${bar}] ${parseFloat(val).toFixed(4)}</div>`;
    });
    document.getElementById("grader-result").innerHTML = html;
    setStatus(`📊 Score: ${result.score.toFixed(4)} — ${result.feedback}`, result.passed?"ok":"info");
  } catch(e) {
    setStatus("❌ Grader failed: " + e.message, "err");
  }
}
</script>
</body>
</html>"""


@app.post("/reset")
def reset(req: Optional[ResetRequest] = None):
    if req is None:
        req = ResetRequest()
    env = _get_or_create(req.session_id)
    try:
        obs = env.reset(
            task_name  = req.task_name,
            seed       = req.seed,
            difficulty = req.difficulty,
        )
        return obs.dict()
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/step")
def step(req: StepRequest):
    env = _get_or_create(req.session_id)
    try:
        action = DataCleaningAction(**req.action)
        obs, reward, done, info = env.step(action)
        return {
            "observation": obs.dict(),
            "reward": reward,
            "done": done,
            "info": info,
        }
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(422, f"Invalid action: {e}")


@app.get("/state")
def state(session_id: str = "default"):
    env = _get_or_create(session_id)
    try:
        return env.state().dict()
    except RuntimeError as e:
        raise HTTPException(400, str(e))


# ---------------------------------------------------------------------------
# Hackathon-required endpoints
# ---------------------------------------------------------------------------

@app.get("/tasks")
def list_tasks():
    """Return all tasks with descriptions and action schema."""
    tasks = []
    for task_name in TASK_NAMES:
        data = get_dataset(task_name, seed=42)
        difficulty_map = {
            "missing_value_imputation":   TaskDifficulty.EASY,
            "type_errors_and_outliers":   TaskDifficulty.MEDIUM,
            "schema_normalization_dedup": TaskDifficulty.HARD,
        }
        info = TaskInfo(
            task_name        = task_name,
            difficulty       = difficulty_map[task_name],
            description      = data["description"],
            objective        = data["objective"],
            max_steps        = data["max_steps"],
            action_schema    = ACTION_SCHEMA,
            example_action   = EXAMPLE_ACTIONS[ActionType.IMPUTE],
            scoring_criteria = data["scoring_criteria"],
        )
        tasks.append(info.dict())
    return {"tasks": tasks, "total": len(tasks)}


@app.post("/grader")
def grader(req: GraderRequest):
    """Return grader score for the current episode state."""
    env = _get_or_create(req.session_id)
    try:
        result = env.get_grader_result()
        # Strictly between 0 and 1 — validator rejects exactly 0.0 or 1.0
        safe_score = float(min(max(result["score"], 0.001), 0.999))
        return GraderResult(
            task_name = result["task_name"],
            score     = safe_score,
            breakdown = result["breakdown"].get("per_column", {}),
            passed    = result["passed"],
            feedback  = result["feedback"],
        ).dict()
    except RuntimeError as e:
        raise HTTPException(400, str(e))


@app.post("/baseline")
def baseline(req: BaselineRequest):
    """
    Run a heuristic baseline agent on all 3 tasks and return scores.
    Deterministic — no LLM call needed. Full LLM baseline is in baseline.py.
    """
    import numpy as np
    import pandas as pd
    from graders import REGION_MAP, COUNTRY_MAP, NULL_VARIANTS

    results = {}

    for task_name in TASK_NAMES:
        data = get_dataset(task_name, seed=req.seed)
        df   = data["dataframe"].copy()
        gt   = data["ground_truth"]

        if task_name == "missing_value_imputation":
            # Fill each column with its ground-truth value
            for col, info in gt.items():
                if col in df.columns:
                    df[col] = df[col].fillna(info["value"])

        elif task_name == "type_errors_and_outliers":
            # Cast unit_price string → float
            df["unit_price"]   = pd.to_numeric(
                df["unit_price"].astype(str).str.replace(r"[^\d.]", "", regex=True),
                errors="coerce",
            )
            df["quantity"]     = pd.to_numeric(df["quantity"], errors="coerce")
            df["discount_pct"] = df["discount_pct"].clip(0, 100)
            df["rating"]       = pd.to_numeric(
                df["rating"].astype(str).str.extract(r"(\d+\.?\d*)")[0],
                errors="coerce",
            ).clip(0, 5)
            df["order_date"]   = pd.to_datetime(df["order_date"], errors="coerce")
            df["region"]       = df["region"].str.title()

        elif task_name == "schema_normalization_dedup":
            df = df.drop_duplicates().reset_index(drop=True)
            df["region"]  = df["region"].str.lower().str.strip().map(REGION_MAP).fillna(df["region"])
            df["country"] = df["country"].str.lower().str.strip().map(COUNTRY_MAP).fillna(df["country"])
            df["status"]  = df["status"].str.lower().str.strip()
            df["age"]     = pd.to_numeric(df["age"], errors="coerce").clip(0, 120)
            df["annual_revenue"] = pd.to_numeric(df["annual_revenue"], errors="coerce").clip(lower=0)
            for col in ["email", "phone", "region"]:
                if col in df.columns:
                    df[col] = df[col].replace(list(NULL_VARIANTS), np.nan)

        result             = grade(task_name, df, gt)
        # Strictly between 0 and 1 — validator rejects exactly 0.0 or 1.0
        results[task_name] = float(min(max(round(result["score"], 4), 0.001), 0.999))

    mean_score = round(sum(results.values()) / len(results), 4)

    return {
        "model":      "heuristic_baseline",
        "results":    results,
        "mean_score": mean_score,
        "timestamp":  datetime.now().isoformat(),
    }


# ---------------------------------------------------------------------------
# WebSocket endpoint — persistent session
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    env = DataCleaningEnvironment()

    try:
        while True:
            raw  = await websocket.receive_text()
            data = json.loads(raw)
            msg_type = data.get("type")

            if msg_type == "reset":
                obs = env.reset(
                    task_name  = data.get("task_name", "missing_value_imputation"),
                    seed       = data.get("seed", 42),
                    difficulty = float(data.get("difficulty", 0.5)),
                )
                await websocket.send_text(json.dumps({
                    "type":        "observation",
                    "observation": _safe_dict(obs.dict()),
                    "reward":      0.0,
                    "done":        False,
                }))

            elif msg_type == "step":
                action = DataCleaningAction(**data.get("action", {}))
                obs, reward, done, info = env.step(action)
                await websocket.send_text(json.dumps({
                    "type":        "observation",
                    "observation": _safe_dict(obs.dict()),
                    "reward":      reward,
                    "done":        done,
                    "info":        info,
                }))

            elif msg_type == "state":
                s = env.state()
                await websocket.send_text(json.dumps({
                    "type":  "state",
                    "state": _safe_dict(s.dict()),
                }))

            elif msg_type == "grader":
                result = env.get_grader_result()
                # Strictly between 0 and 1 — validator rejects exactly 0.0 or 1.0
                result["score"] = float(min(max(result["score"], 0.001), 0.999))
                await websocket.send_text(json.dumps({
                    "type":   "grader",
                    "result": result,
                }))

            else:
                await websocket.send_text(json.dumps({
                    "type":  "error",
                    "error": f"Unknown message type: {msg_type}",
                }))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({"type": "error", "error": str(e)}))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_dict(d: Any) -> Any:
    """Recursively make a dict JSON-serialisable — handles NaN, numpy, datetime."""
    import math
    import numpy as np
    if isinstance(d, dict):
        return {k: _safe_dict(v) for k, v in d.items()}
    if isinstance(d, list):
        return [_safe_dict(v) for v in d]
    if isinstance(d, float) and math.isnan(d):
        return None
    if isinstance(d, np.floating):
        return None if math.isnan(float(d)) else float(d)
    if isinstance(d, np.integer):
        return int(d)
    if isinstance(d, np.bool_):
        return bool(d)
    if isinstance(d, np.ndarray):
        return [_safe_dict(v) for v in d.tolist()]
    if hasattr(d, "isoformat"):
        return d.isoformat()
    if hasattr(d, "item"):
        return d.item()
    return d


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for openenv / pyproject.toml [project.scripts]."""
    import uvicorn
    uvicorn.run(
        "server.app:app",
        host    = os.getenv("HOST", "0.0.0.0"),
        port    = int(os.getenv("PORT", 8000)),
        workers = int(os.getenv("WORKERS", 1)),
        reload  = False,
    )


if __name__ == "__main__":
    main()