"""
web_ui.py — Gradio Web UI for Data Cleaning Pipeline OpenEnv
=============================================================
Interactive dashboard for testing the environment manually.
Mounted at /web on the FastAPI server.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import pandas as pd
import numpy as np

from environment import DataCleaningEnvironment, TASK_NAMES
from dataset_generator import get_dataset
from models import DataCleaningAction, ActionType

# ---------------------------------------------------------------------------
# Session state (one env per Gradio session)
# ---------------------------------------------------------------------------

_sessions: Dict[str, DataCleaningEnvironment] = {}


def _get_env(session_id: str) -> DataCleaningEnvironment:
    if session_id not in _sessions:
        _sessions[session_id] = DataCleaningEnvironment()
    return _sessions[session_id]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TASK_LABELS = {
    "missing_value_imputation":   "🟢 Easy   — Missing Value Imputation",
    "type_errors_and_outliers":   "🟡 Medium — Type Errors + Outliers",
    "schema_normalization_dedup": "🔴 Hard   — Schema Normalization + Dedup",
}

TASK_FROM_LABEL = {v: k for k, v in TASK_LABELS.items()}

ACTION_DESCRIPTIONS = {
    "impute":        "Fill missing values  |  params: {strategy: mean/median/mode}",
    "cast":          "Fix wrong types      |  params: {dtype: int/float/str/date}",
    "normalize":     "Standardize formats  |  params: {format or method or mapping}",
    "clip_outlier":  "Remove outliers      |  params: {lower: 0, upper: 100}",
    "deduplicate":   "Remove duplicates    |  params: {} (no params needed)",
    "execute_code":  "Run Python code      |  params: {code: 'df[...] = ...'}",
    "flag_outlier":  "Flag outliers        |  params: {} (adds _is_outlier column)",
    "drop_rows":     "Drop rows            |  params: {condition: null}",
    "finish":        "End episode          |  params: {} (triggers final grader)",
}


def _df_to_html(df: pd.DataFrame, issues: List[Dict]) -> str:
    """Render dataframe as HTML with null cells highlighted red."""
    if df is None or len(df) == 0:
        return "<p>No data loaded yet. Click Reset to start.</p>"

    # Columns with issues
    issue_cols = {i["column"] for i in issues if i.get("column")}
    sample = df.head(12).copy()

    html = ["<div style='overflow-x:auto;font-size:13px'>"]
    html.append("<table style='border-collapse:collapse;width:100%'>")

    # Header
    html.append("<tr style='background:#1a1a2e;color:white'>")
    for col in sample.columns:
        color = "#e63946" if col in issue_cols else "#1a1a2e"
        html.append(f"<th style='padding:6px 10px;border:1px solid #333;"
                    f"background:{color}'>{col}</th>")
    html.append("</tr>")

    # Rows
    for idx, row in sample.iterrows():
        html.append("<tr>")
        for col in sample.columns:
            val = row[col]
            is_null = val is None or (isinstance(val, float) and np.isnan(val))
            bg    = "#3d0000" if is_null else ("#1e1e2e" if idx % 2 == 0 else "#16213e")
            color = "#ff6b6b" if is_null else "#e0e0e0"
            disp  = "⬜ null" if is_null else str(val)[:20]
            html.append(f"<td style='padding:5px 10px;border:1px solid #2a2a4a;"
                        f"background:{bg};color:{color}'>{disp}</td>")
        html.append("</tr>")

    html.append("</table></div>")
    return "".join(html)


def _issues_html(issues: List[Dict], remaining: int) -> str:
    if not issues:
        return "<p style='color:#4ade80'>✅ No issues detected!</p>"

    severity_color = {"high": "#ef4444", "medium": "#f97316", "low": "#facc15"}
    html = [f"<p style='color:#a0aec0;font-size:12px'>{remaining} issue groups remaining</p>"]
    for issue in issues[:8]:
        color = severity_color.get(issue.get("severity", "low"), "#ccc")
        col   = issue.get("column") or "dataset"
        desc  = issue.get("description", "")[:60]
        count = issue.get("count", 0)
        html.append(
            f"<div style='padding:4px 8px;margin:3px 0;border-left:3px solid {color};"
            f"background:#1a1a2e;font-size:12px;color:#e0e0e0'>"
            f"<b style='color:{color}'>{col}</b> — {desc} "
            f"<span style='color:#666'>({count})</span></div>"
        )
    return "".join(html)


def _log_html(history: List[str]) -> str:
    if not history:
        return "<p style='color:#666;font-size:12px'>No actions taken yet.</p>"
    items = []
    for entry in reversed(history[-10:]):
        items.append(
            f"<div style='padding:3px 8px;margin:2px 0;background:#1a1a2e;"
            f"font-size:12px;color:#a0aec0;border-radius:4px'>{entry}</div>"
        )
    return "".join(items)


# ---------------------------------------------------------------------------
# Core UI actions
# ---------------------------------------------------------------------------

def do_reset(task_label: str, seed: int, session_id: str):
    """Reset the environment and return all UI components."""
    task_name = TASK_FROM_LABEL.get(task_label, "missing_value_imputation")
    env = _get_env(session_id)

    obs = env.reset(task_name=task_name, seed=int(seed))

    df_html     = _df_to_html(env._df, obs.issues_detected)
    issues_html = _issues_html(obs.issues_detected, obs.issues_remaining)
    log_html    = _log_html(obs.action_history)
    columns     = list(env._df.columns)

    status = (
        f"✅ Episode started | Task: **{task_name}** | "
        f"Rows: {obs.total_rows} | Issues: {obs.issues_remaining}"
    )

    return (
        df_html,
        issues_html,
        log_html,
        f"{obs.progress_pct:.0%}",
        f"Reward: +0.00 | Cumulative: 0.00 | Step: 0",
        gr.update(choices=["(all columns)"] + columns, value="(all columns)"),
        status,
    )


def do_step(
    action_type: str,
    column: str,
    params_str: str,
    session_id: str,
):
    """Execute one cleaning action."""
    env = _get_env(session_id)
    if env._state is None:
        return (
            "<p style='color:#ef4444'>⚠️ Please click Reset first!</p>",
            "", "", "0%", "", gr.update(), "❌ Not started"
        )

    # Parse params
    try:
        params = json.loads(params_str) if params_str.strip() else {}
    except json.JSONDecodeError:
        return (
            _df_to_html(env._df, []),
            "", "", f"{0:.0%}", "",
            gr.update(),
            f"❌ Invalid JSON in params: {params_str}",
        )

    col = None if column in ("(all columns)", "", None) else column

    try:
        action = DataCleaningAction(
            action_type=action_type,
            column=col,
            params=params,
        )
        obs, reward, done, info = env.step(action)
    except Exception as e:
        return (
            _df_to_html(env._df, []),
            "", "", "0%", "",
            gr.update(),
            f"❌ Error: {str(e)[:100]}",
        )

    df_html     = _df_to_html(env._df, obs.issues_detected)
    issues_html = _issues_html(obs.issues_detected, obs.issues_remaining)
    log_html    = _log_html(obs.action_history)
    columns     = list(env._df.columns)

    reward_sign = "+" if reward >= 0 else ""
    reward_str  = (
        f"Reward: {reward_sign}{reward:.4f} | "
        f"Cumulative: {obs.cumulative_reward:.4f} | "
        f"Step: {obs.step_count}"
    )

    if done:
        grader = env.get_grader_result()
        status = (
            f"🏁 Episode done! Grader score: **{grader['score']:.4f}** — "
            f"{grader['feedback']}"
        )
    else:
        status = (
            f"{'✅' if reward > 0 else '⚠️' if reward == 0 else '❌'} "
            f"{obs.last_action_result[:80]}"
        )

    return (
        df_html,
        issues_html,
        log_html,
        f"{obs.progress_pct:.0%}",
        reward_str,
        gr.update(choices=["(all columns)"] + columns),
        status,
    )


def do_grader(session_id: str):
    """Get current grader score without finishing episode."""
    env = _get_env(session_id)
    if env._state is None:
        return "⚠️ Reset first."
    result = env.get_grader_result()
    breakdown = result.get("breakdown", {}).get("per_column", {})
    lines = [
        f"**Score: {result['score']:.4f}** — {result['feedback']}",
        "",
        "**Breakdown:**",
    ]
    for col, val in breakdown.items():
        bar = "█" * int(float(val) * 20)
        lines.append(f"  `{col:25s}` [{bar:<20}] {float(val):.4f}")
    return "\n".join(lines)


def update_params_hint(action_type: str) -> str:
    """Return example params JSON for selected action."""
    examples = {
        "impute":        '{"strategy": "median"}',
        "cast":          '{"dtype": "float"}',
        "normalize":     '{"method": "lowercase"}',
        "clip_outlier":  '{"lower": 0, "upper": 100}',
        "deduplicate":   '{}',
        "execute_code":  '{"code": "df[\'age\'] = df[\'age\'].fillna(df[\'age\'].median())"}',
        "flag_outlier":  '{}',
        "drop_rows":     '{"condition": "null"}',
        "finish":        '{}',
    }
    return examples.get(action_type, "{}")


# ---------------------------------------------------------------------------
# Build Gradio app
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    session_id = str(uuid.uuid4())[:8]

    with gr.Blocks(
        title="Data Cleaning Pipeline — OpenEnv",
        theme=gr.themes.Base(
            primary_hue="blue",
            neutral_hue="slate",
        ),
        css="""
        .gradio-container { max-width: 1200px !important; }
        .status-box { font-size: 14px !important; }
        h1 { color: #60a5fa !important; }
        """,
    ) as demo:

        # ── Header ──────────────────────────────────────────────────
        gr.Markdown("""
# 🧹 Data Cleaning Pipeline — OpenEnv

**Interactive environment for training RL agents to clean real-world datasets.**
Select a task, click Reset, then issue cleaning actions one by one.
""")

        with gr.Row():
            # ── Left column ─────────────────────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Controls")

                task_radio = gr.Radio(
                    choices  = list(TASK_LABELS.values()),
                    value    = TASK_LABELS["missing_value_imputation"],
                    label    = "Task",
                )
                seed_slider = gr.Slider(
                    minimum=1, maximum=100, value=42, step=1,
                    label="Seed (controls dataset randomness)",
                )
                reset_btn = gr.Button("🔄 Reset Episode", variant="primary")

                gr.Markdown("### 🎯 Action")

                action_dd = gr.Dropdown(
                    choices = [a.value for a in ActionType],
                    value   = "impute",
                    label   = "Action Type",
                )
                action_desc = gr.Markdown(
                    value=ACTION_DESCRIPTIONS.get("impute", ""),
                    elem_classes=["status-box"],
                )
                column_dd = gr.Dropdown(
                    choices = ["(all columns)"],
                    value   = "(all columns)",
                    label   = "Column",
                )
                params_box = gr.Textbox(
                    value       = '{"strategy": "median"}',
                    label       = "Params (JSON)",
                    placeholder = '{"strategy": "median"}',
                    lines       = 2,
                )
                step_btn   = gr.Button("▶ Execute Action", variant="primary")
                grader_btn = gr.Button("📊 Check Score", variant="secondary")

                gr.Markdown("### 📈 Progress")
                progress_md = gr.Markdown("0%")
                reward_md   = gr.Markdown("Reward: +0.00 | Cumulative: 0.00 | Step: 0")

            # ── Right column ─────────────────────────────────────────
            with gr.Column(scale=2):
                gr.Markdown("### 📋 Dataset (first 12 rows)")
                dataset_html = gr.HTML(
                    value="<p style='color:#666'>Click Reset to load dataset.</p>"
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### ⚠️ Issues Detected")
                        issues_html = gr.HTML(
                            value="<p style='color:#666'>No data loaded.</p>"
                        )
                    with gr.Column(scale=1):
                        gr.Markdown("### 📝 Action Log")
                        log_html = gr.HTML(
                            value="<p style='color:#666'>No actions yet.</p>"
                        )

                gr.Markdown("### 💬 Status")
                status_md = gr.Markdown(
                    "Ready — select a task and click Reset.",
                    elem_classes=["status-box"],
                )

                grader_md = gr.Markdown("")

        # ── Hidden state ────────────────────────────────────────────
        sid_state = gr.State(session_id)

        # ── Event handlers ──────────────────────────────────────────
        reset_btn.click(
            fn      = do_reset,
            inputs  = [task_radio, seed_slider, sid_state],
            outputs = [
                dataset_html, issues_html, log_html,
                progress_md, reward_md, column_dd, status_md,
            ],
        )

        step_btn.click(
            fn      = do_step,
            inputs  = [action_dd, column_dd, params_box, sid_state],
            outputs = [
                dataset_html, issues_html, log_html,
                progress_md, reward_md, column_dd, status_md,
            ],
        )

        grader_btn.click(
            fn      = do_grader,
            inputs  = [sid_state],
            outputs = [grader_md],
        )

        action_dd.change(
            fn      = update_params_hint,
            inputs  = [action_dd],
            outputs = [params_box],
        )

        action_dd.change(
            fn      = lambda a: ACTION_DESCRIPTIONS.get(a, ""),
            inputs  = [action_dd],
            outputs = [action_desc],
        )

    return demo


# ---------------------------------------------------------------------------
# Mount on FastAPI
# ---------------------------------------------------------------------------

def mount_gradio(app):
    """Mount Gradio UI onto existing FastAPI app at /web."""
    import gradio as gr
    demo = build_ui()
    app  = gr.mount_gradio_app(app, demo, path="/web")
    return app
