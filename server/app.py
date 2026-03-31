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
from fastapi.responses import JSONResponse
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
    task_name: str = "missing_value_imputation"
    seed: int = 42
    session_id: str = "default"


class StepRequest(BaseModel):
    action: Dict[str, Any]
    session_id: str = "default"


class GraderRequest(BaseModel):
    session_id: str = "default"


class BaselineRequest(BaseModel):
    model: str = "gpt-4o-mini"
    seed: int = 42


# ---------------------------------------------------------------------------
# Standard OpenEnv endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "healthy", "sessions": len(_sessions)}


@app.post("/reset")
def reset(req: ResetRequest):
    env = _get_or_create(req.session_id)
    try:
        obs = env.reset(task_name=req.task_name, seed=req.seed)
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
        return GraderResult(
            task_name = result["task_name"],
            score     = result["score"],
            breakdown = result["breakdown"].get("per_column", {}),
            passed    = result["passed"],
            feedback  = result["feedback"],
        ).dict()
    except RuntimeError as e:
        raise HTTPException(400, str(e))


@app.post("/baseline")
def baseline(req: BaselineRequest):
    """
    Run a simple heuristic baseline agent on all 3 tasks and return scores.
    Uses deterministic rule-based actions (no LLM call) for reproducibility.
    The full LLM baseline is in baseline.py (requires OPENAI_API_KEY).
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
            for col, info in gt.items():
                if col in df.columns:
                    df[col] = df[col].fillna(info["value"])

        elif task_name == "type_errors_and_outliers":
            df["price"]        = pd.to_numeric(
                df["price"].astype(str).str.replace(r"[^\d.]", "", regex=True),
                errors="coerce",
            )
            df["quantity"]     = pd.to_numeric(df["quantity"], errors="coerce")
            df["discount_pct"] = df["discount_pct"].clip(0, 100)
            df["weight_kg"]    = df["weight_kg"].clip(upper=200)
            df["rating"]       = pd.to_numeric(
                df["rating"].astype(str).str.extract(r"(\d+\.?\d*)")[0],
                errors="coerce",
            ).clip(0, 5)
            df["order_date"]   = pd.to_datetime(df["order_date"], errors="coerce")

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

        result       = grade(task_name, df, gt)
        results[task_name] = round(result["score"], 4)

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
                    task_name = data.get("task_name", "missing_value_imputation"),
                    seed      = data.get("seed", 42),
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host = os.getenv("HOST", "0.0.0.0"),
        port = int(os.getenv("PORT", 8000)),
        workers = int(os.getenv("WORKERS", 1)),
        reload = False,
    )