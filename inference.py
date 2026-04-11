"""
Inference Script — Data Cleaning Pipeline OpenEnv
===================================================
MANDATORY ENVIRONMENT VARIABLES:
    API_BASE_URL   LLM API endpoint  (default: HF Router)
    MODEL_NAME     Model identifier  (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN       HuggingFace token (used as API key)
    SPACE_URL      HF Space URL      (default: deployed space)

STDOUT FORMAT:
    [START] task=<task> env=<benchmark> model=<model>
    [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
"""

import asyncio
import json
import os
import textwrap
import time
from typing import Any, Dict, List, Optional

import requests
import websockets
from openai import OpenAI

# Load variables from .env file if it exists (python-dotenv)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed — fall back to system env vars

# ---------------------------------------------------------------------------
# Configuration — reads from .env file or system environment variables
# ---------------------------------------------------------------------------

API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME   = os.getenv("MODEL_NAME")   or "Qwen/Qwen2.5-72B-Instruct"
SPACE_URL    = os.getenv("SPACE_URL")    or "https://revanth11-data-cleaning-env.hf.space"

BENCHMARK              = "data-cleaning-env"
TEMPERATURE            = 0.0
MAX_TOKENS             = 200
SUCCESS_SCORE_THRESHOLD = 0.5

TASKS = [
    {"name": "missing_value_imputation",   "max_steps": 12, "seed": 42, "difficulty": 0.4},
    {"name": "type_errors_and_outliers",   "max_steps": 18, "seed": 42, "difficulty": 0.5},
    {"name": "schema_normalization_dedup", "max_steps": 22, "seed": 42, "difficulty": 0.6},
    {"name": "data_type_inference",        "max_steps": 15, "seed": 42, "difficulty": 0.5},
    {"name": "text_standardization",       "max_steps": 18, "seed": 42, "difficulty": 0.5},
]

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert data engineer cleaning a real-world dataset.
    You will see column statistics and detected data quality issues.
    Issue ONE cleaning action at a time as a JSON object.

    Available actions:
    {"action_type":"impute",       "column":"col", "params":{"strategy":"median"}}
    {"action_type":"cast",         "column":"col", "params":{"dtype":"float"}}
    {"action_type":"normalize",    "column":"col", "params":{"method":"lowercase"}}
    {"action_type":"clip_outlier", "column":"col", "params":{"lower":0,"upper":100}}
    {"action_type":"deduplicate",  "column":null,  "params":{}}
    {"action_type":"execute_code", "column":null,  "params":{"code":"df['col']=df['col'].fillna(df['col'].median())"}}
    {"action_type":"finish",       "column":null,  "params":{}}

    Rules:
    - Fix HIGH severity issues first
    - Numeric nulls use median; categorical nulls use mode
    - Call finish when progress_pct > 0.85 or no issues remain
    - Never repeat the same action on the same column
    - Reply with ONLY the JSON object, nothing else
""").strip()


# ---------------------------------------------------------------------------
# Exact log format required by evaluator
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)



class DataCleaningWSEnv:
    """
    WebSocket client for the Data Cleaning Pipeline environment.
    Uses /ws endpoint — persistent connection to ONE worker,
    so reset() and step() always hit the same process.
    """

    def __init__(self, base_url: str) -> None:
        ws_url = base_url.rstrip("/").replace("https://", "wss://").replace("http://", "ws://")
        self.ws_url = ws_url + "/ws"
        self._ws    = None

    async def connect(self) -> None:
        self._ws = await websockets.connect(
            self.ws_url,
            ping_interval=20,
            ping_timeout=30,
            open_timeout=30,
        )

    async def _send(self, msg: Dict) -> Dict:
        await self._ws.send(json.dumps(msg))
        raw = await asyncio.wait_for(self._ws.recv(), timeout=45)
        return json.loads(raw)

    async def reset(self, task_name: str, seed: int = 42, difficulty: float = 0.5) -> Dict:
        resp = await self._send({"type": "reset", "task_name": task_name, "seed": seed, "difficulty": difficulty})
        return resp.get("observation", {})

    async def step(self, action: Dict) -> Dict:
        resp = await self._send({"type": "step", "action": action})
        return {
            "observation": resp.get("observation", {}),
            "reward":      resp.get("reward", 0.0),
            "done":        resp.get("done", False),
        }

    async def grader(self) -> Dict:
        resp = await self._send({"type": "grader"})
        return resp.get("result", {})

    async def close(self) -> None:
        if self._ws:
            await self._ws.close()


def build_user_prompt(obs: Dict, step: int, history: List[str]) -> str:
    issues = obs.get("issues_detected", [])
    issue_text = "\n".join(
        f"  [{i.get('severity','').upper():6s}] {i.get('column') or 'dataset'}: {i.get('description','')}"
        for i in issues
    ) or "  None remaining — call finish."

    col_text = "\n".join(
        f"  {s.get('name',''):20s} dtype={s.get('dtype',''):10s} nulls={s.get('null_count',0):3d}"
        for s in obs.get("column_stats", [])
    )

    history_block = "\n".join(history[-4:]) if history else "None"

    return textwrap.dedent(f"""
        Step {step} | Progress: {obs.get('progress_pct', 0):.0%} | Issues left: {obs.get('issues_remaining', 0)}

        REMAINING ISSUES:
        {issue_text}

        COLUMN STATISTICS:
        {col_text}

        RECENT ACTIONS:
        {history_block}

        LAST RESULT: {obs.get('last_action_result', '')}

        Reply with a single JSON cleaning action only.
    """).strip()


def parse_action(text: str) -> Optional[Dict]:
    import re
    text = re.sub(r"```(?:json)?\s*", "", text).strip()
    text = re.sub(r"```\s*$", "", text).strip()
    try:
        action = json.loads(text)
        if action.get("action_type"):
            return action
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.+\}", text, re.DOTALL)
    if match:
        try:
            action = json.loads(match.group())
            if action.get("action_type"):
                return action
        except json.JSONDecodeError:
            pass
    return None


def get_model_action(client: OpenAI, obs: Dict, step: int, history: List[str]) -> Dict:
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": build_user_prompt(obs, step, history)},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text   = (completion.choices[0].message.content or "").strip()
        action = parse_action(text)
        if action:
            return action
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
    return {"action_type": "finish", "column": None, "params": {}}


# ---------------------------------------------------------------------------
# Single task episode
# ---------------------------------------------------------------------------

async def run_task(client: OpenAI, task_name: str, max_steps: int, seed: int, difficulty: float = 0.5) -> None:
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    score   = 0.001
    success = False

    env = DataCleaningWSEnv(base_url=SPACE_URL)

    try:
        await env.connect()

        obs  = await env.reset(task_name=task_name, seed=seed, difficulty=difficulty)
        done = False
        history: List[str] = []

        print(f"[DEBUG] reset ok | rows={obs.get('total_rows')} issues={obs.get('issues_remaining')} diff={difficulty}", flush=True)

        for step in range(1, max_steps + 1):
            if done:
                break

            action     = get_model_action(client, obs, step, history)
            action_str = json.dumps(action, separators=(",", ":"))
            error      = None

            try:
                result = await env.step(action)
                obs    = result["observation"]
                reward = float(result.get("reward", 0.0))
                done   = bool(result.get("done", False))
            except Exception as exc:
                reward = 0.0
                done   = True
                error  = str(exc)[:60]
                print(f"[DEBUG] step {step} error: {exc}", flush=True)

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            history.append(f"Step {step}: {action.get('action_type')} col={action.get('column')} -> {reward:+.2f}")

            if done:
                break

        # Get final grader score — strictly between 0 and 1 (exclusive)
        try:
            grader = await env.grader()
            score  = float(grader.get("score", 0.001))
            score  = min(max(score, 0.001), 0.999)
        except Exception:
            max_total = max_steps * 0.25
            score = min(max(sum(rewards) / max_total, 0.001), 0.999) if max_total > 0 else 0.001

        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode failed: {exc}", flush=True)

    finally:
        try:
            await env.close()
        except Exception:
            pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    if not API_KEY:
        print("[DEBUG] HF_TOKEN not set. Run: set HF_TOKEN=hf_...", flush=True)
        return

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    print(f"[DEBUG] API_BASE_URL : {API_BASE_URL}", flush=True)
    print(f"[DEBUG] MODEL_NAME   : {MODEL_NAME}",   flush=True)
    print(f"[DEBUG] SPACE_URL    : {SPACE_URL}",    flush=True)
    print(f"[DEBUG] Tasks        : {[t['name'] for t in TASKS]}", flush=True)

    try:
        health = requests.get(f"{SPACE_URL}/health", timeout=15).json()
        print(f"[DEBUG] Space health : {health}", flush=True)
    except Exception as exc:
        print(f"[DEBUG] Health check failed: {exc}", flush=True)
        return

    print(flush=True)

    for task in TASKS:
        await run_task(
            client    = client,
            task_name = task["name"],
            max_steps = task["max_steps"],
            seed      = task["seed"],
            difficulty = task.get("difficulty", 0.5),
        )
        print(flush=True)


if __name__ == "__main__":
    asyncio.run(main())