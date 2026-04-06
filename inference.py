"""
Inference Script — Data Cleaning Pipeline OpenEnv
===================================================
MANDATORY ENVIRONMENT VARIABLES:
    API_BASE_URL       LLM API endpoint (default: HF Router)
    MODEL_NAME         Model identifier  (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN           HuggingFace token (used as API key)
    SPACE_URL          HF Space URL      (default: our deployed space)
    IMAGE_NAME         Docker image name (optional — for local Docker mode)

STDOUT FORMAT (strictly followed):
    [START] task=<task> env=<benchmark> model=<model>
    [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>

Run:
    export HF_TOKEN=hf_...
    python inference.py
"""

import asyncio
import json
import os
import textwrap
import time
import uuid
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME   = os.getenv("MODEL_NAME")   or "Qwen/Qwen2.5-72B-Instruct"
SPACE_URL    = os.getenv("SPACE_URL")    or "https://revanth11-data-cleaning-env.hf.space"
IMAGE_NAME   = os.getenv("IMAGE_NAME")  # optional — set for local Docker mode

BENCHMARK    = "data-cleaning-env"
TEMPERATURE  = 0.0
MAX_TOKENS   = 200
SUCCESS_SCORE_THRESHOLD = 0.6

# One entry per task — easy → medium → hard
TASKS = [
    {"name": "missing_value_imputation",   "max_steps": 12, "seed": 42},
    {"name": "type_errors_and_outliers",   "max_steps": 18, "seed": 42},
    {"name": "schema_normalization_dedup", "max_steps": 22, "seed": 42},
]

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert data engineer cleaning a real-world dataset.
    You will see column statistics and a list of data quality issues.
    Issue ONE cleaning action at a time as a JSON object.

    Available actions:
    {"action_type": "impute",       "column": "col", "params": {"strategy": "median"}}
    {"action_type": "cast",         "column": "col", "params": {"dtype": "float"}}
    {"action_type": "normalize",    "column": "col", "params": {"method": "lowercase"}}
    {"action_type": "clip_outlier", "column": "col", "params": {"lower": 0, "upper": 100}}
    {"action_type": "deduplicate",  "column": null,  "params": {}}
    {"action_type": "execute_code", "column": null,  "params": {"code": "df['col'] = df['col'].fillna(df['col'].median())"}}
    {"action_type": "finish",       "column": null,  "params": {}}

    Rules:
    - Fix HIGH severity issues first
    - Numeric nulls → median; categorical nulls → mode
    - Call finish when progress_pct > 0.85 or no issues remain
    - Never repeat the same action on the same column
    - Reply with ONLY the JSON object, no explanation
""").strip()


# ---------------------------------------------------------------------------
# Logging — exact format required by evaluator
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# HTTP Environment Client (no external OpenEnv deps needed)
# ---------------------------------------------------------------------------

class DataCleaningHTTPEnv:
    """
    Thin HTTP client for the Data Cleaning Pipeline OpenEnv.
    Connects to either a running HF Space or a local Docker container.
    """

    def __init__(self, base_url: str) -> None:
        self.base_url   = base_url.rstrip("/")
        self.session_id = str(uuid.uuid4())[:8]
        self._session   = requests.Session()

    def reset(self, task_name: str, seed: int = 42) -> Dict[str, Any]:
        resp = self._session.post(
            f"{self.base_url}/reset",
            json={"task_name": task_name, "seed": seed, "session_id": self.session_id},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute action with retry on transient errors."""
        for attempt in range(2):
            try:
                resp = self._session.post(
                    f"{self.base_url}/step",
                    json={"action": action, "session_id": self.session_id},
                    timeout=45,
                )
                if resp.status_code == 400:
                    detail = resp.json().get("detail", "")
                    # Session expired — not retriable
                    raise requests.exceptions.HTTPError(
                        f"HTTP 400: {detail}", response=resp
                    )
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.HTTPError:
                raise
            except requests.exceptions.Timeout:
                if attempt == 0:
                    continue
                raise
        raise RuntimeError("step() failed after retries")

    def grader(self) -> Dict[str, Any]:
        resp = self._session.post(
            f"{self.base_url}/grader",
            json={"session_id": self.session_id},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def close(self) -> None:
        self._session.close()


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def build_user_prompt(obs: Dict, step: int, history: List[str]) -> str:
    issues = obs.get("issues_detected", [])
    issue_text = "\n".join(
        f"  [{i.get('severity','').upper():6s}] {i.get('column') or 'dataset'}: {i.get('description','')}"
        for i in issues
    ) or "  None remaining — call finish."

    col_stats = obs.get("column_stats", [])
    col_text = "\n".join(
        f"  {s.get('name',''):20s} dtype={s.get('dtype',''):10s} "
        f"nulls={s.get('null_count',0):3d}  outliers={s.get('outlier_count',0):2d}"
        for s in col_stats
    )

    history_block = "\n".join(history[-4:]) if history else "None"
    progress = obs.get("progress_pct", 0)
    last_result = obs.get("last_action_result", "")

    return textwrap.dedent(f"""
        Step {step} | Progress: {progress:.0%} | Issues: {obs.get('issues_remaining', 0)}

        REMAINING ISSUES:
        {issue_text}

        COLUMN STATISTICS:
        {col_text}

        RECENT ACTIONS:
        {history_block}

        LAST RESULT: {last_result}

        Reply with a single JSON cleaning action.
    """).strip()


def parse_action(text: str) -> Optional[Dict]:
    """Extract JSON action from LLM output."""
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


def get_model_action(
    client: OpenAI,
    obs: Dict,
    step: int,
    history: List[str],
) -> Dict[str, Any]:
    """Call LLM and return a parsed cleaning action."""
    user_prompt = build_user_prompt(obs, step, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
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

    # Fallback: finish the episode cleanly
    return {"action_type": "finish", "column": None, "params": {}}


# ---------------------------------------------------------------------------
# Single task episode
# ---------------------------------------------------------------------------

async def run_task(
    client: OpenAI,
    env: DataCleaningHTTPEnv,
    task_name: str,
    max_steps: int,
    seed: int,
) -> None:
    """Run one full episode for a single task."""
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    score  = 0.0
    success = False

    try:
        obs    = env.reset(task_name=task_name, seed=seed)
        done   = obs.get("done", False)
        history: List[str] = []

        for step in range(1, max_steps + 1):
            if done:
                break

            action  = get_model_action(client, obs, step, history)
            action_str = json.dumps(action, separators=(",", ":"))

            try:
                result  = env.step(action)
                obs     = result.get("observation", obs)
                reward  = float(result.get("reward", 0.0))
                done    = bool(result.get("done", False))
                error   = None
            except requests.exceptions.HTTPError as exc:
                # 400 = episode already done or invalid action — end gracefully
                reward = 0.0
                done   = True
                error  = f"HTTP {exc.response.status_code}"
            except Exception as exc:
                reward = 0.0
                done   = True
                error  = str(exc)[:80]

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            history.append(
                f"Step {step}: {action.get('action_type')} "
                f"col={action.get('column')} -> reward {reward:+.2f}"
            )

            if done:
                break

            time.sleep(0.5)  # small pause to avoid HF Space rate limiting

        # Final grader score (0–1) — more accurate than step-reward sum
        try:
            grader = env.grader()
            score  = float(grader.get("score", 0.0))
            score  = min(max(score, 0.0), 1.0)
        except Exception:
            # Fallback: normalize step rewards
            max_total = max_steps * 0.25
            score = min(sum(rewards) / max_total, 1.0) if max_total > 0 else 0.0

        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    if not API_KEY:
        print("[DEBUG] HF_TOKEN not set — set it with: export HF_TOKEN=hf_...", flush=True)
        return

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    print(f"[DEBUG] API_BASE_URL : {API_BASE_URL}", flush=True)
    print(f"[DEBUG] MODEL_NAME   : {MODEL_NAME}",   flush=True)
    print(f"[DEBUG] SPACE_URL    : {SPACE_URL}",    flush=True)
    print(f"[DEBUG] Tasks        : {[t['name'] for t in TASKS]}", flush=True)
    print(flush=True)

    # Verify space is live
    try:
        resp = requests.get(f"{SPACE_URL}/health", timeout=15)
        print(f"[DEBUG] Space health : {resp.json()}", flush=True)
    except Exception as exc:
        print(f"[DEBUG] Space health check failed: {exc}", flush=True)
        return

    for task in TASKS:
        env = DataCleaningHTTPEnv(base_url=SPACE_URL)  # fresh session per task
        try:
            await run_task(
                client    = client,
                env       = env,
                task_name = task["name"],
                max_steps = task["max_steps"],
                seed      = task["seed"],
            )
        except Exception as exc:
            print(f"[DEBUG] Task {task['name']} failed: {exc}", flush=True)
            log_end(success=False, steps=0, score=0.0, rewards=[])
        finally:
            env.close()
        print(flush=True)


if __name__ == "__main__":
    asyncio.run(main())