"""
baseline.py — LLM baseline agent for Data Cleaning Pipeline OpenEnv
====================================================================
Runs a model against all 3 tasks via the HuggingFace Router.
No OpenAI key needed — just your HF token.

Usage:
    export HF_TOKEN=hf_...          # HuggingFace token (free)
    python baseline.py
    python baseline.py --model Qwen/Qwen2.5-72B-Instruct --seed 42
    python baseline.py --task missing_value_imputation

    # Optional: use OpenAI directly
    export OPENAI_API_KEY=sk-...
    export USE_OPENAI=true
    python baseline.py --model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from environment import DataCleaningEnvironment, TASK_NAMES
from models import DataCleaningAction, DataCleaningObservation


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert data engineer specialising in data quality and cleaning.
You will be given a dataset snapshot with column statistics and a list of detected issues.
Your job is to issue cleaning actions one at a time to fix all issues.

## Available actions

| action_type    | Required params                        | When to use                          |
|----------------|----------------------------------------|--------------------------------------|
| impute         | strategy (mean/median/mode/constant)   | Fix missing values in a column       |
| cast           | dtype (int/float/str/date/datetime)    | Fix wrong data types                 |
| normalize      | format OR method OR mapping            | Standardise formats / categories     |
| clip_outlier   | lower and/or upper (numeric bounds)    | Remove statistical outliers          |
| flag_outlier   | none required                          | Mark outliers without removing       |
| deduplicate    | subset (optional list of columns)      | Remove duplicate rows                |
| drop_column    | none required                          | Remove an irrelevant column          |
| drop_rows      | condition (null/invalid_range)         | Remove rows matching a condition     |
| finish         | none required                          | Signal episode complete              |

## Response format

Respond with a single JSON object — no markdown, no explanation:
{
  "action_type": "<type>",
  "column": "<column_name_or_null>",
  "params": {}
}

## Strategy

1. Read the issues_detected list carefully.
2. Fix the highest-severity issues first.
3. For missing values: use median for numeric, mode for categorical.
4. For type errors: cast to the correct dtype.
5. For outliers: clip to sensible domain bounds.
6. For duplicates: deduplicate first, then fix formats.
7. Call finish when all issues are resolved or no more progress is possible.
8. Never repeat the same action on the same column twice.
"""


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class LLMAgent:
    """Thin wrapper around OpenAI chat completions for action generation."""

    def __init__(self, model: str, client: OpenAI) -> None:
        self.model   = model
        self.client  = client
        self._history: List[Dict] = []

    def reset(self) -> None:
        self._history = []

    def act(self, obs: DataCleaningObservation) -> Optional[DataCleaningAction]:
        user_msg = _obs_to_prompt(obs)
        self._history.append({"role": "user", "content": user_msg})

        try:
            response = self.client.chat.completions.create(
                model       = self.model,
                messages    = [{"role": "system", "content": SYSTEM_PROMPT}] + self._history,
                temperature = 0.0,
                max_tokens  = 200,
            )
            raw = response.choices[0].message.content.strip()
            self._history.append({"role": "assistant", "content": raw})

            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]

            payload = json.loads(raw.strip())
            return DataCleaningAction(
                action_type = payload.get("action_type", "finish"),
                column      = payload.get("column"),
                params      = payload.get("params", {}),
            )

        except json.JSONDecodeError:
            # Fallback — finish episode cleanly
            return DataCleaningAction(action_type="finish", column=None, params={})
        except Exception as e:
            print(f"    [LLM error] {e}")
            return None


def _obs_to_prompt(obs: DataCleaningObservation) -> str:
    """Convert observation to a concise prompt string."""
    issues = "\n".join(
        f"  - [{i.severity.upper()}] {i.column or 'dataset'}: {i.description}"
        for i in obs.issues_detected
    ) or "  None detected."

    col_summary = "\n".join(
        f"  {s.name:20s} dtype={s.dtype:10s} nulls={s.null_count:3d} "
        f"outliers={s.outlier_count:2d} unique={s.unique_count}"
        for s in obs.column_stats
    )

    history = "\n".join(f"  {a}" for a in obs.action_history[-5:]) or "  None yet."

    return f"""TASK: {obs.task_name} ({obs.task_difficulty})
OBJECTIVE: {obs.task_description}

PROGRESS: {obs.progress_pct:.0%} complete | Step {obs.step_count} | Reward so far: {obs.cumulative_reward:.3f}

REMAINING ISSUES:
{issues}

COLUMN STATISTICS:
{col_summary}

RECENT ACTIONS:
{history}

LAST RESULT: {obs.last_action_result}

What is your next cleaning action? Respond with JSON only."""


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(
    env: DataCleaningEnvironment,
    agent: LLMAgent,
    task_name: str,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run one full episode. Returns grader score and stats."""
    agent.reset()
    obs = env.reset(task_name=task_name, seed=seed)

    if verbose:
        print(f"\n  Task      : {task_name}")
        print(f"  Rows      : {obs.total_rows} | Issues: {obs.issues_remaining}")

    total_reward = 0.0
    steps        = 0

    while not obs.done:
        action = agent.act(obs)
        if action is None:
            break

        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps        += 1

        if verbose:
            col = f"({action.column})" if action.column else ""
            print(f"  [{steps:2d}] {str(action.action_type):15s}{col:15s} "
                  f"reward={reward:+.4f}  progress={obs.progress_pct:.0%}")

        if done:
            break

        time.sleep(0.1)   # rate-limit buffer

    # Final grader score
    grader = env.get_grader_result()

    if verbose:
        print(f"  ─────────────────────────────────────────────────")
        print(f"  Score     : {grader['score']:.4f}  |  {grader['feedback']}")
        print(f"  Steps     : {steps}  |  Reward: {total_reward:.4f}")

    return {
        "task_name":     task_name,
        "score":         grader["score"],
        "passed":        grader["passed"],
        "steps":         steps,
        "total_reward":  round(total_reward, 4),
        "breakdown":     grader["breakdown"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Data Cleaning Pipeline — LLM Baseline")
    parser.add_argument("--model", default=os.getenv("BASELINE_MODEL", "Qwen/Qwen2.5-72B-Instruct"))
    parser.add_argument("--seed",  default=42, type=int)
    parser.add_argument("--task",  default="all", choices=["all"] + TASK_NAMES)
    parser.add_argument("--url",   default="", help="Environment server URL (optional)")
    args = parser.parse_args()

    # --- API client setup ---
    # Priority: HF Router (free) → OpenAI (paid)
    use_openai = os.getenv("USE_OPENAI", "false").lower() == "true"

    if use_openai:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ USE_OPENAI=true but OPENAI_API_KEY not set.")
            sys.exit(1)
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        print(f"  Using : OpenAI API")
    else:
        # HuggingFace Router — uses HF_TOKEN, no OpenAI key needed
        api_key = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        if not api_key:
            print("❌ HF_TOKEN not set.")
            print("   Get yours at: https://huggingface.co/settings/tokens")
            print("   Then: export HF_TOKEN=hf_...")
            sys.exit(1)
        base_url = "https://router.huggingface.co/v1"
        print(f"  Using : HuggingFace Router (free tier)")

    client = OpenAI(api_key=api_key, base_url=base_url)
    agent  = LLMAgent(model=args.model, client=client)
    env    = DataCleaningEnvironment()
    tasks  = TASK_NAMES if args.task == "all" else [args.task]

    print("=" * 60)
    print("  Data Cleaning Pipeline — LLM Baseline")
    print("=" * 60)
    print(f"  Model : {args.model}")
    print(f"  Seed  : {args.seed}")
    print(f"  Tasks : {tasks}")
    print("=" * 60)

    results  = {}
    failures = []

    for task_name in tasks:
        try:
            result = run_episode(
                env       = env,
                agent     = agent,
                task_name = task_name,
                seed      = args.seed,
                verbose   = True,
            )
            results[task_name] = result["score"]
        except Exception as e:
            print(f"\n  ❌ {task_name} failed: {e}")
            failures.append(task_name)
            results[task_name] = 0.0

    # --- Summary ---
    mean_score = round(sum(results.values()) / max(len(results), 1), 4)

    print("\n" + "=" * 60)
    print("  FINAL SCORES")
    print("=" * 60)
    for task, score in results.items():
        bar    = "█" * int(score * 20)
        status = "✅" if score >= 0.6 else "❌"
        print(f"  {status} {task[:38]:38s} [{bar:<20}] {score:.4f}")

    print(f"\n  Mean score : {mean_score:.4f}")
    print(f"  Model      : {args.model}")
    print(f"  Seed       : {args.seed}")

    if failures:
        print(f"\n  ⚠ Failed tasks: {failures}")

    # --- Write results to file for reproducibility ---
    output = {
        "model":      args.model,
        "seed":       args.seed,
        "results":    results,
        "mean_score": mean_score,
    }
    with open("baseline_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\n  Results saved to baseline_results.json")
    print("=" * 60)

    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    main()