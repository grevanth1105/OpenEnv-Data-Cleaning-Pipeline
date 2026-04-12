from __future__ import annotations

import json
import os
import textwrap
from typing import Optional

import pandas as pd

# Load .env if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration — HF Router + OpenAI client (REQUIRED BY RULES)
# ---------------------------------------------------------------------------

API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
JUDGE_MODEL  = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

JUDGE_MAX_TOKENS = 200
JUDGE_ENABLED    = bool(API_KEY)


# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = textwrap.dedent("""
    You are an expert data quality judge evaluating how well an AI agent
    cleaned a real-world dataset.

    You will receive:
    1. A BEFORE sample (5 rows with issues)
    2. An AFTER sample (5 rows after cleaning)
    3. A summary of what issues were present

    Your job: rate the cleaning quality from 0.0 to 1.0.

    Scoring rubric:
    - 0.9–1.0: Excellent — all issues fixed, no over-cleaning
    - 0.7–0.8: Good — most issues fixed
    - 0.5–0.6: Partial — some issues remain
    - 0.3–0.4: Poor — dataset still messy
    - 0.1–0.2: Minimal — little improvement

    IMPORTANT:
    - NEVER return exactly 0.0 or 1.0
    - Always return between 0.001 and 0.999

    Reply ONLY JSON:
    {"score": 0.0, "reason": "one sentence"}
""").strip()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _df_to_text(df: pd.DataFrame, n_rows: int = 5) -> str:
    sample = df.head(n_rows)
    lines  = [" | ".join(str(v)[:15] for v in sample.columns)]
    lines += [
        " | ".join(str(v)[:15] if v is not None else "NULL"
        for v in row)
        for _, row in sample.iterrows()
    ]
    return "\n".join(lines)


def strict_score(x: float) -> float:
    return float(min(max(x, 0.001), 0.999))


# ---------------------------------------------------------------------------
# Main Judge Function
# ---------------------------------------------------------------------------

def judge_task3(
    df_before: pd.DataFrame,
    df_after:  pd.DataFrame,
    n_dupes_original: int = 0,
) -> float:

    if not JUDGE_ENABLED:
        return 0.5  # safe fallback

    try:
        client = OpenAI(
            api_key=API_KEY,
            base_url=API_BASE_URL
        )

        before_text = _df_to_text(df_before)
        after_text  = _df_to_text(df_after)

        user_prompt = textwrap.dedent(f"""
            TASK: Schema Normalization + Deduplication

            Issues:
            - {n_dupes_original} duplicate rows
            - inconsistent formats
            - null variants
            - invalid values

            BEFORE:
            {before_text}

            AFTER:
            {after_text}

            Give score between 0.001 and 0.999
        """).strip()

        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=JUDGE_MAX_TOKENS,
            temperature=0.0,
        )

        text = (response.choices[0].message.content or "").strip()

        # Clean response
        import re
        text = re.sub(r"```(?:json)?\s*", "", text).strip()
        text = re.sub(r"```\s*$", "", text).strip()

        # Parse JSON
        try:
            result = json.loads(text)
            score  = float(result.get("score", 0.5))
        except:
            match = re.search(r'"score"\s*:\s*([0-9.]+)', text)
            score = float(match.group(1)) if match else 0.5

        score = strict_score(score)

        print(f"[LLM Judge] score={score:.3f}", flush=True)
        return score

    except Exception as exc:
        print(f"[LLM Judge] Failed: {exc} → fallback 0.5", flush=True)
        return 0.5


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running LLM Judge test...")

    import pandas as pd

    df_before = pd.DataFrame({
        "region": ["NORTH", "south", "N"],
        "status": ["ACTIVE", "inactive", "Pending"]
    })

    df_after = pd.DataFrame({
        "region": ["North", "South", "North"],
        "status": ["active", "inactive", "pending"]
    })

    score = judge_task3(df_before, df_after, n_dupes_original=5)
    print(f"Score: {score}")