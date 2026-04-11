

from __future__ import annotations

import json
import os
import textwrap
from typing import Optional

import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration — uses OPENAI_API_KEY (separate from HF_TOKEN)
# ---------------------------------------------------------------------------

OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
JUDGE_MODEL     = os.getenv("JUDGE_MODEL", "gpt-4o-mini")  # cheap and fast
JUDGE_MAX_TOKENS = 200
JUDGE_ENABLED   = bool(OPENAI_API_KEY)  # silently skip if key not set


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
    - 0.9–1.0: Excellent — all issues fixed, no over-cleaning, data looks consistent
    - 0.7–0.8: Good — most issues fixed, minor problems remain
    - 0.5–0.6: Partial — key issues addressed but several remain visible
    - 0.3–0.4: Poor — few issues resolved, dataset still messy
    - 0.1–0.2: Minimal — almost no improvement from original

    Reply with ONLY a JSON object, no explanation:
    {"score": 0.0, "reason": "one sentence"}
""").strip()


def _df_to_text(df: pd.DataFrame, n_rows: int = 5) -> str:
    """Convert dataframe sample to readable text for LLM."""
    sample = df.head(n_rows)
    lines  = [" | ".join(str(v)[:15] for v in sample.columns)]
    lines += [" | ".join(str(v)[:15] if v is not None else "NULL"
                         for v in row) for _, row in sample.iterrows()]
    return "\n".join(lines)


def judge_task3(
    df_before: pd.DataFrame,
    df_after:  pd.DataFrame,
    n_dupes_original: int = 0,
) -> float:
    """
    Call OpenAI LLM to judge the quality of Task 3 cleaning.

    Returns a score in (0.001, 0.999).
    Falls back to 0.5 if OPENAI_API_KEY is not set or call fails.
    """
    if not JUDGE_ENABLED:
        # Silently return neutral score if no key
        return 0.5

    try:
        client = OpenAI(api_key=OPENAI_API_KEY)

        before_text = _df_to_text(df_before)
        after_text  = _df_to_text(df_after)

        user_prompt = textwrap.dedent(f"""
            TASK: Schema Normalization + Deduplication

            Issues that were present:
            - {n_dupes_original} duplicate rows
            - Inconsistent region/status formats (e.g. NORTH, north, N)
            - NULL variants (N/A, none, - instead of actual null)
            - Invalid ages and revenue values

            BEFORE CLEANING (first 5 rows):
            {before_text}

            AFTER CLEANING (first 5 rows):
            {after_text}

            Rate the cleaning quality from 0.0 to 1.0.
            Reply ONLY with: {{"score": 0.0, "reason": "one sentence"}}
        """).strip()

        response = client.chat.completions.create(
            model       = JUDGE_MODEL,
            messages    = [
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens  = JUDGE_MAX_TOKENS,
            temperature = 0.0,
        )

        text = (response.choices[0].message.content or "").strip()

        # Parse score from response
        import re
        text = re.sub(r"```(?:json)?\s*", "", text).strip()
        text = re.sub(r"```\s*$", "", text).strip()

        try:
            result = json.loads(text)
            score  = float(result.get("score", 0.5))
        except (json.JSONDecodeError, ValueError):
            # Try extracting number
            match = re.search(r'"score"\s*:\s*([0-9.]+)', text)
            score = float(match.group(1)) if match else 0.5

        # Clamp strictly between 0 and 1
        score = float(min(max(score, 0.001), 0.999))
        print(f"[LLM Judge] score={score:.3f}", flush=True)
        return score

    except Exception as exc:
        print(f"[LLM Judge] Failed: {exc} — using fallback 0.5", flush=True)
        return 0.5


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as np

    print("=" * 55)
    print("  LLM Judge Smoke Test")
    print("=" * 55)

    if not JUDGE_ENABLED:
        print("⚠️  OPENAI_API_KEY not set — testing fallback path")
        score = judge_task3(
            df_before=pd.DataFrame({"region": ["NORTH", "south", "N"], "status": ["ACTIVE", "inactive", "Pending"]}),
            df_after =pd.DataFrame({"region": ["North", "South", "North"], "status": ["active", "inactive", "pending"]}),
        )
        print(f"  Fallback score: {score} (expected 0.5)")
        assert score == 0.5, "Fallback should return 0.5"
        print("  ✅ Fallback works correctly")
    else:
        print(f"  Using model: {JUDGE_MODEL}")

        # Create test dataframes
        df_before = pd.DataFrame({
            "region":  ["NORTH", "south", "N", "Eastern", "WEST"],
            "status":  ["ACTIVE", "inactive", "Pending", "CHURNED", "active"],
            "email":   ["N/A", "a@b.com", "none", "c@d.com", "-"],
            "age":     [25, -5, 30, 150, 40],
        })
        df_after = pd.DataFrame({
            "region":  ["North", "South", "North", "East", "West"],
            "status":  ["active", "inactive", "pending", "churned", "active"],
            "email":   [None, "a@b.com", None, "c@d.com", None],
            "age":     [25, None, 30, None, 40],
        })

        score = judge_task3(df_before, df_after, n_dupes_original=5)
        ok    = 0.0 < score < 1.0
        print(f"  Score: {score:.4f}  {'✅' if ok else '❌'}")

    print("✅ LLM Judge test complete!")
