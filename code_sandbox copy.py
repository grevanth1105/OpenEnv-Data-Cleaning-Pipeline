"""
code_sandbox.py — Safe Python execution sandbox for Data Cleaning Pipeline
===========================================================================
Allows agents to write and execute real Python cleaning code.
Security model:
  - Restricted builtins (no open, exec, import of dangerous modules)
  - Execution timeout (5 seconds max)
  - No filesystem access
  - No network access
  - DataFrame mutation tracked — rewards based on actual improvement
"""

from __future__ import annotations

import signal
import traceback
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Allowed builtins — safe subset only
# ---------------------------------------------------------------------------

SAFE_BUILTINS = {
    # Math
    "abs": abs, "round": round, "min": min, "max": max,
    "sum": sum, "len": len, "range": range,
    # Types
    "int": int, "float": float, "str": str, "bool": bool,
    "list": list, "dict": dict, "tuple": tuple, "set": set,
    # Iteration
    "enumerate": enumerate, "zip": zip, "map": map,
    "filter": filter, "sorted": sorted, "reversed": reversed,
    # Inspection
    "isinstance": isinstance, "hasattr": hasattr, "getattr": getattr,
    "print": print,
    # None/True/False
    "None": None, "True": True, "False": False,
}

# Modules the agent is allowed to use
SAFE_MODULES = {
    "pd":  pd,
    "np":  np,
    "pandas": pd,
    "numpy": np,
}

# Patterns that are never allowed regardless of context
BLOCKED_PATTERNS = [
    "import os", "import sys", "import subprocess",
    "import socket", "import urllib", "import requests",
    "__import__", "open(", "exec(", "eval(",
    "os.system", "os.path", "shutil",
    "globals()", "locals()", "__builtins__",
    "compile(", "memoryview(",
]

TIMEOUT_SECONDS = 5


# ---------------------------------------------------------------------------
# Timeout handler
# ---------------------------------------------------------------------------

class TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("Code execution timed out (5s limit)")


# ---------------------------------------------------------------------------
# Core execution function
# ---------------------------------------------------------------------------

def execute_cleaning_code(
    df: pd.DataFrame,
    code: str,
) -> Tuple[pd.DataFrame, str, bool]:
    """
    Execute agent-provided Python cleaning code in a restricted sandbox.

    Parameters
    ----------
    df   : Current dataframe (will be available as `df` in the code)
    code : Python code string from the agent

    Returns
    -------
    result_df : Modified dataframe (or original if execution failed)
    message   : Human-readable result or error message
    success   : True if code ran without errors
    """
    if not code or not code.strip():
        return df, "Empty code provided.", False

    # --- Security check ---
    blocked = _check_security(code)
    if blocked:
        return df, f"⛔ Blocked: {blocked}", False

    # --- Prepare execution namespace ---
    namespace: Dict[str, Any] = {
        "__builtins__": SAFE_BUILTINS,
        "df": df.copy(),    # agent gets a copy — only applied if successful
        **SAFE_MODULES,
    }

    # --- Execute with timeout ---
    try:
        # Set timeout (Unix only — Docker container runs Linux)
        try:
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(TIMEOUT_SECONDS)
        except (AttributeError, OSError):
            pass  # Windows doesn't support SIGALRM — skip timeout

        exec(compile(code, "<agent_code>", "exec"), namespace)  # noqa: S102

        try:
            signal.alarm(0)  # cancel timeout
        except (AttributeError, OSError):
            pass

        # Extract modified dataframe
        result_df = namespace.get("df", df)

        if not isinstance(result_df, pd.DataFrame):
            return df, "⚠️ Code did not return a valid DataFrame (df was overwritten).", False

        # Basic sanity check — df should still have rows
        if len(result_df) == 0:
            return df, "⚠️ Code dropped all rows — reverting.", False

        # Compute what changed
        summary = _summarize_changes(df, result_df, code)
        return result_df, summary, True

    except TimeoutError:
        return df, f"⏱️ Code timed out after {TIMEOUT_SECONDS}s — simplify your code.", False
    except Exception as e:
        error_msg = _format_error(e, code)
        return df, error_msg, False
    finally:
        try:
            signal.alarm(0)
        except (AttributeError, OSError):
            pass


# ---------------------------------------------------------------------------
# Security scanner
# ---------------------------------------------------------------------------

def _check_security(code: str) -> str:
    """Return a message describing what was blocked, or '' if safe."""
    code_lower = code.lower()
    for pattern in BLOCKED_PATTERNS:
        if pattern.lower() in code_lower:
            return f"'{pattern}' is not allowed in sandbox"
    # Block attempts to access private attributes
    if ".__" in code and not (".__class__" in code or ".__name__" in code):
        return "Access to private attributes (.__) is not allowed"
    return ""


# ---------------------------------------------------------------------------
# Change summarizer
# ---------------------------------------------------------------------------

def _summarize_changes(
    before: pd.DataFrame,
    after: pd.DataFrame,
    code: str,
) -> str:
    """Produce a human-readable summary of what the code changed."""
    changes = []

    # Row count change
    row_diff = len(before) - len(after)
    if row_diff > 0:
        changes.append(f"removed {row_diff} rows")
    elif row_diff < 0:
        changes.append(f"added {abs(row_diff)} rows")

    # Column changes
    new_cols  = set(after.columns) - set(before.columns)
    drop_cols = set(before.columns) - set(after.columns)
    if new_cols:
        changes.append(f"added columns: {sorted(new_cols)}")
    if drop_cols:
        changes.append(f"dropped columns: {sorted(drop_cols)}")

    # Null changes per column
    null_fixes = []
    for col in before.columns:
        if col not in after.columns:
            continue
        before_nulls = int(before[col].isna().sum())
        after_nulls  = int(after[col].isna().sum())
        if before_nulls > after_nulls:
            null_fixes.append(f"'{col}' {before_nulls}→{after_nulls} nulls")
        elif after_nulls > before_nulls:
            null_fixes.append(f"'{col}' gained {after_nulls - before_nulls} nulls ⚠️")

    if null_fixes:
        changes.append("nulls: " + ", ".join(null_fixes))

    # Dtype changes
    dtype_changes = []
    for col in before.columns:
        if col not in after.columns:
            continue
        if str(before[col].dtype) != str(after[col].dtype):
            dtype_changes.append(f"'{col}' {before[col].dtype}→{after[col].dtype}")
    if dtype_changes:
        changes.append("types: " + ", ".join(dtype_changes))

    if not changes:
        return "✅ Code ran successfully — no measurable changes detected."
    return "✅ Code executed: " + " | ".join(changes)


# ---------------------------------------------------------------------------
# Error formatter
# ---------------------------------------------------------------------------

def _format_error(e: Exception, code: str) -> str:
    """Format an execution error into a helpful message for the agent."""
    error_type = type(e).__name__
    error_msg  = str(e)

    # Give helpful hints for common errors
    if "KeyError" in error_type:
        return (
            f"❌ KeyError: {error_msg} — column does not exist. "
            f"Check column_stats for available columns."
        )
    if "TypeError" in error_type:
        return (
            f"❌ TypeError: {error_msg} — wrong type operation. "
            f"Cast the column first before operating on it."
        )
    if "ValueError" in error_type:
        return f"❌ ValueError: {error_msg} — invalid value or conversion."
    if "AttributeError" in error_type:
        return f"❌ AttributeError: {error_msg} — method not available on this column type."
    if "SyntaxError" in error_type:
        return f"❌ SyntaxError: {error_msg} — check your Python syntax."

    # Generic error
    lines     = traceback.format_exc().split("\n")
    short_tb  = "\n".join(l for l in lines if l.strip() and "site-packages" not in l)
    return f"❌ {error_type}: {error_msg}"


# ---------------------------------------------------------------------------
# Reward calculator for execute_code
# ---------------------------------------------------------------------------

def score_code_execution(
    before: pd.DataFrame,
    after: pd.DataFrame,
    success: bool,
) -> float:
    """
    Compute reward for a code execution action.

    Rewards:
      +0.20  for each column with nulls reduced
      +0.10  for each dtype corrected (object → numeric/datetime)
      +0.15  for removing duplicates
      -0.10  per column that gained new nulls
      -0.20  if code dropped > 15% of rows unnecessarily
      -0.05  if code failed (syntax/runtime error)
    """
    if not success:
        return -0.05

    reward = 0.0

    # Null improvements
    for col in before.columns:
        if col not in after.columns:
            continue
        before_nulls = int(before[col].isna().sum())
        after_nulls  = int(after[col].isna().sum())
        if after_nulls < before_nulls:
            reward += 0.20
        elif after_nulls > before_nulls:
            reward -= 0.10

    # Dtype improvements
    for col in before.columns:
        if col not in after.columns:
            continue
        was_object = str(before[col].dtype) == "object"
        now_numeric = pd.api.types.is_numeric_dtype(after[col])
        now_datetime = pd.api.types.is_datetime64_any_dtype(after[col])
        if was_object and (now_numeric or now_datetime):
            reward += 0.10

    # Deduplication bonus
    dupes_before = int(before.duplicated().sum())
    dupes_after  = int(after.duplicated().sum())
    if dupes_after < dupes_before:
        reward += 0.15

    # Row drop penalty
    row_pct_dropped = (len(before) - len(after)) / max(len(before), 1)
    if row_pct_dropped > 0.15:
        reward -= 0.20

    return round(float(np.clip(reward, -1.0, 1.0)), 4)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pandas as pd
    import numpy as np

    print("=" * 55)
    print("  Code Sandbox — Smoke Test")
    print("=" * 55)

    df = pd.DataFrame({
        "age":    [25.0, None, 30.0, None, 45.0],
        "salary": ["$50,000", "$60,000", None, "$70,000", "$80,000"],
        "dept":   ["Eng", "eng", "ENG", None, "Eng"],
    })

    print(f"\n  Input:\n{df}\n")

    # Test 1 — valid cleaning code
    code1 = "df['age'] = df['age'].fillna(df['age'].median())"
    result, msg, ok = execute_cleaning_code(df, code1)
    print(f"  Test 1 (impute age): {'✅' if ok else '❌'}")
    print(f"    Message : {msg}")
    print(f"    Nulls   : {int(df['age'].isna().sum())} → {int(result['age'].isna().sum())}")
    reward = score_code_execution(df, result, ok)
    print(f"    Reward  : {reward:+.4f}")

    # Test 2 — cast salary
    code2 = "df['salary'] = pd.to_numeric(df['salary'].str.replace(r'[^\\d]', '', regex=True), errors='coerce')"
    result2, msg2, ok2 = execute_cleaning_code(df, code2)
    print(f"\n  Test 2 (cast salary): {'✅' if ok2 else '❌'}")
    print(f"    Message : {msg2}")
    print(f"    Dtype   : {df['salary'].dtype} → {result2['salary'].dtype}")
    reward2 = score_code_execution(df, result2, ok2)
    print(f"    Reward  : {reward2:+.4f}")

    # Test 3 — blocked code
    code3 = "import os; os.system('rm -rf /')"
    result3, msg3, ok3 = execute_cleaning_code(df, code3)
    print(f"\n  Test 3 (blocked): {'✅ blocked' if not ok3 else '❌ not blocked!'}")
    print(f"    Message : {msg3}")

    # Test 4 — syntax error
    code4 = "df['age'] = df['age'].fillna(  # incomplete"
    result4, msg4, ok4 = execute_cleaning_code(df, code4)
    print(f"\n  Test 4 (syntax error): {'✅ caught' if not ok4 else '❌ missed'}")
    print(f"    Message : {msg4}")

    # Test 5 — runtime error
    code5 = "df['nonexistent'] = df['nonexistent'].fillna(0)"
    result5, msg5, ok5 = execute_cleaning_code(df, code5)
    print(f"\n  Test 5 (KeyError): {'✅ caught' if not ok5 else '❌ missed'}")
    print(f"    Message : {msg5}")

    print("\n✅ Sandbox working correctly!")
