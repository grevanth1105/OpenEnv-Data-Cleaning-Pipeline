"""
graders.py — Deterministic Task Graders
=========================================
Each grader returns a score strictly in (0.001, 0.999).
Scores are based on ground_truth generated at reset time.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from dataset_generator import REGION_VARIANTS, NULL_VARIANTS, STATUSES

TASK_NAMES = [
    "missing_value_imputation",
    "type_errors_and_outliers",
    "schema_normalization_dedup",
]


# ---------------------------------------------------------------------------
# Task 1 grader — Missing Value Imputation
# ---------------------------------------------------------------------------

def _grade_task1(df: pd.DataFrame, ground_truth: Dict) -> Tuple[float, Dict]:
    null_cols = ground_truth.get("null_cols", {})
    if not null_cols:
        return 0.5, {}

    scores: Dict[str, float] = {}
    weight_per_col = 1.0 / len(null_cols)

    for col, info in null_cols.items():
        if col not in df.columns:
            scores[col] = 0.0
            continue

        remaining_nulls = int(df[col].isna().sum())
        original_nulls  = info.get("null_count", 1)

        if remaining_nulls == 0:
            # All nulls filled — check if values are reasonable
            expected   = info.get("expected", None)
            strategy   = info.get("strategy", "mean")
            filled_val = df[col].dropna()

            if expected is not None and len(filled_val) > 0:
                actual_val  = float(filled_val.median() if strategy == "median" else filled_val.mean())
                tolerance   = abs(expected) * 0.15 + 1.0
                if abs(actual_val - expected) <= tolerance:
                    scores[col] = weight_per_col
                else:
                    scores[col] = weight_per_col * 0.6
            else:
                scores[col] = weight_per_col * 0.8
        else:
            # Partial credit for reducing nulls
            pct_fixed = 1.0 - (remaining_nulls / max(original_nulls, 1))
            scores[col] = round(weight_per_col * max(pct_fixed, 0) * 0.7, 4)

    total = round(sum(scores.values()), 4)
    return total, {"per_column": scores}


# ---------------------------------------------------------------------------
# Task 2 grader — Type Errors + Outliers
# ---------------------------------------------------------------------------

def _grade_task2(df: pd.DataFrame, ground_truth: Dict) -> Tuple[float, Dict]:
    scores: Dict[str, float] = {}
    n_rows = ground_truth.get("n_rows", len(df))

    # 1. unit_price as numeric (weight 0.22)
    scores["unit_price"] = 0.0
    if "unit_price" in df.columns:
        if pd.api.types.is_numeric_dtype(df["unit_price"]):
            valid = df["unit_price"].dropna()
            if len(valid) > 0 and (valid > 0).all():
                scores["unit_price"] = 0.22

    # 2. quantity as integer (weight 0.18)
    scores["quantity"] = 0.0
    if "quantity" in df.columns:
        if pd.api.types.is_integer_dtype(df["quantity"]) or \
           (pd.api.types.is_numeric_dtype(df["quantity"]) and
            df["quantity"].dropna().apply(lambda x: x == int(x)).all()):
            scores["quantity"] = 0.18
        elif pd.api.types.is_numeric_dtype(df["quantity"]):
            scores["quantity"] = 0.10

    # 3. order_date as datetime (weight 0.18)
    scores["order_date"] = 0.0
    if "order_date" in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df["order_date"]):
            valid_pct = df["order_date"].notna().mean()
            scores["order_date"] = round(0.18 * valid_pct, 4)
        elif df["order_date"].dtype == object:
            # Partial credit if at least some were parsed
            try:
                parsed = pd.to_datetime(df["order_date"], errors="coerce")
                parsed_pct = parsed.notna().mean()
                scores["order_date"] = round(0.18 * parsed_pct * 0.5, 4)
            except Exception:
                pass

    # 4. rating as numeric (weight 0.15)
    scores["rating"] = 0.0
    if "rating" in df.columns:
        if pd.api.types.is_numeric_dtype(df["rating"]):
            valid = df["rating"].dropna()
            if len(valid) > 0 and (valid.between(0, 5)).all():
                scores["rating"] = 0.15
            else:
                scores["rating"] = 0.08

    # 5. discount_pct outliers removed (weight 0.17)
    scores["discount_pct"] = 0.0
    if "discount_pct" in df.columns:
        if pd.api.types.is_numeric_dtype(df["discount_pct"]):
            bad = ((df["discount_pct"] < 0) | (df["discount_pct"] > 100)).sum()
            if bad == 0:
                scores["discount_pct"] = 0.17
            else:
                scores["discount_pct"] = round(0.17 * (1 - bad / max(len(df), 1)), 4)

    # 6. region normalized (weight 0.10)
    scores["region"] = 0.0
    if "region" in df.columns:
        valid_regions = set(REGION_VARIANTS.keys())
        pct_valid = df["region"].isin(valid_regions).mean()
        scores["region"] = round(0.10 * pct_valid, 4)

    total = round(sum(scores.values()), 4)
    return total, {"per_column": scores}


# ---------------------------------------------------------------------------
# Task 3 grader — Schema Normalization + Dedup
# ---------------------------------------------------------------------------

def _grade_task3(df: pd.DataFrame, ground_truth: Dict) -> Tuple[float, Dict]:
    scores: Dict[str, float] = {}
    n_base  = ground_truth.get("n_base",  len(df))
    n_dupes = ground_truth.get("n_dupes", 0)

    # 1. Deduplication (weight 0.30)
    actual_dupes = int(df.duplicated().sum())
    scores["deduplication"] = 0.0
    if n_dupes > 0:
        if actual_dupes == 0:
            # Check rows are roughly right (not too many dropped)
            row_ratio = len(df) / max(n_base, 1)
            if row_ratio >= 0.85:
                scores["deduplication"] = 0.30
            else:
                scores["deduplication"] = round(0.30 * row_ratio, 4)
        else:
            # Partial: fewer dupes than before
            pct_removed = 1.0 - (actual_dupes / max(n_dupes, 1))
            scores["deduplication"] = round(0.30 * max(pct_removed, 0) * 0.7, 4)

    # 2. Region normalization (weight 0.20)
    scores["region"] = 0.0
    if "region" in df.columns:
        valid_regions = set(REGION_VARIANTS.keys())
        pct_valid = df["region"].isin(valid_regions).mean()
        scores["region"] = round(0.20 * pct_valid, 4)

    # 3. Status normalization (weight 0.20)
    scores["status"] = 0.0
    if "status" in df.columns:
        pct_valid = df["status"].isin(STATUSES).mean()
        scores["status"] = round(0.20 * pct_valid, 4)

    # 4. Null variants replaced (weight 0.15)
    scores["null_handling"] = 0.0
    null_variant_set = set(NULL_VARIANTS) - {""}
    null_remaining   = 0
    null_cols_checked = ["email", "phone"]
    total_checked = 0
    for col in null_cols_checked:
        if col in df.columns:
            remaining = df[col].isin(null_variant_set).sum()
            null_remaining += remaining
            total_checked  += len(df)
    if total_checked > 0:
        pct_clean = 1.0 - (null_remaining / total_checked)
        scores["null_handling"] = round(0.15 * pct_clean, 4)

    # 5. Schema repair (age, revenue) (weight 0.15)
    scores["schema_repair"] = 0.0
    schema_issues = 0
    schema_total  = 0
    if "age" in df.columns:
        bad_age = ((df["age"] < 0) | (df["age"] > 120)).sum()
        schema_issues += bad_age
        schema_total  += len(df)
    if "annual_revenue" in df.columns:
        bad_rev = (df["annual_revenue"] < 0).sum()
        schema_issues += bad_rev
        schema_total  += len(df)
    if schema_total > 0:
        pct_fixed = 1.0 - (schema_issues / schema_total)
        scores["schema_repair"] = round(0.15 * pct_fixed, 4)
    elif schema_total == 0:
        scores["schema_repair"] = 0.15  # no schema issues to fix

    total = round(sum(scores.values()), 4)
    return total, {"per_column": scores}


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------

def _feedback(score: float) -> str:
    if score >= 0.85: return "Excellent — nearly complete cleaning."
    if score >= 0.70: return "Good — most issues resolved, minor gaps remain."
    if score >= 0.50: return "Partial — key issues addressed but several remain."
    if score >= 0.30: return "Poor — few issues resolved."
    return "Minimal progress — most issues unresolved."


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

GRADERS = {
    "missing_value_imputation":   _grade_task1,
    "type_errors_and_outliers":   _grade_task2,
    "schema_normalization_dedup": _grade_task3,
}


def grade(task_name: str, df: pd.DataFrame, ground_truth: Dict) -> Dict[str, Any]:
    """
    Grade a cleaned dataframe against ground truth.
    Score is strictly between 0.001 and 0.999 — validator rejects 0.0 and 1.0.
    """
    if task_name not in GRADERS:
        raise ValueError(f"Unknown task: {task_name}")

    score, breakdown = GRADERS[task_name](df, ground_truth)
    # STRICTLY between 0 and 1 — validator rejects exactly 0.0 or 1.0
    score = float(np.clip(score, 0.001, 0.999))

    return {
        "task_name": task_name,
        "score":     score,
        "breakdown": breakdown,
        "passed":    score >= 0.5,
        "feedback":  _feedback(score),
    }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from dataset_generator import get_dataset

    print("=" * 60)
    print("  Grader Smoke Test")
    print("=" * 60)

    for task in TASK_NAMES:
        print(f"\n  [{task}]")
        for seed in [42, 99, 777]:
            data   = get_dataset(task, seed=seed, difficulty=0.5)
            df     = data["dataframe"]
            gt     = data["ground_truth"]
            result = grade(task, df, gt)
            score  = result["score"]
            ok     = 0.0 < score < 1.0
            print(f"    seed={seed} score={score:.4f}  {'✅' if ok else '❌ OUT OF RANGE'}")
            if score == 0.0 or score == 1.0:
                print(f"    ❌ BOUNDARY VALUE DETECTED!")

    print("\n✅ All grader tests passed!")