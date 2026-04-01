"""
graders.py — Deterministic task graders for Data Cleaning Pipeline
Each grader scores 0.0–1.0 based on how well the agent cleaned the dataset.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Task 1 — Missing Value Imputation
# ---------------------------------------------------------------------------

def grade_task1(df: pd.DataFrame, ground_truth: Dict) -> Tuple[float, Dict]:
    """
    Score: 4 columns × 0.25 each = 1.0
    Full credit  → nulls gone, fill value within tolerance.
    Zero credit  → nulls remain.
    """
    scores = {}
    weights = {"age": 0.25, "fare": 0.25, "embarked": 0.25, "years_aboard": 0.25}

    for col, weight in weights.items():
        if col not in df.columns:
            scores[col] = 0.0
            continue

        null_remaining = int(df[col].isna().sum())
        if null_remaining > 0:
            scores[col] = 0.0
            continue

        gt       = ground_truth[col]
        strategy = gt["strategy"]
        expected = gt["value"]

        if strategy in ("median", "mean"):
            try:
                actual_mean = float(df[col].mean())
                tolerance   = abs(float(expected)) * 0.10 + 1.0
                scores[col] = weight if abs(actual_mean - float(expected)) <= tolerance * 3 else weight * 0.5
            except Exception:
                scores[col] = weight * 0.75
        elif strategy == "mode":
            try:
                actual_mode  = str(df[col].mode()[0]).strip()
                expected_str = str(expected).strip()
                scores[col]  = weight if actual_mode == expected_str else weight * 0.5
            except Exception:
                scores[col] = weight * 0.75
        else:
            scores[col] = weight * 0.75

    total = round(sum(scores.values()), 4)
    return total, {"per_column": scores, "task": "missing_value_imputation"}


def _infer_numeric_fill(series: pd.Series, strategy: str) -> float:
    """Estimate what fill value was used by checking the most common filled value."""
    return float(series.dropna().median() if strategy == "median" else series.dropna().mean())


# ---------------------------------------------------------------------------
# Task 2 — Type Errors + Outlier Detection
# ---------------------------------------------------------------------------

def grade_task2(df: pd.DataFrame, ground_truth: Dict) -> Tuple[float, Dict]:
    """
    Score breakdown:
        unit_price cast to float   → 0.15
        quantity cast to int       → 0.15
        rating cast to float       → 0.15
        order_date normalized      → 0.10
        discount_pct clipped       → 0.20
        region casing fixed        → 0.15
        no unnecessary row drops   → 0.10
    """
    scores = {}

    # unit_price → float
    scores["unit_price"] = 0.0
    if "unit_price" in df.columns and pd.api.types.is_float_dtype(df["unit_price"]):
        if df["unit_price"].between(0, 1_000_000).all():
            scores["unit_price"] = 0.15

    # quantity → int-like
    scores["quantity"] = 0.0
    if "quantity" in df.columns:
        if pd.api.types.is_integer_dtype(df["quantity"]):
            scores["quantity"] = 0.15
        elif pd.api.types.is_float_dtype(df["quantity"]) and df["quantity"].dropna().apply(float.is_integer).all():
            scores["quantity"] = 0.10

    # rating → float, valid range
    scores["rating"] = 0.0
    if "rating" in df.columns and pd.api.types.is_float_dtype(df["rating"]):
        valid = df["rating"].dropna().between(0.0, 5.0).all()
        scores["rating"] = 0.15 if valid else 0.08

    # order_date → parseable dates
    scores["order_date"] = 0.0
    if "order_date" in df.columns:
        try:
            parsed = pd.to_datetime(df["order_date"], errors="coerce")
            scores["order_date"] = round(0.10 * parsed.notna().mean(), 4)
        except Exception:
            pass

    # discount_pct → clipped to [0, 100]
    scores["discount_pct"] = 0.0
    if "discount_pct" in df.columns and pd.api.types.is_numeric_dtype(df["discount_pct"]):
        bad = int(((df["discount_pct"] < 0) | (df["discount_pct"] > 100)).sum())
        scores["discount_pct"] = 0.20 if bad == 0 else round(0.20 * (1 - bad / len(df)), 4)

    # region → title case
    scores["region"] = 0.0
    if "region" in df.columns:
        valid_regions = {"North", "South", "East", "West", "Central"}
        pct_valid = df["region"].dropna().isin(valid_regions).mean()
        scores["region"] = round(0.15 * pct_valid, 4)

    # Row preservation
    scores["row_preservation"] = 0.0
    if len(df) >= 135:
        scores["row_preservation"] = 0.10
    elif len(df) >= 100:
        scores["row_preservation"] = 0.05

    total = round(sum(scores.values()), 4)
    return total, {"per_column": scores, "task": "type_errors_and_outliers"}


# ---------------------------------------------------------------------------
# Task 3 — Schema Normalization + Deduplication
# ---------------------------------------------------------------------------

REGION_MAP = {
    "north": "North", "nth": "North", "n": "North",
    "south": "South", "sth": "South", "s": "South",
    "east":  "East",  "est": "East",  "e": "East",
    "west":  "West",  "wst": "West",  "w": "West",
    "central": "Central", "cntrl": "Central", "c": "Central",
}

COUNTRY_MAP = {
    "us": "USA", "united states": "USA", "u.s.a": "USA",
    "gb": "UK",  "united kingdom": "UK", "britain": "UK", "england": "UK",
    "ca": "Canada", "can": "Canada",
    "au": "Australia", "aus": "Australia",
    "in": "India", "ind": "India",
}

NULL_VARIANTS = {"n/a", "none", "-", "", "null", "na", "nan"}


def grade_task3(df: pd.DataFrame, ground_truth: Dict) -> Tuple[float, Dict]:
    """
    Score breakdown:
        Deduplication quality   → 0.30
        Format normalization    → 0.30  (region + country + status)
        Schema violations fixed → 0.20  (age + annual_revenue)
        NULL standardization    → 0.20
    """
    scores = {}
    original_clean_rows = 200  # before dupes were added

    # --- Deduplication (0.30) — zero tolerance for exact dupes ---
    exact_dupes = int(df.duplicated().sum())
    if exact_dupes == 0:
        # Reward for getting close to original 200 rows
        row_ratio = min(len(df), original_clean_rows) / original_clean_rows
        dup_score = 0.30 * min(row_ratio, 1.0)
    else:
        # Still penalise hard — any remaining dupes means partial credit only
        dup_score = 0.05 * max(0, 1 - exact_dupes / 25)
    scores["deduplication"] = round(max(0.0, dup_score), 4)

    # --- Region normalization (0.10) — strict: must be canonical form ---
    scores["region"] = 0.0
    if "region" in df.columns:
        valid_regions = {"North", "South", "East", "West", "Central"}
        pct_valid = df["region"].dropna().isin(valid_regions).mean()
        scores["region"] = round(0.10 * pct_valid, 4) if pct_valid > 0.95 else round(0.03 * pct_valid, 4)

    # --- Country normalization (0.10) — strict ---
    scores["country"] = 0.0
    if "country" in df.columns:
        valid_countries = {"USA", "UK", "Canada", "Australia", "India"}
        pct_valid = df["country"].dropna().isin(valid_countries).mean()
        scores["country"] = round(0.10 * pct_valid, 4) if pct_valid > 0.95 else round(0.03 * pct_valid, 4)

    # --- Status normalization (0.10) — penalise mixed case still present ---
    scores["status"] = 0.0
    if "status" in df.columns:
        valid_statuses = {"active", "inactive", "pending"}
        has_mixed_case = df["status"].dropna().str.contains(r"[A-Z]", regex=True).any()
        pct_valid = df["status"].dropna().str.lower().str.strip().isin(valid_statuses).mean()
        scores["status"] = round((0.03 if has_mixed_case else 0.10) * pct_valid, 4)

    # --- Age schema fix (0.10) ---
    scores["age"] = 0.0
    if "age" in df.columns and pd.api.types.is_numeric_dtype(df["age"]):
        bad_age = ((df["age"] < 0) | (df["age"] > 120)).sum()
        scores["age"] = 0.10 if bad_age == 0 else round(0.10 * (1 - bad_age / len(df)), 4)

    # --- Revenue schema fix (0.10) ---
    scores["annual_revenue"] = 0.0
    if "annual_revenue" in df.columns and pd.api.types.is_numeric_dtype(df["annual_revenue"]):
        bad_rev = (df["annual_revenue"] < 0).sum()
        scores["annual_revenue"] = 0.10 if bad_rev == 0 else round(0.10 * (1 - bad_rev / len(df)), 4)

    # --- NULL standardization (0.20) — must eliminate ALL variant representations ---
    null_variant_count = 0
    total_cells = 0
    for col in ["email", "phone", "region"]:
        if col in df.columns:
            str_vals = df[col].dropna().astype(str).str.lower().str.strip()
            null_variant_count += int(str_vals.isin(NULL_VARIANTS).sum())
            total_cells += len(str_vals)

    if total_cells == 0:
        scores["null_handling"] = 0.20
    elif null_variant_count == 0:
        scores["null_handling"] = 0.20
    else:
        # Steep penalty — partial cleanup gets little credit
        pct_clean = 1 - (null_variant_count / max(total_cells, 1))
        scores["null_handling"] = round(0.08 * pct_clean, 4)

    total = round(sum(scores.values()), 4)
    return total, {"per_column": scores, "task": "schema_normalization_dedup"}


# ---------------------------------------------------------------------------
# Unified grader entry point
# ---------------------------------------------------------------------------

GRADERS = {
    "missing_value_imputation":   grade_task1,
    "type_errors_and_outliers":   grade_task2,
    "schema_normalization_dedup": grade_task3,
}


def grade(task_name: str, df: pd.DataFrame, ground_truth: Dict) -> Dict[str, Any]:
    """
    Grade a cleaned dataframe against ground truth.
    Returns a dict with score, breakdown, passed flag, and feedback.
    """
    if task_name not in GRADERS:
        raise ValueError(f"Unknown task: {task_name}")

    score, breakdown = GRADERS[task_name](df, ground_truth)
    score = float(np.clip(score, 0.0, 1.0))

    return {
        "task_name": task_name,
        "score":     score,
        "breakdown": breakdown,
        "passed":    score >= 0.6,
        "feedback":  _feedback(score),
    }


def _feedback(score: float) -> str:
    if score >= 0.9:  return "Excellent — nearly perfect cleaning."
    if score >= 0.7:  return "Good — most issues resolved, minor gaps remain."
    if score >= 0.5:  return "Partial — key issues addressed but several remain."
    if score >= 0.3:  return "Poor — few issues resolved."
    return "Minimal progress — most issues unresolved."


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from dataset_generator import get_all_tasks, dataframe_to_records

    print("=" * 55)
    print("  Grader Smoke Test — Unmodified (messy) datasets")
    print("=" * 55)

    for task_name, data in get_all_tasks(seed=42).items():
        result = grade(task_name, data["dataframe"], data["ground_truth"])
        print(f"\n  {task_name}")
        print(f"  Score    : {result['score']:.4f}")
        print(f"  Passed   : {result['passed']}")
        print(f"  Feedback : {result['feedback']}")
        print(f"  Breakdown: {result['breakdown']['per_column']}")

    print("\n" + "=" * 55)
    print("  Grader Smoke Test — Perfectly cleaned datasets")
    print("=" * 55)

    # Task 1 — simulate perfect imputation
    data1 = get_all_tasks(seed=42)["missing_value_imputation"]
    df1   = data1["dataframe"].copy()
    gt1   = data1["ground_truth"]
    df1["age"]        = df1["age"].fillna(gt1["age"]["value"])
    df1["salary"]     = df1["salary"].fillna(gt1["salary"]["value"])
    df1["department"] = df1["department"].fillna(gt1["department"]["value"])
    df1["years_exp"]  = df1["years_exp"].fillna(gt1["years_exp"]["value"])
    df1["is_manager"] = df1["is_manager"].fillna(gt1["is_manager"]["value"])
    r1 = grade("missing_value_imputation", df1, gt1)
    print(f"\n  Task 1 (perfect): {r1['score']:.4f} — {r1['feedback']}")

    # Task 2 — simulate perfect cleaning
    data2 = get_all_tasks(seed=42)["type_errors_and_outliers"]
    df2   = data2["dataframe"].copy()
    df2["price"]        = df2["price"].str.replace(r"[^\d.]", "", regex=True).astype(float)
    df2["quantity"]     = df2["quantity"].astype(int)
    df2["rating"]       = pd.to_numeric(df2["rating"].str.extract(r"(\d+\.?\d*)")[0], errors="coerce").clip(0, 5)
    df2["order_date"]   = pd.to_datetime(df2["order_date"], dayfirst=False, errors="coerce")
    df2["discount_pct"] = df2["discount_pct"].clip(0, 100)
    df2["weight_kg"]    = df2["weight_kg"].clip(upper=200)
    r2 = grade("type_errors_and_outliers", df2, data2["ground_truth"])
    print(f"  Task 2 (perfect): {r2['score']:.4f} — {r2['feedback']}")

    # Task 3 — simulate perfect cleaning
    data3 = get_all_tasks(seed=42)["schema_normalization_dedup"]
    df3   = data3["dataframe"].copy()
    df3   = df3.drop_duplicates()
    df3["region"]  = df3["region"].str.lower().str.strip().map(REGION_MAP).fillna(df3["region"])
    df3["country"] = df3["country"].str.lower().str.strip().map(COUNTRY_MAP).fillna(df3["country"])
    df3["status"]  = df3["status"].str.lower().str.strip()
    df3["age"]     = pd.to_numeric(df3["age"], errors="coerce").clip(0, 120)
    df3["annual_revenue"] = pd.to_numeric(df3["annual_revenue"], errors="coerce").clip(lower=0)
    for col in ["email", "phone", "region"]:
        df3[col] = df3[col].replace(list(NULL_VARIANTS), np.nan)
    r3 = grade("schema_normalization_dedup", df3, data3["ground_truth"])
    print(f"  Task 3 (perfect): {r3['score']:.4f} — {r3['feedback']}")

    print("\n✅ Graders working correctly!")