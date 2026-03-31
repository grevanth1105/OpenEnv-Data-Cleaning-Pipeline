"""
dataset_generator.py — Data Cleaning Pipeline OpenEnv Environment
==================================================================
Generates reproducible messy datasets for all 3 tasks.

Task 1 (Easy)   — Missing Value Imputation
Task 2 (Medium) — Type Errors + Outlier Detection
Task 3 (Hard)   — Schema Inference + Normalization + Deduplication

Each generator accepts a `seed` for full reproducibility.
"""

from __future__ import annotations

import copy
import random
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _inject_nulls(
    df: pd.DataFrame,
    column: str,
    pct: float,
    seed: int = 42,
) -> pd.DataFrame:
    """Randomly replace `pct` fraction of values in `column` with NaN."""
    rng = np.random.RandomState(seed)
    mask = rng.random(len(df)) < pct
    df = df.copy()
    # Bool columns can't hold NaN in pandas 3.x — cast to object first
    if df[column].dtype == bool or str(df[column].dtype) == "bool":
        df[column] = df[column].astype(object)
    df.loc[mask, column] = np.nan
    return df


def _inject_outliers(
    df: pd.DataFrame,
    column: str,
    n: int = 5,
    multiplier: float = 10.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Replace `n` random values with extreme outliers."""
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(df), size=min(n, len(df)), replace=False)
    df = df.copy()
    col_max = df[column].dropna().max()
    df.loc[idx, column] = col_max * multiplier
    return df


def _df_to_records(df: pd.DataFrame, max_rows: int = 10) -> List[Dict[str, Any]]:
    """Convert DataFrame to list of dicts, replacing NaN/inf with None."""
    import math
    sample = df.head(max_rows).copy()

    def _clean(val: Any) -> Any:
        if val is None:
            return None
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return None
        if hasattr(val, "item"):   # numpy scalar
            v = val.item()
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return None
            return v
        return val

    records = []
    for row in sample.to_dict(orient="records"):
        records.append({k: _clean(v) for k, v in row.items()})
    return records


def _col_stats(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Compute per-column statistics for the observation."""
    stats = []
    for col in df.columns:
        series = df[col]
        null_count = int(series.isna().sum())
        total = len(series)
        unique_count = int(series.nunique(dropna=True))
        sample_vals = series.dropna().head(5).tolist()

        stat: Dict[str, Any] = {
            "name": col,
            "dtype": str(series.dtype),
            "null_count": null_count,
            "null_pct": round(null_count / total, 4) if total else 0.0,
            "unique_count": unique_count,
            "sample_values": sample_vals,
            "min_value": None,
            "max_value": None,
            "mean_value": None,
            "has_outliers": False,
            "outlier_count": 0,
        }

        if pd.api.types.is_numeric_dtype(series) and series.dtype != bool and str(series.dtype) != 'object':
            clean = series.dropna()
            if len(clean):
                min_val  = float(clean.min())
                max_val  = float(clean.max())
                mean_val = float(clean.mean())
                import math
                stat["min_value"]  = None if math.isnan(min_val)  else min_val
                stat["max_value"]  = None if math.isnan(max_val)  else max_val
                stat["mean_value"] = None if math.isnan(mean_val) else round(mean_val, 4)
                # IQR-based outlier detection
                q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
                iqr = q3 - q1
                outlier_mask = (clean < q1 - 1.5 * iqr) | (clean > q3 + 1.5 * iqr)
                stat["has_outliers"] = bool(outlier_mask.any())
                stat["outlier_count"] = int(outlier_mask.sum())

        stats.append(stat)
    return stats


# ---------------------------------------------------------------------------
# Task 1 — Easy: Missing Value Imputation
# ---------------------------------------------------------------------------

def generate_task1_dataset(seed: int = 42) -> Dict[str, Any]:
    """
    Employee HR dataset with deliberate missing values.

    Columns:
        employee_id  : int        — no nulls (key)
        name         : str        — no nulls
        age          : float      — 20% missing → impute with median
        salary       : float      — 15% missing → impute with mean
        department   : str        — 10% missing → impute with mode
        years_exp    : float      — 25% missing → impute with median
        is_manager   : bool/str   — 5%  missing → impute with mode

    Ground truth imputation strategies are stored so grader can check.
    """
    _set_seed(seed)
    n = 120

    departments = ["Engineering", "Marketing", "Sales", "HR", "Finance"]
    dept_weights = [0.35, 0.20, 0.25, 0.10, 0.10]

    df = pd.DataFrame({
        "employee_id": range(1001, 1001 + n),
        "name": [f"Employee_{i}" for i in range(n)],
        "age": np.random.randint(22, 60, size=n).astype(float),
        "salary": np.round(np.random.uniform(40000, 120000, size=n), 2),
        "department": np.random.choice(departments, size=n, p=dept_weights),
        "years_exp": np.random.randint(0, 30, size=n).astype(float),
        "is_manager": np.random.choice([True, False], size=n, p=[0.2, 0.8]),
    })

    # Store clean reference BEFORE injecting nulls
    clean_df = df.copy()

    # Inject missing values
    df = _inject_nulls(df, "age",        pct=0.20, seed=seed + 1)
    df = _inject_nulls(df, "salary",     pct=0.15, seed=seed + 2)
    df = _inject_nulls(df, "department", pct=0.10, seed=seed + 3)
    df = _inject_nulls(df, "years_exp",  pct=0.25, seed=seed + 4)
    df = _inject_nulls(df, "is_manager", pct=0.05, seed=seed + 5)

    # Ground truth
    ground_truth = {
        "age":        {"strategy": "median", "value": float(clean_df["age"].median())},
        "salary":     {"strategy": "mean",   "value": round(float(clean_df["salary"].mean()), 2)},
        "department": {"strategy": "mode",   "value": clean_df["department"].mode()[0]},
        "years_exp":  {"strategy": "median", "value": float(clean_df["years_exp"].median())},
        "is_manager": {"strategy": "mode",   "value": bool(clean_df["is_manager"].mode()[0])},
    }

    issues = []
    for col in ["age", "salary", "department", "years_exp", "is_manager"]:
        null_count = int(df[col].isna().sum())
        if null_count:
            issues.append({
                "issue_type": "missing_values",
                "column": col,
                "severity": "high" if null_count > 20 else "medium",
                "description": f"Column '{col}' has {null_count} missing values ({round(null_count/n*100, 1)}%)",
                "count": null_count,
            })

    return {
        "task_name":        "missing_value_imputation",
        "difficulty":       "easy",
        "description":      "An HR dataset with missing values across multiple columns. Impute each column using the appropriate strategy (mean/median/mode).",
        "objective":        "Impute all missing values in: age (median), salary (mean), department (mode), years_exp (median), is_manager (mode).",
        "dataframe":        df,
        "clean_dataframe":  clean_df,
        "ground_truth":     ground_truth,
        "issues":           issues,
        "total_rows":       n,
        "max_steps":        15,
        "scoring_criteria": [
            "Each correctly imputed column scores 0.2 (5 columns × 0.2 = 1.0)",
            "Strategy match (mean/median/mode) required for full credit",
            "Partial credit if values are close but wrong strategy",
        ],
    }


# ---------------------------------------------------------------------------
# Task 2 — Medium: Type Errors + Outlier Detection
# ---------------------------------------------------------------------------

def generate_task2_dataset(seed: int = 42) -> Dict[str, Any]:
    """
    E-commerce orders dataset with type errors and outliers.

    Columns:
        order_id     : str        — correct
        customer_id  : str        — correct
        price        : str        — should be float (stored as string with '$')
        quantity     : str        — should be int (stored as string)
        order_date   : str        — should be datetime (mixed formats)
        discount_pct : float      — has outliers (>100% discounts)
        weight_kg    : float      — has outliers (extreme values)
        rating       : str        — should be float (0.0-5.0), some invalid

    Ground truth:
        - price → cast to float (strip '$', ',')
        - quantity → cast to int
        - order_date → normalize to '%Y-%m-%d'
        - discount_pct → clip outliers (0-100 range)
        - weight_kg → clip outliers (IQR method)
        - rating → cast to float, clip to 0.0-5.0
    """
    _set_seed(seed)
    n = 150

    # Clean base data
    prices     = np.round(np.random.uniform(5.0, 500.0, size=n), 2)
    quantities = np.random.randint(1, 50, size=n)
    discounts  = np.round(np.random.uniform(0, 30, size=n), 1)
    weights    = np.round(np.random.uniform(0.1, 50.0, size=n), 2)
    ratings    = np.round(np.random.uniform(1.0, 5.0, size=n), 1)

    base_date  = datetime(2024, 1, 1)
    dates      = [base_date + timedelta(days=int(d)) for d in np.random.randint(0, 365, size=n)]

    df = pd.DataFrame({
        "order_id":     [f"ORD-{1000+i}" for i in range(n)],
        "customer_id":  [f"CUST-{random.randint(100, 999)}" for _ in range(n)],
        "price":        prices,
        "quantity":     quantities,
        "order_date":   dates,
        "discount_pct": discounts,
        "weight_kg":    weights,
        "rating":       ratings,
    })
    clean_df = df.copy()

    # --- Introduce type errors ---
    # price → string with '$' and ','
    df["price"] = df["price"].apply(
        lambda x: f"${x:,.2f}" if random.random() > 0.05 else f"${x:.2f}USD"
    )

    # quantity → string
    df["quantity"] = df["quantity"].astype(str)

    # order_date → mixed formats
    fmt_choices = ["%Y-%m-%d", "%d/%m/%Y", "%m-%d-%Y", "%B %d, %Y"]
    df["order_date"] = df["order_date"].apply(
        lambda d: d.strftime(random.choice(fmt_choices))
    )

    # rating → string, some invalid
    def corrupt_rating(r):
        x = random.random()
        if x < 0.07:
            return "N/A"
        elif x < 0.12:
            return str(r) + " stars"
        return str(r)
    df["rating"] = df["rating"].apply(corrupt_rating)

    # --- Inject outliers ---
    # discount_pct: a few > 100
    outlier_idx = np.random.RandomState(seed).choice(n, size=6, replace=False)
    df.loc[outlier_idx[:3], "discount_pct"] = np.random.uniform(120, 300, size=3).round(1)
    df.loc[outlier_idx[3:], "discount_pct"] = -np.random.uniform(10, 50, size=3).round(1)

    # weight_kg: a few extreme values
    w_idx = np.random.RandomState(seed + 10).choice(n, size=5, replace=False)
    df.loc[w_idx, "weight_kg"] = np.random.uniform(500, 2000, size=5).round(2)

    issues = [
        {"issue_type": "wrong_dtype",  "column": "price",        "severity": "high",   "description": "price stored as string with '$' prefix — must cast to float",  "count": n},
        {"issue_type": "wrong_dtype",  "column": "quantity",     "severity": "high",   "description": "quantity stored as string — must cast to int",                 "count": n},
        {"issue_type": "wrong_dtype",  "column": "order_date",   "severity": "medium", "description": "order_date has mixed formats — normalize to %Y-%m-%d",         "count": n},
        {"issue_type": "wrong_dtype",  "column": "rating",       "severity": "medium", "description": "rating stored as string with invalid entries — cast to float",  "count": n},
        {"issue_type": "outlier",      "column": "discount_pct", "severity": "high",   "description": "discount_pct has values outside 0-100 range",                  "count": 6},
        {"issue_type": "outlier",      "column": "weight_kg",    "severity": "medium", "description": "weight_kg has extreme outliers (>500kg)",                       "count": 5},
    ]

    ground_truth = {
        "price":        {"action": "cast",      "dtype": "float",      "preprocess": "strip_currency"},
        "quantity":     {"action": "cast",      "dtype": "int"},
        "order_date":   {"action": "normalize", "format": "%Y-%m-%d"},
        "discount_pct": {"action": "clip",      "lower": 0.0, "upper": 100.0},
        "weight_kg":    {"action": "clip",      "method": "iqr"},
        "rating":       {"action": "cast",      "dtype": "float",      "clip": [0.0, 5.0]},
    }

    return {
        "task_name":        "type_errors_and_outliers",
        "difficulty":       "medium",
        "description":      "An e-commerce orders dataset with type errors (strings instead of numbers/dates) and statistical outliers. Fix all type issues and handle outliers.",
        "objective":        "Cast price/quantity/rating to correct types, normalize order_date format, clip discount_pct and weight_kg outliers.",
        "dataframe":        df,
        "clean_dataframe":  clean_df,
        "ground_truth":     ground_truth,
        "issues":           issues,
        "total_rows":       n,
        "max_steps":        20,
        "scoring_criteria": [
            "Type fixes: each correct cast scores 0.15 (4 columns × 0.15 = 0.60)",
            "Outlier handling: each correct clip/flag scores 0.15 (2 columns × 0.15 = 0.30)",
            "Normalization: date format correct scores 0.10",
            "Penalty: -0.05 for dropping valid rows unnecessarily",
        ],
    }


# ---------------------------------------------------------------------------
# Task 3 — Hard: Schema Inference + Normalization + Deduplication
# ---------------------------------------------------------------------------

def generate_task3_dataset(seed: int = 42) -> Dict[str, Any]:
    """
    Customer CRM dataset — fully messy.

    Issues:
        1. Duplicates (exact + near-duplicates with slight variations)
        2. Inconsistent formats (phone, email, country codes, names)
        3. Schema violations (age < 0, revenue < 0, invalid email)
        4. Mixed encodings in categorical columns
        5. Referential integrity (invalid region codes)
        6. Multiple NULL representations ('N/A', 'none', '-', '')

    Agent must:
        - Deduplicate (exact + fuzzy)
        - Normalize phone/email/country formats
        - Fix schema violations (clip/drop invalid rows)
        - Standardize categorical values
        - Handle all NULL representations
    """
    _set_seed(seed)
    n = 200

    regions = ["North", "South", "East", "West", "Central"]
    region_variants = {
        "North":   ["North", "north", "N", "NORTH", "Nth"],
        "South":   ["South", "south", "S", "SOUTH", "Sth"],
        "East":    ["East",  "east",  "E", "EAST",  "Est"],
        "West":    ["West",  "west",  "W", "WEST",  "Wst"],
        "Central": ["Central", "central", "C", "CENTRAL", "Cntrl"],
    }

    countries = ["USA", "UK", "Canada", "Australia", "India"]
    country_variants = {
        "USA":       ["USA", "US", "United States", "U.S.A", "united states"],
        "UK":        ["UK",  "GB", "United Kingdom", "Britain", "england"],
        "Canada":    ["Canada", "CA", "CAN", "canada"],
        "Australia": ["Australia", "AU", "AUS", "Aus"],
        "India":     ["India", "IN", "IND", "india"],
    }

    def random_phone():
        formats = [
            f"+1-{random.randint(200,999)}-{random.randint(100,999)}-{random.randint(1000,9999)}",
            f"({random.randint(200,999)}) {random.randint(100,999)}-{random.randint(1000,9999)}",
            f"{random.randint(2000000000,9999999999)}",
            f"+44 {random.randint(1000,9999)} {random.randint(100000,999999)}",
        ]
        return random.choice(formats)

    def random_email(name):
        domains = ["gmail.com", "yahoo.com", "outlook.com", "company.org"]
        variants = [
            f"{name.lower()}@{random.choice(domains)}",
            f"{name.lower()}.{random.randint(1,99)}@{random.choice(domains)}",
            f"{name.lower()[:3]}{random.randint(10,99)}@{random.choice(domains)}",
        ]
        return random.choice(variants)

    names = [f"Customer_{i}" for i in range(n)]

    ages     = np.random.randint(18, 70, size=n).astype(float)
    revenues = np.round(np.random.uniform(1000, 500000, size=n), 2)

    assigned_regions   = [random.choice(regions)    for _ in range(n)]
    assigned_countries = [random.choice(countries)  for _ in range(n)]

    df = pd.DataFrame({
        "customer_id":  [f"CRM-{1000+i}" for i in range(n)],
        "name":         names,
        "email":        [random_email(name) for name in names],
        "phone":        [random_phone() for _ in range(n)],
        "age":          ages,
        "annual_revenue": revenues,
        "region":       [random.choice(region_variants[r])  for r in assigned_regions],
        "country":      [random.choice(country_variants[c]) for c in assigned_countries],
        "status":       np.random.choice(["active","inactive","pending","ACTIVE","Active","INACTIVE"], size=n),
    })

    clean_df = df.copy()
    # Normalize for reference
    clean_df["region"]  = pd.Series(assigned_regions)
    clean_df["country"] = pd.Series(assigned_countries)
    clean_df["status"]  = clean_df["status"].str.lower().str.strip()

    # --- Inject duplicates ---
    # Exact duplicates (15 rows)
    dup_exact = df.sample(15, random_state=seed).copy()
    # Near-duplicates (10 rows — same person, slightly different data)
    dup_near  = df.sample(10, random_state=seed + 1).copy()
    dup_near["email"] = dup_near["email"].apply(
        lambda e: e.replace("@", f".{random.randint(1,9)}@")
    )
    dup_near["phone"] = [random_phone() for _ in range(len(dup_near))]

    df = pd.concat([df, dup_exact, dup_near], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)  # shuffle

    # --- Inject schema violations ---
    bad_idx = np.random.RandomState(seed + 5).choice(len(df), size=8, replace=False)
    df.loc[bad_idx[:4], "age"]            = np.random.choice([-5, -1, 150, 999], size=4)
    df.loc[bad_idx[4:], "annual_revenue"] = np.random.choice([-10000, -500, -1], size=4)

    # --- Inject NULL variants ---
    null_variants = ["N/A", "none", "-", "", "NULL", "n/a"]
    null_idx = np.random.RandomState(seed + 6).choice(len(df), size=20, replace=False)
    for i, idx in enumerate(null_idx):
        col = random.choice(["email", "phone", "region"])
        df.loc[idx, col] = random.choice(null_variants)

    total_dupes = len(dup_exact) + len(dup_near)

    issues = [
        {"issue_type": "duplicate",   "column": None,             "severity": "high",   "description": f"{total_dupes} duplicate/near-duplicate rows detected",           "count": total_dupes},
        {"issue_type": "format",      "column": "region",         "severity": "medium", "description": "region has inconsistent capitalisation/abbreviations",             "count": int((df["region"].nunique()))},
        {"issue_type": "format",      "column": "country",        "severity": "medium", "description": "country has inconsistent names/codes (USA vs US vs United States)", "count": int(df["country"].nunique())},
        {"issue_type": "format",      "column": "status",         "severity": "low",    "description": "status has mixed case (active/ACTIVE/Active)",                     "count": int(df["status"].nunique())},
        {"issue_type": "missing_values","column":"email",         "severity": "medium", "description": "email has NULL variants (N/A, none, -, empty)",                    "count": 7},
        {"issue_type": "missing_values","column":"phone",         "severity": "medium", "description": "phone has NULL variants",                                           "count": 7},
        {"issue_type": "schema",      "column": "age",            "severity": "high",   "description": "age has invalid values (negative or >120)",                        "count": 4},
        {"issue_type": "schema",      "column": "annual_revenue", "severity": "high",   "description": "annual_revenue has negative values",                               "count": 4},
    ]

    ground_truth = {
        "deduplication":    {"exact_dupes": len(dup_exact), "near_dupes": len(dup_near)},
        "region":           {"action": "normalize", "mapping": {v: k for k, variants in region_variants.items() for v in variants}},
        "country":          {"action": "normalize", "mapping": {v: k for k, variants in country_variants.items() for v in variants}},
        "status":           {"action": "normalize", "method": "lowercase"},
        "null_variants":    {"representations": null_variants, "replace_with": None},
        "age":              {"action": "clip", "lower": 0, "upper": 120},
        "annual_revenue":   {"action": "clip", "lower": 0},
    }

    return {
        "task_name":        "schema_normalization_dedup",
        "difficulty":       "hard",
        "description":      "A CRM customer dataset with duplicates, inconsistent formats, schema violations, and multiple NULL representations. A comprehensive cleaning challenge.",
        "objective":        "Deduplicate rows, normalize region/country/status formats, standardize NULL representations, and fix schema violations in age and annual_revenue.",
        "dataframe":        df,
        "clean_dataframe":  clean_df,
        "ground_truth":     ground_truth,
        "issues":           issues,
        "total_rows":       len(df),
        "max_steps":        25,
        "scoring_criteria": [
            "Deduplication: recall × precision of removed rows (0.30)",
            "Format normalization: % of columns correctly standardized (0.30)",
            "Schema fix: % of violations corrected (0.20)",
            "NULL handling: % of null variants replaced with proper NaN (0.20)",
        ],
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

TASK_GENERATORS = {
    "missing_value_imputation":    generate_task1_dataset,
    "type_errors_and_outliers":    generate_task2_dataset,
    "schema_normalization_dedup":  generate_task3_dataset,
}


def get_dataset(task_name: str, seed: int = 42) -> Dict[str, Any]:
    """Return a fresh dataset dict for the given task name."""
    if task_name not in TASK_GENERATORS:
        raise ValueError(f"Unknown task: '{task_name}'. Valid tasks: {list(TASK_GENERATORS)}")
    return TASK_GENERATORS[task_name](seed=seed)


def get_all_tasks(seed: int = 42) -> Dict[str, Dict[str, Any]]:
    """Return datasets for all 3 tasks."""
    return {name: gen(seed=seed) for name, gen in TASK_GENERATORS.items()}


def dataframe_to_records(df: pd.DataFrame, max_rows: int = 10) -> List[Dict[str, Any]]:
    """Convert a DataFrame to JSON-serialisable records."""
    return _df_to_records(df, max_rows=max_rows)


def get_column_stats(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Return per-column statistics."""
    return _col_stats(df)


def detect_issues(df: pd.DataFrame, task_name: str, ground_truth: Dict) -> List[Dict[str, Any]]:
    """
    Re-scan a (partially cleaned) dataframe and return remaining issues.
    Used by the environment to update hints after each step.
    """
    issues = []

    # Missing values
    for col in df.columns:
        null_count = int(df[col].isna().sum())
        if null_count > 0:
            issues.append({
                "issue_type":  "missing_values",
                "column":      col,
                "severity":    "high" if null_count > len(df) * 0.1 else "medium",
                "description": f"'{col}' still has {null_count} missing values",
                "count":       null_count,
            })

    # Type errors — check if numeric columns are still stored as object
    if task_name == "type_errors_and_outliers":
        for col in ["price", "quantity", "rating"]:
            if col in df.columns and df[col].dtype == object:
                issues.append({
                    "issue_type":  "wrong_dtype",
                    "column":      col,
                    "severity":    "high",
                    "description": f"'{col}' is still a string — needs casting",
                    "count":       int(len(df)),
                })
        if "discount_pct" in df.columns and pd.api.types.is_numeric_dtype(df["discount_pct"]) and df["discount_pct"].dtype != bool:
            bad = ((df["discount_pct"] < 0) | (df["discount_pct"] > 100)).sum()
            if bad:
                issues.append({
                    "issue_type":  "outlier",
                    "column":      "discount_pct",
                    "severity":    "high",
                    "description": f"discount_pct still has {bad} out-of-range values",
                    "count":       int(bad),
                })

    # Duplicates
    if task_name == "schema_normalization_dedup":
        dup_count = int(df.duplicated().sum())
        if dup_count:
            issues.append({
                "issue_type":  "duplicate",
                "column":      None,
                "severity":    "high",
                "description": f"{dup_count} duplicate rows remain",
                "count":       dup_count,
            })

        # Schema violations
        if "age" in df.columns and pd.api.types.is_numeric_dtype(df["age"]):
            bad_age = int(((df["age"] < 0) | (df["age"] > 120)).sum())
            if bad_age:
                issues.append({
                    "issue_type":  "schema",
                    "column":      "age",
                    "severity":    "high",
                    "description": f"age still has {bad_age} invalid values",
                    "count":       bad_age,
                })

    return issues


if __name__ == "__main__":
    # Quick smoke test
    print("=" * 60)
    print("  Dataset Generator — Smoke Test")
    print("=" * 60)

    for task_name, generator in TASK_GENERATORS.items():
        data = generator(seed=42)
        df   = data["dataframe"]
        print(f"\n📋 Task: {data['task_name']} ({data['difficulty'].upper()})")
        print(f"   Rows      : {len(df)}")
        print(f"   Columns   : {list(df.columns)}")
        print(f"   Issues    : {len(data['issues'])}")
        print(f"   Max steps : {data['max_steps']}")
        for issue in data["issues"]:
            print(f"   ⚠  [{issue['severity']:6s}] {issue['description']}")

    print("\n✅ All 3 datasets generated successfully!")