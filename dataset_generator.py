"""
dataset_generator.py — Data Cleaning Pipeline OpenEnv
======================================================
Uses real open datasets as base, then injects controlled noise.

Task 1 (Easy)   — Titanic passengers   (real missing values + injected)
Task 2 (Medium) — Sales transaction    (real e-commerce patterns)
Task 3 (Hard)   — CRM customers        (real-world naming + distributions)

All datasets are public domain / CC0. Fallback to synthetic if offline.
"""

from __future__ import annotations

import math
import random
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Real dataset URLs (public domain)
# ---------------------------------------------------------------------------

TITANIC_URL = (
    "https://raw.githubusercontent.com/datasciencedojo/datasets"
    "/master/titanic.csv"
)

NULL_VARIANTS = {"n/a", "none", "-", "", "null", "na", "nan"}

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _inject_nulls(df: pd.DataFrame, column: str, pct: float, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    mask = rng.random(len(df)) < pct
    df = df.copy()
    if df[column].dtype == bool or str(df[column].dtype) == "bool":
        df[column] = df[column].astype(object)
    df.loc[mask, column] = np.nan
    return df


def _df_to_records(df: pd.DataFrame, max_rows: int = 10) -> List[Dict[str, Any]]:
    sample = df.head(max_rows).copy()

    def _clean(val: Any) -> Any:
        if val is None:
            return None
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return None
        if hasattr(val, "item"):
            v = val.item()
            return None if isinstance(v, float) and (math.isnan(v) or math.isinf(v)) else v
        return val

    return [{k: _clean(v) for k, v in row.items()} for row in sample.to_dict(orient="records")]


def _col_stats(df: pd.DataFrame) -> List[Dict[str, Any]]:
    stats = []
    for col in df.columns:
        series = df[col]
        null_count = int(series.isna().sum())
        total = len(series)
        stat: Dict[str, Any] = {
            "name": col,
            "dtype": str(series.dtype),
            "null_count": null_count,
            "null_pct": round(null_count / total, 4) if total else 0.0,
            "unique_count": int(series.nunique(dropna=True)),
            "sample_values": series.dropna().head(5).tolist(),
            "min_value": None, "max_value": None,
            "mean_value": None, "has_outliers": False, "outlier_count": 0,
        }
        is_numeric = (
            pd.api.types.is_numeric_dtype(series)
            and series.dtype != bool
            and str(series.dtype) != "object"
        )
        if is_numeric:
            clean = series.dropna()
            if len(clean):
                mn, mx, mu = float(clean.min()), float(clean.max()), float(clean.mean())
                stat["min_value"]  = None if math.isnan(mn) else mn
                stat["max_value"]  = None if math.isnan(mx) else mx
                stat["mean_value"] = None if math.isnan(mu) else round(mu, 4)
                q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
                iqr = q3 - q1
                outlier_mask = (clean < q1 - 1.5 * iqr) | (clean > q3 + 1.5 * iqr)
                stat["has_outliers"]  = bool(outlier_mask.any())
                stat["outlier_count"] = int(outlier_mask.sum())
        stats.append(stat)
    return stats


def _fetch_url(url: str, timeout: int = 8) -> Optional[pd.DataFrame]:
    """Try to fetch CSV from URL. Returns None if offline."""
    try:
        return pd.read_csv(url, timeout=timeout)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Task 1 — Easy: Missing Value Imputation
# Real base: Titanic passenger dataset
# ---------------------------------------------------------------------------

def generate_task1_dataset(seed: int = 42) -> Dict[str, Any]:
    """
    Base: Titanic dataset — real missing values in Age and Embarked.
    We add extra nulls to Fare and a synthetic years_aboard column.

    Columns:
        passenger_id  int   — no nulls
        name          str   — no nulls
        age           float — real + injected missing → median
        fare          float — injected missing → mean
        pclass        int   — passenger class
        embarked      str   — real missing → mode
        survived      int   — no nulls
        years_aboard  float — synthetic, 25% missing → median
    """
    _set_seed(seed)
    rng = np.random.RandomState(seed)

    raw = _fetch_url(TITANIC_URL)

    if raw is not None and len(raw) >= 100:
        df = raw[["PassengerId", "Name", "Age", "Fare", "Pclass",
                  "Embarked", "Survived"]].copy()
        df.columns = ["passenger_id", "name", "age", "fare",
                      "pclass", "embarked", "survived"]
        df = df.head(200).reset_index(drop=True)
        df["years_aboard"] = rng.randint(0, 20, size=len(df)).astype(float)
        source = "real (Titanic — datasciencedojo)"
    else:
        n = 200
        df = pd.DataFrame({
            "passenger_id": range(1, n + 1),
            "name":         [f"Passenger_{i}" for i in range(n)],
            "age":          rng.normal(29.7, 14.5, n).clip(1, 80).round(1),
            "fare":         rng.exponential(32, n).round(2),
            "pclass":       rng.choice([1, 2, 3], n, p=[0.24, 0.21, 0.55]),
            "embarked":     rng.choice(["S", "C", "Q"], n, p=[0.72, 0.19, 0.09]),
            "survived":     rng.choice([0, 1], n, p=[0.62, 0.38]),
            "years_aboard": rng.randint(0, 20, n).astype(float),
        })
        source = "synthetic (Titanic-like)"

    clean_df = df.copy()

    # Inject extra nulls on top of real ones
    df = _inject_nulls(df, "age",          pct=0.10, seed=seed + 1)
    df = _inject_nulls(df, "fare",         pct=0.15, seed=seed + 2)
    df = _inject_nulls(df, "embarked",     pct=0.05, seed=seed + 3)
    df = _inject_nulls(df, "years_aboard", pct=0.25, seed=seed + 4)

    ground_truth = {
        "age":          {"strategy": "median", "value": float(clean_df["age"].median())},
        "fare":         {"strategy": "mean",   "value": round(float(clean_df["fare"].mean()), 2)},
        "embarked":     {"strategy": "mode",   "value": str(clean_df["embarked"].mode()[0])},
        "years_aboard": {"strategy": "median", "value": float(clean_df["years_aboard"].median())},
    }

    issues = []
    for col in ["age", "fare", "embarked", "years_aboard"]:
        nc = int(df[col].isna().sum())
        if nc:
            issues.append({
                "issue_type": "missing_values", "column": col,
                "severity": "high" if nc > len(df) * 0.1 else "medium",
                "description": f"'{col}' has {nc} missing values ({round(nc/len(df)*100,1)}%)",
                "count": nc,
            })

    return {
        "task_name":        "missing_value_imputation",
        "difficulty":       "easy",
        "source":           source,
        "description":      (
            "Titanic passenger dataset with missing values in 4 columns. "
            "Impute each using the statistically appropriate strategy."
        ),
        "objective":        "Impute: age (median), fare (mean), embarked (mode), years_aboard (median).",
        "dataframe":        df,
        "clean_dataframe":  clean_df,
        "ground_truth":     ground_truth,
        "issues":           issues,
        "total_rows":       len(df),
        "max_steps":        12,
        "scoring_criteria": [
            "age → median (0.25)",
            "fare → mean (0.25)",
            "embarked → mode (0.25)",
            "years_aboard → median (0.25)",
        ],
    }


# ---------------------------------------------------------------------------
# Task 2 — Medium: Type Errors + Outlier Detection
# ---------------------------------------------------------------------------

def generate_task2_dataset(seed: int = 42) -> Dict[str, Any]:
    """
    Sales transaction dataset with injected type errors and outliers.
    Uses realistic product/region distributions.
    """
    _set_seed(seed)
    n = 150
    rng = np.random.RandomState(seed)

    products = ["Laptop", "Mouse", "Keyboard", "Monitor", "Headset",
                "Webcam", "Desk Chair", "Charger", "Speaker", "Tablet"]
    regions  = ["North", "South", "East", "West", "Central"]

    base_date = datetime(2024, 1, 1)
    dates = [(base_date + timedelta(days=int(d))).strftime("%Y-%m-%d")
             for d in rng.randint(0, 365, n)]

    df = pd.DataFrame({
        "order_id":     [f"ORD-{1000+i}" for i in range(n)],
        "product":      rng.choice(products, n),
        "quantity":     rng.randint(1, 50, n),
        "unit_price":   np.round(rng.uniform(5.0, 500.0, n), 2),
        "order_date":   dates,
        "discount_pct": np.round(rng.uniform(0, 30, n), 1),
        "rating":       np.round(rng.uniform(1.0, 5.0, n), 1),
        "region":       rng.choice(regions, n),
    })
    clean_df = df.copy()

    # Type errors
    df["unit_price"] = df["unit_price"].apply(
        lambda x: f"${x:,.2f}" if rng.random() > 0.05 else f"${x:.2f}USD"
    )
    df["quantity"] = df["quantity"].astype(str)

    fmt_choices = ["%Y-%m-%d", "%d/%m/%Y", "%m-%d-%Y", "%B %d, %Y"]
    df["order_date"] = df["order_date"].apply(
        lambda d: pd.to_datetime(d).strftime(random.choice(fmt_choices))
    )

    def corrupt_rating(r):
        x = rng.random()
        if x < 0.07: return "N/A"
        if x < 0.12: return f"{r} stars"
        return str(r)
    df["rating"] = df["rating"].apply(corrupt_rating)

    def corrupt_region(r):
        x = rng.random()
        if x < 0.30: return r.upper()
        if x < 0.50: return r.lower()
        return r
    df["region"] = df["region"].apply(corrupt_region)

    # Outliers
    out_idx = rng.choice(n, size=6, replace=False)
    df.loc[out_idx[:3], "discount_pct"] = np.round(rng.uniform(120, 300, 3), 1)
    df.loc[out_idx[3:], "discount_pct"] = -np.round(rng.uniform(10, 50, 3), 1)

    issues = [
        {"issue_type": "wrong_dtype",  "column": "unit_price",   "severity": "high",
         "description": "unit_price stored as string with '$' prefix",  "count": n},
        {"issue_type": "wrong_dtype",  "column": "quantity",     "severity": "high",
         "description": "quantity stored as string — cast to int",      "count": n},
        {"issue_type": "wrong_dtype",  "column": "order_date",   "severity": "medium",
         "description": "order_date has mixed date formats",            "count": n},
        {"issue_type": "wrong_dtype",  "column": "rating",       "severity": "medium",
         "description": "rating stored as string with N/A entries",     "count": n},
        {"issue_type": "outlier",      "column": "discount_pct", "severity": "high",
         "description": "discount_pct has values outside 0–100",       "count": 6},
        {"issue_type": "format",       "column": "region",       "severity": "low",
         "description": "region has inconsistent casing",              "count": n},
    ]

    ground_truth = {
        "unit_price":   {"action": "cast",      "dtype": "float"},
        "quantity":     {"action": "cast",      "dtype": "int"},
        "order_date":   {"action": "normalize", "format": "%Y-%m-%d"},
        "rating":       {"action": "cast",      "dtype": "float", "clip": [0.0, 5.0]},
        "discount_pct": {"action": "clip",      "lower": 0.0, "upper": 100.0},
        "region":       {"action": "normalize", "method": "titlecase"},
    }

    return {
        "task_name":        "type_errors_and_outliers",
        "difficulty":       "medium",
        "source":           "synthetic (realistic sales patterns)",
        "description":      (
            "Sales transaction dataset with type errors (prices as strings, "
            "mixed date formats, invalid ratings) and outliers in discount_pct."
        ),
        "objective":        "Cast unit_price/quantity/rating, normalize order_date and region, clip discount_pct.",
        "dataframe":        df,
        "clean_dataframe":  clean_df,
        "ground_truth":     ground_truth,
        "issues":           issues,
        "total_rows":       n,
        "max_steps":        20,
        "scoring_criteria": [
            "unit_price cast to float → 0.15",
            "quantity cast to int → 0.15",
            "rating cast to float → 0.15",
            "order_date normalized → 0.10",
            "discount_pct clipped [0,100] → 0.20",
            "region casing normalized → 0.15",
            "row preservation → 0.10",
        ],
    }


# ---------------------------------------------------------------------------
# Task 3 — Hard: Schema Normalization + Deduplication
# ---------------------------------------------------------------------------

def generate_task3_dataset(seed: int = 42) -> Dict[str, Any]:
    """
    CRM customer dataset with realistic real-world naming conventions.
    Real first/last names, real company names, realistic email formats.
    """
    _set_seed(seed)
    n = 200
    rng = np.random.RandomState(seed)

    first_names = ["James", "Mary", "John", "Patricia", "Robert", "Jennifer",
                   "Michael", "Linda", "William", "Barbara", "David", "Susan",
                   "Richard", "Jessica", "Joseph", "Sarah", "Thomas", "Karen"]
    last_names  = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia",
                   "Miller", "Davis", "Wilson", "Taylor", "Anderson", "Thomas"]
    companies   = ["Acme Corp", "GlobalTech", "NexGen", "BlueSky", "DataFlow",
                   "CloudBase", "TechNova", "InnoSys", "CoreLogic", "Apex Ltd"]
    domains     = ["gmail.com", "yahoo.com", "outlook.com",
                   "techcorp.io", "enterprise.net"]

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

    def make_email(fname, lname):
        domain = random.choice(domains)
        style  = rng.randint(0, 3)
        if style == 0: return f"{fname.lower()}.{lname.lower()}@{domain}"
        if style == 1: return f"{fname.lower()[0]}{lname.lower()}@{domain}"
        return f"{fname.lower()}{rng.randint(1,99)}@{domain}"

    def make_phone():
        style = rng.randint(0, 3)
        if style == 0:
            return f"+1-{rng.randint(200,999)}-{rng.randint(100,999)}-{rng.randint(1000,9999)}"
        if style == 1:
            return f"({rng.randint(200,999)}) {rng.randint(100,999)}-{rng.randint(1000,9999)}"
        if style == 2:
            return str(rng.randint(2000000000, 9999999999))
        return f"+44 {rng.randint(1000,9999)} {rng.randint(100000,999999)}"

    assigned_regions   = [random.choice(regions)   for _ in range(n)]
    assigned_countries = [random.choice(countries) for _ in range(n)]
    fnames = [random.choice(first_names) for _ in range(n)]
    lnames = [random.choice(last_names)  for _ in range(n)]

    df = pd.DataFrame({
        "customer_id":    [f"CRM-{1000+i}" for i in range(n)],
        "name":           [f"{f} {l}" for f, l in zip(fnames, lnames)],
        "company":        [random.choice(companies) for _ in range(n)],
        "email":          [make_email(f, l) for f, l in zip(fnames, lnames)],
        "phone":          [make_phone() for _ in range(n)],
        "age":            rng.randint(22, 65, n).astype(float),
        "annual_revenue": np.round(rng.uniform(10000, 500000, n), 2),
        "region":         [random.choice(region_variants[r])  for r in assigned_regions],
        "country":        [random.choice(country_variants[c]) for c in assigned_countries],
        "status":         rng.choice(
                              ["active", "inactive", "pending",
                               "ACTIVE", "Active", "INACTIVE"], n
                          ),
    })

    clean_df = df.copy()
    clean_df["region"]  = pd.Series(assigned_regions)
    clean_df["country"] = pd.Series(assigned_countries)
    clean_df["status"]  = clean_df["status"].str.lower().str.strip()

    # Duplicates
    dup_exact = df.sample(15, random_state=seed).copy()
    dup_near  = df.sample(10, random_state=seed + 1).copy()
    dup_near["email"] = dup_near["email"].apply(
        lambda e: e.replace("@", f".{rng.randint(1,9)}@")
    )
    dup_near["phone"] = [make_phone() for _ in range(len(dup_near))]
    df = pd.concat([df, dup_exact, dup_near], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Schema violations
    bad_idx = rng.choice(len(df), size=8, replace=False)
    df.loc[bad_idx[:4], "age"]            = rng.choice([-5, -1, 150, 999], 4)
    df.loc[bad_idx[4:], "annual_revenue"] = rng.choice([-10000, -500, -1], 4)

    # NULL variants
    null_variants_list = ["N/A", "none", "-", "", "NULL", "n/a"]
    null_idx = rng.choice(len(df), size=20, replace=False)
    for idx in null_idx:
        col = random.choice(["email", "phone", "region"])
        df.loc[idx, col] = random.choice(null_variants_list)

    total_dupes = len(dup_exact) + len(dup_near)
    issues = [
        {"issue_type": "duplicate",     "column": None,
         "severity": "high",   "description": f"{total_dupes} duplicate rows",    "count": total_dupes},
        {"issue_type": "format",        "column": "region",
         "severity": "medium", "description": "region has 5 inconsistent variants", "count": int(df["region"].nunique())},
        {"issue_type": "format",        "column": "country",
         "severity": "medium", "description": "country has inconsistent names/codes","count": int(df["country"].nunique())},
        {"issue_type": "format",        "column": "status",
         "severity": "low",    "description": "status has mixed case",              "count": int(df["status"].nunique())},
        {"issue_type": "missing_values","column": "email",
         "severity": "medium", "description": "email has NULL variant strings",     "count": 7},
        {"issue_type": "missing_values","column": "phone",
         "severity": "medium", "description": "phone has NULL variant strings",     "count": 7},
        {"issue_type": "schema",        "column": "age",
         "severity": "high",   "description": "age has invalid values (neg or >120)","count": 4},
        {"issue_type": "schema",        "column": "annual_revenue",
         "severity": "high",   "description": "annual_revenue has negative values", "count": 4},
    ]

    ground_truth = {
        "deduplication":  {"exact_dupes": len(dup_exact), "near_dupes": len(dup_near)},
        "region":         {"action": "normalize", "mapping": {
            v: k for k, variants in region_variants.items() for v in variants
        }},
        "country":        {"action": "normalize", "mapping": {
            v: k for k, variants in country_variants.items() for v in variants
        }},
        "status":         {"action": "normalize", "method": "lowercase"},
        "null_variants":  {"representations": null_variants_list, "replace_with": None},
        "age":            {"action": "clip", "lower": 0, "upper": 120},
        "annual_revenue": {"action": "clip", "lower": 0},
    }

    return {
        "task_name":        "schema_normalization_dedup",
        "difficulty":       "hard",
        "source":           "real-world-inspired (real names, companies, email patterns)",
        "description":      (
            "CRM customer dataset using real-world naming conventions. "
            "Contains duplicate records, inconsistent formats, schema violations, "
            "and multiple NULL representations."
        ),
        "objective":        "Deduplicate, normalize formats, fix schema violations, standardize NULLs.",
        "dataframe":        df,
        "clean_dataframe":  clean_df,
        "ground_truth":     ground_truth,
        "issues":           issues,
        "total_rows":       len(df),
        "max_steps":        25,
        "scoring_criteria": [
            "Deduplication → 0.30",
            "Format normalization (region/country/status) → 0.30",
            "Schema fixes (age/revenue) → 0.20",
            "NULL standardization → 0.20",
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
    if task_name not in TASK_GENERATORS:
        raise ValueError(f"Unknown task: '{task_name}'. Valid: {list(TASK_GENERATORS)}")
    return TASK_GENERATORS[task_name](seed=seed)


def get_all_tasks(seed: int = 42) -> Dict[str, Dict[str, Any]]:
    return {name: gen(seed=seed) for name, gen in TASK_GENERATORS.items()}


def dataframe_to_records(df: pd.DataFrame, max_rows: int = 10) -> List[Dict[str, Any]]:
    return _df_to_records(df, max_rows=max_rows)


def get_column_stats(df: pd.DataFrame) -> List[Dict[str, Any]]:
    return _col_stats(df)


def detect_issues(df: pd.DataFrame, task_name: str, ground_truth: Dict) -> List[Dict[str, Any]]:
    issues = []

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

    if task_name == "type_errors_and_outliers":
        for col in ["unit_price", "quantity", "rating"]:
            if col in df.columns and df[col].dtype == object:
                issues.append({
                    "issue_type":  "wrong_dtype",
                    "column":      col,
                    "severity":    "high",
                    "description": f"'{col}' is still a string — needs casting",
                    "count":       int(len(df)),
                })
        if "discount_pct" in df.columns \
                and pd.api.types.is_numeric_dtype(df["discount_pct"]) \
                and df["discount_pct"].dtype != bool:
            bad = int(((df["discount_pct"] < 0) | (df["discount_pct"] > 100)).sum())
            if bad:
                issues.append({
                    "issue_type":  "outlier",
                    "column":      "discount_pct",
                    "severity":    "high",
                    "description": f"discount_pct still has {bad} out-of-range values",
                    "count":       bad,
                })

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
    print("=" * 60)
    print("  Dataset Generator — Smoke Test")
    print("=" * 60)

    for task_name, generator in TASK_GENERATORS.items():
        data = generator(seed=42)
        df   = data["dataframe"]
        print(f"\n📋 {data['task_name']} ({data['difficulty'].upper()})")
        print(f"   Source : {data.get('source', 'unknown')}")
        print(f"   Rows   : {len(df)} | Columns: {list(df.columns)}")
        for issue in data["issues"]:
            print(f"   ⚠  [{issue['severity']:6s}] {issue['description']}")

    print("\n✅ All 3 datasets generated successfully!")