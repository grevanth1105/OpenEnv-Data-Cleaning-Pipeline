"""
dataset_generator.py — Procedural Dataset Generator
=====================================================
Every seed generates a completely unique dataset.
difficulty (0.1–1.0) controls how messy the data is.

Task 1 — missing_value_imputation
Task 2 — type_errors_and_outliers
Task 3 — schema_normalization_dedup
"""

from __future__ import annotations

import hashlib
import random
import re
import string
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Name pools for realistic data
# ---------------------------------------------------------------------------

FIRST_NAMES = [
    "James","Mary","John","Patricia","Robert","Jennifer","Michael","Linda",
    "William","Barbara","David","Elizabeth","Richard","Susan","Joseph","Jessica",
    "Thomas","Sarah","Charles","Karen","Priya","Arjun","Vikram","Ananya",
    "Rahul","Deepa","Suresh","Kavya","Ravi","Meera","Chen","Li","Wei","Fang",
    "Mohammed","Fatima","Ali","Aisha","Omar","Zara","Lucas","Emma","Liam","Olivia",
]

LAST_NAMES = [
    "Smith","Johnson","Williams","Brown","Jones","Garcia","Miller","Davis",
    "Wilson","Taylor","Sharma","Patel","Kumar","Singh","Gupta","Reddy",
    "Wang","Zhang","Liu","Chen","Kim","Park","Lee","Tanaka","Suzuki",
    "Ahmed","Khan","Hassan","Ali","Ibrahim","Müller","Schmidt","Fischer",
]

COMPANIES = [
    "TechCorp","DataFlow Inc","Nexus Solutions","Apex Analytics",
    "CloudBase Ltd","InfoTech","Synergy Systems","PeakData","CoreLogic",
    "BlueWave","RedShift","NovaTech","DigitalEdge","FutureSoft","BrightPath",
    "Quantum Labs","Matrix Corp","Fusion Tech","Delta Systems","Sigma AI",
    "Innovate Hub","TurboData","SwiftCloud","AlphaNet","BetaWorks",
]

CITIES = [
    "Mumbai","Delhi","Bangalore","Chennai","Hyderabad","Kolkata","Pune",
    "New York","San Francisco","Seattle","Austin","Boston","Chicago",
    "London","Berlin","Paris","Amsterdam","Singapore","Tokyo","Sydney",
]

REGIONS = ["North","South","East","West","Central"]
REGION_VARIANTS = {
    "North":   ["North","NORTH","north","N","Northern"],
    "South":   ["South","SOUTH","south","S","Southern"],
    "East":    ["East","EAST","east","E","Eastern"],
    "West":    ["West","WEST","west","W","Western"],
    "Central": ["Central","CENTRAL","central","C","Centre"],
}

STATUSES    = ["active","inactive","pending","churned","prospect"]
DEPARTMENTS = ["Engineering","Marketing","Sales","Finance","HR","Operations","Product","Design"]
CATEGORIES  = ["Electronics","Clothing","Food","Books","Sports","Home","Beauty","Toys"]

NULL_VARIANTS = ["N/A","n/a","NA","na","None","none","null","NULL","","—","-","unknown","?"]

# ---------------------------------------------------------------------------
# Seeded RNG helper
# ---------------------------------------------------------------------------

def _rng(seed: int) -> random.Random:
    return random.Random(seed)

def _nprng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)

def _choice(rng: random.Random, pool: list) -> Any:
    return rng.choice(pool)

def _choices(rng: random.Random, pool: list, k: int) -> list:
    return [rng.choice(pool) for _ in range(k)]

# ---------------------------------------------------------------------------
# Task 1 — Missing Value Imputation (procedural)
# ---------------------------------------------------------------------------

def _gen_task1(seed: int, difficulty: float) -> Dict:
    """
    Generate employee/HR dataset with missing values.
    difficulty (0.1–1.0) controls null rate and number of affected columns.
    """
    rng   = _rng(seed)
    nprng = _nprng(seed)

    n_rows   = int(150 + rng.randint(0, 100))           # 150–250 rows
    null_pct = 0.05 + difficulty * 0.30                  # 5–35% nulls

    # Generate base data
    ages       = nprng.integers(22, 65, n_rows).tolist()
    salaries   = (nprng.normal(60000, 15000, n_rows).clip(25000, 150000)).tolist()
    tenures    = nprng.integers(0, 20, n_rows).tolist()
    scores     = nprng.uniform(1.0, 5.0, n_rows).round(1).tolist()
    depts      = _choices(rng, DEPARTMENTS, n_rows)
    cities     = _choices(rng, CITIES, n_rows)
    names      = [f"{_choice(rng, FIRST_NAMES)} {_choice(rng, LAST_NAMES)}" for _ in range(n_rows)]

    # Decide which numeric columns get nulls (difficulty controls count)
    num_null_cols = max(2, int(difficulty * 4))           # 2–4 columns
    null_cols     = rng.sample(["age","salary","tenure","score"], num_null_cols)

    col_data = {
        "employee_id": [f"EMP{seed % 1000:03d}{i:04d}" for i in range(n_rows)],
        "name":        names,
        "department":  depts,
        "city":        cities,
        "age":         ages,
        "salary":      salaries,
        "tenure":      tenures,
        "score":       scores,
    }

    df = pd.DataFrame(col_data)
    df["salary"] = df["salary"].round(2)

    # Inject nulls
    ground_truth_nulls: Dict[str, Dict] = {}
    for col in null_cols:
        n_null  = max(1, int(n_rows * null_pct * rng.uniform(0.5, 1.5)))
        n_null  = min(n_null, n_rows - 5)
        indices = rng.sample(range(n_rows), n_null)

        # Determine correct strategy from data
        if col in ["age","tenure"]:
            strategy = "median"
            expected = float(np.nanmedian(df[col].values))
        else:
            strategy = "mean"
            expected = float(np.nanmean(df[col].values))

        df.loc[indices, col] = np.nan
        ground_truth_nulls[col] = {
            "strategy": strategy,
            "expected": round(expected, 4),
            "null_count": n_null,
            "weight": round(1.0 / num_null_cols, 4),
        }

    return {
        "task_name":    "missing_value_imputation",
        "task_difficulty": _difficulty_label(difficulty),
        "dataframe":    df,
        "ground_truth": {"null_cols": ground_truth_nulls},
        "description":  f"Fix {len(null_cols)} columns with missing values in an HR employee dataset ({n_rows} rows).",
        "n_rows":       n_rows,
        "seed":         seed,
        "difficulty":   difficulty,
    }


# ---------------------------------------------------------------------------
# Task 2 — Type Errors + Outliers (procedural)
# ---------------------------------------------------------------------------

def _gen_task2(seed: int, difficulty: float) -> Dict:
    """
    Generate sales/transactions dataset with type errors and outliers.
    """
    rng   = _rng(seed)
    nprng = _nprng(seed)

    n_rows      = int(100 + rng.randint(0, 80))
    outlier_pct = 0.02 + difficulty * 0.12               # 2–14% outliers

    # Base data
    quantities = nprng.integers(1, 50, n_rows)
    prices     = nprng.uniform(5.0, 500.0, n_rows).round(2)
    discounts  = nprng.uniform(0, 40, n_rows).round(1)
    ratings    = nprng.uniform(1.0, 5.0, n_rows).round(1)
    categories = _choices(rng, CATEGORIES, n_rows)
    regions    = _choices(rng, REGIONS, n_rows)

    # Date formats pool
    date_formats = ["%Y-%m-%d", "%d/%m/%Y", "%m-%d-%Y", "%Y/%m/%d", "%d %b %Y"]
    base_dates = pd.date_range("2023-01-01", periods=n_rows, freq="D")
    fmt_choices = [rng.choice(date_formats) for _ in range(n_rows)]
    date_strs = [d.strftime(f) for d, f in zip(base_dates, fmt_choices)]

    # Type errors
    # price as string with currency symbol
    currency    = rng.choice(["$", "£", "€", "₹"])
    price_strs  = [f"{currency}{p:.2f}" for p in prices]

    # quantity as string
    qty_strs = [str(q) for q in quantities]

    # rating with suffix
    rating_suffix = rng.choice([" stars", " pts", " / 5", ""])
    rating_strs = []
    for r in ratings:
        if rng.random() < 0.15:
            rating_strs.append(rng.choice(["N/A", "n/a", "NA", "unknown"]))
        else:
            rating_strs.append(f"{r}{rating_suffix}" if rating_suffix else str(r))

    # region mixed case
    region_strs = []
    for r in regions:
        variants = REGION_VARIANTS[r]
        region_strs.append(rng.choice(variants))

    df = pd.DataFrame({
        "order_id":    [f"ORD{seed % 100:02d}{i:04d}" for i in range(n_rows)],
        "order_date":  date_strs,
        "category":    categories,
        "quantity":    qty_strs,
        "unit_price":  price_strs,
        "discount_pct": discounts,
        "rating":      rating_strs,
        "region":      region_strs,
    })

    # Inject outliers into discount_pct
    n_outliers = max(1, int(n_rows * outlier_pct))
    out_idx    = rng.sample(range(n_rows), min(n_outliers, n_rows))
    for i in out_idx:
        df.loc[i, "discount_pct"] = rng.choice([-10, -5, 110, 120, 150, 200])

    return {
        "task_name":    "type_errors_and_outliers",
        "task_difficulty": _difficulty_label(difficulty),
        "dataframe":    df,
        "ground_truth": {
            "currency":    currency,
            "date_formats": list(set(fmt_choices)),
            "n_rows":      n_rows,
            "n_outliers":  n_outliers,
        },
        "description":  f"Fix type errors (string prices, mixed dates, messy ratings) and clip outliers ({n_outliers} rows) in a sales dataset ({n_rows} rows).",
        "n_rows":       n_rows,
        "seed":         seed,
        "difficulty":   difficulty,
    }


# ---------------------------------------------------------------------------
# Task 3 — Schema Normalization + Dedup (procedural)
# ---------------------------------------------------------------------------

def _gen_task3(seed: int, difficulty: float) -> Dict:
    """
    Generate CRM/customer dataset with duplicates, inconsistent schema, nulls.
    """
    rng   = _rng(seed)
    nprng = _nprng(seed)

    n_base   = int(150 + rng.randint(0, 80))
    n_dupes  = max(5, int(n_base * (0.05 + difficulty * 0.15)))  # 5–20% dupes
    null_pct = 0.03 + difficulty * 0.10

    # Base records
    first_names  = _choices(rng, FIRST_NAMES, n_base)
    last_names   = _choices(rng, LAST_NAMES,  n_base)
    companies    = _choices(rng, COMPANIES,   n_base)
    ages         = nprng.integers(20, 70, n_base).tolist()
    revenues     = nprng.uniform(10000, 500000, n_base).round(0).tolist()
    regions      = _choices(rng, REGIONS, n_base)
    statuses     = _choices(rng, STATUSES, n_base)

    emails = [f"{fn.lower()}.{ln.lower()}@{rng.choice(['gmail','yahoo','outlook','company'])}.com"
              for fn, ln in zip(first_names, last_names)]
    phones = [f"+91-{nprng.integers(7000000000, 9999999999)}" for _ in range(n_base)]

    df = pd.DataFrame({
        "customer_id": [f"C{seed % 1000:03d}{i:04d}" for i in range(n_base)],
        "first_name":  first_names,
        "last_name":   last_names,
        "email":       emails,
        "phone":       phones,
        "company":     companies,
        "age":         ages,
        "annual_revenue": revenues,
        "region":      regions,
        "status":      statuses,
    })

    # Inject issues
    # 1. Region variants
    for i in range(len(df)):
        r = df.loc[i, "region"]
        if r in REGION_VARIANTS:
            df.loc[i, "region"] = rng.choice(REGION_VARIANTS[r])

    # 2. Status mixed case
    status_variants = {s: [s, s.upper(), s.title(), s[:1].upper() + s[1:]] for s in STATUSES}
    for i in range(len(df)):
        s = df.loc[i, "status"]
        if s in status_variants:
            df.loc[i, "status"] = rng.choice(status_variants[s])

    # 3. Null variants in email/phone
    n_null_email = max(1, int(n_base * null_pct))
    n_null_phone = max(1, int(n_base * null_pct))
    for i in rng.sample(range(n_base), min(n_null_email, n_base)):
        df.loc[i, "email"] = rng.choice(NULL_VARIANTS)
    for i in rng.sample(range(n_base), min(n_null_phone, n_base)):
        df.loc[i, "phone"] = rng.choice(NULL_VARIANTS)

    # 4. Invalid ages and revenues
    n_bad = max(1, int(n_base * difficulty * 0.05))
    for i in rng.sample(range(n_base), min(n_bad, n_base)):
        df.loc[i, "age"] = rng.choice([-5, -1, 130, 150, -20])
    for i in rng.sample(range(n_base), min(n_bad, n_base)):
        df.loc[i, "annual_revenue"] = rng.choice([-1000, -50000, -99])

    # 5. Inject duplicates
    dup_indices = rng.sample(range(n_base), min(n_dupes, n_base))
    dup_rows    = df.iloc[dup_indices].copy()
    dup_rows.index = range(n_base, n_base + len(dup_rows))
    # Slightly modify some duplicates (near-dupes)
    for i in dup_rows.index:
        if rng.random() < 0.3:
            dup_rows.loc[i, "email"] = rng.choice(NULL_VARIANTS)

    df = pd.concat([df, dup_rows], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    return {
        "task_name":    "schema_normalization_dedup",
        "task_difficulty": _difficulty_label(difficulty),
        "dataframe":    df,
        "ground_truth": {
            "n_base":   n_base,
            "n_dupes":  n_dupes,
            "valid_regions":  list(REGION_VARIANTS.keys()),
            "valid_statuses": STATUSES,
            "age_range":      (0, 120),
            "revenue_min":    0,
        },
        "description":  f"Fix {n_dupes} duplicate rows, normalize region/status formats, and repair schema violations in a CRM dataset ({len(df)} rows).",
        "n_rows":       len(df),
        "seed":         seed,
        "difficulty":   difficulty,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _difficulty_label(d: float) -> str:
    if d <= 0.35:  return "easy"
    if d <= 0.65:  return "medium"
    return "hard"


def get_dataset(task_name: str, seed: int = 42, difficulty: float = 0.5) -> Dict:
    """
    Generate a dataset for the given task, seed and difficulty.
    Every unique (task_name, seed, difficulty) produces a unique dataset.
    """
    difficulty = float(np.clip(difficulty, 0.1, 1.0))
    generators = {
        "missing_value_imputation":   _gen_task1,
        "type_errors_and_outliers":   _gen_task2,
        "schema_normalization_dedup": _gen_task3,
        "data_type_inference":        _gen_task4,
        "text_standardization":       _gen_task5,
    }
    if task_name not in generators:
        raise ValueError(f"Unknown task: {task_name}. Choose from: {list(generators)}")
    return generators[task_name](seed, difficulty)


def get_column_stats(df: pd.DataFrame) -> List[Dict]:
    """Compute per-column statistics for the observation."""
    stats = []
    for col in df.columns:
        series  = df[col]
        is_num  = pd.api.types.is_numeric_dtype(series)
        nulls   = int(series.isna().sum())
        n       = max(len(series), 1)

        # Unique count
        unique_count = int(series.nunique(dropna=True))

        # Outlier count for numeric columns
        outliers = 0
        if is_num and len(series.dropna()) > 4:
            q1, q3 = series.quantile([0.25, 0.75])
            iqr    = q3 - q1
            if iqr > 0:
                outliers = int(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum())

        # Min / max / mean for numeric
        min_val  = None
        max_val  = None
        mean_val = None
        if is_num and len(series.dropna()) > 0:
            min_val  = float(series.min())
            max_val  = float(series.max())
            mean_val = round(float(series.mean()), 4)

        # Sample values (non-null, safe)
        sample = []
        for v in series.dropna().head(3).tolist():
            if isinstance(v, float) and np.isnan(v):
                sample.append(None)
            elif isinstance(v, (np.integer,)):
                sample.append(int(v))
            elif isinstance(v, (np.floating,)):
                sample.append(round(float(v), 4))
            else:
                sample.append(str(v)[:30])

        stats.append({
            "name":          col,
            "dtype":         str(series.dtype),
            "null_count":    nulls,
            "null_pct":      round(nulls / n, 3),
            "unique_count":  unique_count,
            "outlier_count": outliers,
            "has_outliers":  outliers > 0,
            "min_value":     min_val,
            "max_value":     max_val,
            "mean_value":    mean_val,
            "sample_values": sample,
            "is_numeric":    bool(is_num),
        })
    return stats


def detect_issues(task_name: str, df: pd.DataFrame, ground_truth: Dict) -> List[Dict]:
    """Detect current data quality issues for the observation."""
    issues = []

    if task_name == "missing_value_imputation":
        for col, info in ground_truth.get("null_cols", {}).items():
            if col in df.columns:
                n_null = int(df[col].isna().sum())
                if n_null > 0:
                    issues.append({
                        "issue_type":   "missing_values",
                        "column":      col,
                        "severity":    "high" if info["weight"] >= 0.33 else "medium",
                        "count":       n_null,
                        "description": f"{col} has {n_null} missing values — use {info['strategy']}",
                    })

    elif task_name == "type_errors_and_outliers":
        # String price
        if "unit_price" in df.columns:
            try:
                bad = df["unit_price"].astype(str).str.contains(r"[^\d.]", regex=True, na=False).sum()
            except Exception:
                bad = 0
            if bad > 0:
                issues.append({"issue_type": "type_error", "column": "unit_price", "severity": "high",
                                "count": int(bad), "description": f"unit_price is a string with currency symbol ({bad} rows)"})
        # String quantity
        if "quantity" in df.columns:
            if df["quantity"].dtype == object:
                issues.append({"issue_type": "type_error", "column": "quantity", "severity": "high",
                                "count": int(len(df)), "description": "quantity is stored as string — cast to int"})
        # Mixed dates
        if "order_date" in df.columns and df["order_date"].dtype == object:
            issues.append({"issue_type": "type_error", "column": "order_date", "severity": "medium",
                           "count": int(len(df)), "description": "order_date has mixed format strings — normalize to datetime"})
        # Messy ratings
        if "rating" in df.columns and df["rating"].dtype == object:
            issues.append({"issue_type": "type_error", "column": "rating", "severity": "medium",
                           "count": int(len(df)), "description": "rating has text suffixes and nulls — cast to float"})
        # Outliers in discount_pct
        if "discount_pct" in df.columns:
            bad_disc = ((df["discount_pct"] < 0) | (df["discount_pct"] > 100)).sum()
            if bad_disc > 0:
                issues.append({"issue_type": "outlier", "column": "discount_pct", "severity": "high",
                                "count": int(bad_disc), "description": f"discount_pct has {bad_disc} values outside [0, 100]"})
        # Region mixed case
        if "region" in df.columns:
            invalid = (~df["region"].str.istitle()).sum() if df["region"].dtype == object else 0
            if invalid > 0:
                issues.append({"issue_type": "format", "column": "region", "severity": "low",
                                "count": int(invalid), "description": f"region has {invalid} rows with inconsistent casing"})

    elif task_name == "schema_normalization_dedup":
        # Duplicates
        n_dupes = int(df.duplicated().sum())
        if n_dupes > 0:
            issues.append({"issue_type": "duplicates", "column": None, "severity": "high",
                           "count": n_dupes, "description": f"{n_dupes} duplicate rows detected"})
        # Region variants
        if "region" in df.columns and df["region"].dtype == object:
            valid_regions = set(REGION_VARIANTS.keys())
            invalid_reg   = (~df["region"].isin(valid_regions)).sum()
            if invalid_reg > 0:
                issues.append({"issue_type": "format", "column": "region", "severity": "medium",
                               "count": int(invalid_reg), "description": f"region has {invalid_reg} non-standard variants"})
        # Status mixed case
        if "status" in df.columns and df["status"].dtype == object:
            invalid_st = (~df["status"].isin(STATUSES)).sum()
            if invalid_st > 0:
                issues.append({"issue_type": "format", "column": "status", "severity": "medium",
                               "count": int(invalid_st), "description": f"status has {invalid_st} non-lowercase variants"})
        # Null variants
        for col in ["email", "phone"]:
            if col in df.columns:
                null_variants_found = df[col].isin(NULL_VARIANTS).sum()
                if null_variants_found > 0:
                    issues.append({"issue_type": "null_variant", "column": col, "severity": "low",
                                   "count": int(null_variants_found),
                                   "description": f"{col} has {null_variants_found} NULL string variants (N/A, none, etc.)"})
        # Invalid ages
        if "age" in df.columns:
            bad_age = ((df["age"] < 0) | (df["age"] > 120)).sum()
            if bad_age > 0:
                issues.append({"issue_type": "schema", "column": "age", "severity": "high",
                               "count": int(bad_age), "description": f"age has {bad_age} values outside [0, 120]"})
        # Invalid revenues
        if "annual_revenue" in df.columns:
            bad_rev = (df["annual_revenue"] < 0).sum()
            if bad_rev > 0:
                issues.append({"issue_type": "schema", "column": "annual_revenue", "severity": "high",
                               "count": int(bad_rev), "description": f"annual_revenue has {bad_rev} negative values"})

    elif task_name == "data_type_inference":
        target_dtypes = ground_truth.get("target_dtypes", {})
        castable_cols = ground_truth.get("castable_cols", [])
        for col in castable_cols:
            if col not in df.columns:
                continue
            target = target_dtypes.get(col, "")
            actual = str(df[col].dtype)
            is_wrong = (
                (target == "float"    and actual not in ("float64","float32")) or
                (target == "int"      and "int" not in actual.lower()) or
                (target == "bool"     and actual not in ("bool",)) or
                (target == "datetime" and "datetime" not in actual.lower())
            )
            if is_wrong:
                issues.append({
                    "issue_type":  "wrong_dtype",
                    "column":      col,
                    "severity":    "high" if target in ("float","int","datetime") else "medium",
                    "count":       int(len(df)),
                    "description": f"'{col}' is stored as {actual} but should be {target}",
                })

    elif task_name == "text_standardization":
        # Phone issues
        if "phone" in df.columns:
            import re as _re
            valid_pattern = ground_truth.get("valid_phone_pattern", r"^\+91-\d{5}-\d{6}$")
            bad_phones = df["phone"].apply(
                lambda p: not bool(_re.match(valid_pattern, str(p)))
            ).sum()
            if bad_phones > 0:
                issues.append({
                    "issue_type":  "format",
                    "column":      "phone",
                    "severity":    "high",
                    "count":       int(bad_phones),
                    "description": f"phone has {bad_phones} non-standard formats — normalize to +91-XXXXX-XXXXXX",
                })
        # Email issues
        if "email" in df.columns:
            valid_domains = ground_truth.get("valid_email_domains", [])
            bad_emails = df["email"].apply(
                lambda e: not any(str(e).endswith(d) for d in valid_domains)
            ).sum()
            if bad_emails > 0:
                issues.append({
                    "issue_type":  "format",
                    "column":      "email",
                    "severity":    "medium",
                    "count":       int(bad_emails),
                    "description": f"email has {bad_emails} invalid/typo domains",
                })
        # Name casing
        if "full_name" in df.columns:
            bad_names = df["full_name"].apply(
                lambda n: str(n) != " ".join(w.capitalize() for w in str(n).split()) or "," in str(n)
            ).sum()
            if bad_names > 0:
                issues.append({
                    "issue_type":  "format",
                    "column":      "full_name",
                    "severity":    "low",
                    "count":       int(bad_names),
                    "description": f"full_name has {bad_names} rows with wrong casing or format",
                })
        # City variants
        if "city" in df.columns:
            valid_cities = ground_truth.get("valid_cities", [])
            bad_cities = (~df["city"].isin(valid_cities)).sum()
            if bad_cities > 0:
                issues.append({
                    "issue_type":  "format",
                    "column":      "city",
                    "severity":    "medium",
                    "count":       int(bad_cities),
                    "description": f"city has {bad_cities} non-standard variants",
                })
        # Country variants
        if "country" in df.columns:
            valid_countries = ground_truth.get("valid_countries", [])
            bad_countries = (~df["country"].isin(valid_countries)).sum()
            if bad_countries > 0:
                issues.append({
                    "issue_type":  "format",
                    "column":      "country",
                    "severity":    "medium",
                    "count":       int(bad_countries),
                    "description": f"country has {bad_countries} non-standard variants",
                })

    return issues


def dataframe_to_records(df: pd.DataFrame, limit: int = 10) -> List[Dict]:
    """Convert first N rows to JSON-safe records."""
    sample = df.head(limit).copy()
    records = []
    for _, row in sample.iterrows():
        rec = {}
        for col, val in row.items():
            if isinstance(val, float) and np.isnan(val):
                rec[col] = None
            elif isinstance(val, (np.integer,)):
                rec[col] = int(val)
            elif isinstance(val, (np.floating,)):
                rec[col] = round(float(val), 4)
            else:
                rec[col] = val
        records.append(rec)
    return records


TASK_NAMES = [
    "missing_value_imputation",
    "type_errors_and_outliers",
    "schema_normalization_dedup",
    "data_type_inference",
    "text_standardization",
]


# ---------------------------------------------------------------------------
# Task 4 — Data Type Inference (procedural)
# ---------------------------------------------------------------------------

def _gen_task4(seed: int, difficulty: float) -> Dict:
    """
    Generate a dataset where ALL columns are stored as strings (object dtype).
    Agent must infer and cast correct types: datetime, int, float, bool, category.

    This tests reasoning: "this column LOOKS like a date, I should cast it."
    difficulty controls: number of columns to fix and ambiguity level.
    """
    rng   = _rng(seed)
    nprng = _nprng(seed)

    n_rows     = int(100 + rng.randint(0, 80))
    n_fix_cols = max(3, int(difficulty * 6))       # 3–6 columns need casting

    # Build a realistic product/inventory dataset
    product_ids  = [f"PROD{seed % 100:02d}{i:04d}" for i in range(n_rows)]
    product_names = [f"Product_{rng.choice(CATEGORIES)}_{i}" for i in range(n_rows)]
    prices       = nprng.uniform(1.0, 999.99, n_rows).round(2)
    quantities   = nprng.integers(0, 500, n_rows)
    in_stock     = nprng.choice([True, False], n_rows)
    weights_kg   = nprng.uniform(0.1, 50.0, n_rows).round(3)
    ratings      = nprng.uniform(1.0, 5.0, n_rows).round(1)
    base_dates   = pd.date_range("2022-01-01", periods=n_rows, freq="D")

    # Choose date format randomly per seed
    date_fmt = rng.choice(["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d"])
    date_strs = [d.strftime(date_fmt) for d in base_dates]

    # Convert EVERYTHING to string (the core challenge)
    df = pd.DataFrame({
        "product_id":   [str(x) for x in product_ids],
        "product_name": [str(x) for x in product_names],
        "price":        [str(round(x, 2)) for x in prices],
        "quantity":     [str(x) for x in quantities],
        "in_stock":     [str(x) for x in in_stock],     # "True"/"False" as string
        "weight_kg":    [str(round(x, 3)) for x in weights_kg],
        "rating":       [str(x) for x in ratings],
        "last_updated": date_strs,
    })

    # Inject harder cases based on difficulty
    if difficulty > 0.4:
        # Add null-like strings in numeric columns
        null_count = max(1, int(n_rows * 0.05))
        for i in rng.sample(range(n_rows), null_count):
            df.loc[i, "price"] = rng.choice(["N/A", "null", "", "unknown"])
        for i in rng.sample(range(n_rows), null_count):
            df.loc[i, "quantity"] = rng.choice(["N/A", "null", "", "TBD"])

    if difficulty > 0.7:
        # Mix in some invalid values
        for i in rng.sample(range(n_rows), max(1, int(n_rows * 0.03))):
            df.loc[i, "rating"] = rng.choice(["5 stars", "4.5/5", "excellent", "N/A"])

    # Ground truth: what each column SHOULD be
    ground_truth = {
        "target_dtypes": {
            "product_id":   "str",       # keep as string
            "product_name": "str",       # keep as string
            "price":        "float",     # currently "1.23" → 1.23
            "quantity":     "int",       # currently "42" → 42
            "in_stock":     "bool",      # currently "True" → True
            "weight_kg":    "float",     # currently "0.123" → 0.123
            "rating":       "float",     # currently "4.5" → 4.5
            "last_updated": "datetime",  # currently date string → datetime
        },
        "date_format": date_fmt,
        "castable_cols": ["price", "quantity", "in_stock", "weight_kg", "rating", "last_updated"],
    }

    return {
        "task_name":       "data_type_inference",
        "task_difficulty": _difficulty_label(difficulty),
        "dataframe":       df,
        "ground_truth":    ground_truth,
        "description":     f"All {len(df.columns)} columns are stored as strings. Infer and cast each column to its correct dtype (float, int, bool, datetime) in this product inventory dataset ({n_rows} rows).",
        "n_rows":          n_rows,
        "seed":            seed,
        "difficulty":      difficulty,
    }


# ---------------------------------------------------------------------------
# Task 5 — Text Standardization (procedural)
# ---------------------------------------------------------------------------

PHONE_FORMATS = [
    "+91-{a}-{b}",          # +91-98765-43210
    "+91 {a} {b}",          # +91 98765 43210
    "0{a}{b}",              # 09876543210
    "{a}-{b}",              # 98765-43210
    "{a}{b}",               # 9876543210
    "({a}) {b}",            # (98765) 43210
    "+91({a}){b}",          # +91(98765)43210
    "{a}.{b}",              # 98765.43210
]

ZIP_FORMATS = [
    "{z}",          # 500001
    "{z}-{s}",      # 500001-0001
    "  {z}  ",      # spaces around
    "{z[:3]} {z[3:]}",  # 500 001
]

EMAIL_ISSUES = [
    ("gmail.com",   ["gmai.com", "gmial.com", "gamil.com", "gmail.con"]),
    ("yahoo.com",   ["yaho.com", "yahooo.com", "yahoo.co"]),
    ("outlook.com", ["outllook.com", "outlok.com", "outlook.con"]),
]

def _fmt_phone(rng: random.Random, nprng: np.random.Generator) -> str:
    area  = str(nprng.integers(7000, 9999))
    local = str(nprng.integers(100000, 999999))
    fmt   = rng.choice(PHONE_FORMATS)
    return fmt.format(a=area, b=local)


def _gen_task5(seed: int, difficulty: float) -> Dict:
    """
    Generate a contacts dataset with severely inconsistent text formatting.
    Agent must standardize: phone numbers, emails, zip codes, names.

    Tests pattern recognition and regex — very different from other tasks.
    difficulty controls the variety and severity of inconsistencies.
    """
    rng   = _rng(seed)
    nprng = _nprng(seed)

    n_rows     = int(100 + rng.randint(0, 70))
    error_rate = 0.3 + difficulty * 0.5          # 30–80% of rows have issues

    first_names = _choices(rng, FIRST_NAMES, n_rows)
    last_names  = _choices(rng, LAST_NAMES,  n_rows)

    # Name formatting issues
    name_variants = []
    for fn, ln in zip(first_names, last_names):
        if rng.random() < error_rate * 0.5:
            # Various name format issues
            variant = rng.choice([
                f"{fn.upper()} {ln.upper()}",      # ALL CAPS
                f"{fn.lower()} {ln.lower()}",      # all lower
                f"{ln}, {fn}",                     # last, first
                f"{fn[0]}. {ln}",                  # initial
            ])
        else:
            variant = f"{fn} {ln}"                 # correct
        name_variants.append(variant)

    # Phone numbers — many different formats
    phones = [_fmt_phone(rng, nprng) for _ in range(n_rows)]

    # Email addresses — with domain typos
    emails = []
    for fn, ln in zip(first_names, last_names):
        domain_correct = rng.choice(["gmail.com", "yahoo.com", "outlook.com"])
        if rng.random() < error_rate * 0.6:
            # Pick a typo version
            for correct, typos in EMAIL_ISSUES:
                if domain_correct == correct:
                    domain = rng.choice(typos)
                    break
            else:
                domain = domain_correct
        else:
            domain = domain_correct
        emails.append(f"{fn.lower()}.{ln.lower()}@{domain}")

    # Zip codes
    zips = []
    for _ in range(n_rows):
        z = str(nprng.integers(100000, 999999))
        if rng.random() < error_rate * 0.5:
            fmt = rng.choice([
                f"  {z}  ",                        # whitespace
                f"{z[:3]}-{z[3:]}",                # hyphen in middle
                f"{z[:3]} {z[3:]}",                # space in middle
                z[:5],                             # truncated to 5 digits
            ])
            zips.append(fmt)
        else:
            zips.append(z)

    # Cities — mixed case and abbreviations
    city_variants = {
        "Mumbai":    ["Mumbai", "MUMBAI", "mumbai", "Bombay", "MUM"],
        "Delhi":     ["Delhi", "DELHI", "delhi", "New Delhi", "DEL"],
        "Bangalore": ["Bangalore", "BANGALORE", "bangalore", "Bengaluru", "BLR"],
        "Chennai":   ["Chennai", "CHENNAI", "chennai", "Madras", "CHN"],
        "Hyderabad": ["Hyderabad", "HYDERABAD", "hyderabad", "HYD"],
    }
    cities_raw = _choices(rng, list(city_variants.keys()), n_rows)
    cities     = [rng.choice(city_variants[c]) for c in cities_raw]

    # Countries — abbreviations vs full names
    country_pool = {
        "India":   ["India", "IN", "IND", "india", "INDIA"],
        "USA":     ["USA", "US", "United States", "U.S.A", "America"],
        "UK":      ["UK", "United Kingdom", "GB", "U.K.", "Britain"],
    }
    countries_raw = _choices(rng, list(country_pool.keys()), n_rows)
    countries     = [rng.choice(country_pool[c]) for c in countries_raw]

    df = pd.DataFrame({
        "contact_id": [f"CON{seed % 100:02d}{i:04d}" for i in range(n_rows)],
        "full_name":   name_variants,
        "email":       emails,
        "phone":       phones,
        "zip_code":    zips,
        "city":        cities,
        "country":     countries,
    })

    # Count actual issues
    phone_issues   = sum(1 for p in phones if not p.replace("+", "").replace("-", "").replace(" ", "").isdigit())
    email_issues   = sum(1 for e in emails if any(t in e for _, typos in EMAIL_ISSUES for t in typos))
    name_issues    = sum(1 for n in name_variants if n != n.title() or "," in n)

    return {
        "task_name":       "text_standardization",
        "task_difficulty": _difficulty_label(difficulty),
        "dataframe":       df,
        "ground_truth":    {
            "valid_phone_pattern": r"^\+91-\d{5}-\d{6}$",   # target format
            "valid_email_domains": ["gmail.com", "yahoo.com", "outlook.com"],
            "valid_cities":        list(city_variants.keys()),
            "valid_countries":     list(country_pool.keys()),
            "phone_issues":        phone_issues,
            "email_issues":        email_issues,
            "name_issues":         name_issues,
            "target_name_format":  "Title Case",
            "target_zip_format":   "6 digits, no spaces",
        },
        "description": (
            f"Standardize contact information in {n_rows} rows: "
            f"normalize {phone_issues} inconsistent phone formats, "
            f"fix {email_issues} email domain typos, "
            f"standardize name casing and city/country variants."
        ),
        "n_rows":      n_rows,
        "seed":        seed,
        "difficulty":  difficulty,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("  Procedural Generator Smoke Test")
    print("=" * 60)

    for task in TASK_NAMES:
        print(f"\n  [{task}]")
        for seed in [42, 99, 777]:
            for diff in [0.2, 0.6, 0.9]:
                data = get_dataset(task, seed=seed, difficulty=diff)
                df   = data["dataframe"]
                print(f"    seed={seed} diff={diff:.1f} → {len(df):3d} rows "
                      f"{len(df.columns):2d} cols "
                      f"label={data['task_difficulty']}")

    print("\n  Uniqueness check:")
    datasets = [get_dataset("missing_value_imputation", seed=s)["dataframe"]
                for s in range(5)]
    all_unique = len({df.shape[0] for df in datasets}) > 1 or \
                 len({df["age"].iloc[0] for df in datasets}) > 1
    print(f"    {'✅ All seeds produce different datasets' if all_unique else '❌ Seeds collide'}")
    print("\n✅ Generator working correctly!")