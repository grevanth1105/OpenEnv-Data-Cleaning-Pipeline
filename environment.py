"""
environment.py — Core RL Environment
======================================
DataCleaningEnvironment manages one episode:
  reset(task_name, seed, difficulty) → observation
  step(action)                       → observation, reward, done, info
  get_grader_result()                → score dict

difficulty (0.1–1.0) controls how messy the dataset is.
Every (task_name, seed, difficulty) produces a unique episode.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from dataset_generator import (
    REGION_VARIANTS, NULL_VARIANTS, STATUSES,
    get_dataset, get_column_stats, detect_issues, dataframe_to_records,
)
from graders import grade
from code_sandbox import execute_cleaning_code, score_code_execution
from models import (
    ActionType,
    DataCleaningAction,
    DataCleaningObservation,
    DataCleaningState,
)

TASK_NAMES = [
    "missing_value_imputation",
    "type_errors_and_outliers",
    "schema_normalization_dedup",
]

# Max steps per difficulty level
MAX_STEPS_TABLE = {
    "easy":   15,
    "medium": 20,
    "hard":   25,
}


class RewardBreakdown:
    def __init__(self):
        self.issues_fixed      = 0.0
        self.column_fixed      = 0.0
        self.milestone_bonus   = 0.0
        self.destructive_action = 0.0
        self.repeated_action   = 0.0
        self.invalid_action    = 0.0
        self.total             = 0.0

    def compute_total(self):
        self.total = round(
            self.issues_fixed + self.column_fixed + self.milestone_bonus
            + self.destructive_action + self.repeated_action + self.invalid_action,
            4,
        )


class DataCleaningEnvironment:

    def __init__(self):
        self._state:      Optional[DataCleaningState] = None
        self._df:         Optional[pd.DataFrame] = None
        self._task_data:  Optional[Dict] = None
        self._action_counts: Dict[str, int] = {}
        self._cumulative_reward: float = 0.0
        self._prev_score: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(
        self,
        task_name:  str   = "missing_value_imputation",
        seed:       int   = 42,
        difficulty: float = 0.5,
    ) -> DataCleaningObservation:
        if task_name not in TASK_NAMES:
            raise ValueError(f"Unknown task '{task_name}'. Choose from: {TASK_NAMES}")

        difficulty = float(np.clip(difficulty, 0.1, 1.0))

        # Generate unique dataset for this (task, seed, difficulty)
        self._task_data          = get_dataset(task_name, seed=seed, difficulty=difficulty)
        self._df                 = self._task_data["dataframe"].copy()
        self._action_counts      = {}
        self._cumulative_reward  = 0.0
        self._prev_score         = 0.0

        diff_label = self._task_data.get("task_difficulty", "medium")
        max_steps  = MAX_STEPS_TABLE.get(diff_label, 20)

        self._state = DataCleaningState(
            task_name        = task_name,
            task_description = self._task_data.get("description", ""),
            task_difficulty  = diff_label,
            seed             = seed,
            difficulty       = difficulty,
            max_steps        = max_steps,
            step_count       = 0,
            done             = False,
            cumulative_reward = 0.0,
            grader_score     = 0.0,
            action_history   = [],
        )

        return self._build_obs(reward=0.0, last_result="Episode started. Inspect issues and begin cleaning.")

    def step(self, action: DataCleaningAction) -> Tuple[DataCleaningObservation, float, bool, Dict]:
        if self._state is None or self._df is None:
            raise RuntimeError("Call reset() before step().")
        if self._state.done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        self._state.step_count += 1

        # Check max steps
        if self._state.step_count >= self._state.max_steps:
            self._state.done = True
            grader = self._get_grader()
            reward = float(np.clip(grader["score"] * 0.3, -1.0, 1.0))
            self._cumulative_reward += reward
            self._state.cumulative_reward = self._cumulative_reward
            self._state.grader_score = grader["score"]
            obs = self._build_obs(reward=reward, last_result="Max steps reached — episode ended.")
            obs.metadata["grader_result"] = grader
            return obs, reward, True, {}

        # Execute action
        if action.action_type == ActionType.FINISH:
            return self._handle_finish()

        rb, msg = self._execute_action(action)
        rb.compute_total()

        reward = float(np.clip(rb.total, -1.0, 1.0))
        self._cumulative_reward += reward
        self._state.cumulative_reward = round(self._cumulative_reward, 4)

        # Log action
        self._state.action_history.append(
            f"[Step {self._state.step_count}] {action.action_type.value}"
            f"({action.column or 'dataset'}) → {msg[:50]}"
        )

        # Track action for repeat penalty
        key = f"{action.action_type.value}:{action.column}"
        self._action_counts[key] = self._action_counts.get(key, 0) + 1

        done = self._state.done
        obs  = self._build_obs(reward=reward, last_result=msg)
        if done:
            grader = self._get_grader()
            obs.metadata["grader_result"] = grader
        return obs, reward, done, {}

    def state(self) -> DataCleaningState:
        if self._state is None:
            raise RuntimeError("Call reset() before state().")
        return self._state

    def get_grader_result(self) -> Dict:
        if self._state is None or self._df is None:
            raise RuntimeError("Call reset() first.")
        result = grade(
            self._state.task_name,
            self._df,
            self._task_data["ground_truth"],
        )
        # Strictly between 0 and 1
        result["score"] = float(min(max(result["score"], 0.001), 0.999))
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_grader(self) -> Dict:
        return self.get_grader_result()

    def _build_obs(self, reward: float, last_result: str) -> DataCleaningObservation:
        issues  = detect_issues(self._state.task_name, self._df, self._task_data["ground_truth"])
        stats   = get_column_stats(self._df)
        snap    = dataframe_to_records(self._df, limit=10)

        # Progress = fraction of issues resolved vs original
        orig_issues  = len(detect_issues(
            self._state.task_name,
            self._task_data["dataframe"],
            self._task_data["ground_truth"],
        ))
        curr_issues  = len(issues)
        progress_pct = round(1.0 - curr_issues / max(orig_issues, 1), 3) if orig_issues > 0 else 0.5

        return DataCleaningObservation(
            dataset_snapshot   = snap,
            total_rows         = len(self._df),
            total_columns      = len(self._df.columns),
            column_stats       = stats,
            issues_detected    = issues,
            issues_remaining   = len(issues),
            action_history     = self._state.action_history[-10:],
            last_action_result = last_result,
            reward             = reward,
            cumulative_reward  = self._state.cumulative_reward,
            done               = self._state.done,
            step_count         = self._state.step_count,
            task_name          = self._state.task_name,
            task_description   = self._state.task_description,
            task_difficulty    = self._state.task_difficulty,
            progress_pct       = progress_pct,
            metadata           = {},
        )

    def _handle_finish(self) -> Tuple[DataCleaningObservation, float, bool, Dict]:
        self._state.done = True
        grader = self._get_grader()
        score  = grader["score"]

        # Reward = grader score minus cumulative already earned (capped)
        finish_reward = float(np.clip(score * 0.4, 0.05, 0.40))
        self._cumulative_reward += finish_reward
        self._state.cumulative_reward = round(self._cumulative_reward, 4)
        self._state.grader_score = score

        self._state.action_history.append(
            f"[Step {self._state.step_count}] finish() → score={score:.4f}"
        )

        obs = self._build_obs(reward=finish_reward, last_result=f"Episode finished. Score: {score:.4f} — {grader['feedback']}")
        obs.metadata["grader_result"] = grader
        return obs, finish_reward, True, {}

    def _execute_action(self, action: DataCleaningAction) -> Tuple[RewardBreakdown, str]:
        rb = RewardBreakdown()

        # Repeat penalty (escalating)
        key    = f"{action.action_type.value}:{action.column}"
        repeats = self._action_counts.get(key, 0)
        if repeats > 0:
            rb.repeated_action = -0.05 * repeats

        try:
            if action.action_type == ActionType.IMPUTE:
                rb, msg = self._do_impute(action)
            elif action.action_type == ActionType.CAST:
                rb, msg = self._do_cast(action)
            elif action.action_type == ActionType.NORMALIZE:
                rb, msg = self._do_normalize(action)
            elif action.action_type == ActionType.CLIP_OUTLIER:
                rb, msg = self._do_clip(action)
            elif action.action_type == ActionType.FLAG_OUTLIER:
                rb, msg = self._do_flag_outlier(action)
            elif action.action_type == ActionType.DEDUPLICATE:
                rb, msg = self._do_deduplicate(action)
            elif action.action_type == ActionType.DROP_COLUMN:
                rb, msg = self._do_drop_column(action)
            elif action.action_type == ActionType.DROP_ROWS:
                rb, msg = self._do_drop_rows(action)
            elif action.action_type == ActionType.RENAME:
                rb, msg = self._do_rename(action)
            elif action.action_type == ActionType.EXECUTE_CODE:
                rb, msg = self._do_execute_code(action)
            else:
                rb.invalid_action = -0.05
                msg = f"Unknown action type: {action.action_type}"
        except Exception as e:
            rb.invalid_action = -0.05
            msg = f"Action failed: {str(e)[:80]}"

        # Apply repeat penalty
        if repeats > 0:
            rb.repeated_action = -0.05 * repeats

        return rb, msg

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _do_impute(self, action: DataCleaningAction) -> Tuple[RewardBreakdown, str]:
        rb  = RewardBreakdown()
        col = action.column

        if not col or col not in self._df.columns:
            rb.invalid_action = -0.05
            return rb, f"Column '{col}' not found."

        null_before = int(self._df[col].isna().sum())
        if null_before == 0:
            rb.invalid_action = -0.02
            return rb, f"Column '{col}' has no missing values."

        strategy = action.params.get("strategy", "mean")
        value    = action.params.get("value", None)

        try:
            if strategy == "mean" and pd.api.types.is_numeric_dtype(self._df[col]):
                fill_val = self._df[col].mean()
            elif strategy == "median" and pd.api.types.is_numeric_dtype(self._df[col]):
                fill_val = self._df[col].median()
            elif strategy == "mode":
                mode_vals = self._df[col].mode()
                fill_val  = mode_vals.iloc[0] if len(mode_vals) > 0 else None
            elif strategy == "constant" and value is not None:
                fill_val = value
            else:
                rb.invalid_action = -0.05
                return rb, f"Invalid strategy '{strategy}' for column '{col}' (dtype={self._df[col].dtype})."

            if fill_val is None:
                rb.invalid_action = -0.02
                return rb, f"Could not compute fill value for '{col}'."

            self._df[col] = self._df[col].fillna(fill_val)
            null_after = int(self._df[col].isna().sum())
            fixed = null_before - null_after

            rb.issues_fixed = 0.15 if fixed > 0 else 0.0
            rb.column_fixed = 0.05 if null_after == 0 else 0.0
            return rb, f"Imputed '{col}' with {strategy} ({fixed} nulls fixed)."
        except Exception as e:
            rb.invalid_action = -0.05
            return rb, f"Impute failed on '{col}': {e}"

    def _do_cast(self, action: DataCleaningAction) -> Tuple[RewardBreakdown, str]:
        rb    = RewardBreakdown()
        col   = action.column
        dtype = action.params.get("dtype", "float")

        if not col or col not in self._df.columns:
            rb.invalid_action = -0.05
            return rb, f"Column '{col}' not found."

        before_dtype = str(self._df[col].dtype)
        try:
            if dtype in ("float", "float64"):
                self._df[col] = pd.to_numeric(
                    self._df[col].astype(str).str.replace(r"[^\d.\-]", "", regex=True),
                    errors="coerce",
                )
            elif dtype in ("int", "int64"):
                self._df[col] = pd.to_numeric(
                    self._df[col].astype(str).str.replace(r"[^\d\-]", "", regex=True),
                    errors="coerce",
                ).astype("Int64")
            elif dtype in ("date", "datetime"):
                self._df[col] = pd.to_datetime(self._df[col], errors="coerce", infer_datetime_format=True)
            elif dtype == "str":
                self._df[col] = self._df[col].astype(str)
            else:
                rb.invalid_action = -0.05
                return rb, f"Unknown dtype '{dtype}'."

            after_dtype = str(self._df[col].dtype)
            rb.issues_fixed = 0.15 if after_dtype != before_dtype else 0.0
            rb.column_fixed = 0.05
            return rb, f"Cast '{col}' from {before_dtype} to {after_dtype}."
        except Exception as e:
            rb.invalid_action = -0.05
            return rb, f"Cast failed: {e}"

    def _do_normalize(self, action: DataCleaningAction) -> Tuple[RewardBreakdown, str]:
        rb  = RewardBreakdown()
        col = action.column

        if not col or col not in self._df.columns:
            rb.invalid_action = -0.05
            return rb, f"Column '{col}' not found."

        method  = action.params.get("method", "")
        mapping = action.params.get("mapping", {})
        fmt     = action.params.get("format", "")

        try:
            if method == "lowercase":
                self._df[col] = self._df[col].astype(str).str.lower().str.strip()
            elif method == "uppercase":
                self._df[col] = self._df[col].astype(str).str.upper().str.strip()
            elif method == "titlecase":
                self._df[col] = self._df[col].astype(str).str.title().str.strip()
            elif method == "strip":
                self._df[col] = self._df[col].astype(str).str.strip()
            elif mapping:
                self._df[col] = self._df[col].map(mapping).fillna(self._df[col])
            elif fmt:
                self._df[col] = pd.to_datetime(self._df[col], errors="coerce").dt.strftime(fmt)
            else:
                # Auto-normalize: map known variants to canonical form
                for canonical, variants in REGION_VARIANTS.items():
                    mask = self._df[col].isin(variants)
                    self._df.loc[mask, col] = canonical
            rb.issues_fixed = 0.15
            rb.column_fixed = 0.05
            return rb, f"Normalized '{col}'."
        except Exception as e:
            rb.invalid_action = -0.05
            return rb, f"Normalize failed: {e}"

    def _do_clip(self, action: DataCleaningAction) -> Tuple[RewardBreakdown, str]:
        rb  = RewardBreakdown()
        col = action.column

        if not col or col not in self._df.columns:
            rb.invalid_action = -0.05
            return rb, f"Column '{col}' not found."

        if not pd.api.types.is_numeric_dtype(self._df[col]):
            rb.invalid_action = -0.05
            return rb, f"Column '{col}' is not numeric."

        lower = action.params.get("lower", None)
        upper = action.params.get("upper", None)

        if lower is None and upper is None:
            rb.invalid_action = -0.05
            return rb, "clip_outlier requires at least one of lower or upper."

        before_bad = 0
        if lower is not None:
            before_bad += int((self._df[col] < lower).sum())
        if upper is not None:
            before_bad += int((self._df[col] > upper).sum())

        self._df[col] = self._df[col].clip(lower=lower, upper=upper)

        rb.issues_fixed = 0.15 if before_bad > 0 else 0.0
        rb.column_fixed = 0.05
        return rb, f"Clipped '{col}' to [{lower}, {upper}] — {before_bad} values fixed."

    def _do_flag_outlier(self, action: DataCleaningAction) -> Tuple[RewardBreakdown, str]:
        rb  = RewardBreakdown()
        col = action.column

        if not col or col not in self._df.columns:
            rb.invalid_action = -0.05
            return rb, f"Column '{col}' not found."

        flag_col = f"{col}_is_outlier"
        q1, q3   = self._df[col].quantile([0.25, 0.75])
        iqr      = q3 - q1
        self._df[flag_col] = (
            (self._df[col] < q1 - 1.5 * iqr) | (self._df[col] > q3 + 1.5 * iqr)
        )
        n_flagged = int(self._df[flag_col].sum())
        rb.issues_fixed = 0.05
        return rb, f"Flagged {n_flagged} outliers in '{col}' → new column '{flag_col}'."

    def _do_deduplicate(self, action: DataCleaningAction) -> Tuple[RewardBreakdown, str]:
        rb     = RewardBreakdown()
        before = len(self._df)
        subset = action.params.get("subset", None)

        if subset and isinstance(subset, list):
            subset = [c for c in subset if c in self._df.columns] or None

        self._df   = self._df.drop_duplicates(subset=subset).reset_index(drop=True)
        after      = len(self._df)
        removed    = before - after

        if removed == 0:
            rb.invalid_action = -0.02
            return rb, "No duplicates found — no rows removed."

        rb.issues_fixed    = 0.15
        rb.milestone_bonus = 0.10
        return rb, f"Removed {removed} duplicate rows ({before} → {after})."

    def _do_drop_column(self, action: DataCleaningAction) -> Tuple[RewardBreakdown, str]:
        rb  = RewardBreakdown()
        col = action.column

        if not col or col not in self._df.columns:
            rb.invalid_action = -0.05
            return rb, f"Column '{col}' not found."

        self._df.drop(columns=[col], inplace=True)
        rb.issues_fixed = 0.05
        return rb, f"Dropped column '{col}'."

    def _do_drop_rows(self, action: DataCleaningAction) -> Tuple[RewardBreakdown, str]:
        rb        = RewardBreakdown()
        condition = action.params.get("condition", "null")
        col       = action.column
        before    = len(self._df)

        if condition == "null":
            if col and col in self._df.columns:
                self._df = self._df.dropna(subset=[col]).reset_index(drop=True)
            else:
                self._df = self._df.dropna().reset_index(drop=True)
        elif condition == "invalid_range":
            lower = action.params.get("lower", None)
            upper = action.params.get("upper", None)
            if col and col in self._df.columns and pd.api.types.is_numeric_dtype(self._df[col]):
                mask = pd.Series([True] * len(self._df))
                if lower is not None:
                    mask &= self._df[col] >= lower
                if upper is not None:
                    mask &= self._df[col] <= upper
                self._df = self._df[mask].reset_index(drop=True)

        removed = before - len(self._df)
        if removed > before * 0.2:
            rb.destructive_action = -0.10
        rb.issues_fixed = 0.05 if removed > 0 else 0.0
        return rb, f"Dropped {removed} rows."

    def _do_rename(self, action: DataCleaningAction) -> Tuple[RewardBreakdown, str]:
        rb       = RewardBreakdown()
        col      = action.column
        new_name = action.params.get("new_name", None)

        if not col or col not in self._df.columns:
            rb.invalid_action = -0.05
            return rb, f"Column '{col}' not found."
        if not new_name:
            rb.invalid_action = -0.05
            return rb, "rename requires 'new_name' in params."

        self._df.rename(columns={col: new_name}, inplace=True)
        rb.issues_fixed = 0.05
        return rb, f"Renamed '{col}' → '{new_name}'."

    def _do_execute_code(self, action: DataCleaningAction) -> Tuple[RewardBreakdown, str]:
        rb   = RewardBreakdown()
        code = action.params.get("code", "").strip()

        if not code:
            rb.invalid_action = -0.05
            return rb, "params.code is required for execute_code."

        before_df = self._df.copy()
        result_df, message, success = execute_cleaning_code(self._df, code)

        if success:
            reward = score_code_execution(before_df, result_df, success)
            self._df = result_df
            rb.issues_fixed    = max(0.0, reward)
            rb.destructive_action = min(0.0, reward)
        else:
            rb.invalid_action = -0.05

        rb.compute_total()
        return rb, message