"""
environment.py — Core RL environment for Data Cleaning Pipeline
Connects dataset generation, action execution, reward shaping, and grading.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from dataset_generator import (
    get_dataset,
    get_column_stats,
    dataframe_to_records,
    detect_issues,
)
from graders import grade
from code_sandbox import execute_cleaning_code, score_code_execution
from models import (
    ActionType,
    DataCleaningAction,
    DataCleaningObservation,
    DataCleaningState,
    RewardBreakdown,
    TaskDifficulty,
    ColumnStats,
    IssueHint,
)

NULL_VARIANTS = {"n/a", "none", "-", "", "null", "na", "nan"}

TASK_NAMES = [
    "missing_value_imputation",
    "type_errors_and_outliers",
    "schema_normalization_dedup",
]


class DataCleaningEnvironment:
    """
    OpenEnv-compliant environment for data cleaning tasks.

    Episode lifecycle:
        reset(task_name, seed) → observation
        step(action)           → observation, reward, done, info
        state()                → DataCleaningState
    """

    def __init__(self) -> None:
        self._df: Optional[pd.DataFrame] = None
        self._task_data: Optional[Dict] = None
        self._state: Optional[DataCleaningState] = None
        self._action_history: list[str] = []
        self._last_action_counts: Dict[str, int] = {}
        self._cumulative_reward: float = 0.0
        self._initial_issue_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(
        self,
        task_name: str = "missing_value_imputation",
        seed: int = 42,
    ) -> DataCleaningObservation:
        if task_name not in TASK_NAMES:
            raise ValueError(f"Unknown task '{task_name}'. Valid: {TASK_NAMES}")

        self._task_data = get_dataset(task_name, seed=seed)
        self._df = self._task_data["dataframe"].copy()
        self._action_history = []
        self._last_action_counts = {}
        self._cumulative_reward = 0.0

        issues = self._task_data["issues"]
        self._initial_issue_count = sum(i["count"] for i in issues)

        self._state = DataCleaningState(
            episode_id=str(uuid.uuid4())[:8],
            task_name=task_name,
            task_difficulty=TaskDifficulty(self._task_data["difficulty"]),
            step_count=0,
            max_steps=self._task_data["max_steps"],
            cumulative_reward=0.0,
            done=False,
            initial_issue_count=self._initial_issue_count,
            resolved_issue_count=0,
            remaining_issue_count=self._initial_issue_count,
        )

        return self._build_observation(
            reward=0.0,
            last_action_result="Episode started. Inspect the dataset and begin cleaning.",
        )

    def step(self, action: DataCleaningAction) -> Tuple[DataCleaningObservation, float, bool, Dict]:
        if self._state is None or self._df is None:
            raise RuntimeError("Call reset() before step().")
        if self._state.done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        self._state.step_count += 1
        reward_breakdown = RewardBreakdown()
        result_msg = ""

        # --- Execute action ---
        if action.action_type == ActionType.FINISH:
            reward_breakdown, result_msg = self._handle_finish()
        else:
            reward_breakdown, result_msg = self._execute_action(action)

        # --- Repetition penalty ---
        action_key = f"{action.action_type}:{action.column}"
        self._last_action_counts[action_key] = self._last_action_counts.get(action_key, 0) + 1
        if self._last_action_counts[action_key] > 1:
            penalty = -0.1 * min(self._last_action_counts[action_key] - 1, 3)
            reward_breakdown.repeated_action += penalty
            reward_breakdown.total += penalty
            result_msg += f" [repeated action penalty: {penalty:.2f}]"

        # --- Max steps check ---
        done = self._state.done or (self._state.step_count >= self._state.max_steps)
        if done and not self._state.done:
            self._state.done = True
            result_msg += " [max steps reached]"

        # --- Update cumulative reward ---
        reward = round(float(np.clip(reward_breakdown.total, -1.0, 1.0)), 4)
        self._cumulative_reward += reward
        self._state.cumulative_reward = round(self._cumulative_reward, 4)

        # --- Update issue tracking ---
        remaining = detect_issues(self._df, self._state.task_name, self._task_data["ground_truth"])
        remaining_count = sum(i["count"] for i in remaining)
        self._state.remaining_issue_count = remaining_count
        self._state.resolved_issue_count = max(0, self._initial_issue_count - remaining_count)

        # --- Log action ---
        self._action_history.append(
            f"[Step {self._state.step_count}] {str(action.action_type)}"
            + (f"({action.column})" if action.column else "")
            + f" → {result_msg[:60]}"
        )

        obs = self._build_observation(reward=reward, last_action_result=result_msg)

        # --- Final grader on done ---
        if self._state.done:
            grader_result = grade(
                self._state.task_name,
                self._df,
                self._task_data["ground_truth"],
            )
            self._state.grader_score = grader_result["score"]
            obs.metadata["grader_result"] = grader_result

        return obs, reward, self._state.done, {"breakdown": reward_breakdown.dict()}

    def state(self) -> DataCleaningState:
        if self._state is None:
            raise RuntimeError("Call reset() first.")
        return self._state

    def get_grader_result(self) -> Dict:
        if self._state is None or self._df is None:
            raise RuntimeError("Call reset() first.")
        return grade(self._state.task_name, self._df, self._task_data["ground_truth"])

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _execute_action(self, action: DataCleaningAction) -> Tuple[RewardBreakdown, str]:
        rb = RewardBreakdown()

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
                msg = f"Unknown action type: {action.action_type}"
                rb.invalid_action = -0.05
        except Exception as e:
            msg = f"Action failed: {str(e)[:80]}"
            rb.invalid_action = -0.05

        rb.total = round(
            rb.issues_fixed + rb.column_fixed + rb.milestone_bonus
            + rb.destructive_action + rb.repeated_action + rb.invalid_action,
            4,
        )
        return rb, msg

    def _do_impute(self, action: DataCleaningAction) -> Tuple[RewardBreakdown, str]:
        rb = RewardBreakdown()
        col = action.column
        if col not in self._df.columns:
            rb.invalid_action = -0.05
            return rb, f"Column '{col}' not found."

        null_before = self._df[col].isna().sum()
        if null_before == 0:
            rb.invalid_action = -0.05
            return rb, f"'{col}' has no missing values."

        strategy = action.get_strategy() or "mean"
        fill_val  = action.get_fill_value()
        series    = self._df[col]

        if strategy == "mean":
            val = series.mean()
        elif strategy == "median":
            val = series.median()
        elif strategy == "mode":
            try:
                # Avoid numpy bool arithmetic issues by working with string repr
                str_series = series.dropna().astype(str)
                if len(str_series) == 0:
                    val = None
                else:
                    raw = str_series.value_counts().idxmax()
                    if raw == "True":
                        val = True
                    elif raw == "False":
                        val = False
                    else:
                        val = raw
            except Exception:
                val = None
        elif strategy == "constant":
            val = fill_val
        elif strategy == "forward_fill":
            self._df[col] = series.ffill()
            val = None
        elif strategy == "drop":
            before = len(self._df)
            self._df.dropna(subset=[col], inplace=True)
            self._df.reset_index(drop=True, inplace=True)
            dropped = before - len(self._df)
            rb.issues_fixed   = 0.10
            rb.destructive_action = -0.05 if dropped > before * 0.1 else 0.0
            return rb, f"Dropped {dropped} rows with null '{col}'."
        else:
            rb.invalid_action = -0.05
            return rb, f"Unknown strategy '{strategy}'."

        if val is not None:
            self._df[col] = series.fillna(val)

        null_after = self._df[col].isna().sum()
        if null_after == 0:
            # Check if strategy matches ground truth
            gt = self._task_data.get("ground_truth", {}).get(col, {})
            gt_strategy = gt.get("strategy", "")
            rb.issues_fixed = 0.15 if strategy == gt_strategy else 0.08
            rb.column_fixed = 0.05
            return rb, f"Imputed '{col}' with {strategy} → {null_before} nulls fixed."
        else:
            rb.issues_fixed = 0.05
            return rb, f"Partially imputed '{col}': {null_after} nulls remain."

    def _do_cast(self, action: DataCleaningAction) -> Tuple[RewardBreakdown, str]:
        rb = RewardBreakdown()
        col   = action.column
        dtype = action.get_dtype()
        if col not in self._df.columns:
            rb.invalid_action = -0.05
            return rb, f"Column '{col}' not found."
        if not dtype:
            rb.invalid_action = -0.05
            return rb, "params.dtype is required for cast."

        try:
            if dtype == "float":
                cleaned = (
                    self._df[col].astype(str)
                    .str.replace(r"[^\d.\-]", "", regex=True)
                    .replace("", np.nan)
                )
                self._df[col] = pd.to_numeric(cleaned, errors="coerce")
            elif dtype == "int":
                cleaned = (
                    self._df[col].astype(str)
                    .str.replace(r"[^\d\-]", "", regex=True)
                    .replace("", np.nan)
                )
                self._df[col] = pd.to_numeric(cleaned, errors="coerce").astype("Int64")
            elif dtype in ("date", "datetime"):
                self._df[col] = pd.to_datetime(self._df[col], errors="coerce")
            elif dtype == "str":
                self._df[col] = self._df[col].astype(str)
            elif dtype == "bool":
                self._df[col] = self._df[col].astype(bool)
            else:
                rb.invalid_action = -0.05
                return rb, f"Unsupported dtype '{dtype}'."

            rb.issues_fixed = 0.15
            rb.column_fixed = 0.05
            return rb, f"Cast '{col}' to {dtype} successfully."
        except Exception as e:
            rb.invalid_action = -0.05
            return rb, f"Cast failed: {e}"

    def _do_normalize(self, action: DataCleaningAction) -> Tuple[RewardBreakdown, str]:
        rb = RewardBreakdown()
        col    = action.column
        fmt    = action.get_format()
        params = action.params

        if col not in self._df.columns:
            rb.invalid_action = -0.05
            return rb, f"Column '{col}' not found."

        method = params.get("method", "")

        if fmt:
            # Date normalization
            try:
                self._df[col] = pd.to_datetime(self._df[col], errors="coerce").dt.strftime(fmt)
                rb.issues_fixed = 0.10
                rb.column_fixed = 0.05
                return rb, f"Normalized '{col}' to format {fmt}."
            except Exception as e:
                rb.invalid_action = -0.05
                return rb, f"Normalization failed: {e}"

        elif method == "lowercase":
            self._df[col] = self._df[col].astype(str).str.lower().str.strip()
            rb.issues_fixed = 0.10
            return rb, f"Lowercased '{col}'."

        elif method == "uppercase":
            self._df[col] = self._df[col].astype(str).str.upper().str.strip()
            rb.issues_fixed = 0.05
            return rb, f"Uppercased '{col}'."

        elif "mapping" in params:
            mapping = params["mapping"]
            self._df[col] = (
                self._df[col].astype(str).str.lower().str.strip().map(mapping)
                .fillna(self._df[col])
            )
            rb.issues_fixed = 0.15
            rb.column_fixed = 0.05
            return rb, f"Applied value mapping to '{col}'."

        elif method == "null_variants":
            before = self._df[col].isna().sum()
            self._df[col] = self._df[col].replace(list(NULL_VARIANTS), np.nan)
            str_mask = self._df[col].astype(str).str.lower().str.strip().isin(NULL_VARIANTS)
            self._df.loc[str_mask, col] = np.nan
            fixed = int(self._df[col].isna().sum() - before)
            rb.issues_fixed = 0.10
            return rb, f"Replaced NULL variants in '{col}': {fixed} values standardized."

        else:
            rb.invalid_action = -0.05
            return rb, "Normalize requires 'format', 'method', or 'mapping' in params."

    def _do_clip(self, action: DataCleaningAction) -> Tuple[RewardBreakdown, str]:
        rb  = RewardBreakdown()
        col = action.column
        if col not in self._df.columns:
            rb.invalid_action = -0.05
            return rb, f"Column '{col}' not found."
        if not pd.api.types.is_numeric_dtype(self._df[col]):
            rb.invalid_action = -0.05
            return rb, f"'{col}' is not numeric — cast first."

        lower = action.params.get("lower")
        upper = action.params.get("upper")

        if lower is None and upper is None:
            # Auto IQR clipping
            q1, q3 = self._df[col].quantile(0.25), self._df[col].quantile(0.75)
            iqr    = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr

        before_outliers = (
            (self._df[col] < lower if lower is not None else pd.Series([False] * len(self._df)))
            | (self._df[col] > upper if upper is not None else pd.Series([False] * len(self._df)))
        ).sum()

        self._df[col] = self._df[col].clip(lower=lower, upper=upper)
        rb.issues_fixed = 0.15 if before_outliers > 0 else 0.0
        rb.column_fixed = 0.05 if before_outliers > 0 else 0.0
        return rb, f"Clipped '{col}' to [{lower}, {upper}]: {before_outliers} outliers fixed."

    def _do_flag_outlier(self, action: DataCleaningAction) -> Tuple[RewardBreakdown, str]:
        rb  = RewardBreakdown()
        col = action.column
        if col not in self._df.columns:
            rb.invalid_action = -0.05
            return rb, f"Column '{col}' not found."

        q1, q3 = self._df[col].quantile(0.25), self._df[col].quantile(0.75)
        iqr    = q3 - q1
        flag_col = f"{col}_is_outlier"
        self._df[flag_col] = (
            (self._df[col] < q1 - 1.5 * iqr) | (self._df[col] > q3 + 1.5 * iqr)
        )
        count = int(self._df[flag_col].sum())
        rb.issues_fixed = 0.08
        return rb, f"Flagged {count} outliers in '{col}' → new column '{flag_col}'."

    def _do_deduplicate(self, action: DataCleaningAction) -> Tuple[RewardBreakdown, str]:
        rb     = RewardBreakdown()
        before = len(self._df)
        subset = action.params.get("subset")  # optional list of columns
        self._df.drop_duplicates(subset=subset, inplace=True)
        self._df.reset_index(drop=True, inplace=True)
        removed = before - len(self._df)
        if removed == 0:
            return rb, "No exact duplicates found."
        rb.issues_fixed   = min(0.20, 0.01 * removed)
        rb.milestone_bonus = 0.10 if removed >= 10 else 0.0
        return rb, f"Removed {removed} duplicate rows. {len(self._df)} rows remain."

    def _do_drop_column(self, action: DataCleaningAction) -> Tuple[RewardBreakdown, str]:
        rb  = RewardBreakdown()
        col = action.column
        if col not in self._df.columns:
            rb.invalid_action = -0.05
            return rb, f"Column '{col}' not found."
        self._df.drop(columns=[col], inplace=True)
        rb.destructive_action = -0.1
        return rb, f"Dropped column '{col}'. Use carefully — may lose valid data."

    def _do_drop_rows(self, action: DataCleaningAction) -> Tuple[RewardBreakdown, str]:
        rb     = RewardBreakdown()
        col    = action.column
        before = len(self._df)
        condition = action.params.get("condition")

        if col and col in self._df.columns:
            if condition == "null":
                self._df.dropna(subset=[col], inplace=True)
            elif condition == "invalid_range":
                lower = action.params.get("lower")
                upper = action.params.get("upper")
                if lower is not None:
                    self._df = self._df[self._df[col] >= lower]
                if upper is not None:
                    self._df = self._df[self._df[col] <= upper]
            else:
                rb.invalid_action = -0.05
                return rb, "params.condition must be 'null' or 'invalid_range'."
        else:
            rb.invalid_action = -0.05
            return rb, "column and params.condition required for drop_rows."

        self._df.reset_index(drop=True, inplace=True)
        removed = before - len(self._df)
        # Penalise heavy row dropping
        pct_dropped = removed / max(before, 1)
        rb.destructive_action = -0.15 * pct_dropped if pct_dropped > 0.15 else 0.0
        rb.issues_fixed = 0.08 if removed > 0 else 0.0
        return rb, f"Dropped {removed} rows ({pct_dropped:.1%} of dataset)."

    def _do_rename(self, action: DataCleaningAction) -> Tuple[RewardBreakdown, str]:
        rb      = RewardBreakdown()
        col     = action.column
        new_name = action.params.get("new_name")
        if not new_name or col not in self._df.columns:
            rb.invalid_action = -0.05
            return rb, "Rename requires valid column and params.new_name."
        self._df.rename(columns={col: new_name}, inplace=True)
        return rb, f"Renamed '{col}' → '{new_name}'."

    def _do_execute_code(self, action: DataCleaningAction) -> Tuple[RewardBreakdown, str]:
        """
        Execute agent-provided Python cleaning code in a sandbox.
        The agent gets `df` (the current dataframe) and `pd`/`np` in scope.
        """
        rb   = RewardBreakdown()
        code = action.params.get("code", "").strip()

        if not code:
            rb.invalid_action = -0.05
            return rb, "params.code is required for execute_code action."

        before_df = self._df.copy()
        result_df, message, success = execute_cleaning_code(self._df, code)

        if success:
            # Compute reward based on what actually changed
            reward = score_code_execution(before_df, result_df, success)
            self._df = result_df  # apply changes
            rb.issues_fixed = max(0.0, reward)
            rb.destructive_action = min(0.0, reward)
        else:
            rb.invalid_action = -0.05

        rb.total = round(
            rb.issues_fixed + rb.column_fixed + rb.milestone_bonus
            + rb.destructive_action + rb.repeated_action + rb.invalid_action,
            4,
        )
        return rb, message

    def _handle_finish(self) -> Tuple[RewardBreakdown, str]:
        rb = RewardBreakdown()
        self._state.done = True
        result = grade(self._state.task_name, self._df, self._task_data["ground_truth"])
        score  = result["score"]

        # Completion bonus proportional to grader score
        rb.milestone_bonus = round(score * 0.3, 4)
        rb.total = rb.milestone_bonus
        self._state.grader_score = score
        return rb, f"Episode finished. Grader score: {score:.4f}. {result['feedback']}"

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _build_observation(self, reward: float, last_action_result: str) -> DataCleaningObservation:
        remaining_issues = detect_issues(
            self._df, self._state.task_name, self._task_data["ground_truth"]
        )
        initial = max(self._initial_issue_count, 1)
        remaining_count = sum(i["count"] for i in remaining_issues)
        progress = round(1.0 - min(remaining_count / initial, 1.0), 4)

        col_stats = [ColumnStats(**s) for s in get_column_stats(self._df)]
        hints     = [IssueHint(**i)   for i in remaining_issues]

        return DataCleaningObservation(
            dataset_snapshot   = dataframe_to_records(self._df, max_rows=10),
            total_rows         = len(self._df),
            total_columns      = len(self._df.columns),
            column_stats       = col_stats,
            issues_detected    = hints,
            issues_remaining   = len(remaining_issues),
            action_history     = self._action_history[-10:],
            last_action_result = last_action_result,
            reward             = reward,
            cumulative_reward  = round(self._cumulative_reward, 4),
            done               = self._state.done,
            step_count         = self._state.step_count,
            task_name          = self._state.task_name,
            task_description   = self._task_data["description"],
            task_difficulty    = self._state.task_difficulty.value,
            progress_pct       = progress,
            metadata           = {"episode_id": self._state.episode_id},
        )


# ------------------------------------------------------------------
# Smoke test
# ------------------------------------------------------------------

if __name__ == "__main__":
    env = DataCleaningEnvironment()

    print("=" * 55)
    print("  Environment Smoke Test")
    print("=" * 55)

    for task in TASK_NAMES:
        print(f"\n📋 Task: {task}")
        obs = env.reset(task_name=task, seed=42)
        print(f"   Rows: {obs.total_rows} | Columns: {obs.total_columns}")
        print(f"   Issues: {obs.issues_remaining} | Progress: {obs.progress_pct:.0%}")

        # Run sample actions per task
        if task == "missing_value_imputation":
            actions = [
                DataCleaningAction(action_type="impute", column="age",          params={"strategy": "median"}),
                DataCleaningAction(action_type="impute", column="fare",         params={"strategy": "mean"}),
                DataCleaningAction(action_type="finish", column=None,           params={}),
            ]
        elif task == "type_errors_and_outliers":
            actions = [
                DataCleaningAction(action_type="cast",         column="unit_price",   params={"dtype": "float"}),
                DataCleaningAction(action_type="clip_outlier", column="discount_pct", params={"lower": 0, "upper": 100}),
                DataCleaningAction(action_type="finish",       column=None,           params={}),
            ]
        else:
            actions = [
                DataCleaningAction(action_type="deduplicate", column=None,     params={}),
                DataCleaningAction(action_type="normalize",   column="status", params={"method": "lowercase"}),
                DataCleaningAction(action_type="finish",      column=None,     params={}),
            ]

        for action in actions:
            obs, reward, done, info = env.step(action)
            print(f"   → {str(action.action_type):15s} reward={reward:+.4f}  progress={obs.progress_pct:.0%}  done={done}")
            if done:
                print(f"   ✅ Grader score: {env.state().grader_score:.4f}")
                break

    print("\n✅ Environment working correctly!")