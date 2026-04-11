"""
models.py — Data Cleaning Pipeline OpenEnv Environment
=======================================================
Typed Pydantic models for Action, Observation, and State.
These form the contract between the agent and the environment.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    """All cleaning operations an agent can perform."""
    IMPUTE        = "impute"         # Fill missing values
    DROP_COLUMN   = "drop_column"    # Remove an entire column
    DROP_ROWS     = "drop_rows"      # Remove rows matching a condition
    CAST          = "cast"           # Change column data type
    NORMALIZE     = "normalize"      # Standardize format (dates, strings, etc.)
    CLIP_OUTLIER  = "clip_outlier"   # Clip outliers to boundary values
    FLAG_OUTLIER  = "flag_outlier"   # Mark outliers with a boolean column
    DEDUPLICATE   = "deduplicate"    # Remove duplicate rows
    RENAME        = "rename"         # Rename a column
    EXECUTE_CODE  = "execute_code"   # Run Python code directly in sandbox
    FINISH        = "finish"         # Signal episode completion


class ImputeStrategy(str, Enum):
    """Strategies for filling missing values."""
    MEAN        = "mean"
    MEDIAN      = "median"
    MODE        = "mode"
    CONSTANT    = "constant"         # Use a specific fill value
    FORWARD_FILL = "forward_fill"
    DROP        = "drop"             # Drop rows where this column is null


class CastDtype(str, Enum):
    """Target data types for casting."""
    INT     = "int"
    FLOAT   = "float"
    STRING  = "str"
    BOOL    = "bool"
    DATE    = "date"
    DATETIME = "datetime"


class TaskDifficulty(str, Enum):
    EASY   = "easy"
    MEDIUM = "medium"
    HARD   = "hard"


# ---------------------------------------------------------------------------
# Action Model
# ---------------------------------------------------------------------------

class DataCleaningAction(BaseModel):
    """
    An action the agent takes to clean the dataset.

    Examples
    --------
    Impute missing values in 'age' with median:
        DataCleaningAction(
            action_type="impute",
            column="age",
            params={"strategy": "median"}
        )

    Cast 'price' column to float:
        DataCleaningAction(
            action_type="cast",
            column="price",
            params={"dtype": "float"}
        )

    Deduplicate the entire dataset:
        DataCleaningAction(
            action_type="deduplicate",
            column=None,
            params={}
        )
    """
    action_type : ActionType            = Field(...,  description="The cleaning operation to perform")
    column      : Optional[str]         = Field(None, description="Target column name (None for row-level ops)")
    params      : Dict[str, Any]        = Field(default_factory=dict, description="Operation-specific parameters")

    # Convenience helpers so the environment can validate params easily
    def get_strategy(self) -> Optional[str]:
        return self.params.get("strategy")

    def get_fill_value(self) -> Optional[Any]:
        return self.params.get("fill_value")

    def get_dtype(self) -> Optional[str]:
        return self.params.get("dtype")

    def get_format(self) -> Optional[str]:
        return self.params.get("format")


# ---------------------------------------------------------------------------
# Observation Models
# ---------------------------------------------------------------------------

class ColumnStats(BaseModel):
    """Statistics for a single column — gives the agent a full picture."""
    name          : str
    dtype         : str
    null_count    : int
    null_pct      : float                   = Field(..., ge=0.0, le=1.0)
    unique_count  : int
    sample_values : List[Any]               = Field(default_factory=list)
    min_value     : Optional[Any]           = None
    max_value     : Optional[Any]           = None
    mean_value    : Optional[float]         = None
    has_outliers  : bool                    = False
    outlier_count : int                     = 0


class IssueHint(BaseModel):
    """A structured hint about a detected data issue."""
    issue_type  : str           # "missing_values" | "wrong_dtype" | "outlier" | "duplicate" | "format"
    column      : Optional[str] = None
    severity    : str           = "medium"   # "low" | "medium" | "high"
    description : str           = ""
    count       : int           = 0


class DataCleaningObservation(BaseModel):
    """
    Everything the agent sees after each step.
    Includes current data state, statistics, hints, and feedback.
    """
    # Data snapshot (first N rows of current dataset)
    dataset_snapshot  : List[Dict[str, Any]]  = Field(default_factory=list,
                            description="Sample rows from the current dataset (up to 10)")
    total_rows        : int                   = Field(0,   description="Total rows in dataset")
    total_columns     : int                   = Field(0,   description="Total columns in dataset")

    # Per-column statistics
    column_stats      : List[ColumnStats]     = Field(default_factory=list,
                            description="Statistics for each column")

    # Issue hints — helps the agent know what's left to fix
    issues_detected   : List[IssueHint]      = Field(default_factory=list,
                            description="Remaining data quality issues detected")
    issues_remaining  : int                  = Field(0,
                            description="Total count of remaining issues")

    # Episode feedback
    action_history    : List[str]            = Field(default_factory=list,
                            description="Human-readable log of actions taken so far")
    last_action_result: str                  = Field("",
                            description="Result message from the last action")
    reward            : float                = Field(0.0, description="Reward from last action")
    cumulative_reward : float                = Field(0.0, description="Total reward this episode")
    done              : bool                 = Field(False)
    step_count        : int                  = Field(0)

    # Task context
    task_name         : str                  = Field("", description="Current task identifier")
    task_description  : str                  = Field("", description="What the agent must accomplish")
    task_difficulty   : str                  = Field("easy")
    progress_pct      : float                = Field(0.0, ge=0.0, le=1.0,
                            description="Estimated % of issues resolved so far")

    # Metadata passthrough
    metadata          : Dict[str, Any]       = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# State Model
# ---------------------------------------------------------------------------

class DataCleaningState(BaseModel):
    """
    Full episode state — returned by state() endpoint.
    """
    task_name         : str                  = "missing_value_imputation"
    task_description  : str                  = ""
    task_difficulty   : str                  = "medium"
    seed              : int                  = 42
    difficulty        : float                = 0.5
    step_count        : int                  = 0
    max_steps         : int                  = 20
    cumulative_reward : float                = 0.0
    done              : bool                 = False
    grader_score      : float                = 0.0
    action_history    : List[str]            = Field(default_factory=list)
    metadata          : Dict[str, Any]       = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Reward Model
# ---------------------------------------------------------------------------

class RewardBreakdown(BaseModel):
    """
    Detailed reward breakdown — useful for debugging and agent training signal.
    """
    total             : float = 0.0

    # Positive signals
    issues_fixed      : float = 0.0    # +0.2 per issue correctly resolved
    column_fixed      : float = 0.0    # +0.1 per column fully cleaned
    milestone_bonus   : float = 0.0    # +0.3 for completing task objective

    # Negative signals
    destructive_action: float = 0.0    # -0.1 for dropping valid data
    repeated_action   : float = 0.0    # -0.2 for repeating same action
    invalid_action    : float = 0.0    # -0.05 for invalid column/params

    description       : str   = ""


# ---------------------------------------------------------------------------
# Task Descriptor (returned by /tasks endpoint)
# ---------------------------------------------------------------------------

class TaskInfo(BaseModel):
    """Describes a task — returned by GET /tasks."""
    task_name         : str
    difficulty        : TaskDifficulty
    description       : str
    objective         : str
    max_steps         : int
    action_schema     : Dict[str, Any]      # JSON schema of DataCleaningAction
    example_action    : Dict[str, Any]      # A concrete example action
    scoring_criteria  : List[str]           # How the grader scores this task


# ---------------------------------------------------------------------------
# API Response Models
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    """Response returned by POST /step."""
    observation : DataCleaningObservation
    reward      : float
    done        : bool
    info        : Dict[str, Any] = Field(default_factory=dict)


class GraderResult(BaseModel):
    """Response returned by POST /grader."""
    task_name   : str
    score       : float = Field(..., ge=0.0, le=1.0)
    breakdown   : Dict[str, float] = Field(default_factory=dict)
    passed      : bool
    feedback    : str


class BaselineResult(BaseModel):
    """Response returned by POST /baseline."""
    model_name  : str
    results     : Dict[str, float]    # task_name → score
    mean_score  : float
    timestamp   : str


# ---------------------------------------------------------------------------
# Action Schema (for /tasks endpoint)
# ---------------------------------------------------------------------------

ACTION_SCHEMA = {
    "type": "object",
    "required": ["action_type"],
    "properties": {
        "action_type": {
            "type": "string",
            "enum": [e.value for e in ActionType],
            "description": "The cleaning operation to perform"
        },
        "column": {
            "type": ["string", "null"],
            "description": "Target column name. Null for dataset-wide ops (deduplicate, finish)"
        },
        "params": {
            "type": "object",
            "description": "Operation-specific parameters",
            "examples": [
                {"strategy": "median"},
                {"dtype": "float"},
                {"format": "%Y-%m-%d"},
                {"fill_value": 0},
                {"lower": 0.05, "upper": 0.95}
            ]
        }
    }
}

EXAMPLE_ACTIONS = {
    ActionType.IMPUTE:        {"action_type": "impute",       "column": "age",    "params": {"strategy": "median"}},
    ActionType.CAST:          {"action_type": "cast",         "column": "price",  "params": {"dtype": "float"}},
    ActionType.NORMALIZE:     {"action_type": "normalize",    "column": "date",   "params": {"format": "%Y-%m-%d"}},
    ActionType.CLIP_OUTLIER:  {"action_type": "clip_outlier", "column": "salary", "params": {"lower": 0.05, "upper": 0.95}},
    ActionType.DEDUPLICATE:   {"action_type": "deduplicate",  "column": None,     "params": {}},
    ActionType.EXECUTE_CODE:  {"action_type": "execute_code", "column": None,     "params": {"code": "df['age'] = df['age'].fillna(df['age'].median())"}},
    ActionType.FINISH:        {"action_type": "finish",       "column": None,     "params": {}},
}