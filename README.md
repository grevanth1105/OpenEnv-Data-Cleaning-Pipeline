# Data Cleaning Pipeline — OpenEnv

An RL environment where AI agents learn to clean real-world datasets.
Agents interact through the standard OpenEnv `reset()` / `step()` / `state()` API,
receiving dense reward signals at every cleaning action — not just at episode end.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![HF Space](https://img.shields.io/badge/HuggingFace-Space-yellow)](https://huggingface.co/spaces)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Why data cleaning?

Data cleaning is one of the most time-consuming tasks in any data-driven organisation —
estimates put it at 60–80% of a data scientist's time. It is rule-rich, context-dependent,
and hard to automate with simple heuristics. This makes it an ideal environment for training
and evaluating LLM-based agents: the task is real, the feedback is measurable, and the
difficulty scales naturally from trivial to frontier-model-challenging.

---

## Tasks

### Task 1 — Missing Value Imputation `easy`

**Dataset:** HR employee records (120 rows, 7 columns)

**Issues injected:**
- `age` — 20% missing → correct strategy: median
- `salary` — 15% missing → correct strategy: mean
- `department` — 10% missing → correct strategy: mode
- `years_exp` — 25% missing → correct strategy: median
- `is_manager` — 5% missing → correct strategy: mode

**Scoring:** Each correctly imputed column scores 0.20. Strategy match required for full credit.

**Expected scores:** random agent ~0.0 | good agent ~1.0

---

### Task 2 — Type Errors + Outlier Detection `medium`

**Dataset:** E-commerce orders (150 rows, 8 columns)

**Issues injected:**
- `price` stored as `"$5.99"` string → cast to float
- `quantity` stored as string → cast to int
- `order_date` mixed formats (`%Y-%m-%d`, `%d/%m/%Y`, `%m-%d-%Y`) → normalize
- `rating` stored as `"4.5 stars"` or `"N/A"` → cast to float, clip 0–5
- `discount_pct` has values > 100% and < 0% → clip to [0, 100]
- `weight_kg` has extreme outliers (>500kg) → clip with IQR

**Scoring:** Type fixes 0.60, outlier handling 0.30, row preservation 0.10.

**Expected scores:** random agent ~0.46 | good agent ~0.93

---

### Task 3 — Schema Normalization + Deduplication `hard`

**Dataset:** CRM customer records (225 rows, 9 columns — 25 rows are duplicates)

**Issues injected:**
- 15 exact duplicate rows + 10 near-duplicate rows
- `region` — 5 inconsistent variants (`North`, `north`, `N`, `Nth`, `NORTH`)
- `country` — 5 inconsistent variants (`USA`, `US`, `United States`, `U.S.A`, `united states`)
- `status` — mixed case (`active`, `ACTIVE`, `Active`)
- `age` — negative values and values > 120
- `annual_revenue` — negative values
- `email`, `phone`, `region` — NULL variants (`N/A`, `none`, `-`, `""`, `NULL`)

**Scoring:** Deduplication 0.30, format normalization 0.30, schema fixes 0.20, null handling 0.20.

**Expected scores:** random agent ~0.34 | good agent ~0.81

---

## Action space

Every action is a JSON object with three fields:

```json
{
  "action_type": "<type>",
  "column": "<column_name_or_null>",
  "params": {}
}
```

| `action_type` | `column` | `params` | Effect |
|---|---|---|---|
| `impute` | required | `strategy`: mean/median/mode/constant/forward_fill/drop | Fill missing values |
| `cast` | required | `dtype`: int/float/str/date/datetime | Change column type |
| `normalize` | required | `format`, `method`, or `mapping` | Standardise formats |
| `clip_outlier` | required | `lower`, `upper` (optional) | Clip to bounds |
| `flag_outlier` | required | — | Add `<col>_is_outlier` boolean column |
| `deduplicate` | optional | `subset`: list of columns | Remove duplicate rows |
| `drop_column` | required | — | Remove a column |
| `drop_rows` | required | `condition`: null/invalid_range | Remove matching rows |
| `rename` | required | `new_name` | Rename a column |
| `finish` | — | — | End episode, trigger final grader |

---

## Observation space

Each `step()` returns a typed `DataCleaningObservation`:

| Field | Type | Description |
|---|---|---|
| `dataset_snapshot` | `list[dict]` | First 10 rows of current dataset |
| `total_rows` | `int` | Current row count |
| `total_columns` | `int` | Current column count |
| `column_stats` | `list[ColumnStats]` | Per-column: dtype, null_count, null_pct, min, max, mean, outlier_count |
| `issues_detected` | `list[IssueHint]` | Remaining issues with type, severity, count |
| `issues_remaining` | `int` | Total issue count remaining |
| `action_history` | `list[str]` | Last 10 actions taken |
| `last_action_result` | `str` | Feedback from the last action |
| `reward` | `float` | Reward from last action |
| `cumulative_reward` | `float` | Total reward this episode |
| `done` | `bool` | Episode complete flag |
| `step_count` | `int` | Steps taken so far |
| `task_name` | `str` | Current task identifier |
| `task_description` | `str` | Natural language task description |
| `task_difficulty` | `str` | easy / medium / hard |
| `progress_pct` | `float` | Estimated fraction of issues resolved |

---

## Reward function

Rewards are dense — the agent receives signal at every step:

| Signal | Value | When |
|---|---|---|
| Issue fixed | `+0.15` | A data quality issue is correctly resolved |
| Column cleaned | `+0.05` | An entire column is fully cleaned |
| Milestone bonus | `+0.10–0.30` | Key milestones (dedup complete, finish with high score) |
| Destructive action | `-0.05 to -0.15` | Dropping valid data unnecessarily |
| Repeated action | `-0.10` per repeat | Same action on same column more than once |
| Invalid action | `-0.05` | Non-existent column, bad params |

Total episode reward range: `[-1.0, 1.0]` per step.

---

## Setup

### Local — Uvicorn

```bash
git clone https://huggingface.co/spaces/revanth11/data-cleaning-env
cd data-cleaning-env
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

### Local — Docker

```bash
docker build -t data-cleaning-env -f server/Dockerfile .
docker run -d -p 8000:7860 \
    -e WORKERS=4 \
    -e MAX_CONCURRENT_ENVS=100 \
    data-cleaning-env

curl http://localhost:8000/health
# {"status": "healthy", "sessions": 0}
```

### HF Spaces

```bash
# Pull directly from registry
docker pull registry.hf.space/revanthk11-data-cleaning-env:latest
docker run -d -p 8000:7860 registry.hf.space/revanthk11-data-cleaning-env:latest
```

---

## Usage

### Python client — sync

```python
from client import DataCleaningEnv
from models import DataCleaningAction

with DataCleaningEnv("http://localhost:8000").sync() as env:
    # List tasks
    tasks = env.tasks()

    # Start episode
    obs = env.reset(task_name="missing_value_imputation", seed=42)
    print(f"Rows: {obs.total_rows} | Issues: {obs.issues_remaining}")

    # Take a step
    action = DataCleaningAction(
        action_type="impute",
        column="age",
        params={"strategy": "median"},
    )
    obs, reward, done, info = env.step(action)
    print(f"Reward: {reward} | Progress: {obs.progress_pct:.0%}")

    # Get grader score at any point
    result = env.grader()
    print(f"Score: {result.score:.4f}")
```

### Python client — async

```python
import asyncio
from client import DataCleaningEnv
from models import DataCleaningAction

async def main():
    async with DataCleaningEnv("http://localhost:8000") as env:
        obs = await env.reset(task_name="type_errors_and_outliers")
        action = DataCleaningAction(
            action_type="cast", column="price", params={"dtype": "float"}
        )
        obs, reward, done, info = await env.step(action)

asyncio.run(main())
```

### Direct HTTP

```bash
# Reset
curl -X POST http://localhost:8000/reset \
     -H "Content-Type: application/json" \
     -d '{"task_name": "missing_value_imputation", "seed": 42}'

# Step
curl -X POST http://localhost:8000/step \
     -H "Content-Type: application/json" \
     -d '{"action": {"action_type": "impute", "column": "age", "params": {"strategy": "median"}}}'

# List tasks + action schema
curl http://localhost:8000/tasks

# Grader score
curl -X POST http://localhost:8000/grader

# Heuristic baseline scores
curl -X POST http://localhost:8000/baseline
```

---

## Baseline

Run the LLM baseline agent (requires `OPENAI_API_KEY`):

```bash
export OPENAI_API_KEY=sk-...
python baseline.py --model gpt-4o-mini --seed 42
```

**Reproducible baseline scores (gpt-4o-mini, seed=42):**

| Task | Score |
|---|---|
| missing_value_imputation | **1.0000** |
| type_errors_and_outliers | **0.9260** |
| schema_normalization_dedup | **0.8054** |
| **Mean** | **0.9105** |

Results are saved to `baseline_results.json` after each run.

---

## API reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Server health check |
| `/reset` | POST | Start a new episode |
| `/step` | POST | Execute a cleaning action |
| `/state` | GET | Full episode state |
| `/tasks` | GET | Task list + action schema |
| `/grader` | POST | Score current dataset state |
| `/baseline` | POST | Run heuristic baseline, return all 3 scores |
| `/ws` | WS | Persistent WebSocket session |
| `/docs` | GET | Interactive API documentation |

---

## Project structure

```
data-cleaning-env/
├── models.py              Pydantic typed models (Action, Observation, State)
├── dataset_generator.py   Reproducible messy dataset generation
├── graders.py             Deterministic task graders (0.0–1.0)
├── environment.py         Core RL environment logic
├── client.py              HTTP + WebSocket client
├── baseline.py            LLM baseline inference script
├── openenv.yaml           OpenEnv manifest
├── requirements.txt       Pinned dependencies
└── server/
    ├── app.py             FastAPI server
    └── Dockerfile         Container definition
```

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `PORT` | `7860` | Server port (7860 for HF Spaces) |
| `HOST` | `0.0.0.0` | Bind address |
| `WORKERS` | `4` | Uvicorn worker processes |
| `MAX_CONCURRENT_ENVS` | `100` | Max WebSocket sessions |
| `OPENAI_API_KEY` | — | Required for `baseline.py` |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | Optional custom endpoint |
