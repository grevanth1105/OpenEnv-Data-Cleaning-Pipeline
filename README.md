---
title: Data Cleaning Pipeline OpenEnv
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - data-cleaning
  - rl-environment
---

# 🧹 Data Cleaning Pipeline — OpenEnv

> An RL environment where AI agents learn to clean real-world datasets through interaction and reward.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/grevanth1105/OpenEnv-Data-Cleaning-Pipeline/blob/main/training_demo.ipynb)
[![HF Space](https://img.shields.io/badge/HuggingFace-Space-yellow?logo=huggingface)](https://huggingface.co/spaces/revanth11/data-cleaning-env)
[![GitHub](https://img.shields.io/badge/GitHub-Source-black?logo=github)](https://github.com/grevanth1105/OpenEnv-Data-Cleaning-Pipeline)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Why Data Cleaning?

Data cleaning consumes **60–80% of a data scientist's time** in real organisations. It is:

- **Rule-rich** — domain knowledge drives every decision
- **Context-dependent** — the right action depends on the full dataset state
- **Measurable** — success is objectively scorable (nulls removed, types correct, duplicates gone)
- **Naturally graded** — easy → medium → hard difficulty emerges from the data itself

This makes it an ideal RL training ground: real task, dense feedback, clear success criteria.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  TRAINING (Colab / local GPU)                                   │
│                                                                  │
│  GRPOTrainer (TRL)                                              │
│    └── rollout_func()                                           │
│          ├── LLM generates JSON action                          │
│          ├── Action sent to environment via HTTP                │
│          └── 4 reward signals collected                         │
└──────────────────────────┬──────────────────────────────────────┘
                           │  HTTP / WebSocket
                           │  POST /reset  POST /step
┌──────────────────────────▼──────────────────────────────────────┐
│  ENVIRONMENT (HF Spaces — Docker)                               │
│                                                                  │
│  FastAPI Server                                                  │
│    ├── /reset   → load real dataset + inject noise              │
│    ├── /step    → execute cleaning action + compute reward      │
│    ├── /grader  → score cleaned dataset (0.0–1.0)               │
│    ├── /tasks   → task list + action schema                     │
│    └── /baseline → heuristic baseline scores                    │
│                                                                  │
│  3 Tasks: Easy → Medium → Hard                                  │
│  Dense reward at every step (not just episode end)              │
└─────────────────────────────────────────────────────────────────┘
```

---

## Tasks

### Task 1 — Missing Value Imputation `easy`

**Dataset:** Titanic passengers (real data, 200 rows, 8 columns)

| Column | Issue | Correct Strategy |
|---|---|---|
| `age` | 10–20% missing | median |
| `fare` | 15% missing | mean |
| `embarked` | 5% missing | mode |
| `years_aboard` | 25% missing | median |

**Scoring:** Each column correctly imputed → +0.25. **Max: 1.0**

**Expected:** random agent ~0.0 → good agent ~1.0

---

### Task 2 — Type Errors + Outlier Detection `medium`

**Dataset:** Sales transactions (150 rows, 8 columns)

| Column | Issue | Fix |
|---|---|---|
| `unit_price` | `"$5.99"` string | cast to float |
| `quantity` | stored as string | cast to int |
| `order_date` | mixed formats (`%Y-%m-%d`, `%d/%m/%Y`, etc.) | normalize |
| `rating` | `"4.5 stars"` / `"N/A"` | cast to float, clip 0–5 |
| `discount_pct` | values > 100% and < 0% | clip to [0, 100] |
| `region` | `NORTH` / `north` / `North` | normalize to title case |

**Scoring:** Type fixes 0.60, outlier handling 0.20, region 0.15, row preservation 0.10. **Max: 1.0**

**Expected:** random agent ~0.39 → good agent ~0.93

---

### Task 3 — Schema Normalization + Deduplication `hard`

**Dataset:** CRM customers with real names/companies (225 rows, 10 columns — 25 duplicates)

| Issue | Count | Fix |
|---|---|---|
| Exact + near-duplicate rows | 25 | deduplicate |
| `region` inconsistent variants | 5 per value | normalize mapping |
| `country` inconsistent names/codes | 5 per value | normalize mapping |
| `status` mixed case | 6 variants | lowercase |
| `email`/`phone` NULL variants (`N/A`, `none`, `-`) | 20 | replace with NaN |
| `age` negative / > 120 | 4 | clip [0, 120] |
| `annual_revenue` negative | 4 | clip [0, ∞) |

**Scoring:** Dedup 0.30, normalization 0.30, schema 0.20, nulls 0.20. **Max: 1.0**

**Expected:** random agent ~0.35 → good agent ~0.80

---

## Action Space

```json
{
  "action_type": "<type>",
  "column": "<column_name_or_null>",
  "params": {}
}
```

| `action_type` | `params` | When to use |
|---|---|---|
| `impute` | `strategy`: mean/median/mode/constant | Missing values |
| `cast` | `dtype`: int/float/str/date/datetime | Wrong data types |
| `normalize` | `format`, `method`, or `mapping` | Format inconsistencies |
| `clip_outlier` | `lower`, `upper` | Numeric outliers |
| `flag_outlier` | — | Mark without removing |
| `deduplicate` | `subset` (optional) | Duplicate rows |
| `drop_column` | — | Remove column |
| `drop_rows` | `condition`: null/invalid_range | Remove rows |
| `rename` | `new_name` | Rename column |
| `finish` | — | Signal episode complete |

---

## Observation Space

Each `step()` returns:

```python
DataCleaningObservation(
    dataset_snapshot   = [...],  # first 10 rows as JSON
    total_rows         = 200,
    total_columns      = 8,
    column_stats       = [...],  # dtype, null_count, outlier_count per column
    issues_detected    = [...],  # remaining issues with severity
    issues_remaining   = 3,
    action_history     = [...],  # last 10 actions taken
    last_action_result = "Imputed 'age' with median — 20 nulls fixed.",
    reward             = 0.20,
    cumulative_reward  = 0.60,
    done               = False,
    step_count         = 3,
    task_name          = "missing_value_imputation",
    task_difficulty    = "easy",
    progress_pct       = 0.75,   # 0.0 → 1.0
)
```

---

## Reward Function

Dense rewards at **every step** — not just episode end:

| Signal | Value | Trigger |
|---|---|---|
| Issue fixed | `+0.15` | Data quality issue correctly resolved |
| Column cleaned | `+0.05` | Entire column fully fixed |
| Milestone bonus | `+0.10–0.30` | Dedup complete, high-score finish |
| Destructive action | `-0.05 to -0.15` | Dropping valid data |
| Repeated action | `-0.10` per repeat | Same action on same column again |
| Invalid action | `-0.05` | Non-existent column, bad params |

**Range:** `[-1.0, 1.0]` per step

---

## Benchmark Scores

Reproducible baseline (heuristic agent, seed=42):

| Task | Score | Difficulty |
|---|---|---|
| missing_value_imputation | **1.0000** | Easy |
| type_errors_and_outliers | **0.9267** | Medium |
| schema_normalization_dedup | **0.8045** | Hard |
| **Mean** | **0.9104** | |

Run yourself:
```bash
curl -X POST https://revanth11-data-cleaning-env.hf.space/baseline \
     -H "Content-Type: application/json" \
     -d '{"seed": 42}'
```

---

## GRPO Training Demo

Train Qwen3-1.7B to clean datasets through RL:

**Open the notebook in Colab:**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/grevanth1105/OpenEnv-Data-Cleaning-Pipeline/blob/main/training_demo.ipynb)

**What the notebook does:**
1. Connects to the live HF Space environment
2. Loads Qwen3-1.7B via TRL + vLLM
3. Defines 4 reward signals (correct, progress, efficiency, valid)
4. Trains with GRPO using curriculum learning (easy → medium → hard)
5. Shows before/after grader scores

**4 reward signals:**
```python
reward_correct    # final grader score    (did it clean well?)
reward_progress   # per-step improvement  (is each action useful?)
reward_efficiency # fewer steps = bonus   (is it fast?)
reward_valid      # parseable JSON output (is output well-formed?)
```

**Hardware:** A100-40GB (~60–90 min) | T4 works with smaller batch

---

## Setup

### Live HF Space (zero setup)

```bash
# Health check
curl https://revanth11-data-cleaning-env.hf.space/health

# Interactive docs
open https://revanth11-data-cleaning-env.hf.space/docs
```

### Local — Uvicorn

```bash
git clone https://github.com/grevanth1105/OpenEnv-Data-Cleaning-Pipeline.git
cd OpenEnv-Data-Cleaning-Pipeline
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

### Local — Docker

```bash
docker build -t data-cleaning-env -f server/Dockerfile .
docker run -d -p 8000:7860 data-cleaning-env
curl http://localhost:8000/health
```

### HF Registry

```bash
docker pull registry.hf.space/revanth11-data-cleaning-env:latest
docker run -d -p 8000:7860 registry.hf.space/revanth11-data-cleaning-env:latest
```

---

## Usage

### Python Client — Sync

```python
from client import DataCleaningEnv
from models import DataCleaningAction

with DataCleaningEnv("https://revanth11-data-cleaning-env.hf.space").sync() as env:
    obs = env.reset(task_name="missing_value_imputation", seed=42)
    print(f"Rows: {obs.total_rows} | Issues: {obs.issues_remaining}")

    obs, reward, done, info = env.step(
        DataCleaningAction(
            action_type="impute",
            column="age",
            params={"strategy": "median"},
        )
    )
    print(f"Reward: {reward:+.3f} | Progress: {obs.progress_pct:.0%}")

    result = env.grader()
    print(f"Score: {result.score:.4f} — {result.feedback}")
```

### Python Client — Async

```python
import asyncio
from client import DataCleaningEnv
from models import DataCleaningAction

async def main():
    async with DataCleaningEnv("https://revanth11-data-cleaning-env.hf.space") as env:
        obs = await env.reset(task_name="type_errors_and_outliers")
        obs, reward, done, _ = await env.step(
            DataCleaningAction(action_type="cast", column="unit_price", params={"dtype": "float"})
        )

asyncio.run(main())
```

### Direct HTTP

```bash
# Reset
curl -X POST https://revanth11-data-cleaning-env.hf.space/reset \
     -H "Content-Type: application/json" \
     -d '{"task_name": "missing_value_imputation", "seed": 42}'

# Step
curl -X POST https://revanth11-data-cleaning-env.hf.space/step \
     -H "Content-Type: application/json" \
     -d '{"action": {"action_type": "impute", "column": "age", "params": {"strategy": "median"}}}'

# Grader score
curl -X POST https://revanth11-data-cleaning-env.hf.space/grader \
     -H "Content-Type: application/json" -d '{"session_id": "default"}'

# All tasks + action schema
curl https://revanth11-data-cleaning-env.hf.space/tasks

# Baseline scores
curl -X POST https://revanth11-data-cleaning-env.hf.space/baseline \
     -H "Content-Type: application/json" -d '{"seed": 42}'
```

### LLM Baseline (HF Router — free)

```bash
export HF_TOKEN=hf_...
python baseline.py --model Qwen/Qwen2.5-72B-Instruct --seed 42
# Saves results to baseline_results.json
```

### LLM Baseline (OpenAI — optional)

```bash
export OPENAI_API_KEY=sk-...
export USE_OPENAI=true
python baseline.py --model gpt-4o-mini --seed 42
```

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Server health — `{"status": "healthy"}` |
| `/reset` | POST | Start new episode, returns initial observation |
| `/step` | POST | Execute cleaning action, returns obs + reward |
| `/state` | GET | Full episode state |
| `/tasks` | GET | All tasks with descriptions + action schema |
| `/grader` | POST | Score current dataset state (0.0–1.0) |
| `/baseline` | POST | Heuristic baseline scores for all 3 tasks |
| `/ws` | WebSocket | Persistent session (low latency) |
| `/docs` | GET | Interactive Swagger UI |

---

## Project Structure

```
data-cleaning-env/
├── models.py              ← Pydantic typed models (Action, Observation, State)
├── dataset_generator.py   ← Real dataset loading + controlled noise injection
├── graders.py             ← Deterministic task graders (0.0–1.0)
├── environment.py         ← Core RL loop + reward shaping
├── client.py              ← Async + sync HTTP/WebSocket client
├── baseline.py            ← LLM baseline inference script
├── training_demo.ipynb    ← GRPO training demo (Colab-ready)
├── openenv.yaml           ← OpenEnv manifest
├── requirements.txt       ← Pinned dependencies
├── pyproject.toml         ← Package metadata
├── README.md              ← This file
└── server/
    ├── app.py             ← FastAPI server (9 endpoints)
    └── Dockerfile         ← Container definition
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PORT` | `7860` | Server port (HF Spaces uses 7860) |
| `HOST` | `0.0.0.0` | Bind address |
| `WORKERS` | `4` | Uvicorn worker processes |
| `MAX_CONCURRENT_ENVS` | `100` | Max WebSocket sessions |
| `HF_TOKEN` | — | HuggingFace token — required for `baseline.py` |
| `USE_OPENAI` | `false` | Set `true` to use OpenAI instead of HF Router |
| `OPENAI_API_KEY` | — | Only needed if `USE_OPENAI=true` |

---

## Resources

- 🌍 [Live Environment — HF Space](https://huggingface.co/spaces/revanth11/data-cleaning-env)
- 💻 [Source Code — GitHub](https://github.com/grevanth1105/OpenEnv-Data-Cleaning-Pipeline)
- 📓 [GRPO Training Demo — Colab](https://colab.research.google.com/github/grevanth1105/OpenEnv-Data-Cleaning-Pipeline/blob/main/training_demo.ipynb)
- 🔧 [OpenEnv Framework](https://github.com/meta-pytorch/OpenEnv)
- 🤗 [TRL GRPO Docs](https://huggingface.co/docs/trl/main/en/grpo_trainer)