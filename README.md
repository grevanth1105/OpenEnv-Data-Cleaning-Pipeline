---
title: Data Cleaning Pipeline OpenEnv
emoji: 🧹
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

> An RL environment where AI agents learn to clean real-world datasets through interaction and dense reward.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/grevanth1105/OpenEnv-Data-Cleaning-Pipeline/blob/main/training_demo.ipynb)
[![HF Space](https://img.shields.io/badge/HuggingFace-Space-yellow?logo=huggingface)](https://huggingface.co/spaces/revanth11/data-cleaning-env)
[![GitHub](https://img.shields.io/badge/GitHub-Source-black?logo=github)](https://github.com/grevanth1105/OpenEnv-Data-Cleaning-Pipeline)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Why Data Cleaning?

Data cleaning consumes **60–80% of a data scientist's time** in real organisations. It is rule-rich, measurable, and has clear success criteria — making it ideal for RL training. Every seed generates a completely unique dataset, giving the agent **infinite unique episodes** to learn from.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  TRAINING  (Colab / Kaggle T4)                              │
│  GRPOTrainer ──► rollout_func() ──► 4 reward signals        │
└────────────────────────┬────────────────────────────────────┘
                         │  WebSocket /ws  (persistent session)
┌────────────────────────▼────────────────────────────────────┐
│  ENVIRONMENT  (HF Spaces — Docker)                          │
│  FastAPI · /reset · /step · /grader · /ws · /web            │
│                                                             │
│  5 Tasks  ·  Procedural Generation  ·  Dense Rewards        │
│  LLM-as-Judge (Task 3)  ·  Code Execution Sandbox           │
└─────────────────────────────────────────────────────────────┘
```

---

## 5 Tasks — Easy to Hard

### 🟢 Task 1 — Missing Value Imputation `easy`
**Dataset:** Procedural HR/employee data (150–250 rows, 8 cols)

Agent must identify null patterns in numeric columns and apply the correct imputation strategy — `median` for skewed columns, `mean` for symmetric ones, `mode` for categoricals.

| Column | Issue | Correct Fix |
|---|---|---|
| `age` | 5–35% missing | median |
| `salary` | 5–35% missing | mean |
| `tenure` | 5–35% missing | median |
| `score` | 5–35% missing | mean |

**Score:** Partial credit per column based on null reduction and correct strategy.

---

### 🟡 Task 2 — Type Errors + Outlier Detection `medium`
**Dataset:** Procedural sales/transactions data (100–180 rows, 8 cols)

Agent must cast string-encoded numbers, normalize mixed date formats, clean messy ratings, and clip outliers.

| Column | Issue | Fix |
|---|---|---|
| `unit_price` | `"$5.99"` string | cast to float |
| `quantity` | stored as string | cast to int |
| `order_date` | 4 mixed formats | normalize to datetime |
| `rating` | `"4.5 stars"` / `"N/A"` | cast to float, clip 0–5 |
| `discount_pct` | values outside [0, 100] | clip outliers |
| `region` | NORTH / north / N | title case |

---

### 🔴 Task 3 — Schema Normalization + Dedup `hard`
**Dataset:** Procedural CRM/customer data (150–250 rows, 10 cols)

Agent must deduplicate rows, normalize inconsistent region/status formats, fix NULL string variants, and repair invalid values. **Scored with LLM-as-judge (30%) + rule-based grader (70%).**

| Issue | Count | Fix |
|---|---|---|
| Exact + near-duplicate rows | 5–20% | deduplicate |
| Region variants (NORTH/north/N) | many | normalize |
| Status mixed case | 6 variants | lowercase |
| NULL variants (N/A, none, -) | 3–10% | replace with NaN |
| Invalid age / negative revenue | 1–5% | clip to valid range |

---

### 🟠 Task 4 — Data Type Inference `medium`
**Dataset:** Procedural product/inventory data (100–180 rows, 8 cols)

All columns are stored as `object` (string) dtype. Agent must infer the correct type for each column and cast it. Tests reasoning: *"this column looks like a date — I should cast it to datetime."*

| Column | Stored As | Should Be |
|---|---|---|
| `price` | `"$9.99"` string | float |
| `quantity` | `"42"` string | int |
| `in_stock` | `"True"` string | bool |
| `weight_kg` | `"1.234"` string | float |
| `rating` | `"4.5"` string | float |
| `last_updated` | date string | datetime |

---

### 🔵 Task 5 — Text Standardization `medium-hard`
**Dataset:** Procedural contacts data (100–170 rows, 7 cols)

Phone numbers in 8 different formats, email domain typos, inconsistent city/country names, name casing issues. Tests pattern recognition and regex reasoning.

| Column | Issue | Fix |
|---|---|---|
| `phone` | 8 different formats | normalize to +91-XXXXX-XXXXXX |
| `email` | domain typos (gmai.com, yahoo.co) | correct domain |
| `full_name` | ALL CAPS / last, first / initial | Title Case |
| `city` | MUMBAI / mumbai / Bombay | canonical name |
| `country` | IN / IND / india / INDIA | canonical name |

---

## Procedural Generation

Every `(task_name, seed, difficulty)` produces a **unique dataset**. No two episodes are the same.

```python
# 5 completely different datasets:
env.reset("missing_value_imputation", seed=42,  difficulty=0.4)
env.reset("missing_value_imputation", seed=99,  difficulty=0.6)
env.reset("missing_value_imputation", seed=777, difficulty=0.9)

# Difficulty controls messiness (0.1 = easy, 1.0 = very hard):
# - null rate: 5% → 35%
# - outlier count: 2% → 14%
# - duplicate rows: 5% → 20%
```

---

## LLM-as-Judge Reward (Task 3)

Task 3 uses a hybrid scoring approach — rule-based grader + OpenAI judge:

```python
# After agent calls finish() on Task 3:
llm_score = judge_task3(df_before, df_after, n_dupes)
final_score = 0.70 * rule_based_score + 0.30 * llm_score
```

The LLM judge evaluates subjective quality: *"Does the dataset look genuinely clean and consistent?"* — something rule-based graders cannot fully capture.

---

## Action Space

```json
{"action_type": "<type>", "column": "<name or null>", "params": {}}
```

| `action_type` | `params` | Use when |
|---|---|---|
| `impute` | `strategy`: mean/median/mode | Missing values |
| `cast` | `dtype`: int/float/str/date/datetime/bool | Wrong types |
| `normalize` | `method` or `mapping` | Format inconsistencies |
| `clip_outlier` | `lower`, `upper` | Numeric outliers |
| `deduplicate` | `subset` (optional) | Duplicate rows |
| `execute_code` | `code`: python string | Complex multi-step fix |
| `finish` | — | Signal episode complete |

---

## Reward Function

Dense rewards at **every step** — not just episode end:

| Signal | Value | Trigger |
|---|---|---|
| Issue fixed | `+0.15` | Data quality issue resolved |
| Column cleaned | `+0.05` | Full column fixed |
| Milestone bonus | `+0.10–0.30` | Dedup complete, finish |
| Destructive action | `-0.05 to -0.15` | Dropping valid data |
| Repeated action | `-0.05 × repeats` | Same action on same column |
| Invalid action | `-0.05` | Bad params or column not found |

**Range:** `[-1.0, 1.0]` per step · **Episode score:** strictly in `(0.001, 0.999)`

---

## Quick Start

### Run Inference
```bash
# 1. Add your token to .env
cp .env.example .env
# Edit .env: HF_TOKEN=hf_...

# 2. Install deps
pip install -r requirements.txt

# 3. Run
python inference.py
```

Expected output:
```
[START] task=missing_value_imputation env=data-cleaning-env model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={"action_type":"impute","column":"age",...} reward=0.20 done=false error=null
[STEP] step=2 action={"action_type":"finish","column":null,...} reward=0.30 done=true error=null
[END] success=true steps=2 rewards=0.20,0.30
```

### Live HF Space
```bash
# Health check
curl https://revanth11-data-cleaning-env.hf.space/health

# Interactive dashboard
open https://revanth11-data-cleaning-env.hf.space/web

# API docs
open https://revanth11-data-cleaning-env.hf.space/docs
```

### Local Docker
```bash
git clone https://github.com/grevanth1105/OpenEnv-Data-Cleaning-Pipeline.git
cd OpenEnv-Data-Cleaning-Pipeline
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Landing page with links |
| `/health` | GET | Server status |
| `/reset` | POST | Start new episode (accepts empty body) |
| `/step` | POST | Execute cleaning action |
| `/state` | GET | Current episode state |
| `/tasks` | GET | All 5 tasks + action schema |
| `/grader` | POST | Score current state — strictly in (0, 1) |
| `/baseline` | POST | Heuristic baseline scores |
| `/ws` | WebSocket | Persistent session (recommended) |
| `/web` | GET | Interactive dashboard |
| `/docs` | GET | Swagger UI |

### Reset with difficulty
```bash
# Available tasks:
# missing_value_imputation  | type_errors_and_outliers   | schema_normalization_dedup
# data_type_inference       | text_standardization

curl -X POST https://revanth11-data-cleaning-env.hf.space/reset \
     -H "Content-Type: application/json" \
     -d '{"task_name": "data_type_inference", "seed": 42, "difficulty": 0.7}'
```

---

## Environment Variables

| Variable | Default | Required | Description |
|---|---|---|---|
| `HF_TOKEN` | — | ✅ | HuggingFace token for inference |
| `OPENAI_API_KEY` | — | For Task 3 LLM judge | OpenAI key |
| `API_BASE_URL` | `https://router.huggingface.co/v1` | Default set | LLM endpoint |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | Default set | LLM model |
| `SPACE_URL` | deployed space URL | No | Override HF Space |

---

## GRPO Training Demo

Train Qwen2.5-1.5B on data cleaning via RL:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/grevanth1105/OpenEnv-Data-Cleaning-Pipeline/blob/main/training_demo.ipynb)

**4 reward signals used in GRPO:**
```python
reward_correct    # final grader score    — did it clean well?
reward_progress   # per-step improvement  — is each action useful?
reward_efficiency # fewer steps = bonus   — is it fast?
reward_valid      # parseable JSON output — is output well-formed?
```

---

## Project Structure

```
data-cleaning-env/
├── inference.py           ← Submission script ([START]/[STEP]/[END] format)
├── models.py              ← Pydantic typed models
├── dataset_generator.py   ← Procedural generation (infinite unique datasets)
├── graders.py             ← Deterministic graders — score in (0.001, 0.999)
├── environment.py         ← RL loop + dense reward shaping
├── code_sandbox.py        ← Sandboxed Python execution (execute_code action)
├── llm_judge.py           ← LLM-as-judge reward for Task 3 (OpenAI client)
├── client.py              ← HTTP/WebSocket client
├── baseline.py            ← Heuristic baseline agent
├── training_demo.ipynb    ← GRPO training (Kaggle T4 x2)
├── openenv.yaml           ← OpenEnv manifest
├── .env.example           ← Environment variables template
├── requirements.txt       ← Dependencies
├── pyproject.toml         ← Package metadata with [project.scripts]
├── Dockerfile             ← Root container definition
└── server/
    ├── app.py             ← FastAPI server (11 endpoints)
    └── Dockerfile         ← Server container
```

---

## Resources

- 🌍 [Live Environment](https://huggingface.co/spaces/revanth11/data-cleaning-env)
- 💻 [Source Code](https://github.com/grevanth1105/OpenEnv-Data-Cleaning-Pipeline)
- 📓 [GRPO Training Demo](https://colab.research.google.com/github/grevanth1105/OpenEnv-Data-Cleaning-Pipeline/blob/main/training_demo.ipynb)
- 🔧 [OpenEnv Framework](https://github.com/meta-pytorch/OpenEnv)
- 🤗 [TRL GRPO Docs](https://huggingface.co/docs/trl/main/en/grpo_trainer)

---

*Built for the [Meta × HuggingFace × PyTorch OpenEnv Hackathon](https://pytorch.org/event/openenv-ai-hackathon/) — India Edition*