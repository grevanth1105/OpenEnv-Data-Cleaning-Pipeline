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
- **Measurable** — success is objectively scorable (nulls removed, types correct, duplicates gone)
- **Naturally graded** — easy → medium → hard difficulty emerges from the data itself
- **Ideal for RL** — dense feedback, clear success criteria, real-world impact

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  TRAINING (Colab / Kaggle T4)                           │
│  GRPOTrainer → rollout_func() → 4 reward signals        │
└──────────────────────┬──────────────────────────────────┘
                       │  HTTP / WebSocket
┌──────────────────────▼──────────────────────────────────┐
│  ENVIRONMENT  (HF Spaces — Docker)                      │
│  FastAPI  ·  /reset  /step  /grader  /ws  /web          │
│  3 Tasks: Easy → Medium → Hard                          │
│  Dense reward at every step                             │
└─────────────────────────────────────────────────────────┘
```

---

## Tasks

### 🟢 Task 1 — Missing Value Imputation `easy`
**Dataset:** Titanic passengers (200 rows, 8 cols)

| Column | Issue | Fix |
|---|---|---|
| `age` | ~10% missing | median |
| `fare` | ~15% missing | mean |
| `embarked` | ~5% missing | mode |
| `years_aboard` | ~25% missing | median |

**Scoring:** Each column correctly imputed → +0.25. **Max: 0.999**

---

### 🟡 Task 2 — Type Errors + Outlier Detection `medium`
**Dataset:** Sales transactions (150 rows, 8 cols)

| Column | Issue | Fix |
|---|---|---|
| `unit_price` | `"$5.99"` string | cast to float |
| `quantity` | stored as string | cast to int |
| `order_date` | mixed formats | normalize |
| `rating` | `"4.5 stars"` / `"N/A"` | cast, clip 0–5 |
| `discount_pct` | values > 100 and < 0 | clip [0, 100] |
| `region` | `NORTH` / `north` mixed | title case |

**Scoring:** Type fixes 0.60 + outlier 0.20 + region 0.15 + preservation 0.10. **Max: 0.999**

---

### 🔴 Task 3 — Schema Normalization + Deduplication `hard`
**Dataset:** CRM customers (225 rows, 10 cols — 25 duplicates)

| Issue | Count | Fix |
|---|---|---|
| Exact + near-duplicate rows | 25 | deduplicate |
| `region` inconsistent variants | 5 per value | normalize |
| `status` mixed case | 6 variants | lowercase |
| NULL variants (`N/A`, `none`, `-`) | 20 | replace with NaN |
| `age` negative / > 120 | 4 | clip [0, 120] |

**Scoring:** Dedup 0.30 + normalize 0.30 + schema 0.20 + nulls 0.20. **Max: 0.999**

---

## Action Space

```json
{"action_type": "<type>", "column": "<name or null>", "params": {}}
```

| `action_type` | `params` | When to use |
|---|---|---|
| `impute` | `strategy`: mean/median/mode | Missing values |
| `cast` | `dtype`: int/float/str/date | Wrong data types |
| `normalize` | `format`, `method`, or `mapping` | Format inconsistencies |
| `clip_outlier` | `lower`, `upper` | Numeric outliers |
| `deduplicate` | `subset` (optional) | Duplicate rows |
| `execute_code` | `code`: python string | Complex multi-step fix |
| `finish` | — | Signal episode complete |

---

## Reward Function

Dense rewards at **every step**:

| Signal | Value | Trigger |
|---|---|---|
| Issue fixed | `+0.15` | Data quality issue resolved |
| Column cleaned | `+0.05` | Full column fixed |
| Milestone bonus | `+0.10–0.30` | Dedup complete, finish |
| Destructive action | `-0.05 to -0.15` | Dropping valid data |
| Repeated action | `-0.10` per repeat | Same action on same column |
| Invalid action | `-0.05` | Non-existent column / bad params |

**Range:** `[-1.0, 1.0]` per step · **Episode score:** strictly in `(0, 1)`

---

## Benchmark Scores

Reproducible heuristic baseline (seed=42):

| Task | Score |
|---|---|
| missing_value_imputation | **0.999** |
| type_errors_and_outliers | **0.927** |
| schema_normalization_dedup | **0.805** |
| **Mean** | **0.910** |

```bash
curl -X POST https://revanth11-data-cleaning-env.hf.space/baseline \
     -H "Content-Type: application/json" -d '{"seed": 42}'
```

---

## GRPO Training Demo

Train Qwen2.5-1.5B on data cleaning via RL:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/grevanth1105/OpenEnv-Data-Cleaning-Pipeline/blob/main/training_demo.ipynb)

**4 reward signals:**
```python
reward_correct    # final grader score    — did it clean well?
reward_progress   # per-step improvement  — is each action useful?
reward_efficiency # fewer steps = bonus   — is it fast?
reward_valid      # parseable JSON output — is output well-formed?
```

---

## Quick Start

### Live HF Space
```bash
curl https://revanth11-data-cleaning-env.hf.space/health
```

### Run Inference
```bash
# Set token
export HF_TOKEN=hf_...         # Mac/Linux
set HF_TOKEN=hf_...            # Windows

# Run
python inference.py
```

Expected output:
```
[START] task=missing_value_imputation env=data-cleaning-env model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={"action_type":"impute","column":"age",...} reward=0.20 done=false error=null
[STEP] step=2 action={"action_type":"finish","column":null,...} reward=0.30 done=true error=null
[END] success=true steps=2 rewards=0.20,0.30
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
| `/` | GET | Landing page |
| `/health` | GET | Server status |
| `/reset` | POST | Start new episode |
| `/step` | POST | Execute cleaning action |
| `/state` | GET | Current episode state |
| `/tasks` | GET | Task list + action schema |
| `/grader` | POST | Score current state (0–1) |
| `/baseline` | POST | Heuristic baseline scores |
| `/ws` | WebSocket | Persistent session (recommended) |
| `/web` | GET | Interactive dashboard |
| `/docs` | GET | Swagger UI |

---

## Environment Variables

| Variable | Default | Required | Description |
|---|---|---|---|
| `HF_TOKEN` | — | ✅ Yes | HuggingFace token |
| `API_BASE_URL` | `https://router.huggingface.co/v1` | Default set | LLM API endpoint |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | Default set | LLM model |
| `SPACE_URL` | deployed space | No | Override HF Space URL |

---

## Project Structure

```
data-cleaning-env/
├── inference.py           ← Submission inference script
├── models.py              ← Pydantic typed models
├── dataset_generator.py   ← Real datasets + noise injection
├── graders.py             ← Deterministic graders (0–1)
├── environment.py         ← RL loop + reward shaping
├── code_sandbox.py        ← Sandboxed Python execution
├── client.py              ← HTTP/WebSocket client
├── baseline.py            ← Heuristic baseline agent
├── training_demo.ipynb    ← GRPO training (Kaggle T4x2)
├── openenv.yaml           ← OpenEnv manifest
├── .env.example           ← Environment variables template
├── requirements.txt       ← Dependencies
├── pyproject.toml         ← Package metadata
├── Dockerfile             ← Container definition
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