---
title: OpenCleanEnv
emoji: ♻️
colorFrom: green
colorTo: blue
sdk: docker
sdk_version: "1.33.0"
app_file: server/app.py
pinned: false
---


# OpenCleanEnv — OpenEnv Data Cleaning Environment

## 🚀 Overview

OpenCleanEnv is a deterministic data-cleaning environment built using the OpenEnv framework.  
It simulates real-world data preprocessing tasks such as:

- Removing duplicate records  
- Handling missing values  
- Fixing invalid data formats  

The environment is designed for benchmarking agents on structured data cleaning workflows.

---

## 🎯 Objective

Transform a dirty dataset into a fully clean dataset by applying the correct sequence of actions.

---

## ⚙️ Available Actions

| Action | Description |
|--------|-------------|
| REMOVE_DUPLICATES | Removes duplicate rows  |
| FILL_MISSING | Replaces missing values with `0` |
| FIX_FORMAT | Fixes invalid email formats |

## ⚙️ Available Tasks

| Task | Description |
|------|------------|
| Task 1 – Easy | Basic data cleaning workflow |
| Task 2 – Medium | Same workflow with stricter evaluation |
| Task 3 – Hard | Same workflow with tighter scoring sensitivity |

## 🧠 Reward System

| Action | Reward |
|--------|--------|
| REMOVE_DUPLICATES (correct) | +0.3 |
| FILL_MISSING (correct) | +0.3 |
| FIX_FORMAT (correct) | +0.4 |

Reward Range - 0.0 to 1.0
---

## 📁 Project Structure

```
opencleanenv/
│
├── server/
│   └── app.py              # FastAPI OpenEnv server
│
├── data/
│   ├── dirty_data.csv     # Input dataset
│   └── clean_data.csv     # Expected output
│
├── environment.py         # Core environment logic
├── models.py              # Action & observation schemas
├── grader.py              # Scoring logic
├── inference.py           # Agent execution script
├── openenv.yaml           # OpenEnv configuration
├── pyproject.toml         # Project config
└── Dockerfile             # Container setup
```

---

## 🐳 Run with Docker

Build the container:

```bash
docker build -t openenv-kernel .
```

Run the environment:

```bash
docker run -p 7860:7860 openenv-kernel
```

---

## 🧪 Run Inference

```bash
python inference.py
```

Expected output:

```
[START] ...
[STEP] ...
[END] ...
```

---

## ✅ Validation

Run OpenEnv validation:

```bash
openenv validate
```

---

## 🌐 API Endpoints

| Endpoint | Description |
|----------|------------|
| POST /reset | Reset environment |
| POST /step | Apply action |
| GET /state | Get current state |
| GET /schema | Get schema |
| WS /ws | WebSocket interface |

---

## 🤖 LLM Integration

The inference script uses an OpenAI-compatible client.

Environment variables required:

```bash
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
HF_TOKEN=your_token_here
```

---

## 🌐 Deployment (Hugging Face Spaces)

1. Create a new Space (Docker type)
2. Push this repository
3. Set environment variables:
   - `API_BASE_URL`
   - `MODEL_NAME`
   - `HF_TOKEN`
4. Wait for build completion
5. Test `/reset` endpoint

---

## 🏆 Features

- ✅ Deterministic environment  
- ✅ Fully OpenEnv compliant  
- ✅ Dockerized  
- ✅ Structured reward system  
- ✅ Real LLM integration  

## 🚀 Future Enhancements

- Support for multiple datasets and difficulty levels  
- Advanced cleaning actions (normalization, outlier removal)  
- Adaptive reward system based on improvement percentage  
- Action penalties and efficiency scoring  
- Dataset export and visualization tools  
- Explainable step-by-step transformations  
---

## 👨‍💻 Authors

Amogh S Y,A Jatin Ram Chowdary and Apeksh A

---

