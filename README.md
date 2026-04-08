---
title: ClinTrial OpenEnv
emoji: 🧪
colorFrom: blue
colorTo: indigo
sdk: docker
app_file: app.py
pinned: false
---

# ClinTrialEnv (OpenEnv-Compatible)

ClinTrialEnv is a clinical trial auditing benchmark environment aligned to an OpenEnv-style interface.

It provides:
- A strict environment API: `reset()`, `step(action)`, `state()`
- Typed data contracts using Pydantic (`Observation`, `Action`, `RewardBreakdown`)
- Three explicit tasks (`easy`, `medium`, `hard`) with separate datasets
- Reward shaping with partial credit and penalties
- A competition-friendly inference runner with strict logs (`[START]`, `[STEP]`, `[END]`)
- Docker + Hugging Face Space deployment path

## What Was Implemented

- Converted the project to a modular environment architecture:
  - `env.py`
  - `models.py`
  - `tasks/`
- Added realistic task datasets in:
  - `data/easy_cases.json`
  - `data/medium_cases.json`
  - `data/hard_cases.json`
- Rewrote `inference.py` to run full episodes with strict benchmark logs.
- Added an OpenAI-client LLM path (`--agent openai`) with provider selection:
  - `--llm-provider gemini-openai` (Gemini via OpenAI-compatible endpoint)
  - `--llm-provider openai` (native OpenAI endpoint)
  and deterministic fallback baseline when credentials are missing.
- Added deployment files:
  - `Dockerfile`
  - `openenv.yaml`
  - `app.py` (Gradio app for HF Space)

## OpenEnv API

Environment class: `ClinTrialOpenEnv` in `env.py`.

Methods:
- `reset(seed=None, options=None) -> (observation, info)`
- `step(action) -> (observation, reward, done, info)`
- `state() -> observation`

Core action types:
- `read_case`
- `submit_reports`
- `finish`

### Runtime HTTP Endpoints (Validator-facing)

The deployed app exposes OpenEnv-compatible API endpoints:

- `POST /reset`
- `POST /step`
- `GET /state`
- `POST /state`

The interactive Gradio demo is still available at:

- `/ui`

## Task Design

Three explicit benchmark levels are defined:

1. Easy (`structured_detection`)
- Objective: obvious single-document deviations
- Max steps: 10
- Dataset: `data/easy_cases.json`

2. Medium (`severity_classification`)
- Objective: cross-document consistency and severity checks
- Max steps: 25
- Dataset: `data/medium_cases.json`

3. Hard (`multi_protocol_contradiction`)
- Objective: temporal and multi-record contradictions
- Max steps: 50
- Dataset: `data/hard_cases.json`

## Difficulty Rationale

The benchmark is intentionally structured to differentiate agent capability across levels.

- Easy: single, obvious deviation in one patient context.
- Medium: multi-deviation detection with cross-document consistency and timing constraints.
- Hard: multi-violation, conflicting signals, and temporal/accountability reasoning across records.

This prevents flat scoring behavior and makes performance differences between agents observable.

## Expected Score Bands

For a standard non-oracle LLM agent, expected ranges are:

- Easy: `0.8` to `1.0`
- Medium: `0.6` to `0.8`
- Hard: `0.4` to `0.7`

These bands are intended as sanity targets for evaluator interpretation, not hard constraints.

## Reward Shaping

Reward shaping includes both positive and negative components.

Positive examples:
- Correct deviation match
- Correct severity match
- Correct regulation reference

Penalties:
- Duplicate submissions
- False positives
- Invalid action/schema
- Empty report submissions (no-op)

Each step reward is clamped to `[-1.0, 1.0]`.

Task grading (`0.0` to `1.0`) is precision-recall based and reported in `info.task_score`.

## Run Inference Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run deterministic baseline:

```bash
python inference.py --task medium --agent baseline --seed 7
```

Run with Gemini through OpenAI client (recommended path for Gemini preference):

```bash
set GEMINI_API_KEY=your_gemini_key_here
python inference.py --task medium --agent openai --llm-provider gemini-openai --model gemini-2.5-flash-lite --seed 7
```

Run with OpenAI endpoint:

```bash
set OPENAI_API_KEY=your_key_here
python inference.py --task medium --agent openai --llm-provider openai --model gpt-4.1-mini --seed 7
```

Notes:
- All LLM calls are made through the OpenAI Python client.
- Gemini mode uses OpenAI-compatible base URL configuration (default: `https://generativelanguage.googleapis.com/v1beta/openai/`).
- Inference includes anti-loop controls: identical repeated report submissions are auto-terminated, and near-perfect task score can auto-trigger `finish` to avoid reward collapse from spam actions.

Expected log pattern:

```text
[START] Episode EP_XXXXXXX | Task: medium | Case: MED-001
[STEP] 1/25
[ACTION] {"action_type":"read_case","case_id":"MED-001"}
[REWARD] 0.0500
...
[END] Episode finished. Total Reward: 0.7500 | Final Task Score: 1.0000
```

## Run With Docker

Build:

```bash
docker build -t clintrial-env .
```

Run (Gradio app on port 7860):

```bash
docker run -p 7860:7860 --env OPENAI_API_KEY=your_key_here clintrial-env
```

For Gemini OpenAI-compatible mode in Docker:

```bash
docker run -p 7860:7860 --env GEMINI_API_KEY=your_gemini_key_here clintrial-env
```

Then open http://localhost:7860.

## Hugging Face Space (Beginner Steps)

1. Create an account at https://huggingface.co.
2. Click `New Space`.
3. Choose:
- Space SDK: `Docker`
- Visibility: your choice (public/private)
4. Create the space.
5. Push this repository to the Space git remote.
6. HF will auto-build from `Dockerfile`.
7. Add secret in Space settings:
- For OpenAI endpoint:
  - Key: `OPENAI_API_KEY`
  - Value: your key
- For Gemini OpenAI-compatible endpoint:
  - Key: `GEMINI_API_KEY`
  - Value: your Gemini key
8. Open the Space URL and run episodes from the UI.

## Suggested Submission Files

Keep these at root for competition submission:
- `env.py`
- `models.py`
- `tasks/`
- `data/`
- `inference.py`
- `openenv.yaml`
- `Dockerfile`
- `requirements.txt`

## Notes

- Baseline mode is deterministic and reproducible with `--seed`.
- OpenAI mode is available but may vary based on model behavior.
- If OpenAI key is missing, inference falls back to deterministic baseline.
- Environment includes robust handling for model API failures to maintain episode continuity and preserve evaluability.

## Final Submission Checklist

- Rotate any previously exposed Gemini key.
- Add the rotated key to Hugging Face Space secrets as `GEMINI_API_KEY`.
- Run one full Space smoke test and confirm:
  - no crash,
  - valid `[START]/[STEP]/[END]` logs,
  - non-zero scores for at least easy and medium.
