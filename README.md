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
- Added optional OpenAI-driven agent path (`--agent openai`) with deterministic fallback baseline.
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

Run with OpenAI API:

```bash
set OPENAI_API_KEY=your_key_here
python inference.py --task medium --agent openai --model gpt-4.1-mini --seed 7
```

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
- Key: `OPENAI_API_KEY`
- Value: your key
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
