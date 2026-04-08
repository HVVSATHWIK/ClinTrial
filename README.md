# ClinTrialEnv: OpenEnv Benchmark

![ClinTrialEnv](https://img.shields.io/badge/OpenEnv-Compliant-success) ![Status](https://img.shields.io/badge/Status-Submission_Ready-blue)

## Overview
ClinTrialEnv is a strict Reinforcement Learning (RL) environment built for the OpenEnv standard. It evaluates an LLM agent's ability to act as a clinical trial auditor, reviewing protocol documents and patient records to identify, classify, and report protocol deviations.

## 🧠 Environment Dynamics & Edge Cases

To ensure strict evaluation and prevent LLM reward hacking, the environment enforces the following rules:

### 1. Termination (`done` signal)
An episode terminates (`done = True`) under two strict conditions:
1. The agent submits an action with `"action_type": "finish"`.
2. The environment reaches `max_steps` (e.g., 10 for Easy, 25 for Medium, 50 for Hard).

### 2. No-Op Penalty
If the agent submits `"action_type": "submit_reports"` but provides an empty `reports` array (`[]`), the step reward evaluates to `0.0`. The agent wastes a step without gaining any score.

### 3. Duplicate Detection Scope
A submission is considered a **duplicate** if the combination of `patient_id` + `clause_violated` exactly matches a previously submitted report in the current episode's history. 
- Duplicates yield a `0.0` reward penalty for that specific report.

### 4. Reward Saturation & Anti-Spam
To prevent an agent from spamming partial matches to inflate its score:
- **Single Claim:** Each ground truth deviation can only be matched *once*. Once a True Positive is claimed for a specific `patient_id` + `clause_violated`, subsequent matches for that same ground truth item are ignored.
- **Strict Clamping:** The final step reward is strictly bounded using `reward = max(0.0, min(1.0, score))`. It is mathematically impossible to exceed 1.0 per step.

## 🛠 Action Schema (Strict Pydantic Validation)

The environment expects a strict JSON payload. Missing fields, wrong data types, or invalid enums (e.g., a severity of "super_bad" instead of "major") will result in a schema validation failure and a `0.0` reward for the step.

```json
{
  "action_type": "submit_reports", // or "finish"
  "reports": [
    {
      "patient_id": "P004",
      "clause_violated": "Section 3.2",
      "severity": "major", // enum: minor, major, critical
      "regulation_ref": "ICH E6 R2 4.5.2"
    }
  ]
}
```

## 🚀 Running Inference

The `inference.py` script runs the evaluation loop. It produces strict, parser-friendly logs.

```bash
python inference.py
```

### Expected Log Format
```text
[START] Episode EP_8492 | Task: medium
[STEP] 1/10
[ACTION] {"action_type": "submit_reports", "reports": 1}
[REWARD] 0.6000
[STEP] 2/10
[ACTION] {"action_type": "finish"}
[END] Episode finished. Total Reward: 0.6000
```
