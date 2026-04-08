from __future__ import annotations

import json
import os
import threading
from typing import Any, Dict, List, Optional

import gradio as gr
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import RedirectResponse

from env import ClinTrialOpenEnv
from inference import run_episode


class EnvRuntime:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.task_level = "medium"
        self.env: Optional[ClinTrialOpenEnv] = None
        self.last_observation: Dict[str, Any] = {}

    def _normalize_task(self, task_level: str) -> str:
        normalized = task_level.lower().strip()
        if normalized not in {"easy", "medium", "hard"}:
            return "medium"
        return normalized

    def _ensure_env(self, task_level: str) -> ClinTrialOpenEnv:
        normalized_task = self._normalize_task(task_level)
        if self.env is None or self.task_level != normalized_task:
            self.env = ClinTrialOpenEnv(task_level=normalized_task)
            self.task_level = normalized_task
        return self.env

    def reset(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        task_level = str(payload.get("task_level") or payload.get("task") or payload.get("level") or self.task_level)

        seed = payload.get("seed")
        if seed is not None:
            try:
                seed = int(seed)
            except (TypeError, ValueError):
                seed = None

        options = payload.get("options") if isinstance(payload.get("options"), dict) else {}
        case_id = payload.get("case_id")
        if case_id and "case_id" not in options:
            options["case_id"] = case_id

        with self._lock:
            env = self._ensure_env(task_level)
            observation, info = env.reset(seed=seed, options=options or None)
            self.last_observation = observation

        return {
            "observation": observation,
            "info": info,
            "state": observation,
            "task_level": self.task_level,
        }

    def step(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        action = payload.get("action") if isinstance(payload.get("action"), dict) else payload
        if not isinstance(action, dict):
            raise ValueError("Request body must be an action object or contain an 'action' object.")

        with self._lock:
            env = self._ensure_env(self.task_level)
            if not self.last_observation:
                observation, _ = env.reset()
                self.last_observation = observation

            observation, reward, done, info = env.step(action)
            self.last_observation = observation

        return {
            "observation": observation,
            "reward": reward,
            "done": done,
            "info": info,
            "state": observation,
        }

    def state(self) -> Dict[str, Any]:
        with self._lock:
            env = self._ensure_env(self.task_level)
            if not self.last_observation:
                observation, _ = env.reset()
                self.last_observation = observation
            else:
                observation = env.state()
                self.last_observation = observation

        return {
            "observation": observation,
            "state": observation,
            "task_level": self.task_level,
            "done": bool(observation.get("done", False)),
        }


runtime = EnvRuntime()
api = FastAPI(title="ClinTrialEnv OpenEnv API", version="1.2.0")


@api.get("/meta")
def metadata() -> Dict[str, Any]:
    return {
        "name": "ClinTrialEnv",
        "status": "ok",
        "api_endpoints": ["POST /reset", "POST /step", "GET /state", "POST /state"],
        "ui": "/",
        "legacy_ui_path": "/ui",
    }


@api.get("/ui", include_in_schema=False)
def ui_redirect() -> RedirectResponse:
    return RedirectResponse(url="/", status_code=307)


@api.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@api.post("/reset")
def reset_endpoint(payload: Dict[str, Any] = Body(default_factory=dict)) -> Dict[str, Any]:
    try:
        return runtime.reset(payload)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"reset_failed: {exc}") from exc


@api.post("/step")
def step_endpoint(payload: Dict[str, Any] = Body(default_factory=dict)) -> Dict[str, Any]:
    try:
        return runtime.step(payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"step_failed: {exc}") from exc


@api.get("/state")
def state_get_endpoint() -> Dict[str, Any]:
    return runtime.state()


@api.post("/state")
def state_post_endpoint() -> Dict[str, Any]:
    return runtime.state()


def _safe_int_seed(seed: Any) -> int:
    try:
        return int(seed)
    except (TypeError, ValueError):
        return 7


def _parse_info_payload(line: str) -> Optional[Dict[str, Any]]:
    prefix = "[INFO] "
    if not line.startswith(prefix):
        return None
    payload = line[len(prefix):].strip()
    try:
        parsed = json.loads(payload)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def _parse_action_raw_payload(line: str) -> Optional[Dict[str, Any]]:
    prefix = "[ACTION_RAW] "
    if not line.startswith(prefix):
        return None
    payload = line[len(prefix):].strip()
    try:
        parsed = json.loads(payload)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def _preview_case_context(task_level: str, seed: int, case_id: Optional[str]) -> Dict[str, Any]:
    preview_env = ClinTrialOpenEnv(task_level=task_level)
    options = {"case_id": case_id} if case_id else None
    observation, _ = preview_env.reset(seed=seed, options=options)

    active_case_id = str(observation.get("active_case_id") or "")
    if not active_case_id:
        return {
            "case_id": "",
            "objective": "",
            "protocol_excerpt": "",
            "patient_records_json": "[]",
        }

    opened_observation, _, _, _ = preview_env.step(
        {
            "action_type": "read_case",
            "case_id": active_case_id,
        }
    )

    patient_records = opened_observation.get("patient_records")
    if not isinstance(patient_records, list):
        patient_records = []

    return {
        "case_id": active_case_id,
        "objective": str(opened_observation.get("objective") or ""),
        "protocol_excerpt": str(opened_observation.get("protocol_excerpt") or ""),
        "patient_records_json": json.dumps(patient_records, indent=2, ensure_ascii=True),
    }


def _extract_detected_deviations(logs: List[str]) -> List[List[str]]:
    detected: Dict[str, List[str]] = {}
    for line in logs:
        payload = _parse_action_raw_payload(line)
        if not payload or payload.get("action_type") != "submit_reports":
            continue

        reports = payload.get("reports")
        if not isinstance(reports, list):
            continue

        for report in reports:
            if not isinstance(report, dict):
                continue

            patient_id = str(report.get("patient_id") or "")
            clause_violated = str(report.get("clause_violated") or "")
            severity = str(report.get("severity") or "")
            regulation_ref = str(report.get("regulation_ref") or "")
            signature = "|".join(
                [
                    patient_id.strip().lower(),
                    clause_violated.strip().lower(),
                    severity.strip().lower(),
                    regulation_ref.strip().lower(),
                ]
            )
            if signature not in detected:
                detected[signature] = [patient_id, clause_violated, severity, regulation_ref]

    return list(detected.values())


def _build_score_breakdown(logs: List[str], total_reward: float, task_score: float, deviation_rows: List[List[str]]) -> str:
    step_count = 0
    positive_steps = 0
    negative_steps = 0
    neutral_steps = 0
    auto_finish_notes: List[str] = []
    runtime_error_notes: List[str] = []

    for line in logs:
        if line.startswith("[STEP]"):
            step_count += 1

        if line.startswith("[REWARD]"):
            try:
                value = float(line.split(" ", 1)[1])
                if value > 0:
                    positive_steps += 1
                elif value < 0:
                    negative_steps += 1
                else:
                    neutral_steps += 1
            except (IndexError, ValueError):
                pass

        info_payload = _parse_info_payload(line)
        if not info_payload:
            continue

        if "auto_finish" in info_payload:
            auto_finish_notes.append(str(info_payload["auto_finish"]))
        if "agent_runtime_error" in info_payload:
            runtime_error_notes.append("Provider/runtime error handled and episode continued.")

    unique_notes = list(dict.fromkeys(auto_finish_notes + runtime_error_notes))

    lines = [
        "### Score Breakdown",
        f"- Steps executed: {step_count}",
        f"- Final Task Score: {task_score:.4f}",
        f"- Total Reward: {total_reward:.4f}",
        f"- Positive reward steps: {positive_steps}",
        f"- Neutral reward steps: {neutral_steps}",
        f"- Negative reward steps: {negative_steps}",
        f"- Detected deviations: {len(deviation_rows)}",
    ]

    if unique_notes:
        lines.append("### Run Notes")
        for note in unique_notes[:4]:
            lines.append(f"- {note}")

    return "\n".join(lines)


def evaluate(
    task_level: str,
    agent_type: str,
    llm_provider: str,
    model_name: str,
    seed: int,
    case_id: str,
):
    normalized_case_id = case_id.strip() or None
    normalized_seed = _safe_int_seed(seed)

    case_context = _preview_case_context(
        task_level=task_level,
        seed=normalized_seed,
        case_id=normalized_case_id,
    )

    result = run_episode(
        task_level=task_level,
        agent_type=agent_type,
        llm_provider=llm_provider,
        model_name=model_name,
        seed=normalized_seed,
        case_id=normalized_case_id,
        debug_actions=True,
        emit_stdout=False,
    )

    logs = result["logs"]
    detected_rows = _extract_detected_deviations(logs)
    score_breakdown = _build_score_breakdown(
        logs=logs,
        total_reward=result["total_reward"],
        task_score=result["task_score"],
        deviation_rows=detected_rows,
    )

    log_text = "\n".join(result["logs"])
    return (
        case_context["case_id"],
        case_context["objective"],
        case_context["protocol_excerpt"],
        case_context["patient_records_json"],
        detected_rows,
        score_breakdown,
        log_text,
        result["total_reward"],
        result["task_score"],
    )


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="ClinTrialEnv OpenEnv Runner") as demo:
        gr.Markdown("# ClinTrialEnv OpenEnv Runner")
        gr.Markdown(
            "This environment simulates clinical trial auditing. The agent reads protocol and patient records, "
            "submits protocol deviations, and is evaluated on both correctness and decision efficiency."
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=1):
                gr.Markdown("### Controls")
                task_level = gr.Dropdown(choices=["easy", "medium", "hard"], value="medium", label="Task")
                agent_type = gr.Dropdown(choices=["baseline", "openai"], value="baseline", label="Agent")
                llm_provider = gr.Dropdown(
                    choices=["gemini-openai", "openai"], value="gemini-openai", label="LLM Provider"
                )
                model_name = gr.Textbox(value="gemini-2.5-flash-lite", label="Model")
                seed = gr.Number(value=7, precision=0, label="Seed")
                case_id = gr.Textbox(value="", label="Optional Case ID")
                run_btn = gr.Button("Run Episode", variant="primary")

            with gr.Column(scale=2):
                with gr.Row():
                    total_reward = gr.Number(label="Total Reward", interactive=False)
                    final_score = gr.Number(label="Final Task Score", interactive=False)
                    selected_case_id = gr.Textbox(label="Selected Case ID", interactive=False)

                score_breakdown = gr.Markdown()

                with gr.Tabs():
                    with gr.Tab("Detected Deviations"):
                        detected_table = gr.Dataframe(
                            headers=["patient_id", "clause_violated", "severity", "regulation_ref"],
                            datatype=["str", "str", "str", "str"],
                            column_count=(4, "fixed"),
                            row_count=(1, "dynamic"),
                            interactive=False,
                        )

                    with gr.Tab("Case Context"):
                        objective = gr.Textbox(label="Objective", lines=2, interactive=False)
                        protocol_excerpt = gr.Textbox(label="Protocol Excerpt", lines=6, interactive=False)
                        patient_records_json = gr.Textbox(label="Patient Records", lines=10, interactive=False)

                    with gr.Tab("Logs"):
                        logs = gr.Textbox(label="Logs", lines=22)

        run_btn.click(
            fn=evaluate,
            inputs=[task_level, agent_type, llm_provider, model_name, seed, case_id],
            outputs=[
                selected_case_id,
                objective,
                protocol_excerpt,
                patient_records_json,
                detected_table,
                score_breakdown,
                logs,
                total_reward,
                final_score,
            ],
        )

    return demo


demo = build_demo()
app = gr.mount_gradio_app(api, demo, path="/")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7860")))
