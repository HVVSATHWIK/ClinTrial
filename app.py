from __future__ import annotations

import html
import json
import os
import threading
from typing import Any, Dict, List, Optional

import gradio as gr
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import RedirectResponse

from env import ClinTrialOpenEnv
from inference import run_episode


UI_CSS = """
.summary-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 10px;
    margin: 8px 0 12px;
}

@media (max-width: 1100px) {
    .summary-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}

.summary-card {
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 10px 12px;
    background: #0f172a;
}

.summary-label {
    font-size: 12px;
    color: #94a3b8;
    margin-bottom: 4px;
}

.summary-value {
    font-size: 20px;
    line-height: 1.1;
    font-weight: 700;
    color: #e2e8f0;
}

.outcome-banner {
    margin-top: 8px;
    border: 1px solid #1d4ed8;
    border-left: 4px solid #2563eb;
    border-radius: 8px;
    padding: 10px 12px;
    background: #0b1220;
    color: #dbeafe;
    font-size: 13px;
}

.violations-table table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
}

.violations-table th,
.violations-table td {
    border: 1px solid #334155;
    padding: 8px 10px;
    text-align: left;
}

.violations-table th {
    background: #0f172a;
    font-weight: 700;
}

.severity-chip {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 700;
}

.severity-critical {
    background: #7f1d1d;
    color: #fecaca;
}

.severity-major {
    background: #78350f;
    color: #fde68a;
}

.severity-minor {
    background: #1f2937;
    color: #cbd5e1;
}
"""


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


def _severity_class(value: str) -> str:
    normalized = value.strip().lower()
    if normalized == "critical":
        return "severity-critical"
    if normalized == "major":
        return "severity-major"
    return "severity-minor"


def _build_violations_html(deviation_rows: List[List[str]]) -> str:
    if not deviation_rows:
        return (
            '<div class="violations-table"><table><thead><tr>'
            "<th>patient_id</th><th>clause_violated</th><th>severity</th><th>regulation_ref</th>"
            "</tr></thead><tbody><tr><td colspan=\"4\">No detected violations in this run.</td></tr></tbody></table></div>"
        )

    body_rows: List[str] = []
    for row in deviation_rows:
        patient_id = html.escape(row[0] if len(row) > 0 else "")
        clause_violated = html.escape(row[1] if len(row) > 1 else "")
        severity = html.escape(row[2] if len(row) > 2 else "")
        regulation_ref = html.escape(row[3] if len(row) > 3 else "")
        severity_css = _severity_class(severity)
        body_rows.append(
            "<tr>"
            f"<td>{patient_id}</td>"
            f"<td>{clause_violated}</td>"
            f"<td><span class=\"severity-chip {severity_css}\">{severity or 'minor'}</span></td>"
            f"<td>{regulation_ref}</td>"
            "</tr>"
        )

    return (
        '<div class="violations-table"><table><thead><tr>'
        "<th>patient_id</th><th>clause_violated</th><th>severity</th><th>regulation_ref</th>"
        f"</tr></thead><tbody>{''.join(body_rows)}</tbody></table></div>"
    )


def _build_insight_text(deviation_rows: List[List[str]], task_score: float) -> str:
    if not deviation_rows:
        return "Insight: No validated protocol violations were detected in this run."

    top_rows = deviation_rows[:2]
    clause_labels = [row[1] for row in top_rows if len(row) > 1 and row[1]]
    clause_text = ", ".join(clause_labels) if clause_labels else "protocol clauses"

    if task_score >= 0.95:
        prefix = "Insight: Agent detected expected protocol violations with high confidence"
    elif task_score >= 0.7:
        prefix = "Insight: Agent detected most protocol violations with partial uncertainty"
    else:
        prefix = "Insight: Agent detected some protocol violations and needs additional review"

    return f"{prefix}. Key clauses: {clause_text}."


def _build_result_summary_html(logs: List[str], total_reward: float, task_score: float, deviation_rows: List[List[str]]) -> str:
    step_count = sum(1 for line in logs if line.startswith("[STEP]"))

    if task_score >= 0.95 and total_reward >= 0:
        outcome = "Successfully detected expected protocol violations with stable decision efficiency."
    elif task_score >= 0.95:
        outcome = "Detected expected protocol violations; efficiency penalties were applied for suboptimal actions."
    elif task_score >= 0.7:
        outcome = "Detected most protocol violations with partial efficiency."
    elif task_score > 0:
        outcome = "Partial violation detection achieved; additional review recommended."
    else:
        outcome = "No validated protocol violations were detected."

    return (
        '<div class="summary-grid">'
        '<div class="summary-card"><div class="summary-label">Final Score</div>'
        f'<div class="summary-value">{task_score:.2f}</div></div>'
        '<div class="summary-card"><div class="summary-label">Total Reward</div>'
        f'<div class="summary-value">{total_reward:.2f}</div></div>'
        '<div class="summary-card"><div class="summary-label">Steps</div>'
        f'<div class="summary-value">{step_count}</div></div>'
        '<div class="summary-card"><div class="summary-label">Detected Violations</div>'
        f'<div class="summary-value">{len(deviation_rows)}</div></div>'
        "</div>"
        f'<div class="outcome-banner">Outcome: {html.escape(outcome)}</div>'
    )


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
    violations_html = _build_violations_html(detected_rows)
    result_summary_html = _build_result_summary_html(
        logs=logs,
        total_reward=result["total_reward"],
        task_score=result["task_score"],
        deviation_rows=detected_rows,
    )
    insight_text = _build_insight_text(detected_rows, result["task_score"])
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
        result_summary_html,
        insight_text,
        violations_html,
        score_breakdown,
        log_text,
    )


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="ClinTrialEnv OpenEnv Runner") as demo:
        gr.HTML(f"<style>{UI_CSS}</style>")
        gr.Markdown("# ClinTrialEnv OpenEnv Runner")
        gr.Markdown(
            "This environment simulates clinical trial auditing. The agent reads protocol and patient records, "
            "submits protocol deviations, and is evaluated on both correctness and decision efficiency."
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=1):
                gr.Markdown("### Controls")
                gr.Markdown(
                    "Select configuration and click **Run Episode**. "
                    "You do not type protocol text or patient data manually; the environment loads case data automatically."
                )
                task_level = gr.Dropdown(
                    choices=["easy", "medium", "hard"],
                    value="medium",
                    label="Task Level",
                    info="Easy has simpler cases; hard has multi-violation reasoning.",
                )
                agent_type = gr.Dropdown(
                    choices=["baseline", "openai"],
                    value="baseline",
                    label="Agent Mode",
                    info="Use openai for LLM behavior. Baseline is deterministic and simpler.",
                )
                llm_provider = gr.Dropdown(
                    choices=["gemini-openai", "openai"],
                    value="gemini-openai",
                    label="LLM Provider",
                    info="Used only when Agent Mode is openai.",
                )
                model_name = gr.Textbox(
                    value="gemini-2.5-flash-lite",
                    label="Model ID",
                    info="Keep default unless you intentionally switch models.",
                )
                seed = gr.Number(
                    value=7,
                    precision=0,
                    label="Seed",
                    info="Same seed helps reproducibility.",
                )
                case_id = gr.Textbox(
                    value="",
                    label="Optional Case ID",
                    placeholder="Examples: EASY-001, MED-002, HARD-003 (leave blank for random case)",
                )
                run_btn = gr.Button("Run Episode", variant="primary")

                gr.Markdown(
                    "**Quick Input Guide**\n"
                    "1. Choose Task Level\n"
                    "2. Choose Agent Mode\n"
                    "3. If openai mode: keep gemini-openai + model id\n"
                    "4. Optional: set a Case ID for a specific case\n"
                    "5. Click Run Episode"
                )

            with gr.Column(scale=2):
                result_summary = gr.HTML()
                run_insight = gr.Markdown()
                selected_case_id = gr.Textbox(label="Selected Case ID", interactive=False)

                with gr.Tabs():
                    with gr.Tab("Detected Violations"):
                        detected_table = gr.HTML(elem_classes=["violations-table"])
                        score_breakdown = gr.Markdown()

                    with gr.Tab("Case Details"):
                        objective = gr.Textbox(label="Objective", lines=2, interactive=False)
                        protocol_excerpt = gr.Textbox(label="Protocol Excerpt", lines=6, interactive=False)
                        patient_records_json = gr.Textbox(label="Patient Records", lines=10, interactive=False)

                    with gr.Tab("Execution Trace"):
                        with gr.Accordion("View Execution Trace", open=False):
                            logs = gr.Textbox(label="Execution Trace", lines=22)

        run_btn.click(
            fn=evaluate,
            inputs=[task_level, agent_type, llm_provider, model_name, seed, case_id],
            outputs=[
                selected_case_id,
                objective,
                protocol_excerpt,
                patient_records_json,
                result_summary,
                run_insight,
                detected_table,
                score_breakdown,
                logs,
            ],
        )

    return demo


demo = build_demo()
app = gr.mount_gradio_app(api, demo, path="/")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7860")))
