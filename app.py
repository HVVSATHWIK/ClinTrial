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
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Serif:wght@500;600&display=swap');

:root {
    --bg-page: #eef4f7;
    --panel-bg: #ffffff;
    --panel-muted: #f5f9fb;
    --border: #cddae3;
    --text-main: #0f172a;
    --text-muted: #334155;
    --accent: #0e7490;
    --accent-strong: #0b5f78;
}

.gradio-container,
.dark .gradio-container {
    --body-background-fill: var(--bg-page);
    --body-text-color: var(--text-main);
    --body-text-color-subdued: var(--text-muted);
    --block-background-fill: var(--panel-bg);
    --block-border-color: var(--border);
    --block-title-text-color: var(--text-main);
    --block-label-text-color: var(--text-main);
    --background-fill-primary: #ffffff;
    --background-fill-secondary: #f4fbff;
    --input-background-fill: #ffffff;
    --input-background-fill-focus: #ffffff;
    --input-border-color: #c0d2df;
    --input-border-color-focus: #8bb9d3;
    --button-primary-background-fill: var(--accent);
    --button-primary-background-fill-hover: var(--accent-strong);
    --button-primary-text-color: #ffffff;
    --button-secondary-background-fill: #ffffff;
    --button-secondary-background-fill-hover: #eaf5fb;
    --button-secondary-border-color: #b8cedd;
    --button-secondary-border-color-hover: #9ec2d8;
    --button-secondary-text-color: #1e293b;
    --button-secondary-text-color-hover: #15384d;
    color-scheme: light !important;
    background:
        radial-gradient(circle at 0% 0%, #d4f0ef 0%, rgba(212, 240, 239, 0) 38%),
        radial-gradient(circle at 100% 0%, #e2ecfb 0%, rgba(226, 236, 251, 0) 44%),
        var(--bg-page);
    color: var(--text-main) !important;
    font-family: "IBM Plex Sans", sans-serif;
}

.gradio-container .prose,
.gradio-container .prose p,
.gradio-container .prose li,
.gradio-container .prose strong,
.gradio-container .prose h1,
.gradio-container .prose h2,
.gradio-container .prose h3,
.gradio-container .label-wrap label,
.gradio-container .label-wrap span,
.gradio-container label,
.gradio-container legend {
    color: var(--text-main) !important;
    opacity: 1 !important;
}

.gradio-container .prose p,
.gradio-container .prose li,
.gradio-container .block-info {
    color: var(--text-muted) !important;
}

.gradio-container h1,
.gradio-container h2,
.gradio-container h3 {
    font-family: "IBM Plex Serif", serif;
    color: #0b1324 !important;
    letter-spacing: 0.2px;
}

.control-panel,
.result-panel {
    background: var(--panel-bg);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 16px;
    box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
}

.result-panel {
    min-height: 640px;
}

.clinical-note {
    border: 1px solid #bae6fd;
    background: #eff9ff;
    border-radius: 10px;
    padding: 10px 12px;
    color: #0b4f63;
    font-size: 13px;
    margin-bottom: 10px;
}

.guide-panel {
    border: 1px solid #a7d8ef;
    background: #f4fbff;
    border-radius: 10px;
    padding: 10px 12px;
    margin-top: 8px;
    color: var(--text-main) !important;
}

.gradio-container input,
.gradio-container textarea,
.gradio-container select {
    background: #ffffff !important;
    color: var(--text-main) !important;
    border-color: #b9ccda !important;
}

.gradio-container input[type="number"]::-webkit-outer-spin-button,
.gradio-container input[type="number"]::-webkit-inner-spin-button {
    -webkit-appearance: none;
    margin: 0;
}

.gradio-container input[type="number"] {
    -moz-appearance: textfield;
    appearance: textfield;
}

.gradio-container select option {
    background: #ffffff !important;
    color: var(--text-main) !important;
}

.gradio-container input::placeholder,
.gradio-container textarea::placeholder {
    color: #64748b !important;
}

.gradio-container [role="listbox"],
.gradio-container .options,
.gradio-container .dropdown-menu,
.gradio-container .choices__list--dropdown {
    background: #ffffff !important;
    color: var(--text-main) !important;
    border: 1px solid #b9ccda !important;
    box-shadow: 0 10px 24px rgba(15, 23, 42, 0.1) !important;
}

.gradio-container [role="option"],
.gradio-container .option,
.gradio-container .dropdown-item,
.gradio-container .choices__item {
    background: #ffffff !important;
    color: var(--text-main) !important;
}

.gradio-container [role="option"]:hover,
.gradio-container [role="option"][aria-selected="true"],
.gradio-container .option:hover,
.gradio-container .option.selected,
.gradio-container .dropdown-item:hover,
.gradio-container .choices__item--selectable.is-highlighted {
    background: #eaf5fb !important;
    color: #0f172a !important;
}

.gradio-container [role="tab"] {
    color: #475569 !important;
    font-weight: 600;
}

.gradio-container [role="tab"][aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
}

.gradio-container button {
    font-weight: 600 !important;
}

.gradio-container .secondary-wrap button:hover,
.gradio-container button.secondary:hover {
    background: #eaf5fb !important;
    color: #15384d !important;
    border-color: #9ec2d8 !important;
}

.gradio-container .primary-wrap button,
.gradio-container button.primary {
    background: var(--accent) !important;
    color: #ffffff !important;
}

.gradio-container .primary-wrap button:hover,
.gradio-container button.primary:hover {
    background: var(--accent-strong) !important;
    color: #ffffff !important;
}

.usage-hero {
    border: 1px solid #b8d5e5;
    border-radius: 16px;
    padding: 14px;
    background: linear-gradient(135deg, #ffffff 0%, #f4fbff 100%);
    margin: 8px 0 16px;
}

.usage-header {
    display: flex;
    justify-content: space-between;
    gap: 10px;
    align-items: start;
    margin-bottom: 10px;
}

.usage-title {
    margin: 0;
    color: #0b3550;
    font-size: 24px;
}

.usage-subtitle {
    margin: 4px 0 0;
    color: #36556d;
    font-size: 14px;
}

.usage-pill {
    border: 1px solid #a7d8ef;
    border-radius: 999px;
    background: #ecf7ff;
    color: #0b4f63;
    font-size: 12px;
    font-weight: 700;
    padding: 6px 10px;
    white-space: nowrap;
}

.usage-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 10px;
}

.usage-card {
    border: 1px solid #cbe0ec;
    border-radius: 12px;
    padding: 12px;
    background: #ffffff;
}

.usage-icon {
    width: 38px;
    height: 38px;
    border-radius: 10px;
    background: #ebf7ff;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 8px;
}

.usage-card-title {
    margin: 0 0 4px;
    color: #12374c;
    font-size: 15px;
    font-weight: 700;
}

.usage-card-text {
    margin: 0;
    color: #3c5a70;
    font-size: 13px;
    line-height: 1.45;
}

.top-guide-row {
    margin: 6px 0 12px;
    align-items: flex-start !important;
}

.top-guide-panel {
    height: auto;
    border: 1px solid #b8d5e5;
    border-radius: 16px;
    padding: 14px;
    background: linear-gradient(180deg, #ffffff 0%, #f7fcff 100%);
}

.top-guide-panel .prose {
    color: #35556d !important;
    font-size: 13px;
    line-height: 1.38;
}

.top-guide-dropdown {
    width: 100%;
    margin-top: 6px;
}

.top-guide-dropdown .label-wrap {
    background: #edf8ff;
    border: 1px solid #b7d8ea;
    border-radius: 10px;
    padding: 6px 10px;
}

.top-guide-dropdown .label-wrap span {
    color: #0f3f5a !important;
    font-weight: 700;
}

.top-guide-dropdown .accordion-content,
.top-guide-dropdown .gradio-accordion-content {
    max-height: none !important;
    overflow: visible !important;
    border: none !important;
    padding-top: 6px !important;
}

.main-run-row {
    align-items: stretch !important;
}

.empty-state {
    border: 1px dashed #b7cfde;
    border-radius: 10px;
    padding: 14px;
    background: #f5fbff;
    color: #36556d;
    font-size: 13px;
}

@media (max-width: 980px) {
    .usage-grid {
        grid-template-columns: 1fr;
    }
}

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
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 10px 12px;
    background: var(--panel-muted);
}

.summary-label {
    font-size: 12px;
    color: var(--text-muted);
    margin-bottom: 4px;
}

.summary-value {
    font-size: 22px;
    line-height: 1.1;
    font-weight: 700;
    color: var(--text-main);
}

.outcome-banner {
    margin-top: 8px;
    border: 1px solid #7dd3fc;
    border-left: 4px solid var(--accent);
    border-radius: 8px;
    padding: 10px 12px;
    background: #f0f9ff;
    color: #0f3f5a;
    font-size: 13px;
}

.violations-table table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    background: #ffffff;
}

.violations-table th,
.violations-table td {
    border: 1px solid var(--border);
    padding: 8px 10px;
    text-align: left;
}

.violations-table th {
    background: #eef6f8;
    color: #1e293b;
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
    background: #fee2e2;
    color: #991b1b;
}

.severity-major {
    background: #ffedd5;
    color: #9a3412;
}

.severity-minor {
    background: #e2e8f0;
    color: #334155;
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


def _task_guide_text(task_level: str) -> str:
    normalized = task_level.lower().strip()

    if normalized == "easy":
        task_focus = "Single, high-signal deviation in one patient profile."
        pattern = "Consent timing miss or prohibited medication."
    elif normalized == "hard":
        task_focus = "Cross-document, multi-violation reasoning with timeline conflicts."
        pattern = "Dose sequencing plus accountability or eligibility conflict."
    else:
        task_focus = "Multi-violation review with consistency checks across evidence."
        pattern = "Delayed SAE report and missing follow-up compliance evidence."

    return (
        "### Clinical Task Snapshot\n"
        f"- **Current Level:** {normalized.title()}\n"
        f"- **Audit Focus:** {task_focus}\n"
        f"- **Typical Pattern:** {pattern}\n"
        "\n"
        "### Configure\n"
        "- **Task Level** and **Agent Mode**.\n"
        "- **Seed** and optional **Case ID**.\n"
        "- In openai mode only: **Provider** and **Model ID**.\n"
        "\n"
        "### Important\n"
        "- Protocol and patient records are loaded automatically from benchmark datasets.\n"
        "- Only **Run Episode** executes the environment."
    )


def _agent_mode_ui(agent_type: str):
    normalized = agent_type.lower().strip()
    if normalized == "openai":
        return (
            gr.update(visible=True, interactive=True),
            gr.update(visible=True, interactive=True),
            "LLM mode is active. Provider and Model ID are visible and used for this run.",
        )

    return (
        gr.update(visible=False, interactive=False),
        gr.update(visible=False, interactive=False),
        "Baseline mode is active. Provider and Model ID are hidden because they are ignored.",
    )


def _usage_hero_html() -> str:
        return """
<section class="usage-hero">
    <div class="usage-header">
        <div>
            <h2 class="usage-title">Clinical Runner Guide</h2>
            <p class="usage-subtitle">This benchmark auto-loads protocol and patient records. You only configure and execute the audit run.</p>
        </div>
        <div class="usage-pill">Clinical Audit Workflow</div>
    </div>

    <div class="usage-grid">
        <article class="usage-card">
            <div class="usage-icon" aria-hidden="true">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <rect x="6" y="3" width="12" height="18" rx="2" stroke="#0b5f78" stroke-width="1.8"/>
                    <path d="M9 8H15" stroke="#0b5f78" stroke-width="1.8" stroke-linecap="round"/>
                    <path d="M9 12H15" stroke="#0b5f78" stroke-width="1.8" stroke-linecap="round"/>
                    <path d="M9 16H13" stroke="#0b5f78" stroke-width="1.8" stroke-linecap="round"/>
                </svg>
            </div>
            <h3 class="usage-card-title">1. Configure Run</h3>
            <p class="usage-card-text">Set task level, choose baseline or openai mode, keep seed for reproducibility, and optionally enter a case id.</p>
        </article>

        <article class="usage-card">
            <div class="usage-icon" aria-hidden="true">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M3 12H7L9 8L12 16L15 10L17 12H21" stroke="#0b5f78" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
                    <rect x="2" y="4" width="20" height="16" rx="3" stroke="#0b5f78" stroke-width="1.5"/>
                </svg>
            </div>
            <h3 class="usage-card-title">2. Run Audit Episode</h3>
            <p class="usage-card-text">Click Run Episode once. The environment reads the case and submits protocol deviation actions step by step.</p>
        </article>

        <article class="usage-card">
            <div class="usage-icon" aria-hidden="true">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M12 3L19 6V11C19 15.1 16.4 18.8 12 20C7.6 18.8 5 15.1 5 11V6L12 3Z" stroke="#0b5f78" stroke-width="1.8" stroke-linejoin="round"/>
                    <path d="M9 11.5L11.2 13.7L15.5 9.5" stroke="#0b5f78" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
            </div>
            <h3 class="usage-card-title">3. Review Evidence</h3>
            <p class="usage-card-text">Check detected deviations, case context, and execution trace. Score reflects clinical correctness and decision efficiency.</p>
        </article>
    </div>
</section>
"""


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

        with gr.Row(equal_height=False, elem_classes=["top-guide-row"]):
            with gr.Column(scale=3):
                gr.HTML(_usage_hero_html())

            with gr.Column(scale=2):
                with gr.Group(elem_classes=["top-guide-panel"]):
                    gr.Markdown("### Task Guide")
                    gr.Markdown("Open the dropdown for task-level clinical guidance.")
                    with gr.Accordion("Task Guide Dropdown", open=False, elem_classes=["top-guide-dropdown"]):
                        task_guide_dropdown = gr.Markdown(value=_task_guide_text("medium"))

        with gr.Row(equal_height=True, elem_classes=["main-run-row"]):
            with gr.Column(scale=1, elem_classes=["control-panel"]):
                gr.Markdown("### Run Setup")
                gr.Markdown(
                    "Use the Clinical Runner Guide above, then configure controls and run."
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
                    visible=False,
                    interactive=False,
                )
                model_name = gr.Textbox(
                    value="gemini-2.5-flash-lite",
                    label="Model ID",
                    info="Keep default unless you intentionally switch models.",
                    visible=False,
                    interactive=False,
                )
                seed = gr.Textbox(
                    value="7",
                    label="Seed",
                    info="Same seed helps reproducibility.",
                    placeholder="Enter integer seed",
                )
                case_id = gr.Textbox(
                    value="",
                    label="Optional Case ID",
                    placeholder="Examples: EASY-001, MED-002, HARD-003 (leave blank for random case)",
                )

                mode_hint = gr.Markdown(
                    "Baseline mode is active. Provider and Model ID are hidden because they are ignored.",
                    elem_classes=["clinical-note"],
                )

                run_btn = gr.Button("Run Episode", variant="primary", size="lg")
                gr.Markdown("Task Guide is available in the top dropdown beside Clinical Runner Guide.")

            with gr.Column(scale=2, elem_classes=["result-panel"]):
                result_summary = gr.HTML(
                    value='<div class="outcome-banner">Run an episode to view a scored clinical audit outcome.</div>'
                )
                run_insight = gr.Markdown("Insight will appear after the first run.")
                selected_case_id = gr.Textbox(label="Selected Case ID", interactive=False)

                with gr.Tabs():
                    with gr.Tab("Detected Deviations"):
                        detected_table = gr.HTML(
                            value='<div class="empty-state">Run an episode to populate detected deviations and scoring details.</div>',
                            elem_classes=["violations-table"],
                        )
                        score_breakdown = gr.Markdown("No run executed yet.")

                    with gr.Tab("Case Context"):
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

        task_level.change(
            fn=_task_guide_text,
            inputs=[task_level],
            outputs=[task_guide_dropdown],
            queue=False,
            show_progress="hidden",
        )

        agent_type.change(
            fn=_agent_mode_ui,
            inputs=[agent_type],
            outputs=[llm_provider, model_name, mode_hint],
            queue=False,
            show_progress="hidden",
        )

    return demo


demo = build_demo()
app = gr.mount_gradio_app(api, demo, path="/")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7860")))
