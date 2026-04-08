from __future__ import annotations

import os
import threading
from typing import Any, Dict, Optional

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


def evaluate(
    task_level: str,
    agent_type: str,
    llm_provider: str,
    model_name: str,
    seed: int,
    case_id: str,
):
    normalized_case_id = case_id.strip() or None
    result = run_episode(
        task_level=task_level,
        agent_type=agent_type,
        llm_provider=llm_provider,
        model_name=model_name,
        seed=seed,
        case_id=normalized_case_id,
        emit_stdout=False,
    )
    log_text = "\n".join(result["logs"])
    return log_text, result["total_reward"], result["task_score"]


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="ClinTrialEnv OpenEnv Runner") as demo:
        gr.Markdown("# ClinTrialEnv OpenEnv Runner")
        gr.Markdown("Run easy, medium, or hard episodes with deterministic baseline or OpenAI agent.")

        with gr.Row():
            task_level = gr.Dropdown(choices=["easy", "medium", "hard"], value="medium", label="Task")
            agent_type = gr.Dropdown(choices=["baseline", "openai"], value="baseline", label="Agent")
            llm_provider = gr.Dropdown(
                choices=["gemini-openai", "openai"], value="gemini-openai", label="LLM Provider"
            )

        model_name = gr.Textbox(value="gemini-2.5-flash-lite", label="Model")

        with gr.Row():
            seed = gr.Number(value=7, precision=0, label="Seed")
            case_id = gr.Textbox(value="", label="Optional Case ID")

        run_btn = gr.Button("Run Episode")

        logs = gr.Textbox(label="Logs", lines=22)
        total_reward = gr.Number(label="Total Reward")
        final_score = gr.Number(label="Final Task Score")

        run_btn.click(
            fn=evaluate,
            inputs=[task_level, agent_type, llm_provider, model_name, seed, case_id],
            outputs=[logs, total_reward, final_score],
        )

    return demo


demo = build_demo()
app = gr.mount_gradio_app(api, demo, path="/")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7860")))
