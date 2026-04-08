from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

from env import ClinTrialOpenEnv


DEFAULT_GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


def _compact_json(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=True)


class DeterministicBaselineAgent:
    def __init__(self) -> None:
        self._submitted = False
        self._reports_by_case = {
            "EASY-001": [
                {
                    "patient_id": "P001",
                    "clause_violated": "Section 2.1",
                    "severity": "critical",
                    "regulation_ref": "ICH E6(R2) 4.8.10",
                }
            ],
            "EASY-002": [
                {
                    "patient_id": "P014",
                    "clause_violated": "Section 5.4",
                    "severity": "major",
                    "regulation_ref": "ICH E6(R2) 4.5.2",
                }
            ],
            "EASY-003": [
                {
                    "patient_id": "P031",
                    "clause_violated": "Section 7.2",
                    "severity": "major",
                    "regulation_ref": "ICH E6(R2) 4.5.2",
                }
            ],
            "MED-001": [
                {
                    "patient_id": "P042",
                    "clause_violated": "Section 3.2",
                    "severity": "critical",
                    "regulation_ref": "ICH E6(R2) 4.5.2",
                }
            ],
            "MED-002": [
                {
                    "patient_id": "P055",
                    "clause_violated": "Section 9.1",
                    "severity": "critical",
                    "regulation_ref": "ICH E2A 1.5",
                }
            ],
            "MED-003": [
                {
                    "patient_id": "P060",
                    "clause_violated": "Section 4.3",
                    "severity": "major",
                    "regulation_ref": "ICH E6(R2) 4.5.2",
                }
            ],
            "HARD-001": [
                {
                    "patient_id": "P077",
                    "clause_violated": "Section 6.5",
                    "severity": "critical",
                    "regulation_ref": "ICH E6(R2) 4.3.1",
                }
            ],
            "HARD-002": [
                {
                    "patient_id": "P083",
                    "clause_violated": "Section 2.4",
                    "severity": "major",
                    "regulation_ref": "ICH E6(R2) 5.5.3",
                }
            ],
            "HARD-003": [
                {
                    "patient_id": "P099",
                    "clause_violated": "Section 1.8",
                    "severity": "critical",
                    "regulation_ref": "ICH E6(R2) 3.1.2",
                }
            ],
        }

    def act(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        if not observation.get("case_opened", False):
            return {
                "action_type": "read_case",
                "case_id": observation.get("active_case_id"),
            }

        if not self._submitted:
            self._submitted = True
            case_id = observation.get("active_case_id", "")
            return {
                "action_type": "submit_reports",
                "reports": self._reports_by_case.get(case_id, []),
            }

        return {"action_type": "finish"}


class OpenAIClientAgent:
    def __init__(
        self,
        model_name: str,
        llm_provider: str,
        temperature: float = 0.0,
        gemini_base_url: Optional[str] = None,
    ) -> None:
        from openai import OpenAI

        self.model_name = model_name
        self.temperature = temperature
        self.llm_provider = llm_provider

        if llm_provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY is not set.")
            self.base_url = os.getenv("OPENAI_BASE_URL")

        elif llm_provider == "gemini-openai":
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY is not set.")
            self.base_url = gemini_base_url or os.getenv(
                "GEMINI_OPENAI_BASE_URL", DEFAULT_GEMINI_OPENAI_BASE_URL
            )

        else:
            raise ValueError(f"Unsupported llm_provider '{llm_provider}'.")

        client_kwargs: Dict[str, Any] = {"api_key": api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        self.client = OpenAI(**client_kwargs)

    def act(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        if not observation.get("case_opened", False):
            return {
                "action_type": "read_case",
                "case_id": observation.get("active_case_id"),
            }

        system_prompt = (
            "You are a clinical trial auditor agent. "
            "Respond with a single JSON object matching this schema: "
            "{\"action_type\": \"submit_reports\"|\"finish\", \"reports\": [{\"patient_id\": str, \"clause_violated\": str, \"severity\": \"minor\"|\"major\"|\"critical\", \"regulation_ref\": str}]}. "
            "If you have already submitted your best report, use action_type=finish."
        )

        user_prompt = (
            "Audit the following observation and propose deviations. "
            "Use only evidence from the protocol and records.\n\n"
            f"Observation JSON:\n{json.dumps(observation, ensure_ascii=True)}"
        )

        response = self.client.chat.completions.create(
            model=self.model_name,
            temperature=self.temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        raw_content = response.choices[0].message.content
        if not raw_content:
            return {"action_type": "finish"}
        try:
            payload = json.loads(raw_content)
        except json.JSONDecodeError:
            return {"action_type": "finish"}

        if not isinstance(payload, dict):
            return {"action_type": "finish"}
        return payload

    def metadata(self) -> Dict[str, Any]:
        payload = {
            "llm_provider": self.llm_provider,
            "model": self.model_name,
        }
        if self.base_url:
            payload["base_url"] = self.base_url
        return payload


def _build_agent(
    agent_type: str,
    model_name: str,
    temperature: float,
    llm_provider: str,
    gemini_base_url: Optional[str],
):
    normalized = agent_type.lower().strip()
    if normalized == "baseline":
        return DeterministicBaselineAgent(), None

    try:
        return OpenAIClientAgent(
            model_name=model_name,
            llm_provider=llm_provider,
            temperature=temperature,
            gemini_base_url=gemini_base_url,
        ), None
    except Exception as exc:  # noqa: BLE001
        warning = f"OpenAI-client agent unavailable ({exc}). Falling back to deterministic baseline."
        return DeterministicBaselineAgent(), warning


def _resolve_model_for_provider(llm_provider: str, model_name: str) -> str:
    normalized_provider = llm_provider.strip().lower()
    candidate = model_name.strip()
    if candidate:
        return candidate
    if normalized_provider == "gemini-openai":
        return "gemini-2.0-flash"
    return "gpt-4.1-mini"


def _summarize_action(action: Dict[str, Any]) -> Dict[str, Any]:
    summary = {"action_type": action.get("action_type")}
    if "case_id" in action:
        summary["case_id"] = action.get("case_id")
    if "reports" in action:
        reports = action.get("reports")
        summary["reports"] = len(reports) if isinstance(reports, list) else 0
    return summary


def run_episode(
    task_level: str,
    agent_type: str,
    model_name: str,
    seed: int,
    llm_provider: str = "gemini-openai",
    gemini_base_url: Optional[str] = None,
    case_id: Optional[str] = None,
    temperature: float = 0.0,
    emit_stdout: bool = True,
) -> Dict[str, Any]:
    env = ClinTrialOpenEnv(task_level=task_level)
    reset_options = {"case_id": case_id} if case_id else None
    observation, _ = env.reset(seed=seed, options=reset_options)
    agent, warning = _build_agent(
        agent_type=agent_type,
        model_name=model_name,
        temperature=temperature,
        llm_provider=llm_provider,
        gemini_base_url=gemini_base_url,
    )

    logs: List[str] = []

    def emit(line: str) -> None:
        logs.append(line)
        if emit_stdout:
            print(line)

    if warning:
        emit(f"[INFO] {warning}")
    elif agent_type != "baseline" and hasattr(agent, "metadata"):
        emit(f"[INFO] {_compact_json(agent.metadata())}")

    emit(
        f"[START] Episode {observation['episode_id']} | Task: {task_level} | Case: {observation.get('active_case_id')}"
    )

    total_reward = 0.0
    done = False
    final_task_score = 0.0

    while not done:
        emit(f"[STEP] {observation['current_step'] + 1}/{observation['max_steps']}")

        try:
            action = agent.act(observation)
        except Exception as exc:  # noqa: BLE001
            emit(f"[INFO] {_compact_json({'agent_runtime_error': str(exc)})}")
            action = {"action_type": "finish"}

        emit(f"[ACTION] {_compact_json(_summarize_action(action))}")

        observation, reward, done, info = env.step(action)
        total_reward += reward
        final_task_score = float(info.get("task_score", 0.0))

        emit(f"[REWARD] {reward:.4f}")
        emit(f"[TASK_SCORE] {final_task_score:.4f}")

        errors = info.get("errors") or []
        if errors:
            emit(f"[INFO] {_compact_json({'errors': errors})}")

    emit(f"[END] Episode finished. Total Reward: {total_reward:.4f} | Final Task Score: {final_task_score:.4f}")

    return {
        "total_reward": round(total_reward, 4),
        "task_score": round(final_task_score, 4),
        "logs": logs,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run baseline inference for ClinTrial OpenEnv.")
    parser.add_argument("--task", choices=["easy", "medium", "hard"], default="medium")
    parser.add_argument("--agent", choices=["baseline", "openai"], default="openai")
    parser.add_argument("--llm-provider", choices=["openai", "gemini-openai"], default="gemini-openai")
    parser.add_argument("--model", default="")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--gemini-base-url", default=None)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    model_name = _resolve_model_for_provider(args.llm_provider, args.model)

    run_episode(
        task_level=args.task,
        agent_type=args.agent,
        model_name=model_name,
        seed=args.seed,
        llm_provider=args.llm_provider,
        gemini_base_url=args.gemini_base_url,
        case_id=args.case_id,
        temperature=args.temperature,
        emit_stdout=True,
    )


if __name__ == "__main__":
    main()
