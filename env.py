from __future__ import annotations

import random
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import ValidationError

from models import Action, CaseData, Observation, RewardBreakdown, StepInfo, TaskLevel
from tasks import create_task


class ClinTrialOpenEnv:
    def __init__(self, task_level: TaskLevel = "medium") -> None:
        self.rng = random.Random()
        self.task = create_task(task_level)
        self.task_level = self.task.level
        self.max_steps = self.task.max_steps
        self._clear_episode_state()

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if seed is not None:
            self.rng.seed(seed)

        self._clear_episode_state()
        options = options or {}
        requested_case_id = options.get("case_id")

        if requested_case_id:
            self.active_case = self.task.get_case(requested_case_id)
        else:
            self.active_case = self.task.sample_case(self.rng)

        self.episode_id = f"EP_{uuid.uuid4().hex[:8].upper()}"
        observation = self._build_observation()
        info = {
            "episode_id": self.episode_id,
            "task_level": self.task_level,
            "case_id": self.active_case.case_id,
            "max_steps": self.max_steps,
        }
        return observation.model_dump(), info

    def state(self) -> Dict[str, Any]:
        return self._build_observation().model_dump()

    def step(self, action_dict: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if self.active_case is None:
            raise RuntimeError("Call reset() before calling step().")

        if self.done:
            info = self._build_info(
                reward_breakdown=RewardBreakdown(),
                errors=["episode_already_finished"],
                episode_reason="already_done",
            )
            return self.state(), 0.0, True, info.model_dump()

        self.current_step += 1
        reward_breakdown = RewardBreakdown()
        errors: List[str] = []
        episode_reason = "running"

        try:
            action = Action.model_validate(action_dict)
        except ValidationError as exc:
            reward_breakdown.invalid_action_penalty = -0.3
            errors.append("schema_validation_failed")
            errors.append(str(exc))
            reward = self._finalize_step_reward(reward_breakdown)
            if self.current_step >= self.max_steps:
                self.done = True
                episode_reason = "max_steps"
            info = self._build_info(reward_breakdown, errors, episode_reason)
            return self.state(), reward, self.done, info.model_dump()

        if action.action_type == "read_case":
            if action.case_id != self.active_case.case_id:
                reward_breakdown.invalid_action_penalty = -0.2
                errors.append("unknown_case_id")
            else:
                self.case_opened = True
                reward_breakdown.read_reward = 0.05

        elif action.action_type == "submit_reports":
            if not self.case_opened:
                reward_breakdown.invalid_action_penalty = -0.25
                errors.append("must_read_case_before_submit")
            elif not action.reports:
                reward_breakdown.noop_penalty = -0.2
            else:
                self.report_history.extend(action.reports)
                _, task_breakdown = self.task.evaluate_reports(
                    reports=action.reports,
                    expected=self.active_case.expected_deviations,
                    seen_submissions=self.seen_submissions,
                    claimed_ground_truth=self.claimed_ground_truth,
                )
                reward_breakdown = self._merge_breakdowns(reward_breakdown, task_breakdown)

        elif action.action_type == "finish":
            self.done = True
            episode_reason = "agent_finish"

        self.task_score = self.task.grade_episode(
            expected=self.active_case.expected_deviations,
            seen_submissions=self.seen_submissions,
            claimed_ground_truth=self.claimed_ground_truth,
        )

        if action.action_type == "finish" and self.task_score >= 0.6:
            reward_breakdown.finish_bonus = 0.1

        reward = self._finalize_step_reward(reward_breakdown)

        if not self.done and self.current_step >= self.max_steps:
            self.done = True
            episode_reason = "max_steps"

        info = self._build_info(
            reward_breakdown=reward_breakdown,
            errors=errors,
            episode_reason=episode_reason,
        )
        return self.state(), reward, self.done, info.model_dump()

    def _clear_episode_state(self) -> None:
        self.episode_id = ""
        self.current_step = 0
        self.done = False
        self.case_opened = False
        self.active_case: Optional[CaseData] = None
        self.report_history = []
        self.seen_submissions: Set[str] = set()
        self.claimed_ground_truth: Set[str] = set()
        self.task_score = 0.0
        self.total_reward = 0.0

    def _build_observation(self) -> Observation:
        if self.active_case is None:
            return Observation(
                episode_id=self.episode_id,
                task_level=self.task_level,
                current_step=self.current_step,
                max_steps=self.max_steps,
                done=self.done,
                available_cases=self.task.list_case_ids(),
                case_opened=False,
            )

        return Observation(
            episode_id=self.episode_id,
            task_level=self.task_level,
            current_step=self.current_step,
            max_steps=self.max_steps,
            done=self.done,
            available_cases=self.task.list_case_ids(),
            case_opened=self.case_opened,
            active_case_id=self.active_case.case_id,
            objective=self.active_case.objective if self.case_opened else None,
            protocol_excerpt=self.active_case.protocol_excerpt if self.case_opened else None,
            patient_records=self.active_case.patient_records if self.case_opened else [],
            report_history=self.report_history,
        )

    def _merge_breakdowns(self, base: RewardBreakdown, update: RewardBreakdown) -> RewardBreakdown:
        merged = base.model_dump()
        for key, value in update.model_dump().items():
            if key == "step_reward":
                continue
            merged[key] = merged.get(key, 0.0) + value
        return RewardBreakdown.model_validate(merged)

    def _finalize_step_reward(self, reward_breakdown: RewardBreakdown) -> float:
        raw_reward = (
            reward_breakdown.read_reward
            + reward_breakdown.true_positive
            + reward_breakdown.severity_match
            + reward_breakdown.regulation_match
            + reward_breakdown.false_positive_penalty
            + reward_breakdown.duplicate_penalty
            + reward_breakdown.severity_mismatch_penalty
            + reward_breakdown.invalid_action_penalty
            + reward_breakdown.noop_penalty
            + reward_breakdown.finish_bonus
        )
        reward_breakdown.step_reward = max(-1.0, min(1.0, raw_reward))
        self.total_reward += reward_breakdown.step_reward
        return reward_breakdown.step_reward

    def _build_info(
        self,
        reward_breakdown: RewardBreakdown,
        errors: List[str],
        episode_reason: str,
    ) -> StepInfo:
        expected_total = len(self.active_case.expected_deviations) if self.active_case else 0
        return StepInfo(
            task_score=self.task_score,
            correct_unique_reports=len(self.claimed_ground_truth),
            submitted_unique_reports=len(self.seen_submissions),
            expected_total=expected_total,
            episode_reason=episode_reason,
            reward_breakdown=reward_breakdown,
            errors=errors,
        )
