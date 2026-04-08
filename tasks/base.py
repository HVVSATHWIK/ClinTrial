from __future__ import annotations

import json
from pathlib import Path
from random import Random
from typing import Dict, List, Set, Tuple

from models import CaseData, DeviationReport, RewardBreakdown, TaskDataset, TaskLevel


class BaseTask:
    def __init__(
        self,
        *,
        level: TaskLevel,
        name: str,
        dataset_path: Path,
        max_steps: int,
        reward_weights: Dict[str, float],
    ) -> None:
        self.level = level
        self.name = name
        self.dataset_path = dataset_path
        self.max_steps = max_steps
        self.reward_weights = reward_weights
        self.dataset = self._load_dataset(dataset_path)
        self._case_map = {case.case_id: case for case in self.dataset.cases}

    def _load_dataset(self, dataset_path: Path) -> TaskDataset:
        raw = json.loads(dataset_path.read_text(encoding="utf-8"))
        dataset = TaskDataset.model_validate(raw)
        if dataset.task_level != self.level:
            raise ValueError(
                f"Dataset level mismatch for {dataset_path}: expected {self.level}, got {dataset.task_level}."
            )
        return dataset

    def list_case_ids(self) -> List[str]:
        return list(self._case_map.keys())

    def get_case(self, case_id: str) -> CaseData:
        if case_id not in self._case_map:
            raise KeyError(f"Unknown case_id '{case_id}' for task {self.level}.")
        return self._case_map[case_id]

    def sample_case(self, rng: Random) -> CaseData:
        case_ids = self.list_case_ids()
        if not case_ids:
            raise ValueError(f"Task {self.level} has no cases in dataset {self.dataset_path}.")
        return self.get_case(rng.choice(case_ids))

    def evaluate_reports(
        self,
        reports: List[DeviationReport],
        expected: List[DeviationReport],
        seen_submissions: Set[str],
        claimed_ground_truth: Set[str],
    ) -> Tuple[float, RewardBreakdown]:
        expected_map = {item.signature(): item for item in expected}
        breakdown = RewardBreakdown()

        for report in reports:
            signature = report.signature()
            if signature in seen_submissions:
                breakdown.duplicate_penalty += self.reward_weights["duplicate_penalty"]
                continue

            seen_submissions.add(signature)
            expected_item = expected_map.get(signature)

            if expected_item is None:
                breakdown.false_positive_penalty += self.reward_weights["false_positive_penalty"]
                continue

            if signature in claimed_ground_truth:
                breakdown.duplicate_penalty += self.reward_weights["duplicate_penalty"]
                continue

            claimed_ground_truth.add(signature)
            breakdown.true_positive += self.reward_weights["true_positive"]

            if report.severity == expected_item.severity:
                breakdown.severity_match += self.reward_weights["severity_match"]
            else:
                breakdown.severity_mismatch_penalty += self.reward_weights["severity_mismatch_penalty"]

            if expected_item.regulation_ref and report.regulation_ref == expected_item.regulation_ref:
                breakdown.regulation_match += self.reward_weights["regulation_match"]

        score = self._clamp_step_reward(breakdown)
        return score, breakdown

    def grade_episode(
        self,
        expected: List[DeviationReport],
        seen_submissions: Set[str],
        claimed_ground_truth: Set[str],
    ) -> float:
        expected_signatures = {item.signature() for item in expected}
        correct = len(claimed_ground_truth.intersection(expected_signatures))
        submitted = len(seen_submissions)

        precision = correct / submitted if submitted else 0.0
        recall = correct / len(expected_signatures) if expected_signatures else 1.0

        score = (precision + recall) / 2.0
        return max(0.0, min(1.0, round(score, 4)))

    def _clamp_step_reward(self, breakdown: RewardBreakdown) -> float:
        raw_score = (
            breakdown.read_reward
            + breakdown.true_positive
            + breakdown.severity_match
            + breakdown.regulation_match
            + breakdown.false_positive_penalty
            + breakdown.duplicate_penalty
            + breakdown.severity_mismatch_penalty
            + breakdown.invalid_action_penalty
            + breakdown.noop_penalty
            + breakdown.finish_bonus
        )
        breakdown.step_reward = max(-1.0, min(1.0, raw_score))
        return breakdown.step_reward
