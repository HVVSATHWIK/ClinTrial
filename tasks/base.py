from __future__ import annotations

import json
import re
from pathlib import Path
from random import Random
from typing import Dict, List, Optional, Set, Tuple

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
        breakdown = RewardBreakdown()

        for report in reports:
            signature = report.signature()
            if signature in seen_submissions:
                breakdown.duplicate_penalty += self.reward_weights["duplicate_penalty"]
                continue

            seen_submissions.add(signature)
            expected_item, match_score = self._find_best_expected(report, expected, claimed_ground_truth)

            if expected_item is None or match_score < 0.45:
                breakdown.false_positive_penalty += self.reward_weights["false_positive_penalty"] * 0.85
                continue

            matched_signature = expected_item.signature()
            if matched_signature in claimed_ground_truth:
                breakdown.duplicate_penalty += self.reward_weights["duplicate_penalty"]
                continue

            claimed_ground_truth.add(matched_signature)

            if match_score >= 0.88:
                tp_factor = 1.0
            elif match_score >= 0.7:
                tp_factor = 0.65
            else:
                tp_factor = 0.4
            breakdown.true_positive += self.reward_weights["true_positive"] * tp_factor

            if report.severity == expected_item.severity:
                breakdown.severity_match += self.reward_weights["severity_match"]
            else:
                mismatch_factor = 0.75 if match_score >= 0.7 else 0.5
                breakdown.severity_mismatch_penalty += self.reward_weights["severity_mismatch_penalty"] * mismatch_factor

            regulation_similarity = self._regulation_similarity(report.regulation_ref, expected_item.regulation_ref)
            if regulation_similarity >= 0.95:
                breakdown.regulation_match += self.reward_weights["regulation_match"]
            elif regulation_similarity >= 0.7:
                breakdown.regulation_match += self.reward_weights["regulation_match"] * 0.35

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

        # Favor recall slightly so near-correct reports still produce a useful task score.
        score = (0.45 * precision) + (0.55 * recall)
        return max(0.0, min(1.0, round(score, 4)))

    def _find_best_expected(
        self,
        report: DeviationReport,
        expected: List[DeviationReport],
        claimed_ground_truth: Set[str],
    ) -> Tuple[Optional[DeviationReport], float]:
        candidates = [item for item in expected if item.signature() not in claimed_ground_truth]
        if not candidates:
            return None, 0.0

        report_patient = report.patient_id.strip().lower()
        best_item: Optional[DeviationReport] = None
        best_score = 0.0

        for item in candidates:
            patient_match = 1.0 if report_patient == item.patient_id.strip().lower() else 0.0
            clause_score = self._clause_similarity(report.clause_violated, item.clause_violated)
            combined = (0.7 * clause_score) + (0.3 * patient_match)
            if combined > best_score:
                best_score = combined
                best_item = item

        return best_item, best_score

    def _clause_similarity(self, submitted: str, expected: str) -> float:
        submitted_norm = self._normalize_text(submitted)
        expected_norm = self._normalize_text(expected)
        if not submitted_norm or not expected_norm:
            return 0.0
        if submitted_norm == expected_norm:
            return 1.0

        submitted_section = self._extract_section_id(submitted_norm)
        expected_section = self._extract_section_id(expected_norm)
        section_score = 0.85 if submitted_section and submitted_section == expected_section else 0.0

        submitted_tokens = set(submitted_norm.split())
        expected_tokens = set(expected_norm.split())
        if not submitted_tokens or not expected_tokens:
            return section_score

        overlap = len(submitted_tokens.intersection(expected_tokens))
        union = len(submitted_tokens.union(expected_tokens))
        jaccard = overlap / union if union else 0.0
        containment = overlap / len(expected_tokens)
        return max(section_score, jaccard, containment * 0.85)

    def _regulation_similarity(self, submitted: Optional[str], expected: Optional[str]) -> float:
        if not submitted or not expected:
            return 0.0
        submitted_norm = self._normalize_text(submitted)
        expected_norm = self._normalize_text(expected)
        if submitted_norm == expected_norm:
            return 1.0
        if submitted_norm in expected_norm or expected_norm in submitted_norm:
            return 0.65

        submitted_tokens = set(submitted_norm.split())
        expected_tokens = set(expected_norm.split())
        overlap = len(submitted_tokens.intersection(expected_tokens))
        return overlap / len(expected_tokens) if expected_tokens else 0.0

    def _normalize_text(self, value: str) -> str:
        lowered = value.lower().strip()
        lowered = re.sub(r"[^a-z0-9. ]+", " ", lowered)
        return re.sub(r"\s+", " ", lowered).strip()

    def _extract_section_id(self, value: str) -> Optional[str]:
        match = re.search(r"section\s*(\d+(?:\.\d+)*)", value)
        return match.group(1) if match else None

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
