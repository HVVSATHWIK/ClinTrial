from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

Severity = Literal["minor", "major", "critical"]
TaskLevel = Literal["easy", "medium", "hard"]


class DeviationReport(BaseModel):
    patient_id: str = Field(min_length=1)
    clause_violated: str = Field(min_length=1)
    severity: Severity
    regulation_ref: Optional[str] = None

    def signature(self) -> str:
        return f"{self.patient_id}::{self.clause_violated}"


class Action(BaseModel):
    action_type: Literal["read_case", "submit_reports", "finish"]
    case_id: Optional[str] = None
    reports: List[DeviationReport] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_action_payload(self) -> "Action":
        if self.action_type == "read_case" and not self.case_id:
            raise ValueError("case_id is required when action_type is 'read_case'.")
        if self.action_type != "submit_reports" and self.reports:
            raise ValueError("reports can only be supplied with action_type='submit_reports'.")
        return self


class CaseData(BaseModel):
    case_id: str
    objective: str
    protocol_excerpt: str
    patient_records: List[Dict[str, Any]]
    expected_deviations: List[DeviationReport]


class TaskDataset(BaseModel):
    description: str
    task_level: TaskLevel
    cases: List[CaseData]


class Observation(BaseModel):
    episode_id: str
    task_level: TaskLevel
    current_step: int
    max_steps: int
    done: bool
    available_cases: List[str]
    case_opened: bool
    active_case_id: Optional[str] = None
    objective: Optional[str] = None
    protocol_excerpt: Optional[str] = None
    patient_records: List[Dict[str, Any]] = Field(default_factory=list)
    report_history: List[DeviationReport] = Field(default_factory=list)


class RewardBreakdown(BaseModel):
    read_reward: float = 0.0
    true_positive: float = 0.0
    severity_match: float = 0.0
    regulation_match: float = 0.0
    false_positive_penalty: float = 0.0
    duplicate_penalty: float = 0.0
    severity_mismatch_penalty: float = 0.0
    invalid_action_penalty: float = 0.0
    noop_penalty: float = 0.0
    finish_bonus: float = 0.0
    step_reward: float = 0.0


class StepInfo(BaseModel):
    task_score: float
    correct_unique_reports: int
    submitted_unique_reports: int
    expected_total: int
    episode_reason: str
    reward_breakdown: RewardBreakdown
    errors: List[str] = Field(default_factory=list)
