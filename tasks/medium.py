from __future__ import annotations

from pathlib import Path

from tasks.base import BaseTask


class MediumTask(BaseTask):
    def __init__(self) -> None:
        super().__init__(
            level="medium",
            name="severity_classification",
            dataset_path=Path("data/medium_cases.json"),
            max_steps=25,
            reward_weights={
                "true_positive": 0.3,
                "severity_match": 0.2,
                "regulation_match": 0.1,
                "duplicate_penalty": -0.5,
                "false_positive_penalty": -0.25,
                "severity_mismatch_penalty": -0.15,
            },
        )
