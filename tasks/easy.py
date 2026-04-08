from __future__ import annotations

from pathlib import Path

from tasks.base import BaseTask


class EasyTask(BaseTask):
    def __init__(self) -> None:
        super().__init__(
            level="easy",
            name="structured_detection",
            dataset_path=Path("data/easy_cases.json"),
            max_steps=10,
            reward_weights={
                "true_positive": 0.35,
                "severity_match": 0.2,
                "regulation_match": 0.1,
                "duplicate_penalty": -0.45,
                "false_positive_penalty": -0.2,
                "severity_mismatch_penalty": -0.1,
            },
        )
