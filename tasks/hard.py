from __future__ import annotations

from pathlib import Path

from tasks.base import BaseTask


class HardTask(BaseTask):
    def __init__(self) -> None:
        super().__init__(
            level="hard",
            name="multi_protocol_contradiction",
            dataset_path=Path("data/hard_cases.json"),
            max_steps=50,
            reward_weights={
                "true_positive": 0.25,
                "severity_match": 0.2,
                "regulation_match": 0.15,
                "duplicate_penalty": -0.5,
                "false_positive_penalty": -0.3,
                "severity_mismatch_penalty": -0.15,
            },
        )
