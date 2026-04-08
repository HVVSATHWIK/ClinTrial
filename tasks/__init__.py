from __future__ import annotations

from typing import Dict, Type

from tasks.base import BaseTask
from tasks.easy import EasyTask
from tasks.hard import HardTask
from tasks.medium import MediumTask

TASK_REGISTRY: Dict[str, Type[BaseTask]] = {
    "easy": EasyTask,
    "medium": MediumTask,
    "hard": HardTask,
}


def create_task(task_level: str) -> BaseTask:
    normalized = task_level.lower().strip()
    if normalized not in TASK_REGISTRY:
        supported = ", ".join(sorted(TASK_REGISTRY.keys()))
        raise ValueError(f"Unsupported task level '{task_level}'. Supported levels: {supported}.")
    return TASK_REGISTRY[normalized]()
