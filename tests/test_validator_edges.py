from __future__ import annotations

import io
import re
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import inference
from tasks.easy import EasyTask


class ValidatorEdgeTests(unittest.TestCase):
    def _run_main(self, argv: list[str]) -> list[str]:
        stream = io.StringIO()
        with patch("sys.argv", argv), redirect_stdout(stream):
            inference.main()
        return [line.strip() for line in stream.getvalue().splitlines() if line.strip()]

    def _extract_scores(self, end_lines: list[str]) -> list[float]:
        scores: list[float] = []
        for line in end_lines:
            match = re.search(r"score=([0-9]*\.?[0-9]+)", line)
            if match:
                scores.append(float(match.group(1)))
        return scores

    def _extract_tasks(self, start_lines: list[str]) -> list[str]:
        tasks: list[str] = []
        for line in start_lines:
            match = re.search(r"task=([a-z]+)", line)
            if match:
                tasks.append(match.group(1))
        return tasks

    def test_main_emits_three_tasks_when_single_task_arg_is_passed(self) -> None:
        lines = self._run_main(
            [
                "inference.py",
                "--task",
                "medium",
                "--agent",
                "baseline",
                "--seed",
                "7",
            ]
        )

        start_lines = [line for line in lines if line.startswith("[START]")]
        end_lines = [line for line in lines if line.startswith("[END]")]

        self.assertEqual(len(start_lines), 3)
        self.assertEqual(len(end_lines), 3)
        self.assertEqual(self._extract_tasks(start_lines), ["easy", "medium", "hard"])

        scores = self._extract_scores(end_lines)
        self.assertEqual(len(scores), 3)
        self.assertTrue(all(0.0 < score < 1.0 for score in scores))

    def test_main_uses_safe_fallback_for_incompatible_case_id(self) -> None:
        lines = self._run_main(
            [
                "inference.py",
                "--task",
                "medium",
                "--case-id",
                "EASY-999",
                "--agent",
                "baseline",
                "--seed",
                "7",
            ]
        )

        end_lines = [line for line in lines if line.startswith("[END]")]
        self.assertEqual(len(end_lines), 3)
        self.assertTrue(any("score=0.500" in line for line in end_lines))

        scores = self._extract_scores(end_lines)
        self.assertEqual(len(scores), 3)
        self.assertTrue(all(0.0 < score < 1.0 for score in scores))

    def test_emit_fallback_episode_outputs_validator_safe_lines(self) -> None:
        stream = io.StringIO()
        with redirect_stdout(stream):
            inference._emit_fallback_episode(
                task_level="easy",
                model_label="deterministic-baseline",
                error_reason="task_initialization_failed",
            )

        lines = [line.strip() for line in stream.getvalue().splitlines() if line.strip()]
        self.assertEqual(len(lines), 3)
        self.assertTrue(lines[0].startswith("[START]"))
        self.assertTrue(lines[1].startswith("[STEP]"))
        self.assertTrue(lines[2].startswith("[END]"))
        self.assertIn("score=0.500", lines[2])

    def test_grade_episode_stays_strictly_inside_zero_one(self) -> None:
        task = EasyTask()
        expected = task.get_case(task.list_case_ids()[0]).expected_deviations
        expected_signatures = {item.signature() for item in expected}

        lower_score = task.grade_episode(expected=expected, seen_submissions=set(), claimed_ground_truth=set())
        upper_score = task.grade_episode(
            expected=expected,
            seen_submissions=set(expected_signatures),
            claimed_ground_truth=set(expected_signatures),
        )

        self.assertAlmostEqual(lower_score, 0.001, places=4)
        self.assertAlmostEqual(upper_score, 0.999, places=4)
        self.assertGreater(lower_score, 0.0)
        self.assertLess(upper_score, 1.0)

    def test_metadata_reports_three_tasks_with_graders(self) -> None:
        from app import metadata

        payload = metadata()
        tasks = payload.get("tasks")
        self.assertIsInstance(tasks, list)
        self.assertEqual(len(tasks), 3)
        self.assertEqual(payload.get("task_count"), 3)

        grader_summary = payload.get("grader_summary")
        self.assertIsInstance(grader_summary, dict)
        self.assertEqual(grader_summary.get("tasks_with_graders"), 3)
        self.assertEqual(grader_summary.get("score_range"), [0.001, 0.999])

        for task in tasks:
            self.assertIn("id", task)
            self.assertIn("grader", task)
            self.assertEqual(task.get("grader_type"), "precision_recall_average")


if __name__ == "__main__":
    unittest.main()
