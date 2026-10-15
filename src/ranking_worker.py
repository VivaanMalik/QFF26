"""Background sweep of every Clifford campaign level and difficulty.

Run from the project root with ``python -m src.ranking_worker``. This is the
only worker file; ranking logic and persistence live in ``strategy_ranking``.
"""
from __future__ import annotations

import os
import traceback
from datetime import datetime, timezone
from typing import Any

from src.game import load_level_configs
from src.strategy_ranking import (
    DATA_DIR, DEFAULT_SHOTS, DIFFICULTIES, apply_difficulty, compute_strategy_ranking,
    enumerate_strategy_sets, maximum_affordable_checks, load_job_status, save_job_status, save_ranking,
)


def level_number(level_id: str) -> int:
    try:
        return int(level_id.rsplit("_", 1)[-1])
    except (ValueError, IndexError):
        return 999


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        all_levels = load_level_configs()
        levels = {
            key: value for key, value in all_levels.items()
            if str(value.get("payload_type", "clifford")).lower() == "clifford"
        }
        level_ids = sorted(levels, key=level_number)
        tasks: list[tuple[str, str, dict[str, Any], int, int]] = []
        for difficulty in DIFFICULTIES:
            for level_id in level_ids:
                cfg = apply_difficulty(levels[level_id], difficulty)
                n = int(cfg.get("num_qubits", 3))
                candidates = len(enumerate_strategy_sets(range(n), maximum_affordable_checks(cfg, n)))
                tasks.append((level_id, difficulty, cfg, DEFAULT_SHOTS[difficulty], candidates))
        total_candidates = sum(task[4] for task in tasks)
        initial = load_job_status()
        save_job_status({
            "status": "running", "pid": os.getpid(), "started_at": started_at,
            "completed_levels": 0, "total_levels": len(tasks),
            "completed_strategies": 0, "total_strategies": total_candidates,
            "current_level_id": None, "current_level_title": None,
            "current_difficulty": None, "current_strategy": None, "error": None,
            "last_completed_level": initial.get("last_completed_level"),
            "cancel_requested": False,
        })

        completed_levels = 0
        completed_strategies = 0
        for task_index, (level_id, difficulty, cfg, shots, candidate_count) in enumerate(tasks, start=1):
            if load_job_status().get("cancel_requested"):
                save_job_status({
                    "status": "cancelled", "completed_levels": completed_levels,
                    "total_levels": len(tasks), "completed_strategies": completed_strategies,
                    "total_strategies": total_candidates, "current_level_id": level_id,
                    "current_difficulty": difficulty,
                })
                return 0

            def progress(done: int, total: int, strategy: str) -> None:
                if done == total or done % 4 == 0:
                    save_job_status({
                        "status": "running", "pid": os.getpid(),
                        "completed_levels": completed_levels, "total_levels": len(tasks),
                        "completed_strategies": completed_strategies + done,
                        "total_strategies": total_candidates, "current_level_id": level_id,
                        "current_level_title": levels[level_id].get("title", level_id),
                        "current_difficulty": difficulty, "current_strategy": strategy,
                        "task_number": task_index, "error": None,
                    })

            ranking = compute_strategy_ranking(level_id, cfg, difficulty, shots, progress)
            save_ranking(ranking)
            completed_levels += 1
            completed_strategies += candidate_count
            save_job_status({
                "status": "running", "pid": os.getpid(),
                "completed_levels": completed_levels, "total_levels": len(tasks),
                "completed_strategies": completed_strategies,
                "total_strategies": total_candidates, "current_level_id": level_id,
                "current_level_title": levels[level_id].get("title", level_id),
                "current_difficulty": difficulty, "current_strategy": "Saved ranking",
                "last_completed_level": f"{level_id} · {difficulty}", "error": None,
            })

        save_job_status({
            "status": "complete", "pid": os.getpid(),
            "completed_levels": completed_levels, "total_levels": len(tasks),
            "completed_strategies": completed_strategies,
            "total_strategies": total_candidates, "current_level_id": None,
            "current_level_title": None, "current_difficulty": None,
            "current_strategy": None,
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "error": None,
        })
        return 0
    except BaseException as exc:
        save_job_status({
            "status": "failed", "pid": os.getpid(),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        return 1


if __name__ == "__main__":
    raise SystemExit(main())