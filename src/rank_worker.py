"""Detached campaign-ranking worker. Run with: python -m src.rank_worker --difficulty Guided"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rankings import (  # noqa: E402; intentionally no Qiskit import here
    _remove_lock_if_owned,
    _update_lock_owner,
    rank_all_levels_and_strategies,
    read_job_status,
    write_job_status,
    utc_now,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Rank all strategies for every campaign level in the background.")
    parser.add_argument("--difficulty", choices=("Guided", "Standard", "Full Challenge"), required=True)
    args = parser.parse_args()
    pid = os.getpid()
    _update_lock_owner(pid)
    prior = read_job_status() or {}
    starting = dict(
        prior, status="running", difficulty=args.difficulty, pid=pid,
        started_at=prior.get("started_at") or utc_now(), finished_at=None, error=None,
    )
    write_job_status(starting)
    try:
        # Import the game and Qiskit modules only after the worker has claimed
        # the job, so dependency failures are reported in the status file/log.
        from play_cli import SHOTS, apply_difficulty, max_affordable_checks
        from src.game import load_level_configs, play_level
        from src.simulator import get_noise_models

        levels = load_level_configs(str(PROJECT_ROOT / "config" / "levels.json"))
        # Cache reuse is allowed only if the current effective noise source
        # matches the source under which those scores were generated.
        expected_noise_source = get_noise_models(3)[2]
        rank_all_levels_and_strategies(
            levels=levels,
            difficulty=args.difficulty,
            shots=SHOTS[args.difficulty],
            apply_difficulty_fn=apply_difficulty,
            max_checks_fn=max_affordable_checks,
            play_level_fn=play_level,
            expected_noise_source=expected_noise_source,
        )
        return 0
    except BaseException as exc:
        status = read_job_status() or {}
        status.update({
            "status": "failed", "difficulty": args.difficulty, "pid": pid,
            "finished_at": utc_now(), "error": f"{type(exc).__name__}: {exc}",
        })
        write_job_status(status)
        print(f"Ranking worker failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        _remove_lock_if_owned(pid)


if __name__ == "__main__":
    raise SystemExit(main())
