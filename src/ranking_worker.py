"""Crash-isolated background sweep for every Clifford strategy.

The parent process never executes a noisy quantum circuit. Each individual
strategy runs in a short-lived child process, so a native Qiskit/Aer/Paulice
segmentation fault marks only that strategy as unavailable and does not kill
the entire level/difficulty sweep.

Run from the project root with ``python -m src.ranking_worker``. Single
mission runs launched by the UI use ``src.simulation_worker``; ranking candidates
are likewise executed in disposable children so native crashes stay contained.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.strategy_ranking import (
    DATA_DIR,
    DEFAULT_SHOTS,
    DIFFICULTIES,
    PROJECT_ROOT,
    apply_difficulty,
    compute_strategy_ranking,
    enumerate_strategy_sets,
    maximum_affordable_checks,
    load_job_status,
    save_job_status,
    save_ranking,
)

PROBE_TIMEOUT_SECONDS = 240


def level_number(level_id: str) -> int:
    try:
        return int(level_id.rsplit("_", 1)[-1])
    except (ValueError, IndexError):
        return 999


def _json_safe(value: Any) -> Any:
    """Convert a small simulation result to strict portable JSON primitives."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        import math
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except Exception:
            pass
    try:
        number = float(value)
        import math
        return number if math.isfinite(number) else None
    except (TypeError, ValueError, OverflowError):
        return str(value)


def _extract_ranking_fields(result: dict[str, Any]) -> dict[str, Any]:
    """Return only fields consumed by the scorer; never pickle a QuantumCircuit."""
    metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
    cost = result.get("cost") if isinstance(result.get("cost"), dict) else {}
    return _json_safe({
        "game_score": result.get("game_score"),
        "metrics": metrics,
        "cost": cost,
        "paulice_cost": result.get("paulice_cost"),
        "check_supports": result.get("check_supports", []),
        "num_checks_requested": result.get("num_checks_requested"),
        "num_checks_used": result.get("num_checks_used"),
        "noise_source": result.get("noise_source", "unknown"),
        "scoring_beta": result.get("scoring_beta", 2.0),
    })


def _run_probe_child(input_path: str, output_path: str) -> int:
    """Child entry point. Its process may crash without taking down the sweep."""
    destination = Path(output_path)
    try:
        payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
        # Import the simulator only in this isolated child process.
        from src.game import _play_level_impl

        result = _play_level_impl(
            str(payload["level_id"]),
            num_checks=len(payload["targets"]),
            target_qubits=[int(q) for q in payload["targets"]],
            shots=int(payload["shots"]),
            config_overrides=dict(payload["config"]),
        )
        response = {"status": "complete", "result": _extract_ranking_fields(result)}
    except BaseException as exc:
        response = {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=12),
        }
    try:
        destination.write_text(json.dumps(response, allow_nan=False), encoding="utf-8")
        return 0
    except BaseException:
        # If writing the result fails, the parent will classify the missing
        # response as N/A and keep processing the remaining strategies.
        return 2


def _crash_message(returncode: int, stderr: str = "") -> str:
    if returncode < 0:
        signal_number = -returncode
        try:
            signal_name = signal.Signals(signal_number).name
        except (ValueError, AttributeError):
            signal_name = f"signal {signal_number}"
        if signal_name == "SIGSEGV":
            return "Native simulator segmentation fault (SIGSEGV). Metrics are N/A; run strategy check to retry."
        return f"Isolated simulator exited via {signal_name}. Metrics are N/A; run strategy check to retry."
    if returncode in {0xC0000005, 0xC0000409, 3221225477, 3221226505}:
        return "Native simulator process crashed on Windows. Metrics are N/A; run strategy check to retry."
    details = stderr.strip().splitlines()[-3:]
    suffix = " | ".join(details)
    return f"Isolated simulator exited with code {returncode}. Metrics are N/A; run strategy check to retry." + (f" Details: {suffix}" if suffix else "")


def run_strategy_isolated(
    level_id: str,
    targets: list[int],
    config: dict[str, Any],
    shots: int,
) -> dict[str, Any]:
    """Run one candidate safely in a child process and return scorer inputs."""
    with tempfile.TemporaryDirectory(prefix="syndrome_strategy_probe_") as temp_dir:
        root = Path(temp_dir)
        input_path = root / "strategy.json"
        output_path = root / "result.json"
        payload = {
            "level_id": str(level_id),
            "targets": [int(q) for q in targets],
            "config": dict(config),
            "shots": int(shots),
        }
        input_path.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
        command = [
            sys.executable, "-m", "src.ranking_worker", "--strategy-probe",
            str(input_path), str(output_path),
        ]
        env = os.environ.copy()
        for thread_var in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "BLIS_NUM_THREADS",
        ):
            env[thread_var] = "1"
        env.setdefault("PYTHONUTF8", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")
        existing = env.get("PYTHONPATH", "")
        root_string = str(PROJECT_ROOT)
        if root_string not in existing.split(os.pathsep):
            env["PYTHONPATH"] = (
                root_string if not existing
                else root_string + os.pathsep + existing
            )
        try:
            completed = subprocess.run(
                command,
                cwd=str(PROJECT_ROOT),
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=PROBE_TIMEOUT_SECONDS,
                env=env,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Isolated strategy check timed out after {PROBE_TIMEOUT_SECONDS}s. Metrics are N/A; run strategy check to retry."
            ) from exc
        if not output_path.exists():
            raise RuntimeError(_crash_message(completed.returncode, completed.stderr))
        try:
            response = json.loads(output_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Isolated strategy returned an unreadable result ({type(exc).__name__}). Metrics are N/A; run strategy check to retry."
            ) from exc
        if completed.returncode != 0:
            raise RuntimeError(_crash_message(completed.returncode, completed.stderr))
        if response.get("status") != "complete" or not isinstance(response.get("result"), dict):
            error = response.get("error", "The isolated strategy did not return a complete result.")
            raise RuntimeError(str(error))
        return response["result"]


def _load_levels_without_importing_the_simulator() -> dict[str, dict[str, Any]]:
    """Read plain JSON in the parent; keep Qiskit imports out of the sweep host."""
    path = PROJECT_ROOT / "config" / "levels.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("config/levels.json must contain an object of campaign levels")
    return value


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--strategy-probe":
        if len(argv) != 3:
            return 2
        return _run_probe_child(argv[1], argv[2])

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        all_levels = _load_levels_without_importing_the_simulator()
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
                candidates = len(enumerate_strategy_sets(
                    range(n), maximum_affordable_checks(cfg, n)
                ))
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
                # Publish periodically, and always publish at candidate boundaries.
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

            ranking = compute_strategy_ranking(
                level_id, cfg, difficulty, shots, progress,
                strategy_runner=run_strategy_isolated,
            )
            save_ranking(ranking)
            completed_levels += 1
            completed_strategies += candidate_count
            failed_count = len(ranking.get("failed_rows") or [])
            save_job_status({
                "status": "running", "pid": os.getpid(),
                "completed_levels": completed_levels, "total_levels": len(tasks),
                "completed_strategies": completed_strategies,
                "total_strategies": total_candidates, "current_level_id": level_id,
                "current_level_title": levels[level_id].get("title", level_id),
                "current_difficulty": difficulty,
                "current_strategy": f"Saved ranking · {failed_count} unavailable candidate(s)" if failed_count else "Saved ranking",
                "last_completed_level": f"{level_id} · {difficulty}",
                "failed_strategy_count": int(load_job_status().get("failed_strategy_count", 0) or 0) + failed_count,
                "error": None,
            })

        final_status = load_job_status()
        save_job_status({
            "status": "complete", "pid": os.getpid(), "completed_levels": completed_levels,
            "total_levels": len(tasks), "completed_strategies": completed_strategies,
            "total_strategies": total_candidates, "current_level_id": None,
            "current_level_title": None, "current_difficulty": None, "current_strategy": None,
            "failed_strategy_count": int(final_status.get("failed_strategy_count", 0) or 0),
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