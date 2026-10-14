"""Persistent, resumable campaign strategy rankings and background-worker tools.

The ranking database is written once per completed level. Ordinary gameplay only
reads this database; it never starts an exhaustive ranking search implicitly.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Iterable

RANKING_ENGINE_VERSION = "3"
SCORING_BETA = 2.0
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RANKINGS_PATH = DATA_DIR / "strategy_rankings.json"
JOB_STATUS_PATH = DATA_DIR / "ranking_status.json"
WORKER_LOCK_PATH = DATA_DIR / ".ranking_worker.lock"
WORKER_LOG_PATH = DATA_DIR / "strategy_ranking_worker.log"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _package_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for package in ("qiskit", "qiskit-aer", "qiskit-ibm-runtime", "qiskit-paulice", "numpy"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "not-installed"
    return result


def config_signature(
    level_id: str, difficulty: str, config: dict[str, Any], shots: int
) -> str:
    """Fingerprint everything that affects a stored ranking's comparability."""
    payload = {
        "ranking_engine_version": RANKING_ENGINE_VERSION,
        "scoring_beta": SCORING_BETA,
        "level_id": level_id,
        "difficulty": difficulty,
        "shots": int(shots),
        "config": config,
        "package_versions": _package_versions(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def ranking_key(level_id: str, difficulty: str, signature: str) -> str:
    # Keep old configurations available in the database rather than silently
    # overwriting them when level settings or package versions change.
    return f"{difficulty}::{level_id}::{signature[:16]}"


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


def load_rankings(path: Path | None = None) -> dict[str, Any]:
    source = path or RANKINGS_PATH
    if not source.exists():
        return {"schema_version": 1, "updated_at": None, "rankings": {}}
    try:
        with source.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(f"Could not read ranking database {source}: {exc}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("rankings"), dict):
        raise RuntimeError(f"Ranking database {source} has an invalid structure.")
    value.setdefault("schema_version", 1)
    return value


def save_level_ranking(entry: dict[str, Any], path: Path | None = None) -> None:
    """Atomically persist all strategies for one level/configuration at once."""
    destination = path or RANKINGS_PATH
    db = load_rankings(destination)
    key = ranking_key(entry["level_id"], entry["difficulty"], entry["config_signature"])
    entry = dict(entry)
    entry["cache_key"] = key
    db["rankings"][key] = entry
    db["updated_at"] = utc_now()
    _atomic_write_json(destination, db)


def load_completed_level_ranking(
    level_id: str,
    difficulty: str,
    config: dict[str, Any],
    shots: int,
    path: Path | None = None,
) -> dict[str, Any] | None:
    signature = config_signature(level_id, difficulty, config, shots)
    key = ranking_key(level_id, difficulty, signature)
    entry = load_rankings(path)["rankings"].get(key)
    if not entry or entry.get("rank_status") != "complete":
        return None
    if entry.get("config_signature") != signature:
        return None
    return entry


def enumerate_strategy_sets(eligible_qubits: Iterable[int], maximum_checks: int) -> list[tuple[int, ...]]:
    """Return the no-check baseline plus every allowed canonical qubit subset."""
    eligible = sorted({int(q) for q in eligible_qubits})
    maximum = min(max(0, int(maximum_checks)), len(eligible))
    candidates: list[tuple[int, ...]] = [()]
    for size in range(1, maximum + 1):
        candidates.extend(combinations(eligible, size))
    return candidates


def _safe_result_record(targets: tuple[int, ...], result: dict[str, Any]) -> dict[str, Any]:
    metrics = result.get("metrics", {})
    return {
        "targets": list(targets),
        "num_checks_requested": len(targets),
        "num_checks_used": int(result.get("num_checks_used", 0)),
        "status": "valid",
        "rank": None,
        "score": None if result.get("game_score") is None else float(result["game_score"]),
        "raw_fidelity": float(metrics.get("raw_fidelity", 0.0)),
        "post_selected_fidelity": float(metrics.get("post_selected_fidelity", 0.0)),
        "retention_rate": float(metrics.get("retention_rate", 0.0)),
        "raw_shots": int(metrics.get("raw_shots", 0)),
        "retained_shots": int(metrics.get("retained_shots", 0)),
        "cost": result.get("cost", {}),
        "paulice_cost": result.get("paulice_cost"),
        "check_supports": result.get("check_supports", []),
        "noise_source": result.get("noise_source", "unknown"),
    }


def _rank_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    valid = [row for row in records if row.get("status") == "valid" and row.get("score") is not None]
    valid.sort(key=lambda row: (-float(row["score"]), len(row["targets"]), tuple(row["targets"])))
    for row in valid:
        score = float(row["score"])
        row["rank"] = 1 + sum(float(other["score"]) > score + 1e-9 for other in valid)
    valid_ids = {tuple(row["targets"]) for row in valid}
    invalid = [row for row in records if tuple(row.get("targets", [])) not in valid_ids]
    return valid, invalid


def _result_entry(
    level_id: str,
    title: str,
    difficulty: str,
    cfg: dict[str, Any],
    shots: int,
    signature: str,
    records: list[dict[str, Any]],
    status: str,
    error: str | None = None,
) -> dict[str, Any]:
    valid, _ = _rank_records(records)
    best = valid[0] if valid else None
    sources = sorted({str(row.get("noise_source", "unknown")) for row in records if row.get("noise_source")})
    return {
        "level_id": level_id,
        "level_title": title,
        "difficulty": difficulty,
        "payload_type": str(cfg.get("payload_type", "clifford")).lower(),
        "config_signature": signature,
        "effective_config": cfg,
        "package_versions": _package_versions(),
        "ranking_engine_version": RANKING_ENGINE_VERSION,
        "scoring_beta": SCORING_BETA,
        "simulator_seed": 2026,
        "paulice_seed": 1764,
        "shots_per_strategy": int(shots),
        "rank_status": status,
        "ranked_at": utc_now(),
        "strategies_attempted": len(records),
        "valid_strategy_count": len(valid),
        "invalid_strategy_count": sum(row.get("status") == "invalid" for row in records),
        "analysis_only_count": sum(row.get("status") == "analysis_only" for row in records),
        "noise_sources": sources,
        "best_strategy": None if best is None else {
            "targets": best["targets"], "score": best["score"], "rank": best["rank"],
            "fidelity": best["post_selected_fidelity"], "retention": best["retention_rate"],
        },
        "error": error,
        # Includes all attempted subsets, including unsupported/failed subsets,
        # so a completed level is auditable and doesn't hide failures.
        "strategies": records,
    }


def _job_status_base(difficulty: str, total_levels: int) -> dict[str, Any]:
    return {
        "status": "running",
        "difficulty": difficulty,
        "pid": os.getpid(),
        "started_at": utc_now(),
        "updated_at": utc_now(),
        "finished_at": None,
        "completed_levels": 0,
        "total_levels": total_levels,
        "skipped_cached_levels": 0,
        "current_level": None,
        "current_strategy": None,
        "strategies_done_in_level": 0,
        "strategies_total_in_level": 0,
        "error": None,
    }


def write_job_status(status: dict[str, Any], path: Path | None = None) -> None:
    destination = path or JOB_STATUS_PATH
    updated = dict(status)
    updated["updated_at"] = utc_now()
    _atomic_write_json(destination, updated)


def read_job_status(path: Path | None = None) -> dict[str, Any] | None:
    source = path or JOB_STATUS_PATH
    if not source.exists():
        return None
    try:
        with source.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def process_is_running(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def launch_background_ranking(difficulty: str) -> tuple[str, int | None]:
    """Start/resume the campaign ranker in a detached process, returning immediately."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    existing_status = read_job_status()
    if existing_status and existing_status.get("status") in {"starting", "running"}:
        if process_is_running(int(existing_status.get("pid") or 0)):
            return "already_running", int(existing_status["pid"])

    # A stale lock is safe to remove when its recorded process is gone.
    if WORKER_LOCK_PATH.exists():
        try:
            owner_pid = int(WORKER_LOCK_PATH.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            owner_pid = -1
        if process_is_running(owner_pid):
            return "already_running", owner_pid
        try:
            WORKER_LOCK_PATH.unlink()
        except FileNotFoundError:
            pass

    try:
        fd = os.open(WORKER_LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        return "already_running", None

    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(str(os.getpid()))

    initial_status = {
        "status": "starting", "difficulty": difficulty, "pid": os.getpid(),
        "started_at": utc_now(), "updated_at": utc_now(), "finished_at": None,
        "completed_levels": 0, "total_levels": 10, "skipped_cached_levels": 0,
        "current_level": None, "current_strategy": None,
        "strategies_done_in_level": 0, "strategies_total_in_level": 0, "error": None,
    }
    write_job_status(initial_status)
    log_handle = WORKER_LOG_PATH.open("a", encoding="utf-8", buffering=1)
    command = [sys.executable, "-m", "src.rank_worker", "--difficulty", difficulty]
    try:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    except Exception as exc:
        log_handle.close()
        try:
            WORKER_LOCK_PATH.unlink()
        except FileNotFoundError:
            pass
        failed = dict(initial_status, status="failed", error=f"Could not start worker: {exc}", finished_at=utc_now())
        write_job_status(failed)
        return "failed", None
    finally:
        # The spawned process inherits a duplicated descriptor on POSIX.
        try:
            log_handle.close()
        except Exception:
            pass
    return "started", process.pid


def _update_lock_owner(pid: int) -> None:
    try:
        WORKER_LOCK_PATH.write_text(str(pid), encoding="utf-8")
    except OSError:
        pass


def _remove_lock_if_owned(pid: int) -> None:
    try:
        if int(WORKER_LOCK_PATH.read_text(encoding="utf-8").strip()) == pid:
            WORKER_LOCK_PATH.unlink()
    except (OSError, ValueError):
        pass


def rank_all_levels_and_strategies(
    levels: dict[str, dict[str, Any]],
    difficulty: str,
    shots: int,
    apply_difficulty_fn: Callable[[dict[str, Any], str], dict[str, Any]],
    max_checks_fn: Callable[[dict[str, Any], int], int],
    play_level_fn: Callable[..., dict[str, Any]] | None = None,
    *,
    rankings_file: Path | None = None,
    status_file: Path | None = None,
    expected_noise_source: str | None = None,
) -> dict[str, Any]:
    """Evaluate and persist every legal strategy for every campaign level.

    This function is intentionally callable independently; ``src.rank_worker``
    invokes it in a detached process so normal CLI play remains responsive.
    A fully completed, matching configuration is reused on subsequent runs.
    """
    if play_level_fn is None:
        from src.game import play_level as play_level_fn  # lazy import for unit tests/tools

    data_path = rankings_file or RANKINGS_PATH
    status_path = status_file or JOB_STATUS_PATH
    status = _job_status_base(difficulty, len(levels))
    write_job_status(status, status_path)

    try:
        for level_index, (level_id, base_cfg) in enumerate(levels.items(), start=1):
            cfg = apply_difficulty_fn(base_cfg, difficulty)
            signature = config_signature(level_id, difficulty, cfg, shots)
            key = ranking_key(level_id, difficulty, signature)
            current_db = load_rankings(data_path)
            existing = current_db["rankings"].get(key)
            status["current_level"] = level_id
            status["strategies_done_in_level"] = 0
            status["strategies_total_in_level"] = 0

            cache_noise_matches = (
                expected_noise_source is None
                or expected_noise_source in (existing or {}).get("noise_sources", [])
            )
            if (
                existing
                and existing.get("rank_status") == "complete"
                and existing.get("config_signature") == signature
                and cache_noise_matches
            ):
                status["completed_levels"] = level_index
                status["skipped_cached_levels"] += 1
                status["current_strategy"] = "Using saved completed ranking"
                write_job_status(status, status_path)
                continue

            title = str(cfg.get("title", level_id))
            if str(cfg.get("payload_type", "clifford")).lower() == "tdoped":
                status["strategies_total_in_level"] = 1
                status["current_strategy"] = "T-doped fidelity-only baseline"
                write_job_status(status, status_path)
                try:
                    result = play_level_fn(
                        level_id, num_checks=0, target_qubits=[], shots=shots, config_overrides=cfg
                    )
                    record = _safe_result_record((), result)
                    record["status"] = "analysis_only"
                    record["rank"] = None
                    record["score"] = None  # Official mode deliberately does not score T-doped checks.
                    records = [record]
                    entry = _result_entry(
                        level_id, title, difficulty, cfg, shots, signature, records, "complete",
                        error="T-doped circuits are stored as fidelity-only studies; Paulice check ranking is Clifford-only.",
                    )
                except Exception as exc:
                    entry = _result_entry(
                        level_id, title, difficulty, cfg, shots, signature,
                        [{"targets": [], "num_checks_requested": 0, "status": "invalid", "rank": None,
                          "score": None, "error": f"{type(exc).__name__}: {exc}"}],
                        "failed", error=f"{type(exc).__name__}: {exc}",
                    )
                save_level_ranking(entry, data_path)
                status["completed_levels"] = level_index
                status["strategies_done_in_level"] = 1
                status["current_strategy"] = "Level record saved"
                write_job_status(status, status_path)
                continue

            eligible = [
                q for q in range(int(cfg["num_qubits"]))
                if q not in {int(item) for item in cfg.get("blocked_channels", [])}
            ]
            # Exhaustively rank every subset of unblocked qubits. Mission minimum
            # retention and normal-play check/energy budgets do not exclude candidates.
            candidates = enumerate_strategy_sets(eligible, len(eligible))
            status["strategies_total_in_level"] = len(candidates)
            write_job_status(status, status_path)
            records: list[dict[str, Any]] = []
            noise_sources: set[str] = set()

            for candidate_index, targets in enumerate(candidates, start=1):
                status["strategies_done_in_level"] = candidate_index - 1
                status["current_strategy"] = "NO CHECKS" if not targets else ",".join(map(str, targets))
                write_job_status(status, status_path)
                try:
                    # Ranking explores all unblocked-qubit subsets, including sets
                    # larger than the normal mission check/energy budget. Keep those
                    # limits relaxed only for this ranking run; the saved signature
                    # still represents the original mission configuration.
                    strategy_cfg = dict(cfg)
                    strategy_cfg["max_checks"] = len(eligible)
                    strategy_cfg["energy_budget"] = max(
                        int(strategy_cfg.get("energy_budget", 0)),
                        len(eligible) * max(1, int(strategy_cfg.get("check_cost", 1))),
                    )
                    result = play_level_fn(
                        level_id,
                        num_checks=len(targets),
                        target_qubits=list(targets),
                        shots=shots,
                        config_overrides=strategy_cfg,
                    )
                    record = _safe_result_record(targets, result)
                    if record.get("score") is None:
                        record["status"] = "invalid"
                        record["error"] = "Strategy returned no comparable Clifford game score."
                    else:
                        noise_sources.add(str(record.get("noise_source", "unknown")))
                    records.append(record)
                except Exception as exc:
                    records.append({
                        "targets": list(targets),
                        "num_checks_requested": len(targets),
                        "num_checks_used": 0,
                        "status": "invalid",
                        "rank": None,
                        "score": None,
                        "error": f"{type(exc).__name__}: {str(exc)[:500]}",
                    })
                status["strategies_done_in_level"] = candidate_index
                write_job_status(status, status_path)

            valid, _ = _rank_records(records)
            level_status = "complete" if valid else "failed"
            level_error = None if valid else "No valid check strategy could be scored; inspect worker log and per-strategy errors."
            entry = _result_entry(
                level_id, title, difficulty, cfg, shots, signature, records,
                level_status, error=level_error,
            )
            entry["noise_sources"] = sorted(noise_sources)
            save_level_ranking(entry, data_path)
            status["completed_levels"] = level_index
            status["current_strategy"] = "Level ranking saved" if valid else "Level ranking saved with failures"
            write_job_status(status, status_path)

        status["status"] = "complete"
        status["finished_at"] = utc_now()
        status["current_level"] = None
        status["current_strategy"] = None
        write_job_status(status, status_path)
        return status
    except BaseException as exc:
        status["status"] = "failed"
        status["error"] = f"{type(exc).__name__}: {exc}"
        status["finished_at"] = utc_now()
        write_job_status(status, status_path)
        raise
