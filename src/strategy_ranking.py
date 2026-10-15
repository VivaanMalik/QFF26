"""Strategy scoring, difficulty presets, campaign progression, and ranking storage.

All reusable ranking logic and persistence live here. The separate
``ranking_worker`` module only orchestrates the background sweep.
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Iterable

PASS_RELATIVE_SCORE_THRESHOLD = 50.0
RANKING_VERSION = 3
DIFFICULTIES = ("Guided", "Standard", "Full Challenge")
DEFAULT_SHOTS = {"Guided": 2048, "Standard": 2048, "Full Challenge": 2048}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RANKINGS_PATH = DATA_DIR / "strategy_rankings.json"
JOB_STATUS_PATH = DATA_DIR / "strategy_ranking_job.json"
WORKER_LOG_PATH = DATA_DIR / "strategy_ranking_worker.log"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def enumerate_strategy_sets(
    qubits: Iterable[int], maximum_checks: int | None = None
) -> list[tuple[int, ...]]:
    """Return canonical qubit subsets, including the no-check baseline."""
    available = sorted({int(q) for q in qubits})
    maximum = len(available) if maximum_checks is None else min(
        max(0, int(maximum_checks)), len(available)
    )
    candidates: list[tuple[int, ...]] = [()]
    for size in range(1, maximum + 1):
        candidates.extend(combinations(available, size))
    return candidates


def maximum_affordable_checks(config: dict[str, Any], num_qubits: int) -> int:
    """Return the maximum strategy size allowed by a mission's real rules."""
    check_cost = max(1, int(config.get("check_cost", 1)))
    budget = max(0, int(config.get("energy_budget", 10**9)))
    budget_limit = budget // check_cost
    configured_limit = max(0, int(config.get("max_checks", num_qubits)))
    return min(max(0, int(num_qubits)), configured_limit, budget_limit)


def normalize_score(score: float, scores: Iterable[float]) -> float | None:
    """Map a score to 0..100 using the worst and best scores in its comparison."""
    values = [float(value) for value in scores]
    if not values:
        return None
    low, high = min(values), max(values)
    if abs(high - low) < 1e-12:
        return None
    return max(0.0, min(100.0, 100.0 * (float(score) - low) / (high - low)))


def apply_difficulty(base_cfg: dict[str, Any], difficulty: str) -> dict[str, Any]:
    """Apply a campaign size preset without introducing T gates.

    The authored energy budget, check cost, and check penalty are preserved
    across difficulties. That keeps a level's resource economics consistent;
    difficulty changes circuit size and the number of check targets allowed.
    """
    import math

    cfg = dict(base_cfg)
    original_qubits = int(base_cfg.get("num_qubits", 3))
    original_depth = int(base_cfg.get("depth_layers", 2))
    original_check_limit = int(base_cfg.get("max_checks", 1))
    cfg["payload_type"] = "clifford"
    cfg["t_count"] = 0

    if difficulty == "Guided":
        cfg.update({
            "num_qubits": 3,
            "depth_layers": min(3, max(2, math.ceil(original_depth / 12))),
            "max_checks": min(2, original_check_limit),
            "target_score": 45.0,
            "min_retention": 0.10,
            "description": "Guided practice: a short 3-qubit Clifford circuit.",
        })
    elif difficulty == "Standard":
        cfg.update({
            "num_qubits": min(5, original_qubits),
            "depth_layers": min(12, max(3, math.ceil(original_depth * 0.40))),
            "max_checks": min(3, original_check_limit),
            "target_score": 52.0,
            "min_retention": 0.12,
            "description": "Standard practice: a smaller instance of the campaign Clifford payload.",
        })
    elif difficulty == "Full Challenge":
        # Keep the authored mission size and resources unchanged.
        pass
    else:
        raise ValueError(f"Unknown difficulty preset {difficulty!r}.")
    return cfg


def classify_relative_score(relative_score: float | None) -> str:
    """PASS at relative score >= 50; equal-score rankings are reported as TIE."""
    if relative_score is None:
        return "TIE"
    return "PASS" if float(relative_score) >= PASS_RELATIVE_SCORE_THRESHOLD else "FAIL"


def calculate_campaign_progress(
    level_ids: list[str],
    completed_by_difficulty: dict[str, Iterable[str]],
    difficulty: str,
) -> tuple[list[str], set[str], bool]:
    """Compute unlocks using only passes earned on the selected difficulty."""
    completed = set(completed_by_difficulty.get(difficulty, []))
    unlocked: list[str] = []
    for level_id in level_ids:
        unlocked.append(level_id)
        if level_id not in completed:
            break
    all_passed = bool(level_ids) and all(level_id in completed for level_id in level_ids)
    if all_passed:
        unlocked = list(level_ids)
    return unlocked, completed, all_passed


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def format_targets(targets: list[int] | tuple[int, ...] | None) -> str:
    values = sorted(set(int(q) for q in (targets or [])))
    return "No checks" if not values else "{" + ", ".join(f"q{q}" for q in values) + "}"


def compute_strategy_ranking(
    level_id: str,
    cfg: dict[str, Any],
    difficulty: str,
    shots: int,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Simulate every budget-valid Clifford strategy and compare with no-check."""
    from src.game import play_level

    n = int(cfg.get("num_qubits", 3))
    max_checks = maximum_affordable_checks(cfg, n)
    candidates = enumerate_strategy_sets(range(n), max_checks)
    rows: list[dict[str, Any]] = []
    noise_sources: set[str] = set()

    for index, targets in enumerate(candidates, start=1):
        label = format_targets(targets)
        if progress_callback:
            progress_callback(index - 1, len(candidates), label)
        try:
            result = play_level(
                level_id,
                num_checks=len(targets),
                target_qubits=list(targets),
                shots=int(shots),
                config_overrides=dict(cfg),
            )
            metrics = result.get("metrics", {})
            score = result.get("game_score")
            if score is None:
                raise ValueError("The simulator did not return a comparable Clifford score.")
            noise_sources.add(str(result.get("noise_source", "unknown")))
            cost = result.get("cost", {})
            rows.append({
                "targets": list(targets),
                "strategy": label,
                "checks": len(targets),
                "score": float(score),
                "harmonic_score": safe_float(metrics.get("harmonic_score_H")),
                "check_overhead_penalty": safe_float(metrics.get("check_overhead_penalty")),
                "raw_fidelity": safe_float(metrics.get("raw_fidelity")),
                "fidelity": safe_float(metrics.get("post_selected_fidelity")),
                "retention": safe_float(metrics.get("retention_rate")),
                "retained_shots": int(metrics.get("retained_shots", 0) or 0),
                "added_two_qubit_gates": int(cost.get("added_two_qubit_gates", 0) or 0),
                "added_depth": int(cost.get("added_depth", 0) or 0),
                "status": "valid",
                "error": "",
            })
            del result
        except Exception as exc:
            rows.append({
                "targets": list(targets), "strategy": label, "checks": len(targets),
                "score": None, "harmonic_score": None, "check_overhead_penalty": None,
                "raw_fidelity": None, "fidelity": None, "retention": None,
                "retained_shots": 0, "added_two_qubit_gates": 0, "added_depth": 0,
                "status": "failed", "error": f"{type(exc).__name__}: {exc}",
            })
        if index % 8 == 0:
            import gc
            gc.collect()
        if progress_callback:
            progress_callback(index, len(candidates), label)

    valid = [row for row in rows if row["status"] == "valid"]
    baseline = next((row for row in valid if not row["targets"]), None)
    baseline_score = baseline["score"] if baseline else None
    valid.sort(key=lambda row: (-row["score"], row["checks"], tuple(row["targets"])))
    scores = [row["score"] for row in valid]
    low = min(scores) if scores else None
    high = max(scores) if scores else None
    max_fidelity = max((row["fidelity"] for row in valid), default=None)
    max_retention = max((row["retention"] for row in valid), default=None)
    beats_baseline = 0

    for row in valid:
        row["rank"] = 1 + sum(other["score"] > row["score"] + 1e-9 for other in valid)
        row["relative_score"] = normalize_score(row["score"], scores)
        row["beats_no_check"] = bool(
            baseline_score is not None and row["targets"] and row["score"] > baseline_score + 1e-9
        )
        beats_baseline += int(row["beats_no_check"])
        row["highest_score"] = high is not None and abs(row["score"] - high) < 1e-9
        row["highest_fidelity"] = max_fidelity is not None and abs(row["fidelity"] - max_fidelity) < 1e-9
        row["highest_retention"] = max_retention is not None and abs(row["retention"] - max_retention) < 1e-9
        row["pass_fail"] = classify_relative_score(row["relative_score"])

    return {
        "ranking_version": RANKING_VERSION,
        "level_id": level_id,
        "difficulty": difficulty,
        "shots": int(shots),
        "rows": valid,
        "failed_rows": [row for row in rows if row["status"] != "valid"],
        "baseline_score": baseline_score,
        "strategies_beating_no_check": beats_baseline,
        "non_baseline_strategy_count": sum(bool(row["targets"]) for row in valid),
        "worst_score": low,
        "best_score": high,
        "noise_sources": sorted(noise_sources),
        "candidate_count": len(candidates),
        "relative_pass_threshold": PASS_RELATIVE_SCORE_THRESHOLD,
    }


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else (default or {})
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default or {}


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    _ensure_data_dir()
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(DATA_DIR))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


def load_all_rankings() -> dict[str, Any]:
    data = _read_json(RANKINGS_PATH, {"schema_version": RANKING_VERSION, "rankings": {}})
    data.setdefault("rankings", {})
    return data


def load_ranking(level_id: str, difficulty: str, shots: int | None = None) -> dict[str, Any] | None:
    ranking = load_all_rankings().get("rankings", {}).get(level_id, {}).get(difficulty)
    if not isinstance(ranking, dict) or int(ranking.get("ranking_version", 0)) != RANKING_VERSION:
        return None
    if shots is not None and int(ranking.get("shots", -1)) != int(shots):
        return None
    return ranking


def save_ranking(ranking: dict[str, Any]) -> None:
    ranking = dict(ranking)
    ranking.setdefault("ranking_version", RANKING_VERSION)
    data = load_all_rankings()
    rankings = data.setdefault("rankings", {})
    rankings.setdefault(str(ranking["level_id"]), {})[str(ranking["difficulty"])] = ranking
    data["schema_version"] = RANKING_VERSION
    data["updated_at"] = utc_now()
    _atomic_write_json(RANKINGS_PATH, data)


def load_job_status() -> dict[str, Any]:
    return _read_json(JOB_STATUS_PATH, {})


def save_job_status(status: dict[str, Any]) -> None:
    current = load_job_status()
    current.update(status)
    current["updated_at"] = utc_now()
    _atomic_write_json(JOB_STATUS_PATH, current)


def process_is_alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    proc_stat = Path(f"/proc/{int(pid)}/stat")
    try:
        fields = proc_stat.read_text(encoding="utf-8").split()
        if len(fields) > 2 and fields[2] == "Z":
            return False
    except (OSError, IndexError):
        pass
    return True