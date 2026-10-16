"""Strategy scoring, difficulty presets, campaign progression, and ranking storage.

All reusable ranking logic and persistence live here. The separate
``ranking_worker`` module only orchestrates the background sweep.
"""
from __future__ import annotations

import json
import math
import os
import tempfile
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Iterable

PASS_RELATIVE_SCORE_THRESHOLD = 50.0
RANKING_VERSION = 5
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
    beta = safe_float(base_cfg.get("beta", 2.0), default=float("nan"))
    if not math.isfinite(beta) or beta <= 0 or beta > 1000.0:
        raise ValueError(
            "Each level must define beta in the range (0, 1000] in config/levels.json."
        )
    cfg["beta"] = beta
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
        # Challenge capacity is separate so Standard/Guided historical limits stay.
        # Uses challenge_max_checks / challenge_energy_budget from levels.json when present.
        original_qubits = int(base_cfg.get("num_qubits", 3))
        authored_budget = max(1, int(base_cfg.get("energy_budget", 10**9)))
        challenge_budget = max(
            0,
            int(base_cfg.get("challenge_energy_budget", base_cfg.get("energy_budget", 10**9))),
        )
        cfg.update({
            "max_checks": min(
                original_qubits,
                max(
                    0,
                    int(
                        base_cfg.get(
                            "challenge_max_checks",
                            base_cfg.get("max_checks", original_qubits),
                        )
                    ),
                ),
            ),
            "energy_budget": challenge_budget,
            "description": (
                "Full Challenge: full authored payload with expanded strategy space; "
                "compare fidelity, retention, and check overhead."
            ),
        })
        # The scoring formula divides the resource penalty by energy_budget.
        # Scale its weight alongside the larger Challenge budget so a check has
        # the same absolute score cost at every difficulty; only affordability
        # and circuit size are meant to change with this preset.
        cfg["check_penalty_weight"] = (
            max(0.0, float(base_cfg.get("check_penalty_weight", 0.0)))
            * challenge_budget / authored_budget
        )
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


def optional_float(value: Any) -> float | None:
    """Return a finite float or None; missing metrics must not become zero."""
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def format_targets(targets: list[int] | tuple[int, ...] | None) -> str:
    values = sorted(set(int(q) for q in (targets or [])))
    return "No checks" if not values else "{" + ", ".join(f"q{q}" for q in values) + "}"


def compute_strategy_ranking(
    level_id: str,
    cfg: dict[str, Any],
    difficulty: str,
    shots: int,
    progress_callback: Callable[[int, int, str], None] | None = None,
    strategy_runner: Callable[[str, list[int], dict[str, Any], int], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Rank every budget-valid Clifford strategy against its no-check control.

    ``strategy_runner`` lets the background worker execute each native quantum
    simulation in its own process. That protects the long-running sweep from a
    Qiskit/Aer/Paulice segmentation fault. Direct calls remain available for
    small local diagnostics and tests; the production worker always isolates.
    """
    beta = safe_float(cfg.get("beta", 2.0), default=float("nan"))
    if not math.isfinite(beta) or beta <= 0 or beta > 1000.0:
        raise ValueError("Ranking beta must be finite and in the range (0, 1000].")

    if strategy_runner is None:
        from src.game import play_level

        def strategy_runner(
            _level_id: str,
            targets: list[int],
            _cfg: dict[str, Any],
            shot_count: int,
        ) -> dict[str, Any]:
            return play_level(
                level_id,
                num_checks=len(targets),
                target_qubits=targets,
                shots=int(shot_count),
                config_overrides=dict(cfg),
            )

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
            result = strategy_runner(level_id, list(targets), dict(cfg), int(shots))
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
                "harmonic_score": optional_float(metrics.get("harmonic_score_H")),
                "check_overhead_penalty": optional_float(metrics.get("check_overhead_penalty")),
                "raw_fidelity": optional_float(metrics.get("raw_fidelity")),
                "fidelity": optional_float(metrics.get("post_selected_fidelity")),
                "retention": optional_float(metrics.get("retention_rate")),
                "retained_shots": int(metrics.get("retained_shots", 0) or 0),
                "raw_shots": int(metrics.get("raw_shots", shots) or shots),
                "ancilla_qubits": (
                    int(cost["ancilla_qubits"])
                    if cost.get("ancilla_qubits") is not None else None
                ),
                "two_qubit_gates_before": (
                    int(cost["two_qubit_gates_before"])
                    if cost.get("two_qubit_gates_before") is not None else None
                ),
                "two_qubit_gates_after": (
                    int(cost["two_qubit_gates_after"])
                    if cost.get("two_qubit_gates_after") is not None else None
                ),
                "depth_before": (
                    int(cost["depth_before"])
                    if cost.get("depth_before") is not None else None
                ),
                "depth_after": (
                    int(cost["depth_after"])
                    if cost.get("depth_after") is not None else None
                ),
                "added_two_qubit_gates": int(cost.get("added_two_qubit_gates", 0) or 0),
                "added_depth": int(cost.get("added_depth", 0) or 0),
                "paulice_cost": (
                    None if result.get("paulice_cost") is None
                    else safe_float(result.get("paulice_cost"))
                ),
                "check_supports": result.get("check_supports", []),
                "num_checks_requested": int(
                    result.get("num_checks_requested", len(targets)) or 0
                ),
                "num_checks_used": int(
                    result.get("num_checks_used", len(targets)) or 0
                ),
                "noise_source": str(result.get("noise_source", "unknown")),
                "status": "valid",
                "error": "",
            })
            del result
        except Exception as exc:
            rows.append({
                "targets": list(targets), "strategy": label, "checks": len(targets),
                "score": None, "harmonic_score": None, "check_overhead_penalty": None,
                "raw_fidelity": None, "fidelity": None, "retention": None,
                "retained_shots": None, "raw_shots": None,
                "ancilla_qubits": None, "two_qubit_gates_before": None,
                "two_qubit_gates_after": None, "depth_before": None, "depth_after": None,
                "added_two_qubit_gates": 0, "added_depth": 0,
                "paulice_cost": None, "check_supports": [],
                "num_checks_requested": len(targets), "num_checks_used": 0,
                "noise_source": None,
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
    fidelity_values = [float(row["fidelity"]) for row in valid if row.get("fidelity") is not None]
    retention_values = [float(row["retention"]) for row in valid if row.get("retention") is not None]
    max_fidelity = max(fidelity_values, default=None)
    max_retention = max(retention_values, default=None)
    beats_baseline = 0

    for row in valid:
        row["rank"] = 1 + sum(other["score"] > row["score"] + 1e-9 for other in valid)
        row["relative_score"] = normalize_score(row["score"], scores)
        row["beats_no_check"] = bool(
            baseline_score is not None
            and row["targets"]
            and row["score"] > baseline_score + 1e-9
        )
        beats_baseline += int(row["beats_no_check"])
        row["highest_score"] = high is not None and abs(row["score"] - high) < 1e-9
        row["highest_fidelity"] = (
            row.get("fidelity") is not None
            and max_fidelity is not None
            and abs(row["fidelity"] - max_fidelity) < 1e-9
        )
        row["highest_retention"] = (
            row.get("retention") is not None
            and max_retention is not None
            and abs(row["retention"] - max_retention) < 1e-9
        )
        row["pass_fail"] = classify_relative_score(row["relative_score"])

    return {
        "ranking_version": RANKING_VERSION,
        "level_id": level_id,
        "difficulty": difficulty,
        "shots": int(shots),
        "scoring_beta": beta,
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
    """Write JSON atomically. On Windows, os.replace can fail with Access Denied
    when another process (Streamlit UI, antivirus, indexer) briefly holds the
    destination open. Retry the replace with short backoff before giving up.
    """
    import time

    _ensure_data_dir()
    fd, temp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(DATA_DIR)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        last_error: Exception | None = None
        for attempt in range(12):
            try:
                os.replace(temp_name, path)
                return
            except PermissionError as exc:
                last_error = exc
                time.sleep(min(1.0, 0.04 * (2 ** attempt)))
            except OSError as exc:
                if (
                    getattr(exc, "winerror", None) in {5, 32}
                    or getattr(exc, "errno", None) in {13, 16}
                ):
                    last_error = exc
                    time.sleep(min(1.0, 0.04 * (2 ** attempt)))
                else:
                    raise
        if last_error is not None:
            raise last_error
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


def load_all_rankings() -> dict[str, Any]:
    data = _read_json(RANKINGS_PATH, {"schema_version": RANKING_VERSION, "rankings": {}})
    data.setdefault("rankings", {})
    return data


def _configured_level_beta(level_id: str) -> float | None:
    """Read the current beta directly so edited levels.json invalidates stale rankings."""
    try:
        config = json.loads(
            (PROJECT_ROOT / "config" / "levels.json").read_text(encoding="utf-8")
        )
        beta = float(config[level_id].get("beta", 2.0))
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError, OverflowError):
        return None
    return beta if math.isfinite(beta) and 0 < beta <= 1000.0 else None


def load_ranking(
    level_id: str, difficulty: str, shots: int | None = None
) -> dict[str, Any] | None:
    ranking = load_all_rankings().get("rankings", {}).get(level_id, {}).get(difficulty)
    if not isinstance(ranking, dict):
        return None
    try:
        if int(ranking.get("ranking_version", 0)) != RANKING_VERSION:
            return None
        if shots is not None and int(ranking.get("shots", -1)) != int(shots):
            return None
    except (TypeError, ValueError, OverflowError):
        return None

    current_beta = _configured_level_beta(level_id)
    saved_beta = optional_float(ranking.get("scoring_beta"))
    if current_beta is not None and (
        saved_beta is None
        or not math.isclose(saved_beta, current_beta, rel_tol=0.0, abs_tol=1e-12)
    ):
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
    """Return True if *pid* appears to be a live process (Windows + Linux/macOS)."""
    if not pid or int(pid) <= 0:
        return False
    pid = int(pid)
    if os.name == "nt":
        try:
            import ctypes

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if handle:
                kernel32.CloseHandle(handle)
                return True
            if kernel32.GetLastError() == 5:
                return True
            return False
        except Exception:
            try:
                os.kill(pid, 0)
                return True
            except OSError:
                return False

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False

    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
        if len(fields) > 2 and fields[2] == "Z":
            return False
    except (OSError, IndexError):
        pass
    return True