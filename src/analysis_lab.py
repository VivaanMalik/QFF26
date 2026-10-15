"""Analysis Lab for Qubit Crown.

This module unifies Clifford strategy analysis, individual target-qubit selection,
no-check controls, and T-gate research. Simulations are delegated to the existing
app/game pipeline and start only after an explicit user action. The game backend
remains the source of truth for quantum metrics and scoring.
"""
from __future__ import annotations

import json
import os
import platform
import re
import tempfile
import time
import uuid
from datetime import datetime, timezone
from importlib import metadata
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd
try:
    import streamlit as st
except ImportError:  # pure data helpers and the validation notebook need not import the UI
    class _StreamlitFallback:
        @staticmethod
        def warning(_message: str) -> None:
            return None
    st = _StreamlitFallback()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
T_GATE_RESULTS_PATH = DATA_DIR / "t_gate_research_runs.json"
BASELINE_RESULTS_PATH = DATA_DIR / "baseline_study_runs.json"
SCHEMA_VERSION = 1


def _first(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def _number(value: Any) -> float | None:
    try:
        if value is None:
            return None
        number = float(value)
        if pd.isna(number):
            return None
        return number
    except (TypeError, ValueError, OverflowError):
        return None


def _integer(value: Any) -> int | None:
    number = _number(value)
    return None if number is None else int(number)


def _targets(row: dict[str, Any]) -> list[int]:
    targets = row.get("targets")
    if isinstance(targets, (list, tuple, set)):
        try:
            return sorted({int(q) for q in targets})
        except (TypeError, ValueError):
            return []
    # Legacy/summary records sometimes preserve only a rendered strategy label.
    label = str(row.get("strategy", row.get("label", "")))
    return sorted({int(q) for q in re.findall(r"q(\d+)", label, flags=re.IGNORECASE)})


def _gate_cost(row: dict[str, Any]) -> dict[str, Any]:
    cost = row.get("cost") or row.get("circuit_cost") or {}
    if not isinstance(cost, dict):
        cost = {}
    return cost


def normalize_strategy_row(
    row: dict[str, Any],
    *,
    level_id: str,
    level_title: str,
    difficulty: str,
    requested_shots: int,
    ranking: dict[str, Any],
) -> dict[str, Any]:
    """Normalize the known ranking variants without fabricating missing values."""
    targets = _targets(row)
    row_status = str(row.get("status", "")).lower()
    # Failed/never-run rows preserve the requested check count. They often have
    # num_checks_used=0 because no valid circuit was produced, which must not
    # make a failed 3-check strategy look like a no-check baseline.
    check_fields = ("checks", "num_checks_requested", "num_checks_used") if row_status in {"failed", "error", "not_run"} else ("num_checks_used", "checks", "num_checks_requested")
    checks = _integer(_first(row, *check_fields, default=len(targets)))
    if checks is None:
        checks = len(targets)
    cost = _gate_cost(row)
    total = _integer(_first(row, "raw_shots", "total_shots", default=ranking.get("shots_per_strategy", requested_shots)))
    unavailable_status = row_status in {"failed", "error", "not_run", "pending", "missing"}
    accepted = None if unavailable_status else _integer(_first(row, "retained_shots", "accepted_shots"))
    recorded_rejected = None if unavailable_status else _integer(_first(row, "rejected_shots", "discarded_shots"))
    # Prefer an explicitly recorded rejected count. If the schema has no such
    # field, derive it from total minus accepted and label the derivation.
    # Never infer 0% retention from a crashed/not-yet-run candidate's default 0
    # accepted shots: that is an unavailable measurement, not an observation.
    rejected = recorded_rejected
    rejected_inferred = False
    if rejected is None and not unavailable_status and total is not None and accepted is not None:
        rejected = total - accepted
        rejected_inferred = True
    retention = None if unavailable_status else _number(_first(row, "retention_rate", "retention"))
    if retention is None and not unavailable_status and total and accepted is not None:
        retention = accepted / total

    raw_fidelity = None if unavailable_status else _number(_first(row, "raw_fidelity", "raw_fidelity_rate"))
    # `fidelity` is the ranking UI's selected-fidelity value. Do not reuse it as
    # raw fidelity unless the row is explicitly a no-check baseline.
    post_fidelity = None if unavailable_status else _number(_first(row, "post_selected_fidelity", "selected_fidelity", "fidelity"))
    if raw_fidelity is None and checks == 0 and not unavailable_status:
        raw_fidelity = _number(_first(row, "fidelity", "post_selected_fidelity"))
    score = None if unavailable_status else _number(_first(row, "score", "game_score"))
    harmonic_score = None if unavailable_status else _number(_first(row, "harmonic_score", "harmonic_score_H"))
    # Ranking versions in the project have stored the same harmonic score on
    # both 0..1 and 0..100 scales. Normalize to score points so the integrity
    # check compares like with like against the 0..100 game score.
    if harmonic_score is not None and 0.0 <= harmonic_score <= 1.0:
        harmonic_score *= 100.0

    gates_before = _integer(_first(cost, "two_qubit_gates_before", default=row.get("two_qubit_gates_before")))
    gates_after = _integer(_first(cost, "two_qubit_gates_after", default=row.get("two_qubit_gates_after")))
    depth_before = _integer(_first(cost, "depth_before", default=row.get("depth_before")))
    depth_after = _integer(_first(cost, "depth_after", default=row.get("depth_after")))
    ancillas = _integer(_first(cost, "ancilla_qubits", default=row.get("ancilla_qubits")))
    added_gates = _integer(_first(cost, "added_two_qubit_gates", default=row.get("added_two_qubit_gates")))
    if added_gates is None and gates_before is not None and gates_after is not None:
        added_gates = gates_after - gates_before
    added_depth = _integer(_first(cost, "added_depth", default=row.get("added_depth")))
    if added_depth is None and depth_before is not None and depth_after is not None:
        added_depth = depth_after - depth_before

    effective_config = ranking.get("effective_config") or {}
    strategy_name = _first(row, "strategy", "label", default=("No checks" if checks == 0 else "{" + ", ".join(f"q{q}" for q in targets) + "}"))
    return {
        "level_id": level_id,
        "level_title": level_title,
        "difficulty": difficulty,
        "shots": total,
        "requested_shots": int(requested_shots),
        "strategy": str(strategy_name),
        "targets": targets,
        "targets_label": "No checks" if not targets else "{" + ", ".join(f"q{q}" for q in targets) + "}",
        "checks": checks,
        "is_baseline": checks == 0 and not targets,
        "score": score,
        "harmonic_score": harmonic_score,
        "check_overhead_penalty": _number(_first(row, "check_overhead_penalty", "check_resource_penalty")),
        "raw_fidelity": raw_fidelity,
        "post_selected_fidelity": post_fidelity,
        "retention_rate": retention,
        "accepted_shots": accepted,
        "rejected_shots": rejected,
        "rejected_shots_inferred": rejected_inferred,
        "relative_score": _number(row.get("relative_score")),
        "rank": _integer(row.get("rank")),
        "outcome": _first(row, "pass_fail", "outcome", default=""),
        "status": _first(row, "status", default=ranking.get("rank_status", "saved")),
        "noise_source": _first(row, "noise_source", default=(ranking.get("noise_sources") if isinstance(ranking.get("noise_sources"), list) and ranking.get("noise_sources") else ["unknown"])[0]),
        "ancilla_qubits": ancillas,
        "two_qubit_gates_before": gates_before,
        "two_qubit_gates_after": gates_after,
        "added_two_qubit_gates": added_gates,
        "depth_before": depth_before,
        "depth_after": depth_after,
        "added_depth": added_depth,
        "paulice_cost": _number(_first(row, "paulice_cost", default=ranking.get("paulice_cost"))),
        "check_supports": row.get("check_supports") if isinstance(row.get("check_supports"), list) else [],
        "payload_type": _first(ranking, "payload_type", default=effective_config.get("payload_type", "clifford")),
        "num_qubits": _integer(_first(effective_config, "num_qubits", default=row.get("num_qubits"))),
        "depth_layers": _integer(_first(effective_config, "depth_layers", default=row.get("depth_layers"))),
        "t_count": _integer(_first(effective_config, "t_count", default=row.get("t_count"))),
        "config_signature": ranking.get("config_signature"),
        "ranking_engine_version": ranking.get("ranking_engine_version"),
        "scoring_beta": ranking.get("scoring_beta"),
        "simulator_seed": _first(row, "simulator_seed", default=ranking.get("simulator_seed")),
        "paulice_seed": _first(row, "paulice_seed", default=ranking.get("paulice_seed")),
        "package_versions": ranking.get("package_versions") if isinstance(ranking.get("package_versions"), dict) else {},
    }


def load_clifford_records(
    levels: dict[str, dict[str, Any]],
    difficulties: Iterable[str],
    shot_counts: dict[str, int],
    ranking_loader: Callable[[str, str, int], dict[str, Any] | None],
) -> tuple[pd.DataFrame, list[str]]:
    """Load cached rankings for all supplied levels/difficulties; never simulates."""
    records: list[dict[str, Any]] = []
    issues: list[str] = []
    for level_id, config in levels.items():
        payload_type = str(config.get("payload_type", "clifford")).lower()
        if payload_type == "tdoped":
            continue
        title = str(config.get("title", level_id))
        for difficulty in difficulties:
            shots = int(shot_counts.get(difficulty, 2048))
            try:
                ranking = ranking_loader(level_id, difficulty, shots)
            except Exception as exc:  # saved files can be absent or from an older schema
                issues.append(f"{difficulty} · {title}: {type(exc).__name__}: {exc}")
                continue
            if not isinstance(ranking, dict) or not ranking:
                continue
            rows = ranking.get("rows") or ranking.get("strategies") or []
            failed_rows = ranking.get("failed_rows") or []
            if not isinstance(rows, list) or not isinstance(failed_rows, list):
                issues.append(f"{difficulty} · {title}: ranking rows or failed_rows are not a list.")
                continue
            # Preserve failed candidates in the observatory/table, but keep
            # their missing metrics as None rather than converting them to 0.
            all_rows = [*rows, *failed_rows]
            for row in all_rows:
                if not isinstance(row, dict):
                    continue
                normalized = normalize_strategy_row(
                    row,
                    level_id=level_id,
                    level_title=title,
                    difficulty=difficulty,
                    requested_shots=shots,
                    ranking=ranking,
                )
                # This view is for Clifford experiments only. Do not mix T-doped
                # analysis-only rows into the check-retention plots.
                if str(normalized.get("payload_type", "clifford")).lower() == "tdoped":
                    continue
                records.append(normalized)
    return pd.DataFrame(records), issues


def baseline_records_frame(records: Iterable[dict[str, Any]]) -> pd.DataFrame:
    """Convert saved standalone no-check runs to comparison-compatible rows."""
    normalized: list[dict[str, Any]] = []
    for row in records:
        if not isinstance(row, dict) or str(row.get("status", "complete")).lower() not in {"valid", "complete", "saved", "success"}:
            continue
        shots = _integer(_first(row, "shots_requested", "raw_shots"))
        normalized.append({
            "run_id": row.get("run_id"),
            "timestamp_utc": row.get("timestamp_utc"),
            "level_id": row.get("level_id"),
            "level_title": row.get("level_title", row.get("level_id", "Unknown")),
            "difficulty": row.get("difficulty"),
            "strategy": "No-check baseline (saved run)",
            "targets": [],
            "targets_label": "No checks",
            "checks": 0,
            "shots": shots,
            "requested_shots": shots,
            "raw_shots": _integer(row.get("raw_shots")),
            "accepted_shots": _integer(_first(row, "accepted_shots", "retained_shots")),
            "rejected_shots": _integer(_first(row, "rejected_shots", "discarded_shots")),
            "raw_fidelity": _number(row.get("raw_fidelity")),
            "post_selected_fidelity": _number(_first(row, "post_selected_fidelity", "raw_fidelity")),
            "retention_rate": _number(row.get("retention_rate")),
            "score": _number(_first(row, "score", "game_score")),
            "noise_source": row.get("noise_source", "unknown"),
            "status": str(row.get("status", "complete")),
            "is_baseline": True,
            "config_signature": row.get("config_signature"),
            "num_qubits": _integer(row.get("num_qubits")),
            "depth_layers": _integer(row.get("depth_layers")),
            "circuit_depth": _integer(row.get("circuit_depth")),
            "payload_type": row.get("payload_type", "clifford"),
            "source": "Saved standalone baseline experiment",
        })
    return pd.DataFrame(normalized)


def compute_baseline_comparisons(
    frame: pd.DataFrame,
    standalone_baselines: Iterable[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Return matched comparisons; explicit saved baselines take priority.

    Matching requires the same level, difficulty, shot count, and noise source.
    If the checked strategy records a configuration signature, a baseline with
    a missing or different signature is not treated as equivalent.
    """
    columns = [
        "level_id", "level_title", "difficulty", "strategy", "checks", "shots",
        "raw_fidelity", "post_selected_fidelity", "retention_rate", "score",
        "baseline_raw_fidelity", "baseline_post_selected_fidelity", "baseline_retention_rate",
        "baseline_score", "raw_fidelity_delta_pp", "post_fidelity_delta_pp",
        "retention_delta_pp", "score_delta", "noise_source", "comparison_status",
    ]
    comparison_frame = frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()
    if standalone_baselines is not None:
        explicit = baseline_records_frame(standalone_baselines)
        if not explicit.empty:
            # Put actual standalone measurements first so the control used for
            # a matched pair is an explicit baseline run rather than a ranking
            # sweep's no-check candidate when both are available.
            if comparison_frame.empty:
                comparison_frame = explicit
            else:
                explicit = explicit.dropna(axis=1, how="all")
                comparison_frame = comparison_frame.dropna(axis=1, how="all")
                comparison_frame = pd.concat([explicit, comparison_frame], ignore_index=True, sort=False)
    frame = comparison_frame
    if frame.empty:
        return pd.DataFrame(columns=columns)
    if "is_baseline" not in frame:
        frame["is_baseline"] = False
    baseline_mask = frame["is_baseline"].fillna(False)
    for required_metric in ("raw_fidelity", "post_selected_fidelity", "retention_rate"):
        if required_metric in frame:
            baseline_mask &= pd.to_numeric(frame[required_metric], errors="coerce").notna()
        else:
            baseline_mask &= False
    if "status" in frame:
        usable_statuses = {"valid", "complete", "saved", "success"}
        baseline_mask &= frame["status"].astype(str).str.lower().isin(usable_statuses)
    baselines = frame[baseline_mask]
    comparisons: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        candidates = baselines[
            (baselines["level_id"] == row.get("level_id"))
            & (baselines["difficulty"] == row.get("difficulty"))
            & (baselines["shots"] == row.get("shots"))
            & (baselines["noise_source"] == row.get("noise_source"))
        ]
        signature = row.get("config_signature")
        if signature is not None and not pd.isna(signature) and len(candidates):
            same_signature = candidates[candidates["config_signature"] == signature]
            if len(same_signature):
                candidates = same_signature
            else:
                candidates = candidates.iloc[0:0]
        baseline = candidates.iloc[0].to_dict() if len(candidates) else None
        rec = {
            "level_id": row.get("level_id"), "level_title": row.get("level_title"),
            "difficulty": row.get("difficulty"), "strategy": row.get("strategy"),
            "checks": row.get("checks"), "shots": row.get("shots"),
            "raw_fidelity": row.get("raw_fidelity"), "post_selected_fidelity": row.get("post_selected_fidelity"),
            "retention_rate": row.get("retention_rate"), "score": row.get("score"),
            "noise_source": row.get("noise_source"),
        }
        if baseline is None:
            rec["comparison_status"] = "No matching recorded zero-check row"
            for name in ("baseline_raw_fidelity", "baseline_post_selected_fidelity", "baseline_retention_rate", "baseline_score", "raw_fidelity_delta_pp", "post_fidelity_delta_pp", "retention_delta_pp", "score_delta"):
                rec[name] = None
        else:
            rec["baseline_raw_fidelity"] = baseline.get("raw_fidelity")
            rec["baseline_post_selected_fidelity"] = baseline.get("post_selected_fidelity")
            rec["baseline_retention_rate"] = baseline.get("retention_rate")
            rec["baseline_score"] = baseline.get("score")
            for value_name, base_name, target_name in (
                ("raw_fidelity", "baseline_raw_fidelity", "raw_fidelity_delta_pp"),
                ("post_selected_fidelity", "baseline_post_selected_fidelity", "post_fidelity_delta_pp"),
                ("retention_rate", "baseline_retention_rate", "retention_delta_pp"),
            ):
                value, base = _number(rec.get(value_name)), _number(rec.get(base_name))
                rec[target_name] = None if value is None or base is None else 100 * (value - base)
            score, base_score = _number(rec.get("score")), _number(rec.get("baseline_score"))
            rec["score_delta"] = None if score is None or base_score is None else score - base_score
            rec["comparison_status"] = "Matched control"
        comparisons.append(rec)
    return pd.DataFrame(comparisons, columns=columns)


def load_t_gate_records(path: Path | None = None) -> list[dict[str, Any]]:
    target = path or T_GATE_RESULTS_PATH
    if not target.exists():
        return []
    try:
        content = json.loads(target.read_text(encoding="utf-8"))
        if isinstance(content, dict):
            content = content.get("runs", [])
        if not isinstance(content, list):
            return []
        return [row for row in content if isinstance(row, dict)]
    except (OSError, json.JSONDecodeError):
        return []


@contextmanager
def _exclusive_log_lock(target: Path):
    """Coordinate multiple Streamlit sessions writing the same T-Gate log.

    The lock is separate from the JSON data file, so readers can continue to
    open the data file atomically. A small sidecar lock file is intentionally
    retained: deleting a lock file while another process has it open can create
    two independent lock inodes on POSIX systems.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target.with_name(target.name + ".lock")
    with lock_path.open("a+b") as handle:
        if os.name == "nt":
            import msvcrt
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def save_t_gate_record(record: dict[str, Any], path: Path | None = None) -> None:
    """Persist one complete T-gate run without lost updates from concurrent sessions."""
    target = path or T_GATE_RESULTS_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    with _exclusive_log_lock(target):
        rows = load_t_gate_records(target)
        entry = dict(record)
        entry.setdefault("run_id", uuid.uuid4().hex)
        entry.setdefault("timestamp_utc", datetime.now(timezone.utc).isoformat())
        entry.setdefault("status", "complete")
        known_ids = {str(row.get("run_id")) for row in rows}
        if str(entry["run_id"]) not in known_ids:
            rows.append(entry)
        payload = {"schema_version": SCHEMA_VERSION, "updated_at_utc": datetime.now(timezone.utc).isoformat(), "runs": rows}

        fd, temporary_name = tempfile.mkstemp(prefix=".t_gate_research_runs.", suffix=".tmp", dir=target.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
                handle.flush()
                os.fsync(handle.fileno())
            last_error: PermissionError | None = None
            for attempt in range(5):
                try:
                    os.replace(temporary_name, target)
                    last_error = None
                    break
                except PermissionError as exc:
                    last_error = exc
                    time.sleep(0.08 * (attempt + 1))
            if last_error is not None:
                raise last_error
        finally:
            try:
                if os.path.exists(temporary_name):
                    os.unlink(temporary_name)
            except OSError:
                pass


def persist_t_gate_result(
    *,
    level_id: str,
    level_title: str,
    difficulty: str,
    shots: int,
    config: dict[str, Any],
    t_count: int,
    result: dict[str, Any],
) -> bool:
    """Convert a genuine play_level result to a durable T-gate research row."""
    metrics = result.get("metrics") or {}
    effective = result.get("effective_config") or config
    row = {
        "run_id": uuid.uuid4().hex,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "experiment_type": "tdoped_no_check",
        "level_id": level_id,
        "level_title": level_title,
        "difficulty": difficulty,
        "shots_requested": int(shots),
        "raw_shots": _integer(metrics.get("raw_shots")),
        "accepted_shots": _integer(metrics.get("retained_shots")),
        "retention_rate": _number(metrics.get("retention_rate")),
        "raw_fidelity": _number(metrics.get("raw_fidelity")),
        # The current T-Gate flow uses no checks. Keep this field separate from
        # the raw metric but only populate it if the backend explicitly returns it.
        "post_selected_fidelity": _number(metrics.get("post_selected_fidelity")),
        "t_count": int(t_count),
        "num_qubits": _integer(effective.get("num_qubits")),
        "depth_layers": _integer(effective.get("depth_layers")),
        "circuit_depth": _integer(result.get("circuit_depth")),
        "noise_source": str(result.get("noise_source", "unknown")),
        "payload_type": str(result.get("payload_type", "tdoped")),
        "requested_checks": int(result.get("num_checks_requested", 0) or 0),
        "checks_used": int(result.get("num_checks_used", 0) or 0),
        "package_versions": package_versions(),
        "effective_config": effective,
        "fidelity_measure": "output-distribution fidelity, as implemented by the project",
    }
    try:
        save_t_gate_record(row)
        return True
    except (OSError, TypeError, ValueError) as exc:
        # Preserve the running UI even if the optional research-log write fails.
        st.warning(f"T-Gate result remains in the current session, but could not be saved to {T_GATE_RESULTS_PATH.name}: {exc}")
        return False



def load_baseline_records(path: Path | None = None) -> list[dict[str, Any]]:
    """Load durable no-check baseline runs without mixing them with T-gate rows."""
    target = path or BASELINE_RESULTS_PATH
    if not target.exists():
        return []
    try:
        content = json.loads(target.read_text(encoding="utf-8"))
        if isinstance(content, dict):
            content = content.get("runs", [])
        if not isinstance(content, list):
            return []
        return [row for row in content if isinstance(row, dict)]
    except (OSError, json.JSONDecodeError):
        return []


def save_baseline_record(record: dict[str, Any], path: Path | None = None) -> None:
    """Append a baseline record atomically using the same cross-process lock pattern as T-gate logs."""
    target = path or BASELINE_RESULTS_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    with _exclusive_log_lock(target):
        rows = load_baseline_records(target)
        entry = dict(record)
        entry.setdefault("run_id", uuid.uuid4().hex)
        entry.setdefault("timestamp_utc", datetime.now(timezone.utc).isoformat())
        entry.setdefault("status", "complete")
        if str(entry["run_id"]) not in {str(row.get("run_id")) for row in rows}:
            rows.append(entry)
        payload = {"schema_version": SCHEMA_VERSION, "updated_at_utc": datetime.now(timezone.utc).isoformat(), "runs": rows}
        fd, temporary_name = tempfile.mkstemp(prefix=".baseline_study_runs.", suffix=".tmp", dir=target.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, target)
        finally:
            try:
                if os.path.exists(temporary_name):
                    os.unlink(temporary_name)
            except OSError:
                pass


def persist_baseline_result(
    *, level_id: str, level_title: str, difficulty: str, shots: int,
    config: dict[str, Any], result: dict[str, Any],
) -> bool:
    """Persist a genuine no-check Clifford control run for cross-session analysis."""
    metrics = result.get("metrics") or {}
    effective = result.get("effective_config") or config
    row = {
        "run_id": uuid.uuid4().hex,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "experiment_type": "zero_check_baseline",
        "level_id": level_id,
        "level_title": level_title,
        "difficulty": difficulty,
        "shots_requested": int(shots),
        "raw_shots": _integer(metrics.get("raw_shots")),
        "accepted_shots": _integer(metrics.get("retained_shots")),
        "rejected_shots": _integer(metrics.get("rejected_shots", metrics.get("discarded_shots"))),
        "retention_rate": _number(metrics.get("retention_rate")),
        "raw_fidelity": _number(metrics.get("raw_fidelity")),
        "post_selected_fidelity": _number(metrics.get("post_selected_fidelity")),
        "score": _number(result.get("game_score")),
        "num_qubits": _integer(effective.get("num_qubits")),
        "depth_layers": _integer(effective.get("depth_layers")),
        "circuit_depth": _integer(result.get("circuit_depth")),
        "noise_source": str(result.get("noise_source", "unknown")),
        "payload_type": str(result.get("payload_type", "clifford")),
        "requested_checks": int(result.get("num_checks_requested", 0) or 0),
        "checks_used": int(result.get("num_checks_used", 0) or 0),
        "effective_config": effective,
        "fidelity_measure": "output-distribution fidelity, as implemented by the project",
    }
    try:
        save_baseline_record(row)
        return True
    except (OSError, TypeError, ValueError) as exc:
        st.warning(f"Baseline run completed, but could not be saved to {BASELINE_RESULTS_PATH.name}: {exc}")
        return False


def package_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for package, label in (("qiskit", "qiskit"), ("qiskit-aer", "qiskit-aer"), ("qiskit-paulice", "qiskit-paulice"), ("streamlit", "streamlit"), ("pandas", "pandas")):
        try:
            versions[label] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[label] = "not installed in this environment"
    return versions


def validate_clifford_records(frame: pd.DataFrame) -> list[str]:
    """Lightweight integrity checks for fields present in saved strategy rows."""
    issues: list[str] = []
    if frame.empty:
        return ["No saved Clifford strategy rows are available to validate."]
    for idx, row in frame.iterrows():
        label = f"{row.get('difficulty')} / {row.get('level_id')} / {row.get('strategy')}"
        for field in ("raw_fidelity", "post_selected_fidelity", "retention_rate"):
            val = _number(row.get(field))
            if val is not None and not 0.0 <= val <= 1.0:
                issues.append(f"{label}: {field}={val} lies outside [0, 1].")
        total, accepted, rejected = row.get("shots"), row.get("accepted_shots"), row.get("rejected_shots")
        if _number(total) is not None and _number(accepted) is not None and _number(rejected) is not None:
            if int(total) != int(accepted) + int(rejected):
                issues.append(f"{label}: total shots do not equal accepted + rejected.")
        retention = _number(row.get("retention_rate"))
        if _number(total) not in (None, 0) and _number(accepted) is not None and retention is not None:
            observed = int(accepted) / int(total)
            if abs(observed - retention) > max(1e-6, 1.0 / int(total)):
                issues.append(f"{label}: recorded retention does not match accepted/total shots.")
        if _number(total) is not None and int(total) < 0:
            issues.append(f"{label}: total shots cannot be negative.")
        if _number(accepted) is not None and int(accepted) < 0:
            issues.append(f"{label}: accepted shots cannot be negative.")
        if _number(rejected) is not None and int(rejected) < 0:
            issues.append(f"{label}: rejected shots cannot be negative.")
        harmonic = _number(row.get("harmonic_score"))
        penalty = _number(row.get("check_overhead_penalty"))
        score = _number(row.get("score"))
        # The saved game score in this project is harmonic score minus the
        # check-resource penalty. Only validate when all three were recorded.
        if harmonic is not None and penalty is not None and score is not None:
            expected = harmonic - penalty
            if abs(expected - score) > 0.02:
                issues.append(
                    f"{label}: score {score:.4f} differs from harmonic score minus penalty ({expected:.4f})."
                )
    return issues


def _json_safe(value: Any) -> Any:
    """Convert pandas/numpy values and nested containers to JSON-safe values."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _fmt_pct(value: Any) -> str:
    number = _number(value)
    return "—" if number is None else f"{100 * number:.2f}%"


def _fmt_num(value: Any, digits: int = 2) -> str:
    number = _number(value)
    return "—" if number is None else f"{number:.{digits}f}"


def _render_numbered_level_buttons(
    level_rows: list[tuple[str, str]], *, state_key: str, columns_per_row: int = 3,
) -> str | None:
    """Show levels as numbered buttons, never as a dropdown."""
    if not level_rows:
        return None
    valid_ids = {level_id for level_id, _ in level_rows}
    selected = st.session_state.get(state_key)
    if selected not in valid_ids:
        selected = level_rows[0][0]
        st.session_state[state_key] = selected
    for start in range(0, len(level_rows), columns_per_row):
        row = level_rows[start:start + columns_per_row]
        cols = st.columns(columns_per_row)
        for index, (level_id, title) in enumerate(row, start=start + 1):
            with cols[index - start - 1]:
                if st.button(
                    f"{index} · {title}",
                    key=f"{state_key}::button::{level_id}",
                    type="primary" if selected == level_id else "secondary",
                    width="stretch",
                ):
                    st.session_state[state_key] = level_id
                    st.rerun()
    return str(st.session_state.get(state_key, selected))


def render_analysis_lab(
    *,
    levels: dict[str, dict[str, Any]],
    current_difficulty: str,
    current_shots: int,
    difficulties: Iterable[str],
    canonical_shots: dict[str, int],
    ranking_loader: Callable[[str, str, int], dict[str, Any] | None],
    difficulty_applier: Callable[[dict[str, Any], str], dict[str, Any]] | None = None,
    experiment_renderer: Callable[..., None] | None = None,
) -> None:
    """Render one focused analysis view at a time; simulations require explicit actions."""
    st.markdown(
        '<div class="hero-kicker">MEASURED EVIDENCE / RESEARCH WORKSPACE</div>'
        '<div class="hero-title">Analysis Lab</div>'
        '<div class="hero-copy">One focused view at a time: Clifford strategy trade-offs, no-check controls, or T-gate experiments. Select individual target qubits directly in Clifford Analysis. Opening this page never launches a simulation.</div><br>',
        unsafe_allow_html=True,
    )
    st.caption(
        f"Current simulation setting: {int(current_shots):,} shots. "
        "Shots are noisy measurement samples: more samples reduce statistical fluctuation but take longer."
    )
    difficulty_list = list(difficulties)
    level_ids = sorted(
        [level_id for level_id, cfg in levels.items() if str(cfg.get("payload_type", "clifford")).lower() == "clifford"],
        key=lambda value: (int(str(value).rsplit("_", 1)[-1]) if str(value).rsplit("_", 1)[-1].isdigit() else 999, str(value)),
    )
    with st.spinner("Loading saved strategy results; no quantum runs are started…"):
        clifford, issues = load_clifford_records(levels, difficulty_list, canonical_shots, ranking_loader)

    def _load_t_runs() -> list[dict[str, Any]]:
        rows = [dict(row) for row in load_t_gate_records() if isinstance(row, dict) and row.get("status", "complete") == "complete"]
        persisted = {_t_fingerprint(row) for row in rows}
        seen: set[tuple[Any, ...]] = set()
        for key, value in list(st.session_state.items()):
            if not isinstance(key, str) or not key.startswith("tdoped_results::") or not isinstance(value, list):
                continue
            try:
                _, level_id, run_difficulty, shot_text = key.split("::", 3)
                run_shots = int(shot_text)
            except (ValueError, TypeError):
                continue
            config = levels.get(level_id, {})
            for record in value:
                if not isinstance(record, dict):
                    continue
                row = {
                    "run_id": f"session::{level_id}::{run_difficulty}::{run_shots}::{record.get('t_count')}",
                    "status": "complete", "experiment_type": "tdoped_no_check",
                    "level_id": level_id, "level_title": str(config.get("title", level_id)),
                    "difficulty": run_difficulty, "shots_requested": run_shots,
                    "raw_shots": _integer(record.get("raw_shots")),
                    "raw_fidelity": _number(record.get("fidelity", record.get("raw_fidelity"))),
                    "t_count": _integer(record.get("t_count")),
                    "num_qubits": _integer(config.get("num_qubits")),
                    "depth_layers": _integer(config.get("depth_layers")),
                    "circuit_depth": _integer(record.get("depth")),
                    "noise_source": record.get("noise_source", "unknown"),
                    "source": "Current session (persistence fallback)",
                }
                fingerprint = _t_fingerprint(row)
                if fingerprint not in persisted and fingerprint not in seen:
                    rows.append(row)
                    seen.add(fingerprint)
        return rows

    def _t_fingerprint(row: dict[str, Any]) -> tuple[Any, ...]:
        return (
            row.get("level_id"), row.get("difficulty"),
            row.get("shots_requested", row.get("raw_shots")), row.get("t_count"),
            row.get("raw_fidelity", row.get("fidelity")),
            row.get("circuit_depth", row.get("depth")), row.get("noise_source"),
        )

    def _load_baseline_runs() -> list[dict[str, Any]]:
        return [dict(row) for row in load_baseline_records() if isinstance(row, dict) and row.get("status", "complete") == "complete"]

    modes = ["Clifford Analysis", "No-check Baseline", "T-Gate Sweep"]
    navigation_col, content_col = st.columns([1.25, 4.0])
    with navigation_col:
        st.markdown("#### Analysis views")
        st.caption("Choose one section. The buttons below do not run simulations.")
        mode_key = "analysis_lab_mode"
        if st.session_state.get(mode_key) not in modes:
            st.session_state[mode_key] = modes[0]
        for index, mode_name in enumerate(modes, start=1):
            if st.button(
                f"{index} · {mode_name}",
                key=f"analysis-mode-button::{mode_name}",
                type="primary" if st.session_state[mode_key] == mode_name else "secondary",
                width="stretch",
            ):
                st.session_state[mode_key] = mode_name
                st.rerun()
        st.divider()
        st.markdown("#### Difficulty")
        st.caption("Changes circuit size, depth and check capacity. Campaign unlocks are not changed.")
        difficulty_key = "analysis_lab_difficulty"
        if st.session_state.get(difficulty_key) not in difficulty_list:
            st.session_state[difficulty_key] = current_difficulty if current_difficulty in difficulty_list else difficulty_list[0]
        for index, value in enumerate(difficulty_list, start=1):
            if st.button(
                f"{index} · {value}",
                key=f"analysis-difficulty-button::{value}",
                type="primary" if st.session_state[difficulty_key] == value else "secondary",
                width="stretch",
            ):
                st.session_state[difficulty_key] = value
                st.rerun()
        chosen_difficulty = str(st.session_state[difficulty_key])
        st.caption({
            "Guided": "Shorter circuit and smaller strategy space for experimentation.",
            "Standard": "Intermediate circuit size and check budget.",
            "Full Challenge": "Largest configured circuit and the full affordable target-subset space.",
        }.get(chosen_difficulty, "Difficulty preset for the next explicit run."))

    with content_col:
        mode = str(st.session_state.get("analysis_lab_mode", modes[0]))

        if mode == "Clifford Analysis":
            st.subheader("Clifford Analysis")
            st.caption("Choose a level, run a selected strategy if needed, then compare fidelity, retained shots and circuit overhead against saved results.")
            level_rows = [(level_id, str(levels[level_id].get("title", level_id))) for level_id in level_ids]
            selected_level = _render_numbered_level_buttons(
                level_rows, state_key="analysis_clifford_level", columns_per_row=2,
            )
            if selected_level is None:
                st.info("No Clifford levels are configured.")
                return
            cfg = difficulty_applier(levels[selected_level], chosen_difficulty) if difficulty_applier else dict(levels[selected_level])
            n = _integer(cfg.get("num_qubits")) or 0
            st.caption(
                f"Level controls the payload circuit ({n} data qubits, {_integer(cfg.get('depth_layers')) or 0} layers). "
                "Target buttons below choose qubits for the requested checks; the backend builds the actual check circuit."
            )
            if experiment_renderer is not None:
                experiment_renderer("Clifford Analysis", selected_level, chosen_difficulty)
            st.divider()

            selected_level_rows = clifford[
                (clifford["level_id"] == selected_level)
                & (clifford["difficulty"] == chosen_difficulty)
            ].copy() if not clifford.empty else pd.DataFrame()

            # Include a freshly executed single strategy in the graphs only when
            # its shot count matches the ranking sweep's shot count. Otherwise
            # keep the independent run visible in its after-action report but
            # do not mix unlike sampling configurations into a comparison plot.
            latest_run = st.session_state.get("latest_strategy_run")
            if (
                isinstance(latest_run, dict)
                and latest_run.get("research_mode")
                and latest_run.get("level_id") == selected_level
                and latest_run.get("difficulty") == chosen_difficulty
            ):
                latest_shots = _integer(latest_run.get("shots"))
                canonical_shots_for_view = int(canonical_shots.get(chosen_difficulty, current_shots))
                if latest_shots == canonical_shots_for_view:
                    latest_result = latest_run.get("result", {})
                    latest_metrics = latest_result.get("metrics") if isinstance(latest_result.get("metrics"), dict) else {}
                    latest_targets = sorted({int(q) for q in (latest_run.get("targets") or [])})
                    latest_label = "No checks" if not latest_targets else "{" + ", ".join(f"q{q}" for q in latest_targets) + "}"
                    latest_raw_row = {
                        "targets": latest_targets,
                        "strategy": latest_label,
                        "checks": len(latest_targets),
                        "score": latest_result.get("game_score"),
                        "raw_fidelity": latest_metrics.get("raw_fidelity"),
                        "fidelity": latest_metrics.get("post_selected_fidelity"),
                        "retention": latest_metrics.get("retention_rate"),
                        "retained_shots": latest_metrics.get("retained_shots"),
                        "raw_shots": latest_metrics.get("raw_shots", latest_shots),
                        "noise_source": latest_result.get("noise_source", "unknown"),
                        "cost": latest_result.get("cost", {}),
                        "check_supports": latest_result.get("check_supports", []),
                        "status": "valid" if _number(latest_result.get("game_score")) is not None else "failed",
                        "error": "" if _number(latest_result.get("game_score")) is not None else "N/A — run strategy check to retry.",
                        "relative_score": None,
                        "rank": None,
                    }
                    latest_ranking = ranking_loader(selected_level, chosen_difficulty, canonical_shots_for_view) or {}
                    latest_row = normalize_strategy_row(
                        latest_raw_row,
                        level_id=selected_level,
                        level_title=str(levels[selected_level].get("title", selected_level)),
                        difficulty=chosen_difficulty,
                        requested_shots=latest_shots,
                        ranking=latest_ranking,
                    )
                    if not selected_level_rows.empty:
                        same_subset = selected_level_rows["targets"].map(lambda value: sorted(value or []) == latest_targets)
                        selected_level_rows = selected_level_rows.loc[~same_subset].copy()
                    selected_level_rows = pd.concat([selected_level_rows, pd.DataFrame([latest_row])], ignore_index=True)
                elif latest_shots is not None:
                    st.caption(
                        f"Latest single run used {latest_shots:,} shots, whereas the saved ranking uses "
                        f"{canonical_shots_for_view:,}. The run remains in its report and is not mixed into the ranking graphs."
                    )

            valid = selected_level_rows[
                (~selected_level_rows["status"].astype(str).str.lower().isin(["failed", "error", "not_run"]))
                & selected_level_rows["score"].notna()
            ].copy() if not selected_level_rows.empty else pd.DataFrame()
            if valid.empty:
                st.info("No completed ranking data is saved for this level and difficulty yet. Run the background Strategy Rankings sweep, or use the strategy runner above for a single check.")
            else:
                fidelity_values = pd.to_numeric(valid["post_selected_fidelity"], errors="coerce").dropna()
                retention_values = pd.to_numeric(valid["retention_rate"], errors="coerce").dropna()
                score_values = pd.to_numeric(valid["score"], errors="coerce").dropna()
                a, b, c, d = st.columns(4)
                a.metric("Completed strategies", f"{len(valid):,}")
                b.metric("Best post-selected fidelity", _fmt_pct(fidelity_values.max() if len(fidelity_values) else None))
                c.metric("Best retention", _fmt_pct(retention_values.max() if len(retention_values) else None))
                d.metric("Best score", _fmt_num(score_values.max()) + ("/100" if len(score_values) else ""))

                tradeoff = valid.dropna(subset=["retention_rate", "post_selected_fidelity"]).copy()
                if not tradeoff.empty:
                    tradeoff["Shots retained (%)"] = pd.to_numeric(tradeoff["retention_rate"], errors="coerce") * 100
                    tradeoff["Post-selected fidelity (%)"] = pd.to_numeric(tradeoff["post_selected_fidelity"], errors="coerce") * 100
                    tradeoff["Check group"] = pd.to_numeric(tradeoff["checks"], errors="coerce").fillna(0).astype(int).map(
                        lambda count: "0 · no checks" if count == 0 else f"{count} check" + ("s" if count != 1 else "")
                    )
                    tradeoff["Marker size"] = pd.to_numeric(tradeoff["checks"], errors="coerce").fillna(0).astype(int) + 1
                    st.markdown("#### 1 · Fidelity versus shots kept")
                    st.scatter_chart(
                        tradeoff, x="Shots retained (%)", y="Post-selected fidelity (%)",
                        color="Check group", size="Marker size", height=350,
                    )
                    st.caption("Top-right is the preferred region: higher conditional fidelity with more shots kept. Each point is one saved target subset; colour and marker size show check count.")
                else:
                    st.info("N/A — no completed strategy in this selection has both fidelity and retention. Run strategy check to create a fresh result.")

                comparisons = compute_baseline_comparisons(selected_level_rows)
                if not comparisons.empty:
                    matched = comparisons[
                        (comparisons["comparison_status"] == "Matched control")
                        & (pd.to_numeric(comparisons["checks"], errors="coerce") > 0)
                    ].dropna(subset=["post_fidelity_delta_pp", "retention_delta_pp"]).copy()
                else:
                    matched = pd.DataFrame()
                st.markdown("#### 2 · Change versus the no-check control")
                if matched.empty:
                    st.info("N/A — no checked strategies have a matched no-check control with the same level, difficulty, shots, noise and configuration. Run the ranking sweep to generate comparable data.")
                else:
                    matched["Fidelity change (percentage points)"] = matched["post_fidelity_delta_pp"]
                    matched["Retention change (percentage points)"] = matched["retention_delta_pp"]
                    matched["Check group"] = matched["checks"].map(lambda count: f"{int(count)} check" + ("s" if int(count) != 1 else ""))
                    st.scatter_chart(
                        matched, x="Retention change (percentage points)",
                        y="Fidelity change (percentage points)", color="Check group", height=330,
                    )
                    st.caption("Above zero means fidelity improved over no-check; left of zero means fewer shots were retained. This graph uses matched saved controls, not estimated values.")

                with st.expander("Additional analysis graphs", expanded=False):
                    # All graphs below are scoped to the selected level/difficulty and
                    # use saved valid runs only. Missing measurements remain NaN.
                    grouped_source = valid.copy()
                    for field in ("checks", "score", "raw_fidelity", "post_selected_fidelity", "added_two_qubit_gates", "added_depth"):
                        if field not in grouped_source.columns:
                            grouped_source[field] = None
                        grouped_source[field] = pd.to_numeric(grouped_source[field], errors="coerce")
                    grouped_source = grouped_source.dropna(subset=["checks"])

                    st.markdown("#### Score versus selected target-qubit count")
                    score_source = grouped_source.dropna(subset=["score"])
                    if not score_source.empty:
                        score_curve = score_source.groupby("checks")["score"].agg(
                            **{"Mean score": "mean", "Best observed score": "max"}
                        ).sort_index()
                        score_curve.index.name = "Selected target qubits"
                        st.line_chart(score_curve, height=270)
                        st.caption("Scores are out of 100. Each point summarizes strategies with the same number of selected target qubits for this level and difficulty; best observed scores are descriptive, not a guarantee of future performance.")
                    else:
                        st.info("N/A — no saved strategy scores are available for this selection.")

                    st.markdown("#### Raw versus post-selected fidelity")
                    fidelity_source = grouped_source.dropna(subset=["raw_fidelity", "post_selected_fidelity"], how="all")
                    if not fidelity_source.empty:
                        fidelity_curve = fidelity_source.groupby("checks").agg(
                            **{
                                "Mean raw fidelity (%)": ("raw_fidelity", lambda col: col.mean() * 100),
                                "Mean post-selected fidelity (%)": ("post_selected_fidelity", lambda col: col.mean() * 100),
                            }
                        ).sort_index()
                        fidelity_curve.index.name = "Selected target qubits"
                        st.line_chart(fidelity_curve, height=270)
                        st.caption("Raw fidelity uses all measured data shots; post-selected fidelity uses accepted shots only. Read this alongside the shot-retention graph above.")
                    else:
                        st.info("N/A — fidelity values are not recorded for this selection.")

                    st.markdown("#### Added two-qubit gates")
                    gate_source = grouped_source.dropna(subset=["added_two_qubit_gates"])
                    if not gate_source.empty:
                        gate_curve = gate_source.groupby("checks")["added_two_qubit_gates"].mean().sort_index().to_frame("Mean added two-qubit gates")
                        gate_curve.index.name = "Selected target qubits"
                        st.bar_chart(gate_curve, height=240)
                        st.caption("Mean extra two-qubit gates relative to the payload circuit. Missing gate counts are excluded rather than treated as zero.")
                    else:
                        st.info("N/A — added two-qubit gate counts are not recorded for these strategies.")

                    st.markdown("#### Added circuit depth")
                    depth_source = grouped_source.dropna(subset=["added_depth"])
                    if not depth_source.empty:
                        depth_curve = depth_source.groupby("checks")["added_depth"].mean().sort_index().to_frame("Mean added circuit depth")
                        depth_curve.index.name = "Selected target qubits"
                        st.bar_chart(depth_curve, height=240)
                        st.caption("Mean depth increase relative to the payload circuit. Depth and gate count are shown separately because they measure different costs.")
                    else:
                        st.info("N/A — comparable before/after depth is not recorded for these strategies.")

                with st.expander("Strategy-level data (including unavailable candidates)", expanded=False):
                    visible = selected_level_rows.copy()
                    visible["Score"] = visible["score"].map(lambda value: _fmt_num(value) if _number(value) is not None else "N/A — run strategy check")
                    for source, target in (("raw_fidelity", "Raw fidelity (%)"), ("post_selected_fidelity", "Post-selected fidelity (%)"), ("retention_rate", "Retention (%)")):
                        visible[target] = visible[source].map(lambda value: _fmt_pct(value) if _number(value) is not None else "N/A — run strategy check")
                    show_columns = [col for col in ("targets_label", "checks", "rank", "Score", "Raw fidelity (%)", "Post-selected fidelity (%)", "Retention (%)", "accepted_shots", "rejected_shots", "added_two_qubit_gates", "added_depth", "status", "error") if col in visible]
                    visible = visible[show_columns].rename(columns={
                        "targets_label": "Strategy", "checks": "Checks", "rank": "Rank",
                        "accepted_shots": "Accepted shots", "rejected_shots": "Rejected shots",
                        "added_two_qubit_gates": "Extra 2-qubit gates", "added_depth": "Extra depth",
                        "status": "Status", "error": "Why unavailable / error",
                    })
                    st.dataframe(visible, hide_index=True, width="stretch")
                    st.download_button(
                        "Download selected level data (CSV)",
                        data=visible.to_csv(index=False).encode("utf-8"),
                        file_name="qubit_crown_clifford_analysis.csv", mime="text/csv", width="stretch",
                    )
                validation = validate_clifford_records(selected_level_rows) if not selected_level_rows.empty else []
                if validation:
                    with st.expander(f"Data checks ({len(validation)} warnings)", expanded=False):
                        st.dataframe(pd.DataFrame({"Check": validation}), hide_index=True, width="stretch")

        elif mode == "No-check Baseline":
            st.subheader("No-check baseline and control comparison")
            st.caption("The baseline runs each Clifford level with zero checks. It is the control condition: no syndrome filtering should occur, so retention is expected to be 100%.")
            if experiment_renderer is not None:
                experiment_renderer("No-check Baseline", None, chosen_difficulty)
            st.divider()
            baseline_runs = _load_baseline_runs()
            baseline_frame = pd.DataFrame(baseline_runs)
            if not baseline_frame.empty:
                if "raw_fidelity" not in baseline_frame:
                    baseline_frame["raw_fidelity"] = None
                baseline_frame["raw_fidelity"] = pd.to_numeric(baseline_frame["raw_fidelity"], errors="coerce")
                if "difficulty" not in baseline_frame:
                    baseline_frame["difficulty"] = "Unknown"
                baseline_frame = baseline_frame[baseline_frame["difficulty"].astype(str) == chosen_difficulty].copy()
                baseline_frame = baseline_frame.dropna(subset=["raw_fidelity"])
            if baseline_frame.empty:
                st.info("No saved no-check runs for this difficulty yet. Use the button above to run the control study.")
            else:
                baseline_frame["fidelity_pct"] = baseline_frame["raw_fidelity"] * 100
                baseline_frame["level_label"] = baseline_frame.apply(lambda row: str(row.get("level_title", row.get("level_id", "Unknown"))), axis=1)
                baseline_frame["shot_label"] = baseline_frame.get("shots_requested", pd.Series([None] * len(baseline_frame), index=baseline_frame.index)).fillna("Unknown").astype(str)
                baseline_frame["noise_label"] = baseline_frame.get("noise_source", pd.Series(["unknown"] * len(baseline_frame), index=baseline_frame.index)).fillna("unknown").astype(str)
                baseline_frame["Run configuration"] = baseline_frame.apply(
                    lambda row: f"{row['level_label']} · {row['shot_label']} shots · {row['noise_label']}", axis=1,
                )
                summary = baseline_frame.groupby("Run configuration", as_index=True)["fidelity_pct"].mean().sort_index().to_frame("No-check fidelity (%)")
                st.markdown("#### No-check fidelity by recorded configuration")
                st.bar_chart(summary, height=320)
                st.caption("Each bar is its own level/shot/noise configuration. Different configurations are not blended together.")
                with st.expander("Saved no-check run details", expanded=False):
                    columns = [name for name in ("timestamp_utc", "level_title", "level_id", "difficulty", "shots_requested", "raw_shots", "raw_fidelity", "retention_rate", "noise_source", "circuit_depth") if name in baseline_frame.columns]
                    view = baseline_frame[columns].copy()
                    for field in ("raw_fidelity", "retention_rate"):
                        if field in view:
                            view[field] = view[field].map(lambda value: _fmt_pct(value) if _number(value) is not None else "N/A")
                    st.dataframe(view, hide_index=True, width="stretch")
                    st.download_button("Download no-check results (CSV)", data=view.to_csv(index=False).encode("utf-8"), file_name="qubit_crown_baseline_results.csv", mime="text/csv", width="stretch")

            st.markdown("#### Checked strategies versus matched no-check controls")
            comparison_source = clifford[clifford["difficulty"] == chosen_difficulty].copy() if not clifford.empty else pd.DataFrame()
            baseline_comparison = compute_baseline_comparisons(comparison_source, standalone_baselines=baseline_runs)
            if not baseline_comparison.empty:
                matched = baseline_comparison[
                    (baseline_comparison["comparison_status"] == "Matched control")
                    & (pd.to_numeric(baseline_comparison["checks"], errors="coerce") > 0)
                ].dropna(subset=["post_fidelity_delta_pp", "retention_delta_pp"]).copy()
            else:
                matched = pd.DataFrame()
            if matched.empty:
                st.info("N/A — no checked strategies have matching no-check controls yet. Run Strategy Rankings to generate comparable pairs.")
            else:
                matched["Fidelity change (percentage points)"] = matched["post_fidelity_delta_pp"]
                matched["Retention change (percentage points)"] = matched["retention_delta_pp"]
                matched["Level"] = matched["level_title"].astype(str)
                st.scatter_chart(matched, x="Retention change (percentage points)", y="Fidelity change (percentage points)", color="Level", height=340)
                st.caption("Only same-level, same-difficulty, same-shot, same-noise controls are plotted. Post-selected fidelity gain is measured in percentage points.")
                with st.expander("Matched comparison rows", expanded=False):
                    show = matched[[col for col in ("level_title", "strategy", "checks", "shots", "post_fidelity_delta_pp", "retention_delta_pp", "score_delta", "noise_source") if col in matched.columns]].rename(columns={
                        "level_title": "Level", "strategy": "Strategy", "checks": "Checks", "shots": "Shots",
                        "post_fidelity_delta_pp": "Fidelity change (pp)", "retention_delta_pp": "Retention change (pp)",
                        "score_delta": "Score change",
                    })
                    st.dataframe(show, hide_index=True, width="stretch")

        elif mode == "T-Gate Sweep":
            st.subheader("T-gate sweep and analysis")
            st.caption("This is a separate no-check experiment on T-doped circuits. T=0 is the study control; these measurements do not enter Clifford strategy rankings or campaign unlocks.")
            if experiment_renderer is not None:
                experiment_renderer("T-Gate Sweep", None, chosen_difficulty)
            # Reload after the run controls so a just-completed sweep appears
            # in its graph without requiring the user to leave this section.
            t_runs = _load_t_runs()
            t_frame = pd.DataFrame(t_runs)
            if not t_frame.empty:
                if "raw_fidelity" not in t_frame:
                    t_frame["raw_fidelity"] = t_frame.get("fidelity")
                if "t_count" not in t_frame:
                    t_frame["t_count"] = None
                t_frame["raw_fidelity"] = pd.to_numeric(t_frame["raw_fidelity"], errors="coerce")
                t_frame["t_count"] = pd.to_numeric(t_frame["t_count"], errors="coerce")
                t_frame = t_frame.dropna(subset=["raw_fidelity", "t_count"])
                t_frame = t_frame[t_frame["difficulty"].astype(str) == chosen_difficulty].copy() if "difficulty" in t_frame else t_frame
            if t_frame.empty:
                st.info("No saved T-gate runs for this difficulty yet. Use Run selected T count or Run T-count sweep above.")
            else:
                key_fields = ["level_id", "difficulty", "shots_requested", "num_qubits", "depth_layers", "noise_source"]
                for field in key_fields:
                    if field not in t_frame.columns:
                        t_frame[field] = "Unknown"
                    t_frame[field] = t_frame[field].where(t_frame[field].notna(), "Unknown")
                t_frame["study_key"] = t_frame[key_fields].astype(str).agg(" | ".join, axis=1)
                study_rows: list[tuple[str, str]] = []
                for study_key in list(dict.fromkeys(t_frame["study_key"].tolist())):
                    sample = t_frame[t_frame["study_key"] == study_key].iloc[0]
                    title = str(sample.get("level_title", sample.get("level_id", "Unknown")))
                    label = f"{title} · {sample.get('difficulty', 'Unknown')} · {sample.get('shots_requested', 'Unknown')} shots · {sample.get('num_qubits', '?')}q × {sample.get('depth_layers', '?')} layers · {sample.get('noise_source', 'unknown')}"
                    study_rows.append((study_key, label))
                st.markdown("#### Saved study configuration")
                st.caption("Select one exact level, difficulty, shot count, circuit size/depth and noise source. Only matching runs share a curve.")
                study_key = _render_numbered_level_buttons(study_rows, state_key="analysis_tgate_study", columns_per_row=1)
                if study_key is not None:
                    selected_t = t_frame[t_frame["study_key"] == study_key].copy()
                    selected_t["T gates"] = pd.to_numeric(selected_t["t_count"], errors="coerce")
                    selected_t["Noisy fidelity (%)"] = selected_t["raw_fidelity"] * 100
                    mean_curve = selected_t.groupby("T gates", as_index=True)["Noisy fidelity (%)"].mean().sort_index().to_frame()
                    st.markdown("#### Noisy output fidelity versus T-gate count")
                    st.line_chart(mean_curve, height=330)
                    st.caption("Fidelity is calculated against each circuit's own ideal reference. T=0 is the control for the selected circuit family; this plot is not a universal law about T gates.")
                    t0 = selected_t[selected_t["T gates"] == 0]["Noisy fidelity (%)"]
                    tmax = selected_t[selected_t["T gates"] == selected_t["T gates"].max()]["Noisy fidelity (%)"]
                    if len(t0) and len(tmax):
                        change = float(tmax.mean() - t0.mean())
                        st.metric("High-count minus T=0 fidelity", f"{change:+.2f} percentage points")
                    with st.expander("T-gate run details", expanded=False):
                        columns = [col for col in ("timestamp_utc", "level_title", "t_count", "raw_fidelity", "shots_requested", "raw_shots", "num_qubits", "depth_layers", "circuit_depth", "noise_source", "source") if col in selected_t.columns]
                        view = selected_t[columns].copy().rename(columns={
                            "timestamp_utc": "Run time", "level_title": "Level", "t_count": "T gates",
                            "raw_fidelity": "Noisy fidelity", "shots_requested": "Requested shots",
                            "raw_shots": "Actual shots", "num_qubits": "Qubits", "depth_layers": "Layers",
                            "circuit_depth": "Circuit depth", "noise_source": "Noise source", "source": "Source",
                        })
                        if "Noisy fidelity" in view:
                            view["Noisy fidelity"] = view["Noisy fidelity"].map(lambda value: _fmt_pct(value / 100) if _number(value) is not None else "N/A")
                        st.dataframe(view, hide_index=True, width="stretch")
                        st.download_button("Download selected T-gate study (CSV)", data=view.to_csv(index=False).encode("utf-8"), file_name="qubit_crown_t_gate_results.csv", mime="text/csv", width="stretch")


    if issues:
        with st.expander(f"Saved ranking load warnings ({len(issues)})", expanded=False):
            st.dataframe(pd.DataFrame({"Warning": issues}), hide_index=True, width="stretch")
