"""Mission configuration and execution using official spacetime-code checks."""
from __future__ import annotations

import json
from functools import lru_cache
import math
from pathlib import Path
from typing import Any


@lru_cache(maxsize=64)
def _cached_ideal_distribution(
    payload_type: str, num_qubits: int, depth_layers: int, reference_shots: int
) -> dict[str, float]:
    """Reuse the identical ideal reference across alternative check strategies."""
    normalized_type = payload_type.lower()
    if normalized_type == "tdoped":
        # The T count is not needed here because this helper is only used for
        # shared level references; T-doped calls use the dedicated helper below.
        raise ValueError("T-doped reference caching requires an explicit T count.")
    # Keep Qiskit/Aer imports inside the isolated simulation process.
    from src.circuits import get_clifford_payload
    from src.simulator import run_ideal_simulation

    payload = get_clifford_payload(num_qubits, depth_layers)
    return run_ideal_simulation(
        payload, shots=reference_shots, payload_type=normalized_type
    )


def load_level_configs(config_path="config/levels.json") -> dict[str, dict[str, Any]]:
    path = Path(config_path)
    if not path.exists():
        path = Path(__file__).resolve().parent.parent / "config" / "levels.json"
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def generate_feedback_message(fidelity, retention, score_100h):
    if retention < 0.15:
        return f"Very selective: only {retention * 100:.1f}% of shots survived. Try fewer checks."
    if fidelity < 0.35:
        return f"The output remains noisy ({fidelity * 100:.1f}% post-selected fidelity). Try a different target-qubit set."
    if score_100h >= 80:
        return f"Strong balance: {fidelity * 100:.1f}% fidelity with {retention * 100:.1f}% retention."
    return f"Trade-off score: {score_100h:.1f}/100. Compare against the no-check baseline."


def _check_spacetime_timeline(circuit, ancilla_indices: list[int]) -> list[dict[str, Any]]:
    """Summarize where each Paulice ancilla interacts across the payload timeline.

    The circuit instruction index is used internally to derive a simple payload
    operation range; this summary is explanatory, not a substitute for the full
    circuit diagram.
    """
    result = []
    position_by_ancilla = {int(anc): [] for anc in ancilla_indices}
    payload_ops_seen = 0
    for instruction in circuit.data:
        qindices = [circuit.find_bit(qubit).index for qubit in instruction.qubits]
        touching = [anc for anc in position_by_ancilla if anc in qindices]
        if touching and instruction.operation.num_qubits == 2:
            for anc in touching:
                position_by_ancilla[anc].append(payload_ops_seen)
        elif not touching and instruction.operation.name not in {"barrier", "measure"}:
            payload_ops_seen += 1
    for ancilla, positions in position_by_ancilla.items():
        result.append({
            "ancilla_qubit": ancilla,
            "payload_positions": positions,
            "interaction_count": len(positions),
            "first_payload_position": min(positions) if positions else None,
            "last_payload_position": max(positions) if positions else None,
        })
    return result


def _play_level_impl(
    level_id: str,
    num_checks: int = 0,
    target_qubits: list[int] | None = None,
    shots: int = 1024,
    config_path: str = "config/levels.json",
    config_overrides: dict[str, Any] | None = None,
    **_unused_compatibility_kwargs,
) -> dict[str, Any]:
    """Perform the actual Qiskit run. Call only from a simulation child process.

    T-doped circuits deliberately run without Paulice checks because the
    supported spacetime-check workflow accepts Clifford circuits only.
    """
    levels = load_level_configs(config_path)
    if level_id not in levels:
        raise ValueError(f"Unknown level {level_id!r}. Available levels: {list(levels)}")
    cfg = dict(levels[level_id])
    if config_overrides:
        cfg.update(config_overrides)

    payload_type = str(cfg.get("payload_type", "clifford")).lower()
    n = int(cfg.get("num_qubits", 3))
    depth_layers = int(cfg.get("depth_layers", 2))
    t_count = int(cfg.get("t_count", 0))
    shots = int(shots)

    # Hard limits protect the child process from malformed configs and
    # accidental exponential statevector/memory requests. Current campaign
    # levels are well below these limits.
    if not 3 <= n <= 12:
        raise ValueError("num_qubits must be between 3 and 12 for safe local simulation.")
    if not 1 <= depth_layers <= 256:
        raise ValueError("depth_layers must be between 1 and 256.")
    if not 1 <= shots <= 100_000:
        raise ValueError("shots must be between 1 and 100000.")
    if not 0 <= t_count <= n * depth_layers:
        raise ValueError("t_count must be between 0 and num_qubits * depth_layers.")

    try:
        beta = float(cfg.get("beta", 2.0))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Level beta must be finite and in the range (0, 1000].") from exc
    if not math.isfinite(beta) or beta <= 0 or beta > 1000.0:
        raise ValueError("Level beta must be finite and in the range (0, 1000].")

    if payload_type not in {"clifford", "tdoped"}:
        raise ValueError(f"Unsupported payload_type {payload_type!r} in {level_id}.")

    target_qubits = [int(q) for q in (target_qubits or [])]
    num_checks = int(num_checks)
    if num_checks < 0:
        raise ValueError("num_checks cannot be negative.")

    if payload_type == "tdoped" and (num_checks != 0 or target_qubits):
        raise ValueError(
            "Official qiskit-paulice checks accept Clifford circuits only. "
            "T-doped missions are run without checks and report their noisy fidelity."
        )

    if payload_type == "clifford":
        if len(target_qubits) != len(set(target_qubits)):
            raise ValueError("Target qubits must be unique.")
        if any(q < 0 or q >= n for q in target_qubits):
            raise ValueError(f"Target qubits must be in 0..{n - 1}.")
        if num_checks != len(target_qubits):
            raise ValueError(
                "The number of checks is determined by the selected target qubits: "
                "pass one target qubit per requested check."
            )
        if num_checks > int(cfg.get("max_checks", n)):
            raise ValueError(f"This mission permits at most {cfg.get('max_checks')} checks.")
        max_budget_checks = int(cfg.get("energy_budget", 10**9)) // max(
            1, int(cfg.get("check_cost", 1))
        )
        if num_checks > max_budget_checks:
            raise ValueError("The requested check count exceeds the mission's energy budget.")

    # Only import native-backed quantum packages after validating the request.
    from src.circuits import get_clifford_payload, get_tdoped_payload
    from src.metrics import compute_all_metrics
    from src.simulator import (
        build_paulice_variants,
        circuit_cost,
        get_noise_models,
        measured_payload,
        run_ideal_simulation,
        run_noisy_simulation,
        T_GATE_DEPOLARIZING_RATE,
    )

    if payload_type == "clifford":
        payload = get_clifford_payload(n, depth_layers)
    else:
        payload = get_tdoped_payload(n, depth_layers, t_count)

    reference_shots = max(shots * 4, 8192)
    if payload_type == "clifford":
        # All strategies for a mission share this same ideal reference. Reusing
        # it avoids resampling the identical circuit once per qubit subset.
        ideal_distribution = _cached_ideal_distribution(
            payload_type, n, depth_layers, reference_shots
        )
    else:
        ideal_distribution = run_ideal_simulation(
            payload, shots=reference_shots, payload_type=payload_type
        )

    paulice_noise, aer_noise, noise_source = get_noise_models(
        n, require_paulice=(payload_type == "clifford")
    )
    if payload_type == "tdoped":
        noise_source = (
            f"{noise_source}; T/Tdg single-qubit depolarizing error="
            f"{T_GATE_DEPOLARIZING_RATE:.1%} (T-study only)"
        )
    bare_measured = measured_payload(payload)

    checked_object = None
    final_circuit = bare_measured
    postselection_method = None
    actual_checks = 0
    chosen_targets: list[int] = []
    paulice_cost = None
    supports: list[list[int]] = []
    spacetime_timeline: list[dict[str, Any]] = []

    if payload_type == "clifford" and num_checks > 0:
        variants = build_paulice_variants(payload, target_qubits, paulice_noise)
        if num_checks >= len(variants):
            raise RuntimeError(
                f"Paulice returned {len(variants)} variants for {len(target_qubits)} targets; "
                f"cannot select {num_checks} checks. Retry with fewer checks."
            )
        checked_object = variants[num_checks]
        final_circuit = checked_object.circuit
        postselection_method = checked_object.get_postselection_method()
        actual_checks = len(getattr(checked_object, "check_qubits", ()) or ())
        chosen_targets = list(getattr(checked_object, "target_qubits", ()) or ())
        paulice_cost = getattr(checked_object, "cost", None)
        supports = [
            list(support)
            for support in (getattr(checked_object, "check_support", ()) or ())
        ]
        spacetime_timeline = _check_spacetime_timeline(
            final_circuit,
            [int(q) for q in (getattr(checked_object, "check_qubits", ()) or ())],
        )

    noisy_counts = run_noisy_simulation(
        final_circuit,
        noise_model=aer_noise,
        shots=shots,
        payload_type=payload_type,
    )
    metrics = compute_all_metrics(
        ideal_distribution,
        noisy_counts,
        postselection_method=postselection_method,
        beta=beta,
    )
    cost = circuit_cost(bare_measured, final_circuit, n)

    # T-doped experiments are analysis-only and never receive a campaign score.
    # Campaign scoring also accounts for check resources: a check is not free,
    # and later missions intentionally apply a stronger resource penalty.
    base_harmonic_score = 100.0 * metrics["harmonic_score"]
    check_penalty_weight = max(0.0, float(cfg.get("check_penalty_weight", 0.0)))
    energy_budget = max(1.0, float(cfg.get("energy_budget", 1.0)))
    check_cost = max(0.0, float(cfg.get("check_cost", 0.0)))
    check_overhead_penalty = (
        100.0 * check_penalty_weight * check_cost * num_checks / energy_budget
        if payload_type == "clifford"
        else 0.0
    )
    game_score = (
        round(max(0.0, base_harmonic_score - check_overhead_penalty), 2)
        if payload_type == "clifford"
        else None
    )

    return {
        "level_id": level_id,
        "level_title": cfg.get("title", level_id),
        "payload_type": payload_type,
        "official_mode": (
            "qiskit-paulice spacetime checks"
            if payload_type == "clifford"
            else "T-doped no-check fidelity study"
        ),
        "effective_config": {
            "num_qubits": n,
            "depth_layers": depth_layers,
            "t_count": t_count,
            "target_score": float(cfg.get("target_score", 60.0)),
            "min_retention": float(cfg.get("min_retention", 0.15)),
            "check_penalty_weight": check_penalty_weight,
            "beta": beta,
        },
        "num_checks_requested": num_checks,
        "num_checks_used": actual_checks,
        "target_qubits": chosen_targets,
        "check_supports": supports,
        "spacetime_timeline": spacetime_timeline,
        "paulice_cost": None if paulice_cost is None else float(paulice_cost),
        "noise_source": noise_source,
        "circuit_depth": int(final_circuit.depth()),
        "total_qubits": int(final_circuit.num_qubits),
        "game_score": game_score,
        "scoring_beta": beta,
        "metrics": {
            "raw_fidelity": round(metrics["raw_fidelity"], 6),
            "post_selected_fidelity": round(metrics["post_selected_fidelity"], 6),
            "retention_rate": round(metrics["retention_rate"], 6),
            "harmonic_score_H": round(metrics["harmonic_score"], 6),
            "check_overhead_penalty": round(check_overhead_penalty, 4),
            "raw_shots": metrics["raw_shots"],
            "retained_shots": metrics["retained_shots"],
        },
        "cost": cost,
        "feedback": (
            generate_feedback_message(
                metrics["post_selected_fidelity"], metrics["retention_rate"], game_score
            )
            if game_score is not None
            else "T-doped payload measured without checks because qiskit-paulice accepts Clifford circuits only."
        ),
        "circuit_obj": final_circuit,
    }


class _CircuitText:
    """Small display-only stand-in for QuantumCircuit returned by the child."""

    def __init__(self, drawing: str):
        self._drawing = str(drawing)

    def draw(self, *args, **kwargs) -> str:
        # The UI uses draw(output="text", fold=120). Ignore rendering options
        # because the real text drawing was generated in the child process.
        return self._drawing


def play_level(
    level_id: str,
    num_checks: int = 0,
    target_qubits: list[int] | None = None,
    shots: int = 1024,
    config_path: str = "config/levels.json",
    config_overrides: dict[str, Any] | None = None,
    **_unused_compatibility_kwargs,
) -> dict[str, Any]:
    """Run a mission in a crash-isolated child process.

    All app entry points (Campaign, baseline, and T-gate study) use this
    wrapper. If a native dependency segfaults, only the child exits; Streamlit
    receives a clear RuntimeError instead of being terminated.
    """
    from src.simulation_worker import run_level_isolated

    result = run_level_isolated(
        level_id=str(level_id),
        targets=[int(q) for q in (target_qubits or [])],
        config=dict(config_overrides or {}),
        shots=int(shots),
        config_path=str(config_path),
        num_checks=int(num_checks),
    )
    drawing = result.pop("_circuit_text", "")
    result["circuit_obj"] = _CircuitText(drawing) if drawing else None
    return result