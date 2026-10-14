"""Mission configuration and execution using official spacetime-code checks."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.circuits import get_clifford_payload, get_tdoped_payload
from src.metrics import compute_all_metrics
from src.simulator import (
    build_paulice_variants,
    circuit_cost,
    get_noise_models,
    measured_payload,
    run_ideal_simulation,
    run_noisy_simulation,
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


def play_level(
    level_id: str,
    num_checks: int = 0,
    target_qubits: list[int] | None = None,
    shots: int = 1024,
    config_path: str = "config/levels.json",
    config_overrides: dict[str, Any] | None = None,
    **_unused_compatibility_kwargs,
) -> dict[str, Any]:
    """Execute a level with qiskit-paulice checks for Clifford payloads only.

    The T-doped branch deliberately performs a no-check fidelity study, as
    explicitly allowed by the official brief because add_pauli_checks accepts
    Clifford circuits only.
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
    if payload_type == "clifford":
        payload = get_clifford_payload(n, depth_layers)
    elif payload_type == "tdoped":
        payload = get_tdoped_payload(n, depth_layers, t_count)
        if int(num_checks) != 0 or list(target_qubits or []):
            raise ValueError(
                "Official qiskit-paulice checks accept Clifford circuits only. "
                "T-doped missions are run without checks and report their noisy fidelity."
            )
    else:
        raise ValueError(f"Unsupported payload_type {payload_type!r} in {level_id}.")

    target_qubits = [int(q) for q in (target_qubits or [])]
    num_checks = int(num_checks)
    if num_checks < 0:
        raise ValueError("num_checks cannot be negative.")
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
        max_budget_checks = int(cfg.get("energy_budget", 10**9)) // max(1, int(cfg.get("check_cost", 1)))
        if num_checks > max_budget_checks:
            raise ValueError("The requested check count exceeds the mission's energy budget.")

    ideal_distribution = run_ideal_simulation(
        payload,
        shots=max(int(shots) * 4, 8192),
        payload_type=payload_type,
    )
    paulice_noise, aer_noise, noise_source = get_noise_models(n)
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
        supports = [list(support) for support in (getattr(checked_object, "check_support", ()) or ())]
        spacetime_timeline = _check_spacetime_timeline(
            final_circuit, [int(q) for q in (getattr(checked_object, "check_qubits", ()) or ())]
        )

    noisy_counts = run_noisy_simulation(
        final_circuit,
        noise_model=aer_noise,
        shots=int(shots),
        payload_type=payload_type,
    )
    metrics = compute_all_metrics(
        ideal_distribution,
        noisy_counts,
        postselection_method=postselection_method,
    )
    cost = circuit_cost(bare_measured, final_circuit, n)
    # T-doped levels are analysis-only per the official permitted route. Avoid
    # presenting a strategy score when no checks were used.
    game_score = (
        round(100.0 * metrics["harmonic_score"], 2)
        if payload_type == "clifford"
        else None
    )

    return {
        "level_id": level_id,
        "level_title": cfg.get("title", level_id),
        "payload_type": payload_type,
        "official_mode": "qiskit-paulice spacetime checks" if payload_type == "clifford" else "T-doped no-check fidelity study",
        "effective_config": {
            "num_qubits": n,
            "depth_layers": depth_layers,
            "t_count": t_count,
            "target_score": float(cfg.get("target_score", 60.0)),
            "min_retention": float(cfg.get("min_retention", 0.15)),
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
        "metrics": {
            "raw_fidelity": round(metrics["raw_fidelity"], 6),
            "post_selected_fidelity": round(metrics["post_selected_fidelity"], 6),
            "retention_rate": round(metrics["retention_rate"], 6),
            "harmonic_score_H": round(metrics["harmonic_score"], 6),
            "raw_shots": metrics["raw_shots"],
            "retained_shots": metrics["retained_shots"],
        },
        "cost": cost,
        "feedback": (
            generate_feedback_message(
                metrics["post_selected_fidelity"], metrics["retention_rate"], game_score
            ) if game_score is not None else
            "T-doped payload measured without checks because qiskit-paulice accepts Clifford circuits only."
        ),
        "circuit_obj": final_circuit,
    }
