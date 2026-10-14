"""Campaign configuration and level execution."""
import json
from pathlib import Path

from src.circuits import get_clifford_payload, get_tdoped_payload
from src.metrics import compute_all_metrics
from src.simulator import (
    apply_spacetime_checks,
    get_default_noise_model,
    run_ideal_simulation,
    run_noisy_simulation,
)


def load_level_configs(config_path="config/levels.json"):
    path = Path(config_path)
    if not path.exists():
        path = Path(__file__).resolve().parent.parent / "config" / "levels.json"
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def generate_feedback_message(fidelity, retention, score_100h):
    if retention < 0.15:
        return f"Over-filtered: only {retention * 100:.1f}% of shots survived. Use fewer checks or place them more carefully."
    if fidelity < 0.35:
        return f"The payload is still heavily corrupted (post-selection fidelity {fidelity * 100:.1f}%). Try checks before more noise accumulates."
    if score_100h >= 80:
        return f"Excellent strategy: {fidelity * 100:.1f}% post-selected fidelity with {retention * 100:.1f}% shot retention."
    return f"Balanced run: {score_100h:.1f}/100. Try moving a check earlier or changing its Pauli type."


def play_level(
    level_id,
    num_checks=0,
    target_qubits=None,
    placed_checks=None,
    shots=4096,
    config_path="config/levels.json",
    config_overrides=None,
    noise_scale=1.0,
):
    levels = load_level_configs(config_path)
    if level_id not in levels:
        raise ValueError(f"Unknown level {level_id!r}. Available levels: {list(levels)}")
    cfg = dict(levels[level_id])
    if config_overrides:
        cfg.update(config_overrides)
    payload_type = cfg.get("payload_type", "clifford")
    n = int(cfg.get("num_qubits", 3))
    depth = int(cfg.get("depth_layers", 2))

    if payload_type == "clifford":
        payload = get_clifford_payload(n, depth)
    elif payload_type == "tdoped":
        payload = get_tdoped_payload(n, depth, int(cfg.get("t_count", 0)))
    else:
        raise ValueError(f"Unsupported payload_type {payload_type!r} in {level_id}.")

    # Compatibility with the old API: default to the known preparation parity.
    if placed_checks is None:
        targets = list(target_qubits or [])[: int(num_checks)]
        default_type = "ZZ" if payload_type == "clifford" else "XX"
        placed_checks = [
            {"qubit": q, "layer": 0, "check_type": default_type} for q in targets
        ]
    placed_checks = list(placed_checks or [])

    ideal_counts = run_ideal_simulation(payload, shots=shots)
    checked_circuit, expected_bits = apply_spacetime_checks(payload, placed_checks)
    noisy_counts = run_noisy_simulation(
        checked_circuit, noise_model=get_default_noise_model(noise_scale), shots=shots
    )
    metrics = compute_all_metrics(ideal_counts, noisy_counts, expected_bits)
    score = round(100.0 * metrics["harmonic_score"], 2)

    return {
        "level_id": level_id,
        "level_title": cfg.get("title"),
        "payload_type": payload_type,
        "effective_config": {
            "num_qubits": n,
            "depth_layers": depth,
            "t_count": int(cfg.get("t_count", 0)),
            "target_score": float(cfg.get("target_score", 60.0)),
            "min_retention": float(cfg.get("min_retention", 0.15)),
        },
        "circuit_depth": checked_circuit.depth(),
        "total_qubits": checked_circuit.num_qubits,
        "game_score": score,
        "expected_syndrome_bits": expected_bits,
        "metrics": {
            "raw_fidelity": round(metrics["raw_fidelity"], 4),
            "post_selected_fidelity": round(metrics["post_selected_fidelity"], 4),
            "retention_rate": round(metrics["retention_rate"], 4),
            "harmonic_score_H": round(metrics["harmonic_score"], 4),
            "raw_shots": metrics["raw_shots"],
            "retained_shots": metrics["retained_shots"],
        },
        "feedback": generate_feedback_message(
            metrics["post_selected_fidelity"], metrics["retention_rate"], score
        ),
        "circuit_obj": checked_circuit,
    }
