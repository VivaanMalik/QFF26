import json
from pathlib import Path
from src.circuits import build_clifford_payload, build_tdoped_payload
from src.simulator import run_ideal_simulation, apply_clifford_spacetime_checks, apply_manual_z_checks, run_noisy_simulation
from src.metrics import compute_all_metrics


def load_level_configs(config_path = "config/levels.json"):
    path = Path(config_path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # default fallback
    return {
        "level_1": {
            "title": "Level 1: Shallow Clifford",
            "payload_type": "clifford",
            "num_qubits": 3,
            "depth_layers": 2,
            "t_count": 0,
            "max_checks": 3,
            "description": "Protect a shallow 3-qubit Clifford state with low gate noise."
        },
        "level_4": {
            "title": "Level 4: Light T-Doping",
            "payload_type": "tdoped",
            "num_qubits": 3,
            "depth_layers": 3,
            "t_count": 1,
            "max_checks": 3,
            "description": "First non-Clifford gate introduced. Place hand-made Z-checks."
        }
    }


def generate_feedback_message(fidelity, retention, score_100h):
    if retention < 0.25:
        return (
            f"Over-protected! Retention fell to {retention*100:.1f}%. "
            "Adding too many checks introduces gate noise that discards most shots."
        )
    if fidelity < 0.65:
        return (
            f"Under-protected! Fidelity is low ({fidelity*100:.1f}%). "
            "Add checks to critical qubits to filter out decoherence errors."
        )
    if score_100h >= 80.0:
        return f"Outstanding strategy! High fidelity ({fidelity*100:.1f}%) and solid retention ({retention*100:.1f}%)."
    
    return f"Good balance! Game Score: {score_100h:.1f}/100. Try fine-tuning check locations for higher fidelity."


def play_level(level_id, num_checks = 0, target_qubits = None, placed_checks = None, shots = 1024, config_path = "config/levels.json"):
    levels = load_level_configs(config_path)
    if level_id not in levels:
        raise ValueError(f"Invalid level_id '{level_id}'. Available levels: {list(levels.keys())}")

    level_cfg = levels[level_id]
    payload_type = level_cfg.get("payload_type", "clifford")
    num_qubits = level_cfg.get("num_qubits", 3)
    depth_layers = level_cfg.get("depth_layers", 2)
    t_count = level_cfg.get("t_count", 0)

    # 1. Build Base Circuit Payload
    if payload_type == "clifford":
        base_qc = build_clifford_payload(num_qubits=num_qubits, depth_layers=depth_layers)
    else:
        base_qc = build_tdoped_payload(num_qubits=num_qubits, depth_layers=depth_layers, t_count=t_count)

    # 2. Reference Ideal Counts (No Noise)
    counts_ideal = run_ideal_simulation(base_qc, shots=shots)

    # 3. Apply Checks
    target_qubits = target_qubits or []
    placed_checks = placed_checks or []

    if payload_type == "clifford":
        checked_qc, ancilla_bits = apply_clifford_spacetime_checks(
            base_qc, num_checks=num_checks, target_qubits=target_qubits
        )
    else:
        checked_qc, ancilla_bits = apply_manual_z_checks(
            base_qc, placed_checks=placed_checks
        )

    # 4. Noisy Simulation
    counts_noisy = run_noisy_simulation(checked_qc, shots=shots)

    # 5. Calculate Metrics & Game Score (100 * H)
    metrics = compute_all_metrics(counts_ideal, counts_noisy, ancilla_bits)
    game_score = round(metrics["harmonic_score"] * 100, 2)

    # 6. Build Feedback
    feedback = generate_feedback_message(
        metrics["post_selected_fidelity"],
        metrics["retention_rate"],
        game_score
    )

    return {
        "level_id": level_id,
        "level_title": level_cfg.get("title"),
        "payload_type": payload_type,
        "circuit_depth": checked_qc.depth(),
        "total_qubits": checked_qc.num_qubits,
        "game_score": game_score,
        "metrics": {
            "raw_fidelity": round(metrics["raw_fidelity"], 4),
            "post_selected_fidelity": round(metrics["post_selected_fidelity"], 4),
            "retention_rate": round(metrics["retention_rate"], 4),
            "harmonic_score_H": round(metrics["harmonic_score"], 4),
            "raw_shots": metrics["raw_shots"],
            "retained_shots": metrics["retained_shots"]
        },
        "feedback": feedback,
        "circuit_obj": checked_qc  # Retained internally for SVG rendering in app.py
    }