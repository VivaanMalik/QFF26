"""Difficulty presets shared by the Streamlit UI and ranking worker."""
from __future__ import annotations

import math
from typing import Any

DIFFICULTIES = ("Guided", "Standard", "Full Challenge")
DEFAULT_SHOTS = {"Guided": 2048, "Standard": 2048, "Full Challenge": 2048}


def apply_difficulty(base_cfg: dict[str, Any], difficulty: str) -> dict[str, Any]:
    """Apply the campaign difficulty preset to a base Clifford level config."""
    cfg = dict(base_cfg)
    original_qubits = int(base_cfg.get("num_qubits", 3))
    original_depth = int(base_cfg.get("depth_layers", 2))
    original_t_count = int(base_cfg.get("t_count", 0))

    if difficulty == "Guided":
        cfg.update({
            "num_qubits": 3,
            "depth_layers": min(3, max(2, math.ceil(original_depth / 12))),
            "t_count": min(original_t_count, max(1, math.ceil(original_t_count * 0.25))) if original_t_count else 0,
            "max_checks": 2,
            "energy_budget": 100,
            "check_cost": 35,
            "target_score": 45.0,
            "min_retention": 0.10,
            "description": "Guided practice: same payload family in a short 3-qubit circuit, using the shared reference noise model.",
        })
    elif difficulty == "Standard":
        cfg.update({
            "num_qubits": min(5, original_qubits),
            "depth_layers": min(12, max(3, math.ceil(original_depth * 0.40))),
            "t_count": min(original_t_count, max(1, math.ceil(original_t_count * 0.50))) if original_t_count else 0,
            "max_checks": min(3, int(base_cfg.get("max_checks", 3))),
            "energy_budget": 120,
            "check_cost": 35,
            "target_score": 52.0,
            "min_retention": 0.12,
            "description": "Standard practice: smaller instance, same FakeTorino-derived (or explicitly labelled fallback) noise rates.",
        })
    elif difficulty != "Full Challenge":
        raise ValueError(f"Unknown difficulty preset {difficulty!r}.")
    return cfg