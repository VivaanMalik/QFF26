"""Count parsing, post-selection, and campaign scoring."""
from __future__ import annotations

import numpy as np


def calculate_hellinger_fidelity(counts_ideal: dict[str, int], counts_exp: dict[str, int]) -> float:
    """Squared Bhattacharyya overlap of two measured output distributions."""
    ideal_total = sum(counts_ideal.values())
    exp_total = sum(counts_exp.values())
    if ideal_total <= 0 or exp_total <= 0:
        return 0.0

    p_ideal = {key.replace(" ", ""): value / ideal_total for key, value in counts_ideal.items()}
    p_exp = {key.replace(" ", ""): value / exp_total for key, value in counts_exp.items()}
    all_keys = set(p_ideal) | set(p_exp)
    affinity = sum(np.sqrt(p_ideal.get(key, 0.0) * p_exp.get(key, 0.0)) for key in all_keys)
    return float(np.clip(affinity * affinity, 0.0, 1.0))


def _split_count_key(bitstring: str) -> tuple[str, str]:
    """Return (syndrome register, data register) from Qiskit count keys."""
    parts = bitstring.split()
    if len(parts) >= 2:
        return parts[0], parts[-1]
    return "", parts[0] if parts else ""


def filter_syndrome_shots(
    counts: dict[str, int], expected_syndrome_bits: list[int] | None = None
) -> tuple[dict[str, int], float]:
    """Keep shots whose measured syndrome equals the ideal expected syndrome."""
    total_shots = sum(counts.values())
    if total_shots <= 0:
        return {}, 0.0

    expected = list(expected_syndrome_bits or [])
    expected_display = "".join(str(bit) for bit in reversed(expected))
    accepted: dict[str, int] = {}
    retained = 0

    for bitstring, count in counts.items():
        syndrome, data = _split_count_key(bitstring)
        if not expected:
            is_valid = True
        else:
            # Multi-register Aer keys display the syndrome register first.
            # The fallback covers providers that flatten classical registers.
            if syndrome:
                is_valid = syndrome == expected_display
            else:
                compact = bitstring.replace(" ", "")
                is_valid = compact[: len(expected_display)] == expected_display
                data = compact[len(expected_display) :]
        if is_valid:
            accepted[data] = accepted.get(data, 0) + count
            retained += count

    return accepted, retained / total_shots


def marginalize_raw_counts(
    counts: dict[str, int], expected_syndrome_bits: list[int] | None = None
) -> dict[str, int]:
    raw_data: dict[str, int] = {}
    num_checks = len(expected_syndrome_bits or [])
    for bitstring, count in counts.items():
        syndrome, data = _split_count_key(bitstring)
        if not syndrome and num_checks:
            compact = bitstring.replace(" ", "")
            data = compact[num_checks:]
        raw_data[data] = raw_data.get(data, 0) + count
    return raw_data


def calculate_harmonic_score(fidelity: float, retention: float) -> float:
    """Simple harmonic mean H=2FR/(F+R), returned on a 0..1 scale."""
    fidelity = float(np.clip(fidelity, 0.0, 1.0))
    retention = float(np.clip(retention, 0.0, 1.0))
    denominator = fidelity + retention
    return 0.0 if denominator == 0 else (2.0 * fidelity * retention) / denominator


def compute_all_metrics(
    counts_ideal: dict[str, int],
    counts_noisy_raw: dict[str, int],
    expected_syndrome_bits: list[int] | None = None,
) -> dict:
    raw_counts = marginalize_raw_counts(counts_noisy_raw, expected_syndrome_bits)
    raw_fidelity = calculate_hellinger_fidelity(counts_ideal, raw_counts)
    selected_counts, retention = filter_syndrome_shots(counts_noisy_raw, expected_syndrome_bits)
    selected_fidelity = calculate_hellinger_fidelity(counts_ideal, selected_counts)
    score = calculate_harmonic_score(selected_fidelity, retention)
    return {
        "raw_fidelity": raw_fidelity,
        "post_selected_fidelity": selected_fidelity,
        "retention_rate": retention,
        "harmonic_score": score,
        "raw_shots": sum(counts_noisy_raw.values()),
        "retained_shots": sum(selected_counts.values()),
    }
