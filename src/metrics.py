"""Output-distribution fidelity, Paulice syndrome post-selection, and scoring."""
from __future__ import annotations

from typing import Callable

import numpy as np


def calculate_hellinger_fidelity(
    counts_ideal: dict[str, float], counts_exp: dict[str, float]
) -> float:
    """Squared Bhattacharyya overlap (classical fidelity) of two distributions."""
    ideal_total = float(sum(counts_ideal.values()))
    exp_total = float(sum(counts_exp.values()))
    if ideal_total <= 0 or exp_total <= 0:
        return 0.0

    p_ideal = {key.replace(" ", ""): value / ideal_total for key, value in counts_ideal.items()}
    p_exp = {key.replace(" ", ""): value / exp_total for key, value in counts_exp.items()}
    all_keys = set(p_ideal) | set(p_exp)
    affinity = sum(
        np.sqrt(max(0.0, p_ideal.get(key, 0.0) * p_exp.get(key, 0.0)))
        for key in all_keys
    )
    return float(np.clip(affinity * affinity, 0.0, 1.0))


def data_register_from_count_key(bitstring: str) -> str:
    """Extract the final data-measurement register from a Qiskit count key.

    If the circuit has multiple classical registers, Qiskit separates them with
    spaces and the payload data register is the final field. Paulice commonly
    encodes its syndrome as parity constraints on the measured data bits, so a
    one-register key is also valid. The full original key must be sent to the
    Paulice post-selection method before data extraction.
    """
    parts = bitstring.split()
    return parts[-1] if parts else ""


def marginalize_raw_counts(counts: dict[str, int | float], expected_syndrome_bits=None) -> dict[str, float]:
    """Sum counts over syndrome registers, preserving only the data output."""
    raw_data: dict[str, float] = {}
    for bitstring, count in counts.items():
        data = data_register_from_count_key(bitstring)
        raw_data[data] = raw_data.get(data, 0.0) + float(count)
    return raw_data


def filter_pauli_check_shots(
    counts: dict[str, int], postselection_method: Callable | None
) -> tuple[dict[str, int], float]:
    """Use CheckedCircuit.get_postselection_method and retain only zero syndromes.

    In Paulice, each check's syndrome is the parity of the measured data bits in
    that check's support. A shot passes iff every syndrome bit is zero. This is
    intentionally not the old hand-built +1/-1 checkpoint test.
    """
    total = sum(int(value) for value in counts.values())
    if total <= 0:
        return {}, 0.0

    accepted: dict[str, int] = {}
    retained = 0
    for bitstring, count in counts.items():
        if postselection_method is not None:
            syndrome = np.asarray(postselection_method(bitstring), dtype=bool)
            if bool(syndrome.any()):
                continue
        data = data_register_from_count_key(bitstring)
        accepted[data] = accepted.get(data, 0) + int(count)
        retained += int(count)
    return accepted, retained / total


def filter_syndrome_shots(
    counts: dict[str, int], expected_syndrome_bits: list[int] | None = None
) -> tuple[dict[str, int], float]:
    """Legacy unit-test helper for a single displayed syndrome register.

    The official game path does not use this helper; it delegates syndrome
    interpretation to qiskit-paulice's CheckedCircuit post-selection method.
    """
    total = sum(counts.values())
    if total <= 0:
        return {}, 0.0
    expected = list(expected_syndrome_bits or [])
    expected_display = "".join(str(bit) for bit in reversed(expected))
    accepted: dict[str, int] = {}
    retained = 0
    for bitstring, count in counts.items():
        parts = bitstring.split()
        if expected:
            if len(parts) >= 2:
                syndrome, data = parts[0], parts[-1]
                valid = syndrome == expected_display
            else:
                compact = bitstring.replace(" ", "")
                valid = compact[:len(expected_display)] == expected_display
                data = compact[len(expected_display):]
        else:
            valid = True
            data = parts[-1] if parts else ""
        if valid:
            accepted[data] = accepted.get(data, 0) + count
            retained += count
    return accepted, retained / total


def calculate_harmonic_score(
    fidelity: float, retention: float, beta: float = 2.0
) -> float:
    """Fidelity-weighted harmonic mean, returned on the 0..1 scale.

    H_beta = (1 + beta^2) F R / (F + beta^2 R).
    beta=2 weights fidelity four times as heavily as retention in the
    reciprocal-weighted harmonic mean. The caller converts this to 0..100.
    """
    fidelity = float(np.clip(fidelity, 0.0, 1.0))
    retention = float(np.clip(retention, 0.0, 1.0))
    beta = float(beta)
    if beta <= 0:
        raise ValueError("beta must be positive")
    beta_squared = beta * beta
    denominator = fidelity + beta_squared * retention
    return (
        0.0
        if denominator == 0
        else ((1.0 + beta_squared) * fidelity * retention) / denominator
    )


def compute_all_metrics(
    counts_ideal: dict[str, int | float],
    counts_noisy_raw: dict[str, int],
    postselection_method: Callable | None = None,
) -> dict:
    """Calculate raw fidelity, post-selected fidelity, retention and score."""
    raw_data_counts = marginalize_raw_counts(counts_noisy_raw)
    raw_fidelity = calculate_hellinger_fidelity(counts_ideal, raw_data_counts)

    selected_counts, retention = filter_pauli_check_shots(counts_noisy_raw, postselection_method)
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