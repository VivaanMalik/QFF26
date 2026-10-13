import numpy as np


def calculate_hellinger_fidelity(counts_ideal, counts_exp):
    ideal_tot = sum(counts_ideal.values())
    exp_tot = sum(counts_exp.values())

    if ideal_tot == 0 or ideal_tot == 0:
        return 0

    p_ideal = {k: v / ideal_tot for k, v in counts_ideal.items()}
    p_exp = {k: v / exp_tot for k, v in counts_exp.items()}

    all_keys = set(p_ideal.keys()).union(set(p_exp.keys()))
    overlap = sum(np.sqrt(p_ideal.get(k, 0.0) * p_exp.get(k, 0.0)) for k in all_keys)

    return float(overlap ** 2)


def filter_syndrome_shots(counts, ancilla_bit_indices = None):
    total_shots = sum(counts.values())
    if total_shots == 0:
        return {}, 0

    if not ancilla_bit_indices:
        # R =1
        return counts, 1

    post_selected_counts = {}

    for bitstring, count in counts.items():
        # Remove spaces if counts came from multi-register measurement
        clean_bitstring = bitstring.replace(" ", "")
        
        # Check if all specified ancilla bits equal '0'
        # Qiskit index mapping: qubit i -> bitstring[-(i + 1)]
        syndrome_valid = all(
            clean_bitstring[-(idx + 1)] == '0'
            for idx in ancilla_bit_indices
            if idx < len(clean_bitstring)
        )

        if syndrome_valid:
            post_selected_counts[bitstring] = count

    retained_shots = sum(post_selected_counts.values())
    retention_fraction = retained_shots / total_shots

    return post_selected_counts, float(retention_fraction)


def calculate_harmonic_score(F, R):
    if (F + R) == 0:
        return 0.0
    return float((2 * F * R) / (F + R))


def compute_all_metrics(counts_ideal, counts_noisy_raw, ancilla_bit_indices = None):
    raw_fidelity = calculate_hellinger_fidelity(counts_ideal, counts_noisy_raw)
    
    post_selected_counts, retention = filter_syndrome_shots(counts_noisy_raw, ancilla_bit_indices)
    post_selected_fidelity = calculate_hellinger_fidelity(counts_ideal, post_selected_counts)
    
    score = calculate_harmonic_score(post_selected_fidelity, retention)

    return {
        "raw_fidelity": raw_fidelity,
        "post_selected_fidelity": post_selected_fidelity,
        "retention_rate": retention,
        "harmonic_score": score,
        "raw_shots": sum(counts_noisy_raw.values()),
        "retained_shots": sum(post_selected_counts.values())
    }