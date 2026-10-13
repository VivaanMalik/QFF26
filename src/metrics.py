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

def extract_data_bitstring(clean_bitstring, ancilla_indices):
    # remove ancilla, get logical qubits only
    return "".join(
        char for idx, char in enumerate(reversed(clean_bitstring))
        if idx not in ancilla_indices
    )[::-1]

def filter_syndrome_shots(counts, ancilla_bit_indices = None):
    total_shots = sum(counts.values())
    if total_shots == 0:
        return {}, 0.0

    anc_set = set(ancilla_bit_indices or [])
    post_selected_counts = {}
    retained_shots = 0

    for bitstring, count in counts.items():
        clean_bitstring = bitstring.replace(" ", "")
        
        # check syndrome: qubit i -> clean_bitstring[-(i + 1)]
        syndrome_valid = all(
            clean_bitstring[-(idx + 1)] == '0'
            for idx in anc_set
            if idx < len(clean_bitstring)
        )

        if syndrome_valid:
            data_key = extract_data_bitstring(clean_bitstring, anc_set)
            post_selected_counts[data_key] = post_selected_counts.get(data_key, 0) + count
            retained_shots += count

    retention_fraction = retained_shots / total_shots if total_shots > 0 else 0.0
    return post_selected_counts, float(retention_fraction)

def marginalize_raw_counts(counts, ancilla_bit_indices = None):
    # remove ancilla for raw counts
    if not ancilla_bit_indices:
        return {k.replace(" ", ""): v for k, v in counts.items()}

    anc_set = set(ancilla_bit_indices)
    raw_data_counts = {}

    for bitstring, count in counts.items():
        clean_bitstring = bitstring.replace(" ", "")
        data_key = extract_data_bitstring(clean_bitstring, anc_set)
        raw_data_counts[data_key] = raw_data_counts.get(data_key, 0) + count

    return raw_data_counts

def calculate_harmonic_score(fidelity, retention, beta=2):
    if (fidelity + beta * beta * retention) == 0:
        return 0.0
        
    beta_sq = beta * beta
    score = ((1 + beta_sq) * fidelity * retention) / (fidelity + beta_sq * retention)
    return float(score)


def compute_all_metrics(counts_ideal, counts_noisy_raw, ancilla_bit_indices = None, beta=2):
    raw_data_counts = marginalize_raw_counts(counts_noisy_raw, ancilla_bit_indices)
    raw_fidelity = calculate_hellinger_fidelity(counts_ideal, raw_data_counts)
    
    post_selected_counts, retention = filter_syndrome_shots(counts_noisy_raw, ancilla_bit_indices)
    post_selected_fidelity = calculate_hellinger_fidelity(counts_ideal, post_selected_counts)

    score = calculate_harmonic_score(post_selected_fidelity, retention, beta)

    return {
        "raw_fidelity": raw_fidelity,
        "post_selected_fidelity": post_selected_fidelity,
        "retention_rate": retention,
        "harmonic_score": score,
        "raw_shots": sum(counts_noisy_raw.values()),
        "retained_shots": sum(post_selected_counts.values())
    }