"""Original payload families with explicit spacetime checkpoints.

Checkpoint 0 is after state preparation. Checkpoint k is after payload layer k.
Barriers are simulator metadata and do not change the payload's quantum state.
"""
from qiskit import QuantumCircuit


def _checkpoint(qc: QuantumCircuit, num_qubits: int) -> None:
    qc.barrier(*range(num_qubits))


def get_clifford_payload(num_qubits: int = 3, depth_layers: int = 2) -> QuantumCircuit:
    """Build the deep Clifford payload used by the campaign."""
    if num_qubits < 3:
        raise ValueError("Payloads require at least 3 qubits.")
    if depth_layers < 1:
        raise ValueError("depth_layers must be at least 1.")

    qc = QuantumCircuit(num_qubits, name="Stabilizer_Clifford")
    qc.h(0)
    for q in range(num_qubits - 1):
        qc.cx(q, q + 1)
    _checkpoint(qc, num_qubits)  # checkpoint 0

    for layer in range(depth_layers):
        for q in range(num_qubits):
            selector = (layer + q) % 3
            if selector == 0:
                qc.z(q)
            elif selector == 1:
                qc.s(q)
            else:
                qc.sdg(q)

        for q in range(num_qubits - 1):
            qc.cx(q, q + 1)
        if num_qubits >= 3:
            qc.cx(num_qubits - 1, 0)

        _checkpoint(qc, num_qubits)  # checkpoint layer + 1

    return qc


def get_tdoped_payload(
    num_qubits: int = 3, depth_layers: int = 3, t_count: int = 2
) -> QuantumCircuit:
    """Build the Clifford+T payload used by the campaign."""
    if num_qubits < 3:
        raise ValueError("Payloads require at least 3 qubits.")
    if depth_layers < 1:
        raise ValueError("depth_layers must be at least 1.")
    if not 0 <= t_count <= num_qubits * depth_layers:
        raise ValueError("t_count must be between 0 and num_qubits * depth_layers.")

    qc = QuantumCircuit(num_qubits, name="UltraNoisy_TDoped")
    for q in range(num_qubits):
        qc.h(q)
    _checkpoint(qc, num_qubits)  # checkpoint 0

    t_inserted = 0
    for layer in range(depth_layers):
        for q in range(num_qubits - 1):
            qc.cx(q, q + 1)
        if num_qubits >= 3:
            qc.cx(num_qubits - 1, 0)
        if num_qubits >= 4:
            qc.cz(1, num_qubits - 2)

        for q in range(num_qubits):
            if t_inserted < t_count:
                if (layer + q) % 2 == 0:
                    qc.t(q)
                else:
                    qc.tdg(q)
                t_inserted += 1

        _checkpoint(qc, num_qubits)

    # Preserve the original final basis-conversion layer.
    for q in range(num_qubits):
        qc.h(q)
    return qc
