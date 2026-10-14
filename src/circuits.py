"""Payload families used in the QFF spacetime-error-detection game.

The Clifford payload is the base circuit.  A T-doped payload inserts T/Tdg
operations into the same Clifford backbone so the family differs by the
non-Clifford resource being studied, rather than by an unrelated circuit.
Barriers mark layers for display/debugging only; Paulice input removes them.
"""
from __future__ import annotations

from qiskit import QuantumCircuit


def _checkpoint(qc: QuantumCircuit) -> None:
    """Add a metadata barrier after a completed payload stage."""
    qc.barrier(*range(qc.num_qubits))


def _prepare_clifford_backbone(num_qubits: int, depth_layers: int, t_count: int = 0) -> QuantumCircuit:
    if num_qubits < 3:
        raise ValueError("Payloads require at least 3 qubits.")
    if depth_layers < 1:
        raise ValueError("depth_layers must be at least 1.")
    if not 0 <= t_count <= num_qubits * depth_layers:
        raise ValueError("t_count must be between 0 and num_qubits * depth_layers.")

    qc = QuantumCircuit(num_qubits, name="Clifford_TDoped" if t_count else "Clifford_Payload")

    # Shared state preparation from the original Clifford payload.
    qc.h(0)
    for q in range(num_qubits - 1):
        qc.cx(q, q + 1)
    _checkpoint(qc)  # Checkpoint 0: state preparation is complete.

    remaining_t = t_count
    for layer in range(depth_layers):
        # Original Clifford phase pattern.
        for q in range(num_qubits):
            selector = (layer + q) % 3
            if selector == 0:
                qc.z(q)
            elif selector == 1:
                qc.s(q)
            else:
                qc.sdg(q)

        # T/Tdg gates are inserted into this shared circuit backbone.
        # Alternating signs gives the family different non-Clifford phase layers.
        for q in range(num_qubits):
            if remaining_t <= 0:
                break
            if (layer + q) % 2 == 0:
                qc.t(q)
            else:
                qc.tdg(q)
            remaining_t -= 1

        # Original entangling layer, including the wraparound interaction.
        for q in range(num_qubits - 1):
            qc.cx(q, q + 1)
        qc.cx(num_qubits - 1, 0)
        _checkpoint(qc)

    return qc


def get_clifford_payload(num_qubits: int = 3, depth_layers: int = 2) -> QuantumCircuit:
    """Generate the Clifford-only payload accepted by qiskit-paulice."""
    return _prepare_clifford_backbone(num_qubits, depth_layers, t_count=0)


def get_tdoped_payload(
    num_qubits: int = 3, depth_layers: int = 3, t_count: int = 2
) -> QuantumCircuit:
    """Generate a T-doped member of the same Clifford payload family."""
    # t_count=0 is intentionally supported for controlled sweeps; it is the
    # Clifford reference member of this payload family.
    return _prepare_clifford_backbone(num_qubits, depth_layers, t_count=t_count)
