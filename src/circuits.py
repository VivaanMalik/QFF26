"""Payload circuits for the Clifford campaign and the separate T-gate study.

Campaign payloads remain Clifford-only. The T-gate study has its own matched
Clifford control family with fixed Hadamard basis changes between layers. Those
basis changes let inserted T/Tdg phases interfere before Z-basis measurement;
without them the T gates in the original diagonal-only sections were invisible
to output probabilities.
"""
from __future__ import annotations

from qiskit import QuantumCircuit


def _checkpoint(qc: QuantumCircuit) -> None:
    """Add a metadata barrier after a completed payload stage."""
    qc.barrier(*range(qc.num_qubits))


def _prepare_payload(
    num_qubits: int,
    depth_layers: int,
    t_count: int = 0,
    *,
    t_study: bool = False,
) -> QuantumCircuit:
    if num_qubits < 3:
        raise ValueError("Payloads require at least 3 qubits.")
    if depth_layers < 1:
        raise ValueError("depth_layers must be at least 1.")
    if not 0 <= t_count <= num_qubits * depth_layers:
        raise ValueError("t_count must be between 0 and num_qubits * depth_layers.")
    if t_count and not t_study:
        raise ValueError("T gates are only allowed in the separate T-gate study circuit.")

    if t_study:
        name = "T_Study_Clifford_Control" if t_count == 0 else "T_Study_TDoped"
    else:
        name = "Clifford_Payload"
    qc = QuantumCircuit(num_qubits, name=name)

    # Shared initialization for both families.
    qc.h(0)
    for q in range(num_qubits - 1):
        qc.cx(q, q + 1)
    _checkpoint(qc)

    # Place the requested T gates at evenly spread deterministic slots, rather
    # than clustering them at the start. The slot map is the same for every run.
    total_slots = num_qubits * depth_layers
    if t_count == 1:
        selected_t_slots = {total_slots // 2}
    elif t_count > 1:
        selected_t_slots = {
            round(index * (total_slots - 1) / (t_count - 1))
            for index in range(t_count)
        }
    else:
        selected_t_slots = set()

    for layer in range(depth_layers):
        # Clifford phase pattern shared by the campaign and study families.
        for q in range(num_qubits):
            selector = (layer + q) % 3
            if selector == 0:
                qc.z(q)
            elif selector == 1:
                qc.s(q)
            else:
                qc.sdg(q)

        # Non-Clifford phases are introduced only in the research family.
        for q in range(num_qubits):
            if layer * num_qubits + q not in selected_t_slots:
                continue
            if (layer + q) % 2 == 0:
                qc.t(q)
            else:
                qc.tdg(q)

        # Fixed study-only basis changes make diagonal phases observably affect
        # the final output distribution. This scaffold is identical for every
        # T count, including the T=0 control; campaign circuits never use it.
        if t_study:
            for q in range(num_qubits):
                qc.h(q)

        # Original entangling layer, including the wraparound interaction.
        for q in range(num_qubits - 1):
            qc.cx(q, q + 1)
        qc.cx(num_qubits - 1, 0)
        _checkpoint(qc)

    return qc


def get_clifford_payload(num_qubits: int = 3, depth_layers: int = 2) -> QuantumCircuit:
    """Generate the Clifford-only campaign payload accepted by qiskit-paulice."""
    return _prepare_payload(num_qubits, depth_layers, t_count=0, t_study=False)


def get_tdoped_payload(
    num_qubits: int = 3, depth_layers: int = 3, t_count: int = 2
) -> QuantumCircuit:
    """Generate a controlled T-study circuit with exactly ``t_count`` T/Tdg gates.

    Its T=0 member is a matched Clifford control for the T-study family, not a
    campaign payload. This keeps the experiment isolated from campaign levels.
    """
    return _prepare_payload(num_qubits, depth_layers, t_count=t_count, t_study=True)