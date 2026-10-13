from qiskit import QuantumCircuit

def get_clifford_payload(num_qubits = 3, depth_layers = 2):
    qc = QuantumCircuit(num_qubits, name="GHZ_Clifford_Payload")

    # 1. GHZ Entanglement Core
    qc.h(0)
    for q in range(num_qubits - 1):
        qc.cx(q, q + 1)

    # 2. Layered Clifford Rotations
    for layer in range(depth_layers):
        for q in range(num_qubits):
            if (layer + q) % 2 == 0:
                qc.h(q)
            else:
                qc.s(q)

        # Entangling ladder
        for q in range(num_qubits - 1):
            if layer % 2 == 0:
                qc.cx(q, q + 1)
            else:
                qc.cz(q, q + 1)

    return qc

def get_tdoped_payload(num_qubits = 3, depth_layers = 3, t_count = 2):
    qc = QuantumCircuit(num_qubits, name="T_Doped_Payload")

    # 1. Superposition initialization
    for q in range(num_qubits):
        qc.h(q)

    # 2. Entangling core
    for q in range(num_qubits - 1):
        qc.cx(q, q + 1)

    # 3. Deterministic T-gate doping across target depth layers
    t_gates_inserted = 0
    for layer in range(depth_layers):
        for q in range(num_qubits):
            if t_gates_inserted < t_count:
                if (layer + q) % 2 == 0:
                    qc.t(q)
                else:
                    qc.tdg(q)
                t_gates_inserted += 1

        # CNOT coupling between layers
        if layer < depth_layers - 1:
            for q in range(num_qubits - 1):
                qc.cx(q, q + 1)

    # 4. Final basis change
    for q in range(num_qubits):
        qc.h(q)

    return qc