from qiskit import QuantumCircuit


def get_clifford_payload(num_qubits=3, depth_layers=2):
    qc = QuantumCircuit(num_qubits, name="Deep_Clifford_Payload")

    # generate ghz entangling baseline
    qc.h(0)
    for q in range(num_qubits - 1):
        qc.cx(q, q + 1)

    # deep layers of clifford operations to drive down raw fidelity
    for layer in range(depth_layers):
        for q in range(num_qubits):
            if (layer + q) % 2 == 0:
                qc.h(q)
            else:
                qc.s(q)

        # heavy cnot entangling ladder
        for q in range(num_qubits - 1):
            qc.cx(q, q + 1)

        # wrap around long range coupling
        if num_qubits >= 3:
            qc.cx(num_qubits - 1, 0)

    return qc


def get_tdoped_payload(num_qubits=3, depth_layers=3, t_count=2):
    qc = QuantumCircuit(num_qubits, name="Deep_TDoped_Payload")

    # start in equal superposition
    for q in range(num_qubits):
        qc.h(q)

    t_inserted = 0

    # loop through depth layers to build up bit and phase errors
    for layer in range(depth_layers):
        # entangle adjacent data lines
        for q in range(num_qubits - 1):
            qc.cx(q, q + 1)

        # cross-couple endpoints
        if num_qubits >= 3:
            qc.cx(0, num_qubits - 1)

        # inject t and tdg gates
        for q in range(num_qubits):
            if t_inserted < t_count:
                if (layer + q) % 2 == 0:
                    qc.t(q)
                else:
                    qc.tdg(q)
                t_inserted += 1

        # hadamards convert phase noise into bit flips that zz checks can detect
        for q in range(num_qubits):
            qc.h(q)

    return qc