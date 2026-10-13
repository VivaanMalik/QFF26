from qiskit import QuantumCircuit

def get_clifford_payload(num_qubits = 3, depth_layers = 2):
    qc = QuantumCircuit(num_qubits, name="clifford_payload")
    
    for _ in range(depth_layers):
        for q in range(num_qubits):
            qc.h(q)
            qc.s(q)
        for q in range(num_qubits - 1):
            qc.cx(q, q + 1)
            
    qc.measure_all() 
    return qc

def get_tdoped_payload(num_qubits = 3, t_count = 0, depth_layers = 2):
    qc = QuantumCircuit(num_qubits, name=f"tdoped_{t_count}T")
    t_placed = 0
    
    for _ in range(depth_layers):
        for q in range(num_qubits):
            if t_placed < t_count:
                qc.t(q)
                t_placed += 1
            else:
                qc.h(q)
                qc.s(q)
        for q in range(num_qubits - 1):
            qc.cx(q, q + 1)
            
    qc.measure_all()
    return qc