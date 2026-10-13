import copy
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error
from qiskit_ibm_runtime.fake_provider import FakeTorino
import qiskit_paulice as paulice
from qiskit_paulice.noise_models import NoiseModel as PauliceNoiseModel
from qiskit_ibm_runtime.fake_provider import FakeTorino

def get_default_noise_model():
    try:
        backend = FakeTorino()
        return NoiseModel.from_backend(backend)
    
    except Exception:
        noise_model = NoiseModel()
        p1 = 0.001  # 1-qubit gate error rate
        p2 = 0.01   # 2-qubit gate error rate
        
        err1 = depolarizing_error(p1, 1)
        err2 = depolarizing_error(p2, 2)
        
        noise_model.add_all_qubit_quantum_error(err1, ["h", "s", "t", "rz", "x"])
        noise_model.add_all_qubit_quantum_error(err2, ["cx", "cz"])
        return noise_model


def run_ideal_simulation(circuit, shots = 1024):
    # ideal
    ideal_sim = AerSimulator(method="statevector")
    qc = circuit.copy()
    
    # ensure measurement
    if qc.num_clbits == 0:
        qc.measure_all()
        
    result = ideal_sim.run(qc, shots=shots).result()
    return result.get_counts()


def apply_clifford_spacetime_checks(circuit, num_checks, target_qubits):
    if num_checks == 0:
        qc = circuit.copy()
        if qc.num_clbits == 0:
            qc.measure_all()
        return qc, []

    targets = target_qubits if target_qubits else list(range(circuit.num_qubits))

    try:
        import qiskit_paulice as paulice
        from qiskit_paulice.noise_models import NoiseModel as PauliceNoiseModel
        from qiskit_ibm_runtime.fake_provider import FakeTorino

        # 1. Prepare measured circuit copy for paulice
        qc_measured = circuit.copy()
        if qc_measured.num_clbits == 0:
            qc_measured.measure_all()

        layout = list(range(circuit.num_qubits))
        paulice_noise = PauliceNoiseModel.from_backend(FakeTorino(), layout)
        data_qubit_indices = list(range(circuit.num_qubits))

        # 2. Pass measured_qubits explicitly to add_pauli_checks
        results = paulice.add_pauli_checks(
            qc_measured,
            targets,
            paulice_noise,
            measured_qubits=data_qubit_indices
        )

        selected_idx = min(num_checks, len(results) - 1)
        checked_item = results[selected_idx]
        checked_qc = checked_item.circuit if hasattr(checked_item, "circuit") else checked_item

        ancilla_indices = list(range(circuit.num_qubits, checked_qc.num_clbits))
        return checked_qc, ancilla_indices

    except Exception:
        # Fallback uses unmeasured circuit to avoid duplicate measurement gates
        return _manual_fallback_check_insertion(circuit, num_checks, targets)

def apply_manual_z_checks(circuit, placed_checks):
    
    if not placed_checks:
        qc = circuit.copy()
        if qc.num_clbits == 0:
            qc.measure_all()
        return qc, []

    num_data_qubits = circuit.num_qubits
    num_ancillas = len(placed_checks)

    qr_data = QuantumRegister(num_data_qubits, "q")
    qr_ancilla = QuantumRegister(num_ancillas, "ancilla")
    cr_data = ClassicalRegister(num_data_qubits, "meas")
    cr_syndrome = ClassicalRegister(num_ancillas, "syndrome")

    checked_qc = QuantumCircuit(qr_data, qr_ancilla, cr_data, cr_syndrome)

    # Reconstruct circuit layer with injected Z-checks
    ancilla_idx = 0
    ancilla_bit_indices = []

    # Copy original operations
    for instr, qargs, cargs in circuit.data:
        checked_qc.append(instr, qargs, cargs)

    # Attach manual Z checks
    for check in placed_checks:
        target_q = check["qubit"]
        if target_q < num_data_qubits:
            anc = qr_ancilla[ancilla_idx]
            data_q = qr_data[target_q]

            # Z-basis check: CZ entitlement
            checked_qc.cz(anc, data_q)
            checked_qc.measure(anc, cr_syndrome[ancilla_idx])

            # Classical bit index calculation in combined string
            # Qiskit registers stack: [cr_data, cr_syndrome]
            cbit_global_index = num_data_qubits + ancilla_idx
            ancilla_bit_indices.append(cbit_global_index)
            ancilla_idx += 1

    # Measure data qubits into cr_data
    checked_qc.measure(qr_data, cr_data)

    return checked_qc, ancilla_bit_indices


def _manual_fallback_check_insertion(circuit, num_checks, target_qubits):
    num_data_qubits = circuit.num_qubits
    num_ancillas = min(num_checks, len(target_qubits)) if target_qubits else num_checks

    if num_ancillas == 0:
        qc = circuit.copy()
        if qc.num_clbits == 0:
            qc.measure_all()
        return qc, []

    qr_data = QuantumRegister(num_data_qubits, "q")
    qr_ancilla = QuantumRegister(num_ancillas, "ancilla")
    cr_data = ClassicalRegister(num_data_qubits, "meas")
    cr_syndrome = ClassicalRegister(num_ancillas, "syndrome")

    checked_qc = QuantumCircuit(qr_data, qr_ancilla, cr_data, cr_syndrome)

    for instr, qargs, cargs in circuit.data:
        checked_qc.append(instr, qargs, cargs)

    ancilla_bit_indices = []
    targets = target_qubits[:num_ancillas] if target_qubits else list(range(num_ancillas))

    for idx, t_q in enumerate(targets):
        if t_q < num_data_qubits:
            checked_qc.cx(qr_data[t_q], qr_ancilla[idx])
            checked_qc.measure(qr_ancilla[idx], cr_syndrome[idx])
            ancilla_bit_indices.append(num_data_qubits + idx)

    checked_qc.measure(qr_data, cr_data)
    return checked_qc, ancilla_bit_indices


def run_noisy_simulation(circuit, noise_model = None, shots = 1024):
    if noise_model is None:
        noise_model = get_default_noise_model()

    sim = AerSimulator(noise_model=noise_model)
    result = sim.run(circuit, shots=shots).result()
    return result.get_counts()