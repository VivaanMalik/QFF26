import copy
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_ibm_runtime.fake_provider import FakeTorino


def get_default_noise_model():
    try:
        backend = FakeTorino()
        return NoiseModel.from_backend(backend)
    except Exception:
        noise_model = NoiseModel()
        err1 = depolarizing_error(0.001, 1)
        err2 = depolarizing_error(0.01, 2)
        noise_model.add_all_qubit_quantum_error(err1, ["h", "s", "t", "rz", "x"])
        noise_model.add_all_qubit_quantum_error(err2, ["cx", "cz"])
        return noise_model

def run_ideal_simulation(circuit, shots=1024):
    # run exact statevector sim without hardware noise
    ideal_sim = AerSimulator(method="statevector")
    qc = circuit.copy()
    
    # make sure we have measurements attached
    if qc.num_clbits == 0:
        qc.measure_all()
        
    result = ideal_sim.run(qc, shots=shots).result()
    return result.get_counts()

def apply_manual_z_checks(circuit: QuantumCircuit, placed_checks: list[dict]):
    """
    Applies phase-safe ZZ parity checks between target data qubits and ancillas.
    Preserves superpositions (e.g. GHZ states) while flagging X bit-flip errors.
    """
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

    # 1. Copy original operations
    for instr, qargs, cargs in circuit.data:
        checked_qc.append(instr, qargs, cargs)

    ancilla_bit_indices = []

    # 2. Attach non-demolition ZZ parity checks
    for idx, check in enumerate(placed_checks):
        target_q1 = check["qubit"]
        target_q2 = (target_q1 + 1) % num_data_qubits  # Pair with adjacent qubit for ZZ parity

        anc = qr_ancilla[idx]
        
        # ZZ parity extraction onto ancilla
        checked_qc.cx(qr_data[target_q1], anc)
        checked_qc.cx(qr_data[target_q2], anc)
        checked_qc.measure(anc, cr_syndrome[idx])

        # Track global classical bit index for post-selection filtering
        cbit_global_index = num_data_qubits + idx
        ancilla_bit_indices.append(cbit_global_index)

    # 3. Final data qubit readout
    checked_qc.measure(qr_data, cr_data)
    return checked_qc, ancilla_bit_indices


def _manual_fallback_check_insertion(circuit: QuantumCircuit, num_checks: int, target_qubits: list[int]):
    """
    Fallback ZZ parity check builder for Clifford circuits when qiskit-paulice is unavailable.
    """
    num_data_qubits = circuit.num_qubits
    targets = target_qubits if target_qubits else list(range(num_data_qubits))
    num_ancillas = min(num_checks, len(targets))

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

    for idx in range(num_ancillas):
        q1 = targets[idx]
        q2 = (q1 + 1) % num_data_qubits

        # Non-demolition ZZ parity check
        checked_qc.cx(qr_data[q1], qr_ancilla[idx])
        checked_qc.cx(qr_data[q2], qr_ancilla[idx])
        checked_qc.measure(qr_ancilla[idx], cr_syndrome[idx])
        ancilla_bit_indices.append(num_data_qubits + idx)

    checked_qc.measure(qr_data, cr_data)
    return checked_qc, ancilla_bit_indices


def apply_clifford_spacetime_checks(circuit: QuantumCircuit, num_checks: int, target_qubits: list[int]):
    if num_checks == 0:
        qc = circuit.copy()
        if qc.num_clbits == 0:
            qc.measure_all()
        return qc, []

    targets = target_qubits if target_qubits else list(range(circuit.num_qubits))

    try:
        import qiskit_paulice as paulice
        from qiskit_paulice.noise_models import NoiseModel as PauliceNoiseModel

        qc_measured = circuit.copy()
        if qc_measured.num_clbits == 0:
            qc_measured.measure_all()

        layout = list(range(circuit.num_qubits))
        paulice_noise = PauliceNoiseModel.from_backend(FakeTorino(), layout)
        data_qubit_indices = list(range(circuit.num_qubits))

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
        return _manual_fallback_check_insertion(circuit, num_checks, targets)


def run_noisy_simulation(circuit: QuantumCircuit, noise_model=None, shots: int = 1024):
    if noise_model is None:
        noise_model = get_default_noise_model()

    sim = AerSimulator(noise_model=noise_model)
    result = sim.run(circuit, shots=shots).result()
    return result.get_counts()