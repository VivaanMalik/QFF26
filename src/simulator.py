"""Noisy Aer simulation and checkpoint parity-measurement insertion."""
from __future__ import annotations

from typing import Iterable

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Pauli, Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

VALID_CHECK_TYPES = {"XX", "YY", "ZZ"}


def get_default_noise_model(scale: float = 1.0) -> NoiseModel:
    """Return an intentionally amplified, reproducible challenge noise model.

    These are game-tuned rates rather than a claim to reproduce a particular
    chip's calibration. Noise is applied after each listed gate and at readout.
    Rates can be tuned via the constants below for easier/harder campaigns.
    """
    scale = max(0.0, float(scale))
    noise = NoiseModel()
    one_qubit_error = depolarizing_error(min(0.75, 0.006 * scale), 1)
    two_qubit_error = depolarizing_error(min(0.75, 0.018 * scale), 2)
    one_qubit_gates = ["h", "x", "y", "z", "s", "sdg", "t", "tdg", "rz", "sx"]
    two_qubit_gates = ["cx", "cz", "swap"]
    noise.add_all_qubit_quantum_error(one_qubit_error, one_qubit_gates)
    noise.add_all_qubit_quantum_error(two_qubit_error, two_qubit_gates)
    p_readout = min(0.5, 0.04 * scale)
    noise.add_all_qubit_readout_error(
        ReadoutError([[1.0 - p_readout, p_readout], [p_readout, 1.0 - p_readout]])
    )
    return noise


def run_ideal_simulation(circuit: QuantumCircuit, shots: int = 4096) -> dict[str, int]:
    qc = circuit.copy()
    if qc.num_clbits:
        raise ValueError("Ideal payload simulation expects a circuit without classical bits.")
    qc.measure_all()
    result = AerSimulator(method="statevector").run(qc, shots=shots).result()
    return result.get_counts()


def _checkpoint_prefix(circuit: QuantumCircuit, layer: int) -> QuantumCircuit:
    """Return the unitary prefix ending at checkpoint ``layer`` (inclusive)."""
    if layer < 0:
        raise ValueError("Check layer must be non-negative.")
    prefix = QuantumCircuit(circuit.num_qubits)
    barriers_seen = 0
    target_barrier_count = layer + 1
    found = False
    for item in circuit.data:
        if item.operation.name == "barrier":
            barriers_seen += 1
            # Barriers are checkpoint markers, not physical gates. Do not put
            # them in the statevector prefix being simulated.
            if barriers_seen == target_barrier_count:
                found = True
                break
        else:
            qargs = [prefix.qubits[circuit.find_bit(q).index] for q in item.qubits]
            prefix.append(item.operation, qargs)
    if not found:
        raise ValueError(
            f"Layer {layer} does not exist. This circuit has checkpoints 0 through {barriers_seen - 1}."
        )
    return prefix


def expected_check_bit(circuit: QuantumCircuit, check: dict) -> int:
    """Return expected ancilla bit (0 for +1, 1 for -1) for a deterministic check.

    Only deterministic Pauli-pair checks are allowed. This prevents a check
    from collapsing an arbitrary superposition and then falsely calling that
    outcome an error syndrome.
    """
    q = int(check["qubit"])
    layer = int(check["layer"])
    check_type = str(check.get("check_type", "ZZ")).upper()
    n = circuit.num_qubits
    if check_type not in VALID_CHECK_TYPES:
        raise ValueError(f"Invalid check type {check_type!r}; choose XX, YY, or ZZ.")
    if not 0 <= q < n:
        raise ValueError(f"Qubit index {q} is outside 0..{n - 1}.")
    if not 0 <= layer <= sum(item.operation.name == "barrier" for item in circuit.data) - 1:
        raise ValueError("Check layer is outside the circuit's checkpoint range.")

    q2 = (q + 1) % n
    prefix = _checkpoint_prefix(circuit, layer)
    state = Statevector.from_instruction(prefix)
    label = ["I"] * n
    label[n - 1 - q] = check_type[0]
    label[n - 1 - q2] = check_type[1]
    expectation = complex(state.expectation_value(Pauli("".join(label))))
    if abs(expectation.imag) > 1e-7 or abs(abs(expectation.real) - 1.0) > 1e-7:
        raise ValueError(
            f"{check_type} check on q{q}-q{q2} at layer {layer} is not deterministic "
            f"(ideal expectation {expectation.real:+.3f}). Choose another type or checkpoint."
        )
    return 0 if expectation.real > 0 else 1



def find_deterministic_check_candidates(
    circuit: QuantumCircuit, blocked_channels: Iterable[int] = ()
) -> list[dict]:
    """Find safe adjacent Pauli checks at every explicit circuit checkpoint.

    Each returned item contains ``qubit``, ``layer``, ``check_type`` and
    ``expected_bit``. Prefix statevectors are computed once per checkpoint,
    so the CLI can show a menu of valid placements without re-simulating each candidate.
    ``blocked_channels`` excludes any pair touching a blocked data qubit.
    """
    blocked = {int(q) for q in blocked_channels}
    n = circuit.num_qubits
    if n < 2:
        return []

    prefix = QuantumCircuit(n)
    candidates: list[dict] = []
    layer = 0
    for item in circuit.data:
        if item.operation.name == "barrier":
            state = Statevector.from_instruction(prefix)
            for q in range(n):
                q2 = (q + 1) % n
                if q in blocked or q2 in blocked:
                    continue
                for check_type in sorted(VALID_CHECK_TYPES):
                    label = ["I"] * n
                    label[n - 1 - q] = check_type[0]
                    label[n - 1 - q2] = check_type[1]
                    expectation = complex(state.expectation_value(Pauli("".join(label))))
                    if (
                        abs(expectation.imag) <= 1e-7
                        and abs(abs(expectation.real) - 1.0) <= 1e-7
                    ):
                        candidates.append({
                            "qubit": q,
                            "layer": layer,
                            "check_type": check_type,
                            "expected_bit": 0 if expectation.real > 0 else 1,
                        })
            layer += 1
        else:
            qargs = [prefix.qubits[circuit.find_bit(q).index] for q in item.qubits]
            prefix.append(item.operation, qargs)
    return candidates

def _check_pairwise_commuting(placed_checks: list[dict], num_qubits: int) -> None:
    """Reject pairs of Pauli-product measurements that anticommute.

    Two Pauli strings commute when the number of shared qubits on which their
    Pauli axes differ is even. Comparing the two-letter labels alone is not
    sufficient when checks overlap in only one qubit.
    """
    def axes(check: dict) -> dict[int, str]:
        q1 = int(check["qubit"])
        q2 = (q1 + 1) % num_qubits
        check_type = str(check.get("check_type", "ZZ")).upper()
        return {q1: check_type[0], q2: check_type[1]}

    for i, first in enumerate(placed_checks):
        first_axes = axes(first)
        for second in placed_checks[i + 1 :]:
            if int(first["layer"]) != int(second["layer"]):
                continue
            second_axes = axes(second)
            shared = set(first_axes) & set(second_axes)
            anti_commutations = sum(
                first_axes[q] != second_axes[q] for q in shared
            )
            if anti_commutations % 2:
                raise ValueError(
                    f"Checks {first['check_type']}@q{first['qubit']} and "
                    f"{second['check_type']}@q{second['qubit']} at layer "
                    f"{first['layer']} anticommute. Change a check type or "
                    "place the checks at different checkpoints."
                )


def validate_check_placements(circuit: QuantumCircuit, placed_checks: list[dict]) -> list[int]:
    if len(placed_checks) > circuit.num_qubits:
        raise ValueError("At most one check per data qubit is allowed in this campaign build.")
    normalized = []
    for check in placed_checks:
        normalized.append({
            "qubit": int(check["qubit"]),
            "layer": int(check["layer"]),
            "check_type": str(check.get("check_type", "ZZ")).upper(),
        })
    # Validate type, location, and ideal determinism before testing commutation,
    # so malformed user input receives a useful ValueError rather than IndexError.
    expected_bits = [expected_check_bit(circuit, check) for check in normalized]
    _check_pairwise_commuting(normalized, circuit.num_qubits)
    return expected_bits


def _rotate_to_z(qc: QuantumCircuit, q: int, pauli: str) -> None:
    if pauli == "X":
        qc.h(q)
    elif pauli == "Y":
        qc.sdg(q)
        qc.h(q)


def _rotate_from_z(qc: QuantumCircuit, q: int, pauli: str) -> None:
    if pauli == "X":
        qc.h(q)
    elif pauli == "Y":
        qc.h(q)
        qc.s(q)


def apply_spacetime_checks(circuit: QuantumCircuit, placed_checks: list[dict]):
    """Insert ancilla-mediated Pauli-pair checks at selected layer barriers.

    Returns (measured_circuit, expected_syndrome_bits), where expected bits are
    in check-placement order. Classical-register output displays them reversed.
    """
    if circuit.num_clbits:
        raise ValueError("Payload circuit must not already contain classical bits.")
    placed_checks = [
        {"qubit": int(c["qubit"]), "layer": int(c["layer"]), "check_type": str(c.get("check_type", "ZZ")).upper()}
        for c in (placed_checks or [])
    ]
    expected_bits = validate_check_placements(circuit, placed_checks)
    n = circuit.num_qubits
    m = len(placed_checks)

    qr_data = QuantumRegister(n, "data")
    qr_anc = QuantumRegister(m, "anc") if m else None
    cr_data = ClassicalRegister(n, "data_out")
    cr_syn = ClassicalRegister(m, "syndrome") if m else None
    regs = [qr_data]
    if qr_anc is not None:
        regs.append(qr_anc)
    regs.append(cr_data)
    if cr_syn is not None:
        regs.append(cr_syn)
    out = QuantumCircuit(*regs, name=f"{circuit.name}_checked")

    by_layer: dict[int, list[int]] = {}
    for idx, check in enumerate(placed_checks):
        by_layer.setdefault(check["layer"], []).append(idx)

    barriers_seen = 0
    for item in circuit.data:
        if item.operation.name == "barrier":
            layer = barriers_seen
            for idx in by_layer.get(layer, []):
                check = placed_checks[idx]
                q1 = check["qubit"]
                q2 = (q1 + 1) % n
                p1, p2 = check["check_type"]
                _rotate_to_z(out, q1, p1)
                _rotate_to_z(out, q2, p2)
                out.cx(qr_data[q1], qr_anc[idx])
                out.cx(qr_data[q2], qr_anc[idx])
                out.measure(qr_anc[idx], cr_syn[idx])
                _rotate_from_z(out, q2, p2)
                _rotate_from_z(out, q1, p1)
            out.barrier(*qr_data)
            barriers_seen += 1
        else:
            qargs = [qr_data[circuit.find_bit(q).index] for q in item.qubits]
            out.append(item.operation, qargs)

    out.measure(qr_data, cr_data)
    return out, expected_bits


def run_noisy_simulation(circuit: QuantumCircuit, noise_model: NoiseModel | None = None, shots: int = 4096):
    if noise_model is None:
        noise_model = get_default_noise_model()
    sim = AerSimulator(noise_model=noise_model)
    result = sim.run(circuit, shots=shots).result()
    if result.success is False:
        raise RuntimeError(result.status)
    return result.get_counts()
