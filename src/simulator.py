"""Qiskit Aer simulation and official qiskit-paulice spacetime checks."""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Sequence

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel as AerNoiseModel
from qiskit_aer.noise import ReadoutError, depolarizing_error

# The official spacetime-code implementation is imported explicitly here so it
# is visible during code review and easy to verify. The optional-import guard
# only lets non-quantum unit tests (for example, the scoring tests) run in a
# lightweight environment. The game raises a clear error if Paulice is needed
# but is not installed.
try:
    from qiskit_paulice import add_pauli_checks
    from qiskit_paulice.noise_models import NoiseModel as PauliceNoiseModel
except ImportError:  # pragma: no cover - depends on the user's environment
    add_pauli_checks = None
    PauliceNoiseModel = None


@lru_cache(maxsize=1)
def _reference_noise_parameters() -> tuple[float, float, str]:
    """Get one reproducible uniform noise level derived from FakeTorino.

    The same uniform rates are reused across all level widths, so comparisons
    between missions are not silently changing noise just because their size
    changed.  The official brief also permits a synthetic 1% model; that is an
    explicitly labelled fallback if FakeTorino calibration data is unavailable.
    """
    try:
        from qiskit_ibm_runtime.fake_provider import FakeTorino
        if PauliceNoiseModel is None:
            raise ImportError("qiskit-paulice is not installed")

        backend = FakeTorino()
        layout = _connected_layout(backend.coupling_map, min(7, backend.num_qubits))
        inferred = PauliceNoiseModel.from_backend(
            backend, layout=layout, uniform_gate_noise=True
        )
        gate_noise = float(inferred.gate_noise)
        readout_noise = inferred.readout_noise
        readout_noise = 0.01 if readout_noise is None else float(readout_noise)
        if not (0.0 <= gate_noise <= 1.0 and 0.0 <= readout_noise <= 0.5):
            raise ValueError("FakeTorino returned noise probabilities outside valid ranges.")
        return gate_noise, readout_noise, "FakeTorino-derived uniform noise"
    except Exception as exc:
        # This fallback is part of the brief's allowed synthetic-noise option.
        # Label it in results so it cannot be mistaken for backend-derived data.
        return 0.01, 0.01, f"synthetic fallback (gate=1%, readout=1%; {type(exc).__name__})"


def _connected_layout(coupling_map: Any, count: int) -> list[int]:
    """Pick a small connected set of physical qubits for reading FakeTorino rates."""
    try:
        edges = list(coupling_map.get_edges())
    except Exception:
        edges = list(coupling_map)
    adjacency: dict[int, set[int]] = {}
    for a, b in edges:
        adjacency.setdefault(int(a), set()).add(int(b))
        adjacency.setdefault(int(b), set()).add(int(a))
    if not adjacency:
        raise ValueError("Backend coupling map has no readable edges.")

    # Start at a high-degree qubit and grow a connected set. Physical indices
    # are used only to obtain a representative uniform error rate, not as a
    # hard-coded layout for the logical game circuit.
    starts = sorted(adjacency, key=lambda q: (-len(adjacency[q]), q))
    for start in starts:
        chosen = [start]
        seen = {start}
        queue = [start]
        while queue and len(chosen) < count:
            q = queue.pop(0)
            for neighbor in sorted(adjacency.get(q, ())):
                if neighbor not in seen:
                    seen.add(neighbor)
                    chosen.append(neighbor)
                    queue.append(neighbor)
                    if len(chosen) == count:
                        break
        if len(chosen) == count:
            return chosen
    raise ValueError(f"Could not find {count} connected backend qubits.")


def get_noise_models(num_qubits: int = 7):
    """Return (Paulice planner model, Aer model, human-readable source label).

    Both models use the same uniform two-qubit depolarizing and readout rates.
    This follows the Paulice quickstart pattern and the hackathon's permitted
    FakeTorino-derived / synthetic-noise setup.
    """
    if PauliceNoiseModel is None:
        raise RuntimeError(
            "qiskit-paulice is required for the official Clifford check workflow. "
            "Install project dependencies with: python -m pip install -r requirements.txt"
        )

    p_gate, p_readout, source = _reference_noise_parameters()
    paulice_model = PauliceNoiseModel(gate_noise=p_gate, readout_noise=p_readout)

    aer_model = AerNoiseModel()
    if p_gate > 0:
        two_qubit_error = depolarizing_error(min(p_gate, 1.0), 2)
        # The payload uses CX/CZ; the extra names make the model robust to
        # decomposition/transpilation into common two-qubit operations.
        aer_model.add_all_qubit_quantum_error(
            two_qubit_error, ["cx", "cz", "ecr", "swap"]
        )
    if p_readout > 0:
        aer_model.add_all_qubit_readout_error(
            ReadoutError(
                [
                    [1.0 - p_readout, p_readout],
                    [p_readout, 1.0 - p_readout],
                ]
            )
        )
    return paulice_model, aer_model, source


def get_default_noise_model(num_qubits: int = 7) -> AerNoiseModel:
    """Compatibility helper returning the Aer side of the official noise model."""
    return get_noise_models(num_qubits)[1]


def remove_barriers(circuit: QuantumCircuit) -> QuantumCircuit:
    """Copy a payload while removing checkpoint barriers and classical bits."""
    if circuit.num_clbits:
        raise ValueError("Expected an unmeasured payload circuit.")
    out = QuantumCircuit(circuit.num_qubits, name=circuit.name)
    for instruction in circuit.data:
        if instruction.operation.name == "barrier":
            continue
        qargs = [out.qubits[circuit.find_bit(qubit).index] for qubit in instruction.qubits]
        out.append(instruction.operation, qargs)
    return out


def measured_payload(circuit: QuantumCircuit) -> QuantumCircuit:
    """Return a Paulice-ready circuit with terminal data measurements."""
    out = remove_barriers(circuit)
    out.measure_all()
    return out


def run_ideal_simulation(
    circuit: QuantumCircuit,
    shots: int = 8192,
    payload_type: str = "clifford",
) -> dict[str, float]:
    """Reference distribution: stabilizer sampling for Clifford, statevector for T.

    The T-doped path returns exact probabilities (as weights) rather than noisy
    finite-shot estimates, which reduces reference-sampling error.
    """
    clean = remove_barriers(circuit)
    if payload_type.lower() == "tdoped":
        state = Statevector.from_instruction(clean)
        return {key: float(value) for key, value in state.probabilities_dict().items()}

    measured = clean.copy()
    measured.measure_all()
    simulator = AerSimulator(method="stabilizer")
    result = simulator.run(
        measured, shots=max(int(shots), 8192), seed_simulator=1764
    ).result()
    return {key: float(value) for key, value in result.get_counts().items()}


def build_paulice_variants(
    circuit: QuantumCircuit,
    target_qubits: Sequence[int],
    paulice_noise_model: Any,
):
    """Return qiskit-paulice CheckedCircuit variants for 0..N selected targets.

    Paulice itself picks the spacetime Pauli-check support and locations. The
    user chooses target data qubits and the number of checks, as the brief asks.
    """
    if add_pauli_checks is None:
        raise RuntimeError(
            "qiskit-paulice is required for Clifford missions. Install with: "
            "python -m pip install -r requirements.txt"
        )

    targets = [int(q) for q in target_qubits]
    if len(targets) != len(set(targets)):
        raise ValueError("Target qubits must be unique.")
    if any(q < 0 or q >= circuit.num_qubits for q in targets):
        raise ValueError(f"Target qubits must be between 0 and {circuit.num_qubits - 1}.")

    prepared = measured_payload(circuit)
    if not targets:
        # Even the 0-check result is represented by a CheckedCircuit by Paulice;
        # use one empty target list so its own postselection API remains in play.
        return add_pauli_checks(
            prepared, [], paulice_noise_model, cost="gamma", method="windowed", seed=1764
        )
    return add_pauli_checks(
        prepared,
        targets,
        paulice_noise_model,
        cost="gamma",
        method="windowed",
        seed=1764,
    )


def run_noisy_simulation(
    circuit: QuantumCircuit,
    noise_model: AerNoiseModel | None = None,
    shots: int = 1024,
    payload_type: str = "clifford",
) -> dict[str, int]:
    """Run the already-measured circuit with the matching Aer noise model."""
    if noise_model is None:
        noise_model = get_default_noise_model(circuit.num_qubits)
    method = "statevector" if payload_type.lower() == "tdoped" else "stabilizer"
    simulator = AerSimulator(method=method, noise_model=noise_model)
    result = simulator.run(
        circuit, shots=int(shots), seed_simulator=2026
    ).result()
    if not result.success:
        raise RuntimeError(f"Aer simulation failed: {result.status}")
    return result.get_counts()


def circuit_cost(payload_measured: QuantumCircuit, final_circuit: QuantumCircuit, data_qubits: int) -> dict[str, int]:
    """Compute measurable added overhead: ancillas, 2Q gates and depth."""
    def count_two_qubit_gates(qc: QuantumCircuit) -> int:
        return sum(
            1 for item in qc.data
            if item.operation.name not in {"barrier", "measure"}
            and item.operation.num_qubits == 2
        )

    base_two_qubit = count_two_qubit_gates(payload_measured)
    final_two_qubit = count_two_qubit_gates(final_circuit)
    return {
        "ancilla_qubits": max(0, final_circuit.num_qubits - int(data_qubits)),
        "two_qubit_gates_before": base_two_qubit,
        "two_qubit_gates_after": final_two_qubit,
        "added_two_qubit_gates": max(0, final_two_qubit - base_two_qubit),
        "depth_before": int(payload_measured.depth()),
        "depth_after": int(final_circuit.depth()),
        "added_depth": max(0, int(final_circuit.depth()) - int(payload_measured.depth())),
    }
