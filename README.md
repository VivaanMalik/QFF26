# Qubit Crown
## Qiskit Fall Fest 2026 Hackathon | Advanced Track

### 1. Overview

Qubit Crown is a strategy game that lets players explore quantum error detection. Players choose which qubits to check, run noisy quantum circuits, and compare the results with a run that uses no checks. The goal is to understand when checks help and when the cost of discarding measurement results outweighs the benefit.

The project uses Qiskit and Qiskit Aer for circuit simulation, and `qiskit-paulice` to build spacetime checks for Clifford circuits. It is an educational tool for exploring error detection.

### 2. How the Game Works

The campaign has five 'waves', with larger circuits and greater depth as players progress. In each level, players select target qubits for checks. The app runs a noisy simulation, discards outcomes that fail the checks, and compares the results.

| Level | Name | Data Qubits | Circuit Layers |
|---|---|---:|---:|
| 1 | Outpost Sector | 3 | 4 |
| 2 | Hazardous Corridor | 4 | 20 |
| 3 | Deep Lattice Storm | 5 | 20 |
| 4 | Thermal Decay Nexus | 6 | 28 |
| 5 | Clifford Core Meltdown | 7 | 36 |

These are the base settings. Difficulty options may change the effective settings.

### 3. What the App Measures

The app tracks a few key metrics:

- **Raw fidelity:** How closely the noisy results match the ideal results before checks.
- **Post-selected fidelity:** How closely the remaining results match the ideal results after checks.
- **Retention:** The percentage of measurement shots that remain after checks.
- **Circuit cost:** The extra gates or circuit depth introduced by checks, where available.
- **Game score:** A combined score based on fidelity and retention.

Checks can improve the accuracy of the results that remain, but they can also discard many shots. This means that higher fidelity does not always lead to a better score.

The app measures similarity between probability distributions using:

$$F(P,Q)=\left(\sum_x\sqrt{P(x)Q(x)}\right)^2$$

We use the Bhattacharyya-based formula to compare ideal and noisy output probabilities, giving a similarity score between 0 and 1. Alternatives include total variation distance and KL divergence, but this formula provides a simple, intuitive measure of similarity.

The game combines post-selected fidelity and retention using a weighted harmonic score, where $\beta$ is configurable for each level through `levels.json`.

$$
H_{\beta_\ell}=
\frac{(1+\beta_\ell^2)FR}
{F+\beta_\ell^2R}
$$

Here, $F$ is the post-selected fidelity, $R$ is the retention rate, and $\beta_\ell$ controls the balance between them for level $\ell$. A configurable penalty for check cost is then subtracted from the score.

### 4. Circuits and Noise

The main campaign uses Clifford circuits, which are supported by the project's `qiskit-paulice` check workflow. Qiskit Aer runs the noisy simulations.

The app attempts to use a noise model based on FakeTorino calibration data. If that is unavailable, it can use a synthetic model with 1% gate error and 1% readout error.

The app also includes a separate T-gate study. It varies the number of T and Tdg gates and includes a T=0 control. This study does not use Paulice checks because the supported checking workflow is limited to Clifford circuits. It uses statevector probabilities as the ideal reference and assumes 1% depolarizing noise for T/Tdg gates.

### 5. Features and Expected Outcomes

Qubit Crown allows players to compare noisy quantum circuits with and without error-detection checks. It tracks fidelity, retention, and circuit cost to show how different strategies affect the final results.

The project demonstrates three key ideas:
- **Error detection:** Checks can improve the fidelity of accepted measurement results.
- **Post-selection trade-off:** Discarding more shots can improve fidelity but reduce the amount of usable data.
- **Circuit complexity:** Adding checks introduces extra overhead, so using more checks does not always lead to a better score.

The separate T-gate study explores how adding non-Clifford gates affects circuit output under a chosen noise model. Together, these features provide an interactive way to explore the trade-offs involved in quantum error detection.

### 6. Key Conceptual Insights

Qubit Crown explores an important trade-off in quantum error detection: improving the quality of accepted results does not always mean obtaining a better overall outcome. Post-selection can increase fidelity, but it also discards measurement shots, reducing the amount of usable data.

The project also highlights that adding more checks is not always beneficial, as the extra circuit overhead can outweigh the improvement in fidelity. By comparing checked and unchecked circuits, Qubit Crown makes these trade-offs visible through interactive experiments. Its separate T-gate study extends this exploration to non-Clifford circuits under a chosen noise model.

## 8. How to Run

### Requirements

- Python installed
- The project repository, including `app.py` and `requirements.txt`

### Setup

Clone the repository and enter its directory:

```bash
git clone https://github.com/VivaanMalik/QFF26
cd QFF26
```

### Launch the app

```bash
./scripts/run_linux.sh # linux
\scripts\run_windows.bat # windows
```

Open the local URL printed in the terminal. From the app, use Campaign to run a level, Strategy Rankings to compare strategies, and Analysis Lab to explore Clifford Analysis, No-check Baseline, and T-Gate Sweep.

## 9. Limitations and next steps

1. **Statistical uncertainty:** the reported diagnostic probes use only 128 shots per configuration. Repeat runs with multiple seeds and report variation or confidence intervals.
2. **T-gate findings:** run a documented T-count sweep and include matched T=0 controls before making claims about how T count changes fidelity or retention.
3. **Noise comparability:** record and disclose whether each result uses FakeTorino-derived uniform noise or the synthetic fallback.
4. **Score transparency:** explain the harmonic metric and the configurable check-overhead penalty separately, and test that game balance does not obscure the underlying physical metrics.
5. **Final runtime verification:** run the final package in the target environment, navigate every page, execute at least one checked Clifford run, one no-check baseline, and one T-gate sweep, and archive the outputs.

### 10. Conclusion

Qubit Crown turns quantum error detection into a hands-on strategy game. It lets players compare noisy and ideal results, see how checks affect fidelity and retention, and consider the extra cost of using them. The preliminary results demonstrate this trade-off, although more repeated runs and fresh testing are needed before drawing broader conclusions.

### References

1. Qiskit Fall Fest 2026 Hackathon, Advanced Track brief supplied by the organizers.
2. `qiskit-paulice`, used for the Clifford spacetime-check workflow.
3. Qiskit and Qiskit Aer, used for quantum circuit simulation.