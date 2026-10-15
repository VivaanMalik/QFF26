# Qubit Crown

**Qubit Crown** is a Qiskit-based strategy game for exploring low-overhead quantum error detection. In Clifford campaign missions, players choose data-qubit targets and `qiskit-paulice` constructs spacetime checks. Qiskit Aer simulates the noisy circuits; the game compares post-selected output-distribution fidelity against the fraction of shots retained. A separate no-check T-gate study explores non-Clifford circuit behaviour without applying Clifford-only spacetime checks to T-doped circuits.

## Launch

Python 3.10+ is recommended. Run these commands from the project root.

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt
streamlit run app.py
```

The application runs locally in Streamlit. It does not require an IBM Quantum account. The simulator first attempts to derive a uniform noise level from FakeTorino calibration data and explicitly labels its synthetic 1% gate / 1% readout fallback when that backend-derived model is unavailable.

## App navigation

The six top-level destinations are:

1. **Main Menu**: a command deck linking to every app section.
2. **Profile**: view personal run history and difficulty-specific campaign progress.
3. **Campaign**: play five Clifford missions; completion unlocks levels only within the same difficulty.
4. **Strategy Rankings**: launch or inspect the background sweep. Each target subset runs in a child process so a native simulator crash marks that strategy unavailable without killing the sweep.
5. **Analysis Lab**: one workspace with three views selected using vertical-numbered buttons:
   - **Clifford Analysis**: select individual target qubits, run one strategy, and inspect fidelity-versus-retention, matched baseline deltas, score by target count, raw versus post-selected fidelity, added two-qubit gates, added depth, and downloadable strategy data.
   - **No-check Baseline**: explicitly run no-check controls, inspect the saved control curve, and compare checked strategies against matched no-check rows.
   - **T-Gate Sweep**: run the separate T/Tdg-count experiment and compare only matching circuit/noise configurations. T=0 is the control; these runs never enter Clifford rankings or campaign progression.
6. **How To**: navigation, control meanings, and metric definitions.

The Analysis Lab shows one view at a time. The sidebar difficulty buttons are vertical and fully labelled, levels use numbered buttons, and Clifford target qubits are selected individually with q0, q1, … buttons. Opening a page does not launch a quantum simulation; runs begin only after an explicit run button is pressed.

## Physics and metric interpretation

For a Clifford mission, players choose a subset of data qubits as targets. The number of selected targets is the requested check count. `qiskit-paulice` constructs the spacetime-check variants; the application uses the resulting checked circuits' post-selection method rather than a hand-built syndrome approximation.

The output-distribution fidelity is the squared Bhattacharyya overlap between the ideal and measured data distributions. **Raw fidelity** is measured before filtering. **Post-selected fidelity** is measured on accepted shots. **Retention** is accepted shots divided by total shots. Fidelity and retention should be read together: a higher post-selected fidelity can come at the cost of discarding more shots.

The backend score uses the following harmonic score, with \(\beta=2\), returned as a 0–1 value by `src/metrics.py` and converted to points by the caller:

\[
H_\beta = \frac{(1+\beta^2)FR}{F+\beta^2R}
\]

where \(F\) is post-selected fidelity and \(R\) is retention. The implementation's exact scoring and any check-overhead penalty are defined by the current source and ranking version. Do not compare saved rankings from different scoring versions as though they were equivalent.

The T-gate study is deliberately independent of campaign progress and Clifford rankings. T-doped circuits run without Paulice checks because the official spacetime-check workflow is intended for Clifford circuits. The T-study adds an explicit 1% single-qubit depolarizing assumption on T/Tdg gates; saved rows label the noise source and this study-only assumption.

## Saved data

- `data/strategy_rankings.json`: saved campaign ranking records.
- `data/strategy_ranking_job.json`: state for the ranking worker.
- `data/t_gate_research_runs.json`: saved T-gate observations; future experiments append records.
- `data/baseline_study_runs.json`: saved no-check controls; future experiments append records.
- `data/player_profiles.sqlite3`: local player/profile database, created on first launch and intentionally not bundled in source releases.

Baseline matching requires comparable level, difficulty, shot count and noise source, along with a configuration signature where present. The app leaves unmatched controls unreported instead of inventing a comparison. Stored analysis can be exported as CSV from the Analysis Lab.

## Tests and validation

Run the lightweight tests from the project root:

```bash
python -m py_compile app.py src/*.py tests/*.py
python -m pytest -q
```

The tests cover data normalization, matched-control calculations, metric validation, saved research logs, and player profiles. End-to-end quantum simulation requires the packages in `requirements.txt`. A syntax/test pass alone is not a substitute for running the Streamlit app and noisy Qiskit/Aer/Paulice experiments in the target environment.

The bundled cache contains 278 strategy rows across five levels and passed saved-data checks for metric integrity, score consistency, and level/difficulty coverage. The uploaded diagnostic recorded 20 isolated strategy probes with zero segfaults in the target environment. The ranking worker isolates every candidate so a future native crash should leave that strategy unavailable and continue the sweep.

For release validation, run the test commands above, launch the app, and perform a fresh Clifford check, no-check baseline, and T-gate sweep in the project's `.venv`.

## Hackathon assessment

See [`docs/HACKATHON_ASSESSMENT.md`](docs/HACKATHON_ASSESSMENT.md) for a provisional code-and-artifact review against the supplied Qiskit Fall Fest 2026 Advanced Track rubric. It is an estimate, not an official judge score.
