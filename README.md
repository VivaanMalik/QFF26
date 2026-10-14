# Syndrome Siege: QFF 2026 Advanced Track CLI

Syndrome Siege is a terminal game about low-overhead quantum error detection using Qiskit and `qiskit-paulice`. Clifford missions let the player choose a set of target data qubits. Paulice constructs the spacetime checks, Aer simulates the noisy circuit, Paulice post-selection rejects shots with nonzero syndrome, and the game scores the resulting fidelity-retention trade-off.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python play_cli.py
```

The game runs locally; no IBM Quantum account is needed. It first tries to build a uniform noise model from FakeTorino calibration data. If unavailable, it explicitly labels the permitted synthetic 1% gate / 1% readout noise fallback.

## Main menu

- `1`–`5`: Clifford spacetime-check missions.
- `6`–`10`: T-doped fidelity-study missions without checks, as allowed by the official brief.
- `A`: play through the levels.
- `B`: run no-check baseline measurements across levels.
- `R`: start/resume exhaustive ranking of every strategy for every level in the background.
- `D`: change difficulty.
- `H`: instructions.
- `Q`: quit.

There are no separate `T` or `O` commands. Exhaustive strategy ranking starts only when you choose `R` from the main menu; ordinary level play reads the saved ranking if one exists.

## Player choice

For a Clifford level, enter only the qubits you want to target, for example:

```text
Qubits to check: 0,2,4
```

This means **three target qubits and three requested checks**. There is no separate check-count prompt. Blocked and repeated qubit IDs are rejected, and the selected IDs are normalized into ascending order. This design treats the input as a set rather than making target ordering a separate game decision.

Normal play only runs the strategy the player selected. It does not run an exhaustive ranking search after every mission. A strategy rank is shown only when a completed ranking for the exact level, difficulty, configuration, shot count, scoring version, and package versions is already saved.

Use the main-menu **`R`** command to start or resume the exhaustive campaign ranking in a detached background process. For each Clifford level, the worker evaluates every canonical qubit subset up to the level's check and energy-budget limits, including the no-check baseline. It stores each candidate's result and rank, and saves the complete level table after that level finishes. Paulice-invalid target sets remain in the file with an error status and explanation. T-doped levels are stored as fidelity-only studies because the official path does not apply Paulice checks to non-Clifford circuits.

The search covers **qubit subsets in ascending order**; it does not enumerate every permutation of a target set. Since Paulice's internal search can depend on target order, the result is the best strategy among the game's canonical sorted subsets, not a claim that every possible target ordering was explored. The same number of shots is used for every candidate (2,048 by default), with seeded simulator and Paulice calls. Ranking is still subject to finite-shot sampling; small score differences may not be statistically meaningful.

## Score

Let `F` be post-selected output-distribution fidelity and `R` be the retained-shot fraction. The game uses the fidelity-weighted harmonic mean with `beta = 2`:

\[
S_\beta = 100\frac{(1+\beta^2)FR}{F+\beta^2R}, \qquad \beta=2.
\]

So the implemented score is

\[
S_2 = 100\frac{5FR}{F+4R}.
\]

This weighting gives fidelity more influence than retention. The score does not replace reporting both metrics: retention is also displayed, and each level may require a minimum retention separately.

Fidelity is the squared Bhattacharyya overlap (classical fidelity) between the ideal output distribution and the measured data-output distribution. The ideal Clifford reference uses the stabilizer simulator; the T-doped reference uses an exact statevector probability distribution. The raw fidelity is calculated before filtering, while post-selected fidelity uses only shots accepted by Paulice's all-zero-syndrome post-selection.

## Noise and circuit overhead

The noise model uses a uniform two-qubit depolarizing error and readout error, both matched to a FakeTorino-derived uniform model when possible or to the clearly labelled synthetic fallback. The report includes raw fidelity, post-selected fidelity, retention, weighted score, ancilla count, added two-qubit gates, added depth, and Paulice's gamma cost where available.

## Official method boundaries

- Clifford levels call `qiskit_paulice.add_pauli_checks(...)` to construct spacetime-check variants. The player chooses the target qubit subset; Paulice chooses the checks and their spacetime structure.
- Post-selection uses each returned checked circuit's `get_postselection_method()`; shots with nonzero syndrome are discarded.
- T-doped levels are measured without checks because `qiskit-paulice`'s official workflow accepts Clifford-only circuits. These levels report no-check fidelity, not error-detection gains.
- This is error detection and post-selection, not error correction: the game rejects flagged shots; it does not repair a quantum state.

## Tests

Run the lightweight tests with:

```bash
python -m pytest -q
```

The Paulice smoke tests require Qiskit, Qiskit Aer, and `qiskit-paulice` to be installed. If those dependencies are absent, they are skipped; an end-to-end run should then be performed in the actual project environment.

## Persistent background strategy ranking

Normal gameplay **does not exhaustively rank strategies**. It only looks up a
completed ranking from `data/strategy_rankings.json`; if the current level,
difficulty, shot count, circuit configuration, scoring version, or relevant
package versions do not match a saved record, the CLI says that a rank is not
available yet.

From the main menu, press **`R`** to start or resume the campaign ranker for the
currently selected difficulty. It runs in a detached Python process, so the CLI
returns to the menu immediately and you can continue playing while the scan runs.
The worker visits every campaign level. For Clifford levels it evaluates the
no-check baseline and every legal subset of eligible target qubits up to the
level's check/budget limit. For T-doped levels it stores the permitted no-check
fidelity study only; these levels are not assigned a Paulice strategy score.

After each level has been fully evaluated, all its attempted strategy rows are
written atomically to `data/strategy_rankings.json`, including valid results,
ranks, metrics, circuit overhead, and reasons for strategies that Paulice could
not construct. Thus a completed level remains available if the worker is
interrupted later. A subsequent `R` run skips matching completed records and
continues missing levels. Entries are kept separately for different difficulty
presets/configuration signatures.

Background progress is stored in `data/ranking_status.json`; stdout and errors
are logged to `data/strategy_ranking_worker.log`. Both files and the rankings
JSON are generated locally and are ignored by Git. The worker requires the
project's Qiskit, Qiskit Aer, FakeTorino provider, and `qiskit-paulice`
dependencies to be installed in the same Python environment used to launch the
CLI.
