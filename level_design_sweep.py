#!/usr/bin/env python3
"""Empirically sweep Clifford level designs against their no-check baseline.

Run from the project root:
    python level_design_sweep.py

Outputs are written to data/:
    level_design_sweep_summary.csv      one row per level configuration
    level_design_sweep_strategies.csv   one row per tested strategy
    level_design_sweep.json             complete results and suggested sequence
    suggested_levels.json               copyable level config if five designs are found

This script does not edit config/levels.json. It is a diagnostic tool for
choosing level parameters from actual Qiskit/Aer simulations.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def parse_int_list(value: str) -> list[int]:
    try:
        values = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use comma-separated integers, e.g. 3,4,5") from exc
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("All values must be positive integers")
    return values


def parse_float_list(value: str) -> list[float]:
    try:
        values = sorted({float(part.strip()) for part in value.split(",") if part.strip()})
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use comma-separated numbers, e.g. 0,0.25,0.5,1.0") from exc
    if not values or any(not (item >= 0 and item < float("inf")) for item in values):
        raise argparse.ArgumentTypeError("Penalty weights must be finite and non-negative")
    return values


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def choose_decreasing_sequence(summaries: list[dict[str, Any]], count: int = 5) -> list[dict[str, Any]]:
    """Choose the best complexity-ordered, non-increasing win-count sequence.

    A win means a checked strategy scores strictly above that design's no-check
    baseline. Strict drops are preferred, but a plateau at zero is allowed: a
    strict 5-stage sequence may not exist if several later levels have no winning
    checked strategies.
    """
    eligible = [
        row for row in summaries
        if row.get("baseline_score") is not None
        and row.get("better_than_no_check") is not None
        and row.get("error") is None
        and int(row.get("failed_strategies", 0) or 0) == 0
    ]
    eligible.sort(key=lambda row: (
        int(row["num_qubits"]), int(row["depth_layers"]), int(row.get("max_checks", 0)),
    ))
    if count < 1 or len(eligible) < count:
        return []

    def path_quality(path: list[dict[str, Any]]) -> tuple[float, ...]:
        # Prefer more distinct drops, then harder levels and a broader difficulty
        # range, while avoiding needless inflation of the winning-strategy count.
        wins = [int(row["better_than_no_check"]) for row in path]
        drops = sum(wins[i - 1] > wins[i] for i in range(1, len(wins)))
        qubit_span = int(path[-1]["num_qubits"]) - int(path[0]["num_qubits"])
        depth_span = int(path[-1]["depth_layers"]) - int(path[0]["depth_layers"])
        total_wins = sum(wins)
        return (drops, qubit_span, depth_span, -total_wins)

    # Keep the best path for each (ending design, length, number of strict drops).
    # The grid is deliberately small (usually tens of designs), so this DP is
    # clear and cheap while allowing plateaus where the winning count is zero.
    best_ending: list[dict[tuple[int, int], list[dict[str, Any]]]] = []
    for i, current in enumerate(eligible):
        states: dict[tuple[int, int], list[dict[str, Any]]] = {(1, 0): [current]}
        current_wins = int(current["better_than_no_check"])
        for j in range(i):
            previous = eligible[j]
            if (
                int(previous["num_qubits"]) > int(current["num_qubits"])
                or int(previous["depth_layers"]) > int(current["depth_layers"])
                or (
                    int(previous["num_qubits"]) == int(current["num_qubits"])
                    and int(previous["depth_layers"]) == int(current["depth_layers"])
                )
                or int(previous["better_than_no_check"]) < current_wins
            ):
                continue
            for (length, drops), path in best_ending[j].items():
                if length >= count:
                    continue
                new_key = (length + 1, drops + int(int(previous["better_than_no_check"]) > current_wins))
                proposal = path + [current]
                old = states.get(new_key)
                if old is None or path_quality(proposal) > path_quality(old):
                    states[new_key] = proposal
        best_ending.append(states)

    finalists = [
        path for states in best_ending for (length, _drops), path in states.items()
        if length == count
    ]
    if not finalists:
        return []
    finalists.sort(key=path_quality, reverse=True)
    return finalists[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubits", type=parse_int_list, default=[3, 4, 5, 6, 7],
                        help="comma-separated data-qubit counts (default: 3,4,5,6,7)")
    parser.add_argument("--depths", type=parse_int_list, default=[4, 12, 20, 28, 36],
                        help="comma-separated payload depths (default: 4,12,20,28,36)")
    parser.add_argument("--max-checks", type=parse_int_list, default=[1, 2],
                        help="check caps to test (default: 1,2; keep small for a faster sweep)")
    parser.add_argument("--shots", type=int, default=512,
                        help="noisy shots per strategy (default: 512; ideal reference still uses at least 8192)")
    parser.add_argument("--check-cost", type=int, default=40,
                        help="energy cost per check (default: 40)")
    penalty_group = parser.add_mutually_exclusive_group()
    penalty_group.add_argument("--check-penalty-weight", type=float, default=None,
                        help="use one score penalty weight for every design instead of the campaign's tiered weights")
    penalty_group.add_argument("--penalty-weights", type=parse_float_list, default=None,
                        help="sweep a comma-separated grid of per-design penalty weights, e.g. 0,0.25,0.5,0.75,1,1.5")
    parser.add_argument("--output-dir", default="data",
                        help="output directory, relative to project root (default: data)")
    args = parser.parse_args()

    if args.shots < 1 or args.check_cost < 1:
        parser.error("--shots and --check-cost must be positive")
    if any(n < 3 for n in args.qubits):
        parser.error("--qubits must be 3 or greater for the project payload circuit")

    try:
        from src.game import load_level_configs
        from src.strategy_ranking import compute_strategy_ranking
    except ImportError as exc:
        print(
            "ERROR: install the project dependencies before running the real sweep: "
            f"{type(exc).__name__}: {exc}", file=sys.stderr
        )
        return 2

    base_levels = load_level_configs()
    if not base_levels:
        print("ERROR: no levels found in config/levels.json", file=sys.stderr)
        return 2
    template_level_id = next(iter(base_levels))
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    base_candidates = list(itertools.product(args.qubits, args.depths, args.max_checks))
    if args.penalty_weights is None:
        candidates = [(n, depth, checks, None) for n, depth, checks in base_candidates]
    else:
        candidates = [
            (n, depth, checks, weight)
            for n, depth, checks in base_candidates
            for weight in args.penalty_weights
        ]
    summaries: list[dict[str, Any]] = []
    strategy_rows: list[dict[str, Any]] = []
    start_all = time.time()
    print(f"Testing {len(candidates)} level designs using {args.shots} noisy shots per strategy.")
    print("This runs real simulations; it may take a while. Existing campaign config is not changed.\n")
    if args.penalty_weights is not None:
        print(f"Penalty grid: {args.penalty_weights}")

    for design_index, (n, depth, max_checks, swept_penalty_weight) in enumerate(candidates, start=1):
        # Match the energy budgets and penalty tiers used by config/levels.json.
        # These are intentionally independent of --max-checks so the simulation
        # tests the same resource economics as the actual campaign.
        staged_budgets = {3: 100, 4: 100, 5: 120, 6: 150, 7: 160}
        staged_weights = {3: 0.0, 4: 0.1125, 5: 0.27, 6: 0.75, 7: 1.0}
        energy_budget = staged_budgets.get(n, max(100, 4 * args.check_cost))
        penalty_weight = args.check_penalty_weight
        if penalty_weight is None:
            penalty_weight = swept_penalty_weight
        if penalty_weight is None:
            penalty_weight = staged_weights.get(n, min(1.0, max(0.0, (n - 3) * 0.25)))
        configured_cap = min(max_checks, n)
        affordable_checks = min(configured_cap, energy_budget // args.check_cost)
        cfg = dict(base_levels[template_level_id])
        cfg.update({
            "title": f"Sweep: {n} qubits / depth {depth}",
            "payload_type": "clifford",
            "num_qubits": n,
            "depth_layers": depth,
            "t_count": 0,
            "max_checks": affordable_checks,
            "check_cost": args.check_cost,
            "energy_budget": energy_budget,
            "check_penalty_weight": penalty_weight,
        })
        design_id = f"n{n}_d{depth}_c{affordable_checks}"
        if swept_penalty_weight is not None:
            weight_label = f"{swept_penalty_weight:g}".replace(".", "p")
            design_id += f"_w{weight_label}"
        print(f"[{design_index}/{len(candidates)}] {design_id}: ranking strategies...", flush=True)
        started = time.time()
        try:
            result = compute_strategy_ranking(
                level_id=template_level_id,
                cfg=cfg,
                difficulty="Design sweep",
                shots=args.shots,
            )
            rows = result.get("rows", [])
            baseline = next((row for row in rows if not row.get("targets")), None)
            baseline_score = baseline.get("score") if baseline else None
            better = sum(
                1 for row in rows
                if row.get("targets") and baseline_score is not None
                and row.get("score") is not None
                and row["score"] > baseline_score + 1e-9
            )
            tied = sum(
                1 for row in rows
                if row.get("targets") and baseline_score is not None
                and row.get("score") is not None
                and abs(row["score"] - baseline_score) <= 1e-9
            )
            worse = sum(
                1 for row in rows
                if row.get("targets") and baseline_score is not None
                and row.get("score") is not None
                and row["score"] < baseline_score - 1e-9
            )
            summary = {
                "design_id": design_id,
                "num_qubits": n,
                "depth_layers": depth,
                "max_checks": affordable_checks,
                "check_cost": args.check_cost,
                "check_penalty_weight": penalty_weight,
                "energy_budget": cfg["energy_budget"],
                "shots": args.shots,
                "candidate_strategies": result.get("candidate_count", len(rows)),
                "valid_strategies": len(rows),
                "failed_strategies": len(result.get("failed_rows", [])),
                "baseline_score": baseline_score,
                "best_score": result.get("best_score"),
                "worst_score": result.get("worst_score"),
                "better_than_no_check": better,
                "tied_with_no_check": tied,
                "worse_than_no_check": worse,
                "elapsed_seconds": round(time.time() - started, 2),
            }
            summaries.append(summary)
            for row in rows + result.get("failed_rows", []):
                strategy_rows.append({
                    "design_id": design_id,
                    "num_qubits": n,
                    "depth_layers": depth,
                    "max_checks": affordable_checks,
                    "baseline_score": baseline_score,
                    **row,
                })
            print(
                f"    baseline={baseline_score}; better={better}, tied={tied}, worse={worse}; "
                f"{summary['elapsed_seconds']}s",
                flush=True,
            )
        except Exception as exc:
            summary = {
                "design_id": design_id, "num_qubits": n, "depth_layers": depth,
                "max_checks": affordable_checks, "check_cost": args.check_cost,
                "check_penalty_weight": penalty_weight,
                "energy_budget": cfg["energy_budget"], "shots": args.shots,
                "candidate_strategies": None, "valid_strategies": 0,
                "failed_strategies": None, "baseline_score": None, "best_score": None,
                "worst_score": None, "better_than_no_check": None,
                "tied_with_no_check": None, "worse_than_no_check": None,
                "elapsed_seconds": round(time.time() - started, 2),
                "error": f"{type(exc).__name__}: {exc}",
            }
            summaries.append(summary)
            print(f"    ERROR: {summary['error']}", flush=True)

        # Persist after every design so partial results survive interruption.
        write_csv(output_dir / "level_design_sweep_summary.csv", summaries)
        write_csv(output_dir / "level_design_sweep_strategies.csv", strategy_rows)

    sequence = choose_decreasing_sequence(summaries, count=5)
    report = {
        "description": "Actual simulated scores; better_than_no_check counts checked strategies with a score strictly above that design's no-check baseline.",
        "shots": args.shots,
        "grid": {"qubits": args.qubits, "depths": args.depths, "max_checks": args.max_checks,
                 "check_cost": args.check_cost, "check_penalty_weight_override": args.check_penalty_weight,
                 "penalty_weights_swept": args.penalty_weights,
                 "default_energy_budget_by_qubits": {"3": 100, "4": 100, "5": 120, "6": 150, "7": 160},
                 "default_penalty_weight_by_qubits": {"3": 0.0, "4": 0.1125, "5": 0.27, "6": 0.75, "7": 1.0}},
        "designs_tested": len(summaries),
        "elapsed_seconds": round(time.time() - start_all, 2),
        "summaries": summaries,
        "suggested_five_level_sequence": [
            {key: row[key] for key in (
                "design_id", "num_qubits", "depth_layers", "max_checks", "energy_budget", "check_penalty_weight",
                "candidate_strategies", "baseline_score", "best_score", "better_than_no_check",
                "tied_with_no_check", "worse_than_no_check"
            )}
            for row in sequence
        ],
        "suggested_level_configs": {
            f"level_{index}": {
                "title": f"Level {index}: {('Outpost Sector', 'Hazardous Corridor', 'Deep Lattice Storm', 'Thermal Decay Nexus', 'Clifford Core Meltdown')[index - 1]}",
                "payload_type": "clifford",
                "num_qubits": int(row["num_qubits"]),
                "depth_layers": int(row["depth_layers"]),
                "max_checks": int(row["max_checks"]),
                "energy_budget": int(row["energy_budget"]),
                "check_cost": int(row["check_cost"]),
                "check_penalty_weight": float(row["check_penalty_weight"]),
                "target_score": 55.0 + 2.5 * (index - 1),
                "min_retention": max(0.10, 0.25 - 0.025 * (index - 1)),
                "description": f"Suggested from a {int(row['shots'])}-shot design sweep; re-run with more shots before finalizing.",
            }
            for index, row in enumerate(sequence, start=1)
        },
        "sequence_note": (
            "Found five increasingly complex designs with strictly fewer winning checked strategies at every stage."
            if sequence and all(int(sequence[i - 1]["better_than_no_check"]) > int(sequence[i]["better_than_no_check"]) for i in range(1, len(sequence))) else
            ("Found five increasingly complex designs with a non-increasing count of winning checked strategies. Equal adjacent counts indicate a plateau (often at zero), so inspect the CSV and confirm the result with a larger shot count."
             if sequence else
             "No five-design non-increasing sequence was found in this grid. Inspect the CSV and expand --depths/--max-checks or adjust --check-penalty-weight." )
        ),
    }
    (output_dir / "level_design_sweep.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print("\nSweep complete.")
    print(f"Summary CSV:    {output_dir / 'level_design_sweep_summary.csv'}")
    print(f"Strategy CSV:   {output_dir / 'level_design_sweep_strategies.csv'}")
    print(f"Full JSON:      {output_dir / 'level_design_sweep.json'}")
    print(report["sequence_note"])
    if sequence:
        print("\nSuggested sequence:")
        for i, row in enumerate(sequence, start=1):
            plateau = " (plateau)" if i > 1 and row["better_than_no_check"] == sequence[i - 2]["better_than_no_check"] else ""
            print(
                f"  Level {i}: {row['num_qubits']} qubits, depth {row['depth_layers']}, "
                f"max checks {row['max_checks']}, penalty weight {row.get('check_penalty_weight', 'n/a')}, "
                f"strategies beating no-check: {row['better_than_no_check']}{plateau}"
            )
        config_path = output_dir / "suggested_levels.json"
        config_path.write_text(json.dumps(report["suggested_level_configs"], indent=2), encoding="utf-8")
        print(f"Suggested level configs: {config_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
