"""CLI campaign for QFF 2026 spacetime error detection with qiskit-paulice."""
from __future__ import annotations

import math
from typing import Any

from src.game import load_level_configs, play_level
from src.rankings import (
    load_completed_level_ranking,
    launch_background_ranking,
    process_is_running,
    read_job_status,
)

DIFFICULTIES = {"g": "Guided", "s": "Standard", "c": "Full Challenge"}
SHOTS = {"Guided": 2048, "Standard": 2048, "Full Challenge": 2048}


def print_hud(title: str, char: str = "=") -> None:
    border = char * 72
    print(f"\n{border}\n{title.center(72)}\n{border}")


def apply_difficulty(base_cfg: dict[str, Any], difficulty: str) -> dict[str, Any]:
    """Shrink the same payload family for practice; never rescale noise rates."""
    cfg = dict(base_cfg)
    original_qubits = int(base_cfg.get("num_qubits", 3))
    original_depth = int(base_cfg.get("depth_layers", 2))
    original_t_count = int(base_cfg.get("t_count", 0))

    if difficulty == "Guided":
        n_qubits = 3
        depth = min(3, max(2, math.ceil(original_depth / 12)))
        t_count = min(original_t_count, max(1, math.ceil(original_t_count * 0.25))) if original_t_count else 0
        cfg.update({
            "num_qubits": n_qubits,
            "depth_layers": depth,
            "t_count": t_count,
            "max_checks": 2,
            "energy_budget": 100,
            "check_cost": 35,
            "blocked_channels": [],
            "target_score": 45.0,
            "min_retention": 0.10,
            "description": "Guided practice: same payload family in a short 3-qubit circuit, using the shared reference noise model.",
        })
    elif difficulty == "Standard":
        n_qubits = min(5, original_qubits)
        depth = min(12, max(3, math.ceil(original_depth * 0.40)))
        t_count = min(original_t_count, max(1, math.ceil(original_t_count * 0.50))) if original_t_count else 0
        cfg.update({
            "num_qubits": n_qubits,
            "depth_layers": depth,
            "t_count": t_count,
            "max_checks": min(3, int(base_cfg.get("max_checks", 3))),
            "energy_budget": 120,
            "check_cost": 35,
            "blocked_channels": [int(q) for q in base_cfg.get("blocked_channels", []) if int(q) < n_qubits],
            "target_score": 52.0,
            "min_retention": 0.12,
            "description": "Standard practice: smaller instance, same FakeTorino-derived (or explicitly labelled fallback) noise rates.",
        })
    elif difficulty != "Full Challenge":
        raise ValueError(f"Unknown difficulty preset {difficulty!r}.")
    return cfg


def max_affordable_checks(cfg: dict[str, Any], num_available_targets: int) -> int:
    check_cost = max(1, int(cfg.get("check_cost", 1)))
    budget_limit = int(cfg.get("energy_budget", 10**9)) // check_cost
    return min(int(cfg.get("max_checks", num_available_targets)), budget_limit, num_available_targets)


def prompt_for_targets(cfg: dict[str, Any]) -> list[int]:
    """Ask only for the qubits to target; target count equals check count.

    The UI treats the entry as a set and sorts it into ascending order so that
    strategy ranking compares qubit sets rather than an extra ordering choice.
    """
    n = int(cfg["num_qubits"])
    blocked = {int(q) for q in cfg.get("blocked_channels", [])}
    eligible = [q for q in range(n) if q not in blocked]
    max_targets = max_affordable_checks(cfg, len(eligible))

    print("\nTARGET-QUBIT MAP")
    print("  " + "  ".join(f"q{q}{' [BLOCKED]' if q in blocked else ''}" for q in range(n)))
    print(f"Eligible qubits: {', '.join('q'+str(q) for q in eligible) or 'none'}")
    print(f"Choose up to {max_targets} qubits. Each selected qubit corresponds to one Paulice check.")
    print("Example: 0,2,4 means 3 target qubits and therefore 3 requested checks.")
    print("The game will evaluate your set against every other valid qubit set.")

    raw = input("Qubits to check [e.g. 0,2,4; H=hint; N=no checks]: ").strip().lower()
    if raw in {"n", "none", "no", ""}:
        return []
    if raw in {"h", "hint", "?"}:
        take = min(max_targets, len(eligible))
        if take <= 0:
            print("No eligible targets are available.")
            return []
        if take == 1:
            suggestion = [eligible[len(eligible) // 2]]
        else:
            indexes = [round(i * (len(eligible) - 1) / (take - 1)) for i in range(take)]
            suggestion = sorted({eligible[i] for i in indexes})
        print("Suggested qubit set:", ",".join(map(str, suggestion)))
        if input("Use this set? [Y/n]: ").strip().lower() in {"n", "no"}:
            return prompt_for_targets(cfg)
        return suggestion

    try:
        targets = [int(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError:
        print("Please enter integer qubit IDs separated by commas.")
        return prompt_for_targets(cfg)
    if not targets:
        return []
    if len(targets) != len(set(targets)):
        print("Each qubit may appear only once.")
        return prompt_for_targets(cfg)
    if any(q not in eligible for q in targets):
        print(f"Targets must be eligible qubit IDs: {eligible}. Blocked or out-of-range qubits are not allowed.")
        return prompt_for_targets(cfg)
    if len(targets) > max_targets:
        print(f"Choose at most {max_targets} qubits to stay within the check budget.")
        return prompt_for_targets(cfg)

    # The player selects a set, not a permutation. Canonical ordering makes all
    # subset comparisons consistent and avoids duplicate choices such as 0,2 vs 2,0.
    targets = sorted(targets)
    print(f"Selected {len(targets)} qubit(s): {targets}. Requested check count = {len(targets)}.")
    return targets


def describe_metrics(result: dict[str, Any], label: str) -> None:
    metrics = result["metrics"]
    print(f"\n{label}")
    print(f"  Raw output-distribution fidelity : {100 * metrics['raw_fidelity']:6.2f}%")
    print(f"  Post-selected fidelity            : {100 * metrics['post_selected_fidelity']:6.2f}%")
    print(
        f"  Retained shots                    : {metrics['retained_shots']}/{metrics['raw_shots']} "
        f"({100 * metrics['retention_rate']:.2f}%)"
    )
    if result["game_score"] is not None:
        print(f"  Fidelity-weighted score (β=2)      : {result['game_score']:.2f}/100")


def print_cost(result: dict[str, Any]) -> None:
    cost = result["cost"]
    print("\nADDED CIRCUIT COST")
    print(f"  Ancilla qubits       : {cost['ancilla_qubits']}")
    print(f"  Two-qubit gates      : {cost['two_qubit_gates_before']} -> {cost['two_qubit_gates_after']} "
          f"(+{cost['added_two_qubit_gates']})")
    print(f"  Circuit depth        : {cost['depth_before']} -> {cost['depth_after']} "
          f"(+{cost['added_depth']})")
    if result.get("paulice_cost") is not None:
        print(f"  Paulice gamma cost   : {result['paulice_cost']:.6g} (lower is better during check search)")
    supports = result.get("check_supports", [])
    timelines = result.get("spacetime_timeline", [])
    if supports:
        print("  Paulice check details:")
        for index, support in enumerate(supports, start=1):
            support_text = ", ".join(f"q{q}" for q in support) or "empty"
            line = f"    Check {index}: output parity support {{{support_text}}}"
            if index <= len(timelines):
                timeline = timelines[index - 1]
                if timeline["interaction_count"]:
                    line += (
                        f"; ancilla q{timeline['ancilla_qubit']} interacts "
                        f"{timeline['interaction_count']} times across payload-op range "
                        f"{timeline['first_payload_position']}..{timeline['last_payload_position']}"
                    )
            print(line)


def _run_t_doped_analysis(level_id: str, cfg: dict[str, Any], shots: int) -> dict[str, Any]:
    print("\nOFFICIAL MODE: T-DOPED FIDELITY STUDY (NO CHECKS)")
    print("qiskit-paulice accepts Clifford circuits only. The brief explicitly allows")
    print("T-doped circuits to be reported without checks; this level will not claim a")
    print("post-selection gain or a check-placement score.")
    result = play_level(level_id, num_checks=0, target_qubits=[], shots=shots, config_overrides=cfg)
    metrics = result["metrics"]
    print(f"Noise source: {result['noise_source']}")
    print(f"T gates: {cfg.get('t_count', 0)} | Depth: {cfg['depth_layers']} | Data qubits: {cfg['num_qubits']}")
    print(f"No-check output fidelity: {metrics['raw_fidelity'] * 100:.2f}%")
    print(f"Shots sampled: {metrics['raw_shots']}; no post-selection was applied.")
    print(f"Added cost: {result['cost']['ancilla_qubits']} ancillas, 0 added check gates.")
    print("Interpretation: compare this fidelity with the Clifford missions and other T-doped levels.")
    return result


def _format_qubit_set(targets: tuple[int, ...] | list[int]) -> str:
    return "NO CHECKS" if not targets else "{" + ", ".join(f"q{q}" for q in targets) + "}"


def display_saved_strategy_ranking(
    level_id: str,
    difficulty: str,
    cfg: dict[str, Any],
    shots: int,
    user_targets: list[int],
    live_result: dict[str, Any],
) -> None:
    """Display a previously computed rank; never run simulations here."""
    if cfg.get("payload_type") != "clifford":
        print("\nSTRATEGY RANKING: Not applicable to T-doped levels (fidelity-only mode).")
        return

    try:
        entry = load_completed_level_ranking(level_id, difficulty, cfg, shots)
    except Exception as exc:
        print(f"\n[ RANKING DATABASE ] Could not read saved ranking data: {exc}")
        return

    if entry is None:
        print("\nSTRATEGY RANKING: Not calculated for this level and difficulty yet.")
        print("Use [R] from the main menu to start/resume the full campaign ranking in the background.")
        print("This mission did not trigger any ranking simulations.")
        return

    noise_sources = entry.get("noise_sources", [])
    if noise_sources and live_result.get("noise_source") not in noise_sources:
        print("\nSTRATEGY RANKING: Saved results use a different noise source, so ranks were not compared.")
        print("Run [R] from the main menu to rank the campaign under the current environment.")
        return

    if entry.get("rank_status") != "complete":
        print("\nSTRATEGY RANKING: This level's saved ranking is incomplete or failed.")
        print("Use [R] from the main menu to retry/resume the background ranking.")
        return

    strategies = entry.get("strategies", [])
    valid = [
        row for row in strategies
        if row.get("status") == "valid" and isinstance(row.get("score"), (int, float))
    ]
    user_key = sorted(int(q) for q in user_targets)
    user_row = next((row for row in valid if row.get("targets", []) == user_key), None)
    print_hud("SAVED STRATEGY RANKING", "-")
    print(f"Stored in data/strategy_rankings.json | ranked at {entry.get('ranked_at', 'unknown time')}")
    print(f"Scored strategies: {len(valid)} | candidate strategies stored: {len(strategies)}")

    if user_row is None:
        print(f"Your set {_format_qubit_set(user_key)} has no valid score in the saved table.")
        print("It may have failed to construct a valid Paulice variant during the ranking run.")
    else:
        stored_score = float(user_row["score"])
        print("\nYOUR STRATEGY (using saved ranking data)")
        print(f"  Qubit set       : {_format_qubit_set(user_key)}")
        print(f"  Saved score     : {stored_score:.2f}/100")
        print(f"  Rank            : {user_row.get('rank')} of {len(valid)} valid strategies")
        print(f"  Live score      : {float(live_result['game_score']):.2f}/100")
        difference = float(live_result["game_score"]) - stored_score
        print(f"  Live vs saved   : {difference:+.2f} points")
        if abs(difference) > 0.05:
            print("  Note: live and saved scores differ; check that the environment and shot settings match.")

    ranked = sorted(valid, key=lambda row: (int(row.get("rank") or 10**9), tuple(row.get("targets", []))))
    if ranked:
        best = ranked[0]
        print("\nBEST SAVED STRATEGY")
        print(f"  Qubits    : {_format_qubit_set(best['targets'])}")
        print(f"  Score     : {float(best['score']):.2f}/100")
        print(f"  Fidelity  : {100*float(best['post_selected_fidelity']):.2f}%")
        print(f"  Retention : {100*float(best['retention_rate']):.2f}%")
        print("\nTOP STRATEGIES")
        print("Rank | Qubit set      | Checks | Score  | Fidelity | Retention")
        print("-----+----------------+--------+--------+----------+----------")
        for row in ranked[:5]:
            print(
                f"{int(row['rank']):>4} | {_format_qubit_set(row['targets']):<14} | "
                f"{len(row['targets']):>6} | {float(row['score']):>6.2f} | "
                f"{100*float(row['post_selected_fidelity']):>7.2f}% | "
                f"{100*float(row['retention_rate']):>7.2f}%"
            )
    failed = [row for row in strategies if row.get("status") == "invalid"]
    if failed:
        print(f"\n{len(failed)} attempted strategy set(s) were also saved with their failure reason.")


def print_background_ranking_status() -> None:
    status = read_job_status()
    if not status:
        print("Background campaign ranking: not started yet (use [R] to start it).")
        return
    state = status.get("status", "unknown")
    difficulty = status.get("difficulty", "unknown")
    if state in {"starting", "running"} and not process_is_running(int(status.get("pid") or 0)):
        print("Background campaign ranking: interrupted or stale status; use [R] to resume missing levels.")
    elif state in {"starting", "running"}:
        print(
            f"Background campaign ranking: {state} | {difficulty} | "
            f"levels {status.get('completed_levels', 0)}/{status.get('total_levels', 10)} | "
            f"current {status.get('current_level') or '-'} | "
            f"strategies {status.get('strategies_done_in_level', 0)}/"
            f"{status.get('strategies_total_in_level', 0)}"
        )
    elif state == "complete":
        print(
            f"Background campaign ranking: complete for {difficulty} | "
            f"{status.get('completed_levels', 0)}/{status.get('total_levels', 10)} levels."
        )
    elif state == "failed":
        print(f"Background campaign ranking: failed ({status.get('error', 'see worker log')}).")
    else:
        print(f"Background campaign ranking: {state} for {difficulty}.")


def start_campaign_ranking(difficulty: str) -> None:
    print_hud("BACKGROUND CAMPAIGN RANKING")
    print("This evaluates every allowed qubit subset for every level in the current difficulty.")
    print("A level's full strategy table is saved atomically to data/strategy_rankings.json when done.")
    print("The worker runs in a detached process; you can return to the menu and keep playing.")
    state, pid = launch_background_ranking(difficulty)
    if state == "started":
        print(f"Started background ranker (PID {pid}). It will continue if you leave the game.")
        print("Progress/log: data/ranking_status.json and data/strategy_ranking_worker.log")
    elif state == "already_running":
        print(f"A background ranking worker is already running (PID {pid or 'unknown'}).")
    else:
        print("Could not start the worker. See data/strategy_ranking_worker.log and ranking_status.json.")


def run_level(level_id: str, base_cfg: dict[str, Any], difficulty: str) -> None:
    cfg = apply_difficulty(base_cfg, difficulty)
    shots = SHOTS[difficulty]
    title = cfg.get("title", level_id)
    print_hud(f"{title} | {difficulty.upper()}")
    print(cfg.get("description", ""))
    print(
        f"Payload: {cfg['payload_type'].upper()} | {cfg['num_qubits']} data qubits | "
        f"{cfg['depth_layers']} layers | {cfg.get('t_count', 0)} T gates | {shots} shots"
    )

    while True:
        try:
            if cfg["payload_type"] == "tdoped":
                _run_t_doped_analysis(level_id, cfg, shots)
                print("\nT-doped analysis complete. This is a fidelity comparison, not a checked mission.")
            else:
                targets = prompt_for_targets(cfg)
                count = len(targets)  # One entered qubit corresponds to one requested check.
                print(f"\n[SIM] Target count = check count = {count}.")
                kwargs = {"shots": shots, "config_overrides": cfg}
                print("[SIM] Running the no-check baseline...")
                baseline = play_level(level_id, num_checks=0, target_qubits=[], **kwargs)
                if count:
                    print("[SIM] Running your Paulice strategy...")
                    result = play_level(
                        level_id, num_checks=count, target_qubits=targets, **kwargs
                    )
                else:
                    result = baseline

                print_hud("MISSION DEBRIEF", "-")
                print(f"Noise source: {result['noise_source']}")
                if count:
                    describe_metrics(baseline, "NO-CHECK BASELINE")
                describe_metrics(result, "YOUR PAULICE STRATEGY" if count else "NO-CHECK RUN")
                if count:
                    delta = result["game_score"] - baseline["game_score"]
                    print(f"\n  Score change vs baseline: {delta:+.2f} points")
                print(f"  Selected qubits: {_format_qubit_set(targets)}")
                print(f"  Checks requested/used: {count}/{result['num_checks_used']}")
                print_cost(result)

                target_score = float(cfg.get("target_score", 60.0))
                min_retention = float(cfg.get("min_retention", 0.15))
                metrics = result["metrics"]
                print(f"\nMission target: score >= {target_score:.1f}, retention >= {100*min_retention:.1f}%")
                if result["game_score"] >= target_score and metrics["retention_rate"] >= min_retention:
                    stars = "★★★" if result["game_score"] >= target_score + 15 else "★★☆" if result["game_score"] >= target_score + 5 else "★☆☆"
                    print(f"[SUCCESS] Mission complete: {stars}")
                else:
                    print("[PRACTICE] Target not reached yet. Try another qubit set; cached ranks appear after campaign ranking completes.")
                print("Scoring: weighted harmonic mean with beta=2, giving fidelity greater weight.")
                display_saved_strategy_ranking(
                    level_id, difficulty, cfg, shots, targets, result
                )
        except Exception as exc:
            print(f"\n[ERROR] Mission failed: {exc}")
            print("Check that requirements are installed and retry. See README troubleshooting.")

        action = input("\n[R]etry this level, [M]enu: ").strip().lower()
        if action != "r":
            return


def run_baseline_study(levels: dict[str, dict[str, Any]]) -> None:
    """Show no-check output fidelity across the configured campaign levels."""
    print_hud("CAMPAIGN BASELINE STUDY")
    print("All missions are sampled without checks. T-doped rows are fidelity studies per the brief.")
    print("Level | Family      | Qubits | Depth | T count | Fidelity | Retained")
    print("------+-------------+--------+-------+---------+----------+---------")
    for level_id, base_cfg in levels.items():
        cfg = dict(base_cfg)
        result = play_level(level_id, num_checks=0, target_qubits=[], shots=1024, config_overrides=cfg)
        metrics = result["metrics"]
        number = level_id.removeprefix("level_").zfill(2)
        print(
            f"{number:>5} | {cfg['payload_type']:<11} | {cfg['num_qubits']:>6} | "
            f"{cfg['depth_layers']:>5} | {cfg.get('t_count', 0):>7} | "
            f"{100*metrics['raw_fidelity']:7.2f}% | {100*metrics['retention_rate']:6.2f}%"
        )
    print("\nNote: the campaign levels also change width and depth, so fidelity differences reflect multiple factors.")


def choose_difficulty(current: str) -> str:
    print_hud("CHOOSE DIFFICULTY")
    print("[G] Guided        3 qubits, short circuits, no blocked targets")
    print("[S] Standard      up to 5 qubits and medium-depth circuits")
    print("[C] Full Challenge original ten-level sizes and depths")
    print("All presets use the same reference noise model; only circuit size/depth changes.")
    print(f"Current preset: {current}")
    while True:
        choice = input("Choose G/S/C, or ENTER to keep current: ").strip().lower()
        if not choice:
            return current
        if choice in DIFFICULTIES:
            return DIFFICULTIES[choice]
        print("Please choose G, S, C, or ENTER.")


def show_help() -> None:
    print_hud("HOW TO PLAY")
    print("1. Choose a Clifford mission to place spacetime checks, or a T-doped mission to study fidelity.")
    print("2. For a Clifford mission, enter only the qubit IDs you want to target, e.g. 0,2,4.")
    print("3. The number of requested checks equals the number of qubits you enter; there is no second count prompt.")
    print("4. A cached strategy rank is shown only if the background campaign-ranking function has completed for that level/configuration.")
    print("5. qiskit-paulice chooses the valid spacetime Pauli checks; its all-zero-syndrome filter rejects flagged shots.")
    print("6. The score is a beta=2 weighted harmonic mean that gives fidelity greater weight than retention.")
    print("7. Added circuit cost reports ancilla qubits, extra two-qubit gates, and depth overhead.")
    print("\nT-doped levels run without checks because qiskit-paulice accepts Clifford circuits only.")
    print("[B] runs a no-check baseline study. [R] starts/resumes a background ranking of all levels and strategies. [D] changes difficulty.")
    print("Blocked channels cannot be selected. Input is treated as a qubit set and normalized to ascending order.")


def main() -> None:
    levels = load_level_configs()
    difficulty = "Guided"
    while True:
        print_hud("SYNDROME SIEGE | QFF 2026 OFFICIAL-METHOD CLI", "*")
        print(f"Difficulty: {difficulty} (same noise model in every preset)")
        for level_id, cfg in levels.items():
            number = level_id.removeprefix("level_").zfill(2)
            mode = "spacetime checks + strategy ranking" if cfg["payload_type"] == "clifford" else "T-doped fidelity study"
            print(f"  [{number}] {cfg['title']}  ({mode})")
        print("\n  [A] Play all levels")
        print("  [B] No-check baseline study across all levels")
        print("  [R] Start/resume background ranking for every level/strategy")
        print("  [D] Change difficulty")
        print_background_ranking_status()
        print("  [H] How to play")
        print("  [Q] Quit")
        choice = input("Select a level or command: ").strip().lower()
        if choice in {"q", "quit", "exit"}:
            print("Exiting Syndrome Siege.")
            return
        if choice in {"d", "difficulty"}:
            difficulty = choose_difficulty(difficulty)
        elif choice in {"h", "help", "?"}:
            show_help()
            input("\nPress ENTER to return to the menu...")
        elif choice in {"b", "baseline"}:
            run_baseline_study(levels)
            input("\nPress ENTER to return to the menu...")
        elif choice in {"r", "rank", "ranking"}:
            start_campaign_ranking(difficulty)
            input("\nPress ENTER to return to the menu...")
        elif choice in {"a", "all"}:
            for level_id, cfg in levels.items():
                run_level(level_id, cfg, difficulty)
        else:
            level_id = f"level_{choice}" if choice.isdigit() else choice
            if level_id not in levels:
                print("Invalid selection. Enter a level number, A, B, R, D, H, or Q.")
            else:
                run_level(level_id, levels[level_id], difficulty)


if __name__ == "__main__":
    main()
