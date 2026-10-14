"""Beginner-friendly terminal campaign for Syndrome Siege.

This stays deliberately CLI-only: difficulty presets, a short numbered check
shortlist, negative-parity choices surfaced first, hints, and a no-check comparison.
"""
from __future__ import annotations

import math
from typing import Any

from src.circuits import get_clifford_payload, get_tdoped_payload
from src.game import load_level_configs, play_level
from src.simulator import find_deterministic_check_candidates, validate_check_placements

MAX_OPTIONS_PER_PARITY = 4
DIFFICULTIES = {
    "g": "Guided",
    "s": "Standard",
    "c": "Full Challenge",
}


def print_hud(title: str, char: str = "=") -> None:
    border = char * 68
    print(f"\n{border}\n{title.center(68)}\n{border}")


def build_payload(cfg: dict[str, Any]):
    """Build a payload from the exact effective settings shown to the player."""
    if cfg["payload_type"] == "clifford":
        return get_clifford_payload(
            num_qubits=int(cfg["num_qubits"]),
            depth_layers=int(cfg["depth_layers"]),
        )
    if cfg["payload_type"] == "tdoped":
        return get_tdoped_payload(
            num_qubits=int(cfg["num_qubits"]),
            depth_layers=int(cfg["depth_layers"]),
            t_count=int(cfg.get("t_count", 0)),
        )
    raise ValueError(f"Unknown payload type: {cfg['payload_type']!r}")


def apply_difficulty(base_cfg: dict[str, Any], difficulty: str) -> tuple[dict[str, Any], float]:
    """Return an effective level config and noise multiplier.

    The gates inside each payload generator stay the same. Guided and Standard
    simply run smaller instances of that payload family and use gentler noise.
    Full Challenge uses the original level configuration and full game noise.
    """
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
            "description": (
                "Guided practice remix: the same payload family in a short 3-qubit circuit. "
                "Use this run to learn checkpoints and compare checks with the no-check baseline."
            ),
        })
        return cfg, 0.55

    if difficulty == "Standard":
        n_qubits = min(5, original_qubits)
        depth = min(12, max(3, math.ceil(original_depth * 0.40)))
        t_count = min(original_t_count, max(1, math.ceil(original_t_count * 0.50))) if original_t_count else 0
        blocked = [int(q) for q in base_cfg.get("blocked_channels", []) if int(q) < n_qubits]
        cfg.update({
            "num_qubits": n_qubits,
            "depth_layers": depth,
            "t_count": t_count,
            "max_checks": min(3, int(base_cfg.get("max_checks", 3))),
            "energy_budget": 120,
            "check_cost": 35,
            "blocked_channels": blocked,
            "target_score": 52.0,
            "min_retention": 0.12,
            "description": (
                "Standard remix: a smaller instance of the original mission with moderate noise. "
                "Some blocked channels remain, so check both endpoints before choosing a placement."
            ),
        })
        return cfg, 0.80

    if difficulty == "Full Challenge":
        return cfg, 1.0

    raise ValueError(f"Unknown difficulty preset: {difficulty!r}")


def format_qubit_ring(num_qubits: int, blocked: set[int]) -> str:
    """Render a small text map of the circular adjacent-check layout."""
    labels = [f"q{q}{'*' if q in blocked else ''}" for q in range(num_qubits)]
    return " -- ".join(labels) + " -- (back to q0)"


def _spread_pick(items: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Choose a time/qubit-spread sample instead of overwhelming players."""
    ordered = sorted(
        items,
        key=lambda item: (
            int(item["layer"]),
            int(item["qubit"]),
            {"ZZ": 0, "XX": 1, "YY": 2}.get(item["check_type"], 9),
        ),
    )
    if len(ordered) <= limit:
        return ordered
    if limit <= 1:
        return [ordered[len(ordered) // 2]]
    indexes = [round(i * (len(ordered) - 1) / (limit - 1)) for i in range(limit)]
    return [ordered[index] for index in dict.fromkeys(indexes)]


def eligible_candidates(payload, blocked_channels: list[int]) -> list[dict[str, Any]]:
    """Find safe checks, cap the menu, and put valid -1 checks first.

    A negative expected parity is a real stabilizer eigenvalue, not an invalid
    choice: it corresponds to ancilla measurement bit 1. We deliberately show
    examples of both signs when the payload has them, rather than allowing a
    long list of +1 choices to hide the less-obvious -1 cases.
    """
    blocked = set(int(q) for q in blocked_channels)
    found = find_deterministic_check_candidates(payload, blocked_channels=blocked)
    negative = [item for item in found if int(item["expected_bit"]) == 1]
    positive = [item for item in found if int(item["expected_bit"]) == 0]
    # A maximum of six choices per sign keeps the menu manageable. The actual
    # sign is calculated from the ideal circuit; this never invents a -1 option.
    negative = _spread_pick(negative, MAX_OPTIONS_PER_PARITY)
    positive = _spread_pick(positive, MAX_OPTIONS_PER_PARITY)
    type_rank = {"ZZ": 0, "XX": 1, "YY": 2}
    negative.sort(key=lambda item: (int(item["layer"]), int(item["qubit"]), type_rank.get(item["check_type"], 9)))
    positive.sort(key=lambda item: (int(item["layer"]), int(item["qubit"]), type_rank.get(item["check_type"], 9)))
    return negative + positive


def parity_sign(expected_bit: int) -> str:
    return "+1" if int(expected_bit) == 0 else "-1"


def print_candidate_options(candidates: list[dict[str, Any]], num_qubits: int) -> None:
    print("\nSAFE CHECK OPTIONS (curated shortlist)")
    print("Only checks with a deterministic ideal result are shown.")
    print("A -1 parity is valid too: it predicts ancilla bit 1; +1 predicts bit 0.")
    if not candidates:
        print("No eligible deterministic checks were found for this mission.")
        return

    for expected_bit, heading in (
        (1, "EXPECTED PARITY -1  |  EXPECTED ANCILLA BIT 1"),
        (0, "EXPECTED PARITY +1  |  EXPECTED ANCILLA BIT 0"),
    ):
        rows = [(index, item) for index, item in enumerate(candidates, start=1)
                if int(item["expected_bit"]) == expected_bit]
        if not rows:
            continue
        print(f"\n{heading}")
        print("ID   CHECKPOINT   PAIR       TYPE")
        print("---  ----------   ---------  ----")
        for index, item in rows:
            q1 = int(item["qubit"])
            q2 = (q1 + 1) % num_qubits
            print(f"{index:>3}  {int(item['layer']):>10}   q{q1}-q{q2:<5}  {item['check_type']}")


def find_hint(candidates: list[dict[str, Any]], selected: list[dict[str, Any]], payload) -> dict[str, Any] | None:
    """Suggest a deterministic check around the middle-to-late circuit."""
    depth = sum(item.operation.name == "barrier" for item in payload.data) - 1
    target_layer = max(0, min(depth, round(depth * 0.65)))
    rank = {"ZZ": 0, "XX": 1, "YY": 2}
    ordered = sorted(
        candidates,
        key=lambda item: (
            abs(int(item["layer"]) - target_layer),
            0 if int(item["expected_bit"]) == 1 else 1,
            rank.get(item["check_type"], 9),
            int(item["qubit"]),
        ),
    )
    already = {
        (int(item["qubit"]), int(item["layer"]), str(item["check_type"]))
        for item in selected
    }
    for candidate in ordered:
        signature = (int(candidate["qubit"]), int(candidate["layer"]), str(candidate["check_type"]))
        if signature in already:
            continue
        try:
            validate_check_placements(payload, selected + [candidate])
            return candidate
        except ValueError:
            continue
    return None


def prompt_for_checks(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    payload = build_payload(cfg)
    num_qubits = int(cfg["num_qubits"])
    depth = int(cfg["depth_layers"])
    max_checks = int(cfg.get("max_checks", num_qubits))
    cost = int(cfg.get("check_cost", 30))
    budget = int(cfg.get("energy_budget", 100))
    blocked = set(int(q) for q in cfg.get("blocked_channels", []))
    candidates = eligible_candidates(payload, list(blocked))

    print("\nDATA-QUBIT MAP")
    print("  " + format_qubit_ring(num_qubits, blocked))
    print("  * = blocked channel. A check measures adjacent q and (q+1) mod N.")
    print(f"  Checkpoints: 0 through {depth}; 0 is after preparation, {depth} is after the last payload layer.")
    print(f"  Budget: {budget} energy | {cost} per check | up to {max_checks} checks.")
    print("  The menu is intentionally short: up to four examples of each parity sign.")
    print("  -1 means the ideal ancilla bit should be 1; +1 means it should be 0.")

    selected: list[dict[str, Any]] = []
    if not candidates:
        print("No legal deterministic checks are available. Continuing without checks.")
        return selected

    print_candidate_options(candidates, num_qubits)
    while len(selected) < max_checks:
        print(f"\nSelected: {len(selected)}/{max_checks} checks | Energy used: {len(selected) * cost}/{budget}")
        command = input("Add option ID(s) (e.g. 1 or 1,4), [H]int, [0] done: ").strip().lower()

        if command in {"0", "done", "d", ""}:
            break
        if command in {"h", "hint", "?"}:
            hint = find_hint(candidates, selected, payload)
            if hint is None:
                print("No additional compatible hint is available.")
                continue
            q1 = int(hint["qubit"])
            q2 = (q1 + 1) % num_qubits
            print(
                f"HINT: try {hint['check_type']} on q{q1}-q{q2} at checkpoint {hint['layer']} "
                f"(ideal parity {parity_sign(hint['expected_bit'])}; ancilla bit {hint['expected_bit']})."
            )
            if input("Add this hinted check? [Y/n]: ").strip().lower() not in {"n", "no"}:
                if (len(selected) + 1) * cost > budget:
                    print("Not enough energy for another check.")
                else:
                    try:
                        validate_check_placements(payload, selected + [hint])
                        selected.append(hint)
                        print("Hinted check added.")
                    except ValueError as exc:
                        print(f"Hint conflicts with the current plan: {exc}")
            continue

        try:
            ids = [int(part.strip()) for part in command.split(",") if part.strip()]
            if not ids:
                raise ValueError
            if any(candidate_id < 1 or candidate_id > len(candidates) for candidate_id in ids):
                print(f"Choose IDs shown in the shortlist (1-{len(candidates)}).")
                continue
            if len(set(ids)) != len(ids):
                print("Duplicate IDs were entered. Choose each option only once.")
                continue
            additions = [candidates[candidate_id - 1] for candidate_id in ids]
            signatures = {
                (int(item["qubit"]), int(item["layer"]), str(item["check_type"])) for item in selected
            }
            if any(
                (int(item["qubit"]), int(item["layer"]), str(item["check_type"])) in signatures
                for item in additions
            ):
                print("That check is already in your plan.")
                continue
            if len(selected) + len(additions) > max_checks:
                print(f"That would exceed the {max_checks}-check limit.")
                continue
            if (len(selected) + len(additions)) * cost > budget:
                print("That would exceed your energy budget.")
                continue
            validate_check_placements(payload, selected + additions)
            selected.extend(additions)
            for item in additions:
                q1 = int(item["qubit"])
                q2 = (q1 + 1) % num_qubits
                print(
                    f"Added: {item['check_type']} on q{q1}-q{q2}, checkpoint {item['layer']}, "
                    f"expected parity {parity_sign(item['expected_bit'])} "
                    f"(ancilla bit {item['expected_bit']})."
                )
        except ValueError as exc:
            if str(exc):
                print(f"That selection is invalid: {exc}")
            else:
                print("Enter option IDs separated by commas, or use H or 0.")

    return selected

def describe_metrics(result: dict[str, Any], label: str) -> None:
    metrics = result["metrics"]
    print(f"\n{label}")
    print(f"  Raw output fidelity       : {metrics['raw_fidelity'] * 100:6.2f}%")
    print(f"  Post-selected fidelity    : {metrics['post_selected_fidelity'] * 100:6.2f}%")
    print(
        f"  Retained shots            : {metrics['retained_shots']}/{metrics['raw_shots']} "
        f"({metrics['retention_rate'] * 100:.2f}%)"
    )
    print(f"  Harmonic score            : {result['game_score']:.2f}/100")


def run_level(level_id: str, base_cfg: dict[str, Any], difficulty: str) -> None:
    cfg, noise_scale = apply_difficulty(base_cfg, difficulty)
    shots = 2048 if difficulty in {"Guided", "Standard"} else 1024

    print_hud(f"{cfg.get('title', level_id)}  |  {difficulty.upper()}")
    print(cfg.get("description", ""))
    print(
        f"Payload: {cfg['payload_type'].upper()} | {cfg['num_qubits']} data qubits | "
        f"{cfg['depth_layers']} layers | {cfg.get('t_count', 0)} T gates | "
        f"noise x{noise_scale:.2f} | {shots} shots"
    )

    while True:
        checks = prompt_for_checks(cfg)
        print("\n[SIM] Running the noisy circuit...")
        try:
            kwargs = {
                "level_id": level_id,
                "shots": shots,
                "config_overrides": cfg,
                "noise_scale": noise_scale,
            }
            if checks:
                print("[SIM] Running the no-check baseline for comparison...")
                baseline = play_level(placed_checks=[], **kwargs)
                result = play_level(placed_checks=checks, **kwargs)
            else:
                result = play_level(placed_checks=[], **kwargs)
                baseline = result

            print_hud("MISSION DEBRIEF", "-")
            if checks:
                describe_metrics(baseline, "NO-CHECK BASELINE")
                describe_metrics(result, "YOUR CHECK PLAN")
                delta = result["game_score"] - baseline["game_score"]
                if delta > 0.5:
                    print(f"\n  Strategy impact: +{delta:.2f} score points vs no checks.")
                elif delta < -0.5:
                    print(f"\n  Strategy impact: {delta:.2f} score points vs no checks.")
                else:
                    print("\n  Strategy impact: effectively tied with the no-check baseline.")
            else:
                describe_metrics(result, "NO-CHECK RUN")

            target = float(cfg.get("target_score", 60.0))
            min_retention = float(cfg.get("min_retention", 0.15))
            metrics = result["metrics"]
            print(f"\n  Mission target             : score >= {target:.1f}, retention >= {min_retention * 100:.1f}%")
            passed = result["game_score"] >= target and metrics["retention_rate"] >= min_retention
            if passed:
                stars = "★★★" if result["game_score"] >= target + 15 else "★★☆" if result["game_score"] >= target + 5 else "★☆☆"
                print(f"  [SUCCESS] Mission complete: {stars}")
            else:
                print("  [PRACTICE] Target not reached yet. This is a tuning puzzle, not a dead end.")
                if result["game_score"] < target:
                    print("    Try moving a check, changing its Pauli type, or using fewer checks.")
                if metrics["retention_rate"] < min_retention:
                    print("    Too many shots were rejected. Remove a check or place checks at fewer checkpoints.")
            print("\n  Reading the results:")
            print("    Fidelity measures how similar the output distribution is to the ideal circuit.")
            print("    Retention is the fraction of shots whose measured syndromes match the ideal ones.")
            print("    The score balances both: 100 x 2 x F x R / (F + R).")
            print("    Checks add noisy gates, so more checks are not automatically better.")
        except Exception as exc:
            print(f"\n[ERROR] Simulation failed: {exc}")

        action = input("\n[R]etry this level, [M]enu: ").strip().lower()
        if action != "r":
            return


def choose_difficulty(current: str) -> str:
    print_hud("CHOOSE DIFFICULTY")
    print("  [G] Guided        3 qubits, short circuit, gentle noise, no blocked channels")
    print("  [S] Standard      up to 5 qubits, medium circuit, medium noise")
    print("  [C] Full Challenge original ten-level sizes and full noise")
    print(f"Current preset: {current}")
    while True:
        choice = input("Choose G/S/C, or press ENTER to keep current: ").strip().lower()
        if not choice:
            return current
        if choice in DIFFICULTIES:
            selected = DIFFICULTIES[choice]
            print(f"Difficulty set to {selected}.")
            return selected
        print("Please choose G, S, C, or ENTER.")


def show_help() -> None:
    print_hud("HOW TO PLAY")
    print("1. Pick a mission and a difficulty preset.")
    print("2. Choose from a short numbered shortlist of checks and checkpoints.")
    print("3. The menu shows both -1 and +1 expected parities when the ideal circuit supports them.")
    print("   -1 predicts ancilla bit 1; +1 predicts ancilla bit 0. Both are valid expected outcomes.")
    print("4. Use H for a suggested check, or 0 when done.")
    print("5. The game compares a chosen plan with a run that has no checks.")
    print("\nA checkpoint is a point between circuit layers. Checks placed at different checkpoints detect")
    print("different parts of the error history. Earlier checks add less late-stage overhead; later checks")
    print("observe errors accumulated for longer. A check can still hurt the score because its gates")
    print("are noisy and its syndrome filter discards shots.")


def main() -> None:
    levels = load_level_configs()
    difficulty = "Guided"

    while True:
        print_hud("SYNDROME SIEGE: QUANTUM ERROR DETECTION", "*")
        print(f"Difficulty preset: {difficulty}")
        for level_id, cfg in levels.items():
            number = level_id.removeprefix("level_").zfill(2)
            print(f"  [{number}] {cfg['title']} ({cfg['payload_type'].upper()})")
        print("\n  [A] Play all levels")
        print("  [D] Change difficulty")
        print("  [H] How to play")
        print("  [Q] Quit")
        choice = input("Select a level or command: ").strip().lower()

        if choice in {"q", "quit", "exit"}:
            print("Exiting Syndrome Siege.")
            return
        if choice in {"d", "difficulty"}:
            difficulty = choose_difficulty(difficulty)
            continue
        if choice in {"h", "help", "?"}:
            show_help()
            input("\nPress ENTER to return to the menu...")
            continue
        if choice in {"a", "all"}:
            for level_id, cfg in levels.items():
                run_level(level_id, cfg, difficulty)
            continue

        level_id = f"level_{choice}" if choice.isdigit() else choice
        if level_id not in levels:
            print("Invalid selection. Enter a level number, A, D, H, or Q.")
            continue
        run_level(level_id, levels[level_id], difficulty)


if __name__ == "__main__":
    main()
