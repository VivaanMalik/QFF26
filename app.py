"""Streamlit frontend for Syndrome Siege.

This module is a UI layer over the existing CLI/backend. It intentionally
reuses the same difficulty presets, gameplay function, metrics, and ranking
files instead of reimplementing simulation logic.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from play_cli import SHOTS, apply_difficulty, max_affordable_checks
from src.game import load_level_configs, play_level
from src.rankings import (
    JOB_STATUS_PATH,
    RANKINGS_PATH,
    WORKER_LOG_PATH,
    launch_background_ranking,
    load_completed_level_ranking,
    load_rankings,
    process_is_running,
    read_job_status,
)

st.set_page_config(
    page_title="Syndrome Siege",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { color-scheme: dark; }
    .stApp { background: #080d18; }
    [data-testid="stHeader"] { background: rgba(8,13,24,0.92); }
    [data-testid="stSidebar"] { background: #0d1423; border-right: 1px solid #1e2b43; }
    [data-testid="stSidebar"] * { color: #dce7f8; }
    h1, h2, h3 { color: #edf4ff; letter-spacing: -0.025em; }
    p, li, label, .stMarkdown { color: #c4d0e3; }
    .hero-kicker { color: #6ee7f9; text-transform: uppercase; letter-spacing: .16em;
                   font-size: .74rem; font-weight: 700; }
    .hero-title { font-size: clamp(2rem, 4vw, 3.3rem); line-height: 1.05;
                  font-weight: 800; color: #f4f8ff; margin: .25rem 0 .65rem 0; }
    .hero-copy { color: #9fb0ca; max-width: 780px; font-size: 1.02rem; }
    .panel { background: linear-gradient(145deg, #111c30, #0c1424); border: 1px solid #263651;
             border-radius: 14px; padding: 1rem 1.1rem; margin-bottom: .8rem; }
    .muted { color: #91a2bc; font-size: .9rem; }
    .tag { display:inline-block; border:1px solid #284b69; background:#10273a; color:#80e6f4;
           border-radius:999px; padding:.2rem .55rem; font-size:.76rem; margin-right:.35rem; }
    div[data-testid="stMetric"] { background: #101a2c; border: 1px solid #24344f;
        border-radius: 12px; padding: 14px 16px; }
    div[data-testid="stMetricLabel"] { color: #94a9c6; }
    div[data-testid="stMetricValue"] { color: #f3f7ff; }
    .stButton > button[kind="primary"] { background: #1c9db4; border: 1px solid #42c8dc;
        color: #04111c; font-weight: 750; border-radius: 9px; }
    .stButton > button { border-radius: 9px; }
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
        background: #101a2c; border-color: #2a3b57; }
    hr { border-color: #25344c; }
    </style>
    """,
    unsafe_allow_html=True,
)

DIFFICULTIES = ("Guided", "Standard", "Full Challenge")
DEFAULT_SHOTS = {difficulty: SHOTS[difficulty] for difficulty in DIFFICULTIES}


@st.cache_data(show_spinner=False)
def get_levels() -> dict[str, dict[str, Any]]:
    return load_level_configs()


def level_number(level_id: str) -> int:
    try:
        return int(level_id.rsplit("_", 1)[-1])
    except (ValueError, IndexError):
        return 999


def format_targets(targets: list[int] | tuple[int, ...] | None) -> str:
    values = list(targets or [])
    return "No checks" if not values else "{" + ", ".join(f"q{q}" for q in sorted(values)) + "}"


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def render_hero(kicker: str, title: str, copy: str) -> None:
    st.markdown(
        f'<div class="hero-kicker">{kicker}</div><div class="hero-title">{title}</div>'
        f'<div class="hero-copy">{copy}</div><br>',
        unsafe_allow_html=True,
    )


def render_metrics(result: dict[str, Any], prefix: str = "") -> None:
    metrics = result.get("metrics", {})
    score = result.get("game_score")
    columns = st.columns(4)
    columns[0].metric("Score", f"{score:.2f}/100" if score is not None else "N/A")
    columns[1].metric("Post-selected fidelity", f"{100 * safe_float(metrics.get('post_selected_fidelity')):.2f}%")
    columns[2].metric("Raw fidelity", f"{100 * safe_float(metrics.get('raw_fidelity')):.2f}%")
    columns[3].metric("Retention", f"{100 * safe_float(metrics.get('retention_rate')):.2f}%")
    total = int(metrics.get("raw_shots", 0) or 0)
    retained = int(metrics.get("retained_shots", 0) or 0)
    rejected = max(0, total - retained)
    st.caption(f"{prefix}Shots: {total:,} total · {retained:,} accepted · {rejected:,} rejected")


def render_cost(result: dict[str, Any]) -> None:
    cost = result.get("cost") or {}
    with st.expander("Circuit overhead and check details", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.metric("Ancilla qubits", cost.get("ancilla_qubits", "N/A"))
        before = cost.get("two_qubit_gates_before")
        after = cost.get("two_qubit_gates_after")
        c2.metric("Two-qubit gates", f"{before} → {after}" if before is not None and after is not None else "N/A")
        depth_before, depth_after = cost.get("depth_before"), cost.get("depth_after")
        c3.metric("Circuit depth", f"{depth_before} → {depth_after}" if depth_before is not None and depth_after is not None else result.get("circuit_depth", "N/A"))
        if result.get("paulice_cost") is not None:
            st.write(f"**Paulice gamma cost:** {result['paulice_cost']:.6g}")
        supports = result.get("check_supports") or []
        timelines = result.get("spacetime_timeline") or []
        if supports:
            st.write("**Generated check supports**")
            for i, support in enumerate(supports, start=1):
                text = ", ".join(f"q{q}" for q in support) if support else "empty support"
                line = f"Check {i}: {{{text}}}"
                if i <= len(timelines):
                    timeline = timelines[i - 1]
                    if timeline.get("interaction_count"):
                        line += f" · {timeline['interaction_count']} ancilla interactions"
                st.write(line)
        st.caption(f"Noise source: {result.get('noise_source', 'unknown')}")


def render_ranking_comparison(
    level_id: str,
    difficulty: str,
    cfg: dict[str, Any],
    shots: int,
    targets: list[int],
    result: dict[str, Any],
) -> None:
    if str(cfg.get("payload_type", "clifford")).lower() != "clifford":
        st.info("T-doped levels are fidelity-only studies. Pauli-check strategy rankings do not apply.")
        return
    try:
        entry = load_completed_level_ranking(level_id, difficulty, cfg, shots)
    except Exception as exc:
        st.warning(f"Could not read saved ranking data: {exc}")
        return
    if not entry:
        st.info("No compatible completed ranking is saved for this level and difficulty yet. Open **Strategy Rankings** to start or resume the campaign scan.")
        return
    sources = entry.get("noise_sources", [])
    if sources and result.get("noise_source") not in sources:
        st.warning("Saved rankings use a different noise source, so the saved rank is not compared with this live result.")
        return

    strategies = entry.get("strategies", [])
    valid = [row for row in strategies if row.get("status") == "valid" and isinstance(row.get("score"), (int, float))]
    targets_key = sorted(int(q) for q in targets)
    user_row = next((row for row in valid if sorted(row.get("targets", [])) == targets_key), None)
    st.subheader("Saved strategy comparison")
    if user_row:
        a, b, c = st.columns(3)
        a.metric("Saved rank", f"{user_row.get('rank', 'N/A')} / {len(valid)}")
        a.caption(f"Strategy {format_targets(targets)}")
        b.metric("Saved score", f"{safe_float(user_row.get('score')):.2f}/100")
        live_score = result.get("game_score")
        delta = safe_float(live_score, float("nan")) - safe_float(user_row.get("score")) if live_score is not None else None
        c.metric("Live vs saved", f"{delta:+.2f} points" if delta is not None else "N/A")
    else:
        st.warning("This selected set has no valid saved score, so a rank is unavailable.")

    ranked = sorted(valid, key=lambda row: (int(row.get("rank") or 10**9), tuple(row.get("targets", []))))
    if ranked:
        best = ranked[0]
        st.markdown(f"**Best saved strategy:** {format_targets(best.get('targets'))} · **{safe_float(best.get('score')):.2f}/100**")
        table = []
        for row in ranked[:10]:
            table.append({
                "Rank": row.get("rank"),
                "Qubits": format_targets(row.get("targets")),
                "Checks": len(row.get("targets", [])),
                "Score": safe_float(row.get("score")),
                "Post-selected fidelity (%)": 100 * safe_float(row.get("post_selected_fidelity")),
                "Retention (%)": 100 * safe_float(row.get("retention_rate")),
                "Accepted shots": row.get("retained_shots", 0),
            })
        st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)
        st.caption(f"{len(valid)} valid strategies scored; {len(strategies)} candidate records stored. Ranking timestamp: {entry.get('ranked_at', 'unknown')}.")


def page_campaign(levels: dict[str, dict[str, Any]], difficulty: str) -> None:
    render_hero("Campaign / Mission control", "Choose your checks.", "Select a mission, choose the data qubits to target, and see how syndrome post-selection changes the fidelity–retention trade-off.")
    level_ids = sorted(levels, key=level_number)
    labels = {key: f"{level_number(key):02d} · {levels[key].get('title', key)}" for key in level_ids}
    selected_id = st.selectbox("Mission", level_ids, format_func=lambda key: labels[key], key="campaign_level")
    cfg = apply_difficulty(levels[selected_id], difficulty)
    payload_type = str(cfg.get("payload_type", "clifford")).lower()
    blocked = {int(q) for q in cfg.get("blocked_channels", [])}
    n = int(cfg.get("num_qubits", 3))
    eligible = [q for q in range(n) if q not in blocked]
    maximum = max_affordable_checks(cfg, len(eligible))
    shots = DEFAULT_SHOTS[difficulty]

    with st.container(border=True):
        st.markdown(f"### {cfg.get('title', selected_id)}")
        st.write(cfg.get("description", "No description available."))
        st.markdown(
            f'<span class="tag">{payload_type.upper()}</span><span class="tag">{n} data qubits</span>'
            f'<span class="tag">{cfg.get("depth_layers", "?")} layers</span>'
            f'<span class="tag">{shots:,} shots</span>',
            unsafe_allow_html=True,
        )
        st.caption(f"Mission target: score ≥ {safe_float(cfg.get('target_score', 60)):.1f}/100 · minimum retention {100*safe_float(cfg.get('min_retention', .15)):.1f}%")
        if blocked:
            st.caption("Blocked channels: " + ", ".join(f"q{q}" for q in sorted(blocked)))

    selected_targets: list[int] = []
    if payload_type == "clifford":
        selected_targets = st.multiselect(
            "Qubits to check",
            options=eligible,
            default=[],
            format_func=lambda q: f"q{q}",
            help=f"Choose up to {maximum} qubits. Each selected qubit corresponds to one requested check. Selection order does not matter.",
            key=f"targets_{selected_id}_{difficulty}",
        )
        selected_targets = sorted(set(int(q) for q in selected_targets))
        col1, col2, col3 = st.columns([1, 1, 2])
        col1.metric("Selected qubits", len(selected_targets))
        col2.metric("Requested checks", len(selected_targets))
        col3.markdown(f"**Strategy:** {format_targets(selected_targets)}")
        if len(selected_targets) > maximum:
            st.error(f"This mission permits at most {maximum} checks under the current budget.")
    else:
        st.info("This is a T-doped fidelity study. The official workflow runs it without Pauli checks.")
        st.caption(f"T gates configured: {cfg.get('t_count', 0)}")

    run = st.button("Run strategy" if payload_type == "clifford" else "Run fidelity study", type="primary", use_container_width=True)
    if run:
        if payload_type == "clifford" and len(selected_targets) > maximum:
            st.error("Reduce the selected qubits to fit the mission's check budget.")
        else:
            try:
                with st.spinner("Simulating ideal reference, noise, and post-selection…"):
                    if payload_type == "tdoped":
                        result = play_level(selected_id, num_checks=0, target_qubits=[], shots=shots, config_overrides=cfg)
                        baseline = None
                    else:
                        if selected_targets:
                            baseline = play_level(selected_id, num_checks=0, target_qubits=[], shots=shots, config_overrides=cfg)
                            result = play_level(selected_id, num_checks=len(selected_targets), target_qubits=selected_targets, shots=shots, config_overrides=cfg)
                        else:
                            result = play_level(selected_id, num_checks=0, target_qubits=[], shots=shots, config_overrides=cfg)
                            baseline = None
                # Avoid keeping the QuantumCircuit object in session state; UI needs only result data.
                result_for_state = {key: value for key, value in result.items() if key != "circuit_obj"}
                baseline_for_state = None if baseline is None else {key: value for key, value in baseline.items() if key != "circuit_obj"}
                st.session_state["latest_run"] = {
                    "level_id": selected_id,
                    "difficulty": difficulty,
                    "config": cfg,
                    "targets": selected_targets,
                    "result": result_for_state,
                    "baseline": baseline_for_state,
                    "shots": shots,
                }
            except Exception as exc:
                st.error(f"Simulation failed: {type(exc).__name__}: {exc}")
                st.exception(exc)

    saved = st.session_state.get("latest_run")
    if saved:
        # Do not present a result for a different currently selected configuration as current.
        if saved.get("level_id") == selected_id and saved.get("difficulty") == difficulty:
            st.divider()
            result = saved["result"]
            metrics = result.get("metrics", {})
            st.subheader("Mission debrief")
            st.caption(f"Noise source: {result.get('noise_source', 'unknown')} · Strategy: {format_targets(saved.get('targets'))}")
            render_metrics(result)
            if saved.get("baseline"):
                base = saved["baseline"]
                st.markdown("#### Your strategy vs no-check baseline")
                left, right = st.columns(2)
                with left:
                    st.markdown("**No-check baseline**")
                    render_metrics(base)
                with right:
                    st.markdown("**Selected strategy**")
                    render_metrics(result)
                score = result.get("game_score")
                base_score = base.get("game_score")
                if score is not None and base_score is not None:
                    st.metric("Score change vs baseline", f"{score - base_score:+.2f} points")
            st.success(result.get("feedback", "Simulation complete."))
            target_score = safe_float(saved["config"].get("target_score", 60))
            min_retention = safe_float(saved["config"].get("min_retention", .15))
            retention = safe_float(metrics.get("retention_rate"))
            score = result.get("game_score")
            if score is not None:
                if score >= target_score and retention >= min_retention:
                    st.success("Mission target reached.")
                else:
                    st.info("Mission target not reached yet. Try another valid qubit set and compare the trade-off.")
            render_cost(result)
            render_ranking_comparison(saved["level_id"], difficulty, saved["config"], saved["shots"], saved["targets"], result)
        else:
            st.caption("A previous result is saved for a different level or difficulty. Run the selected mission to update it.")


def ranking_status_summary(status: dict[str, Any] | None) -> tuple[str, str, float]:
    if not status:
        return "Not started", "Start the campaign scan to evaluate candidate strategies.", 0.0
    state = str(status.get("status", "unknown"))
    pid = int(status.get("pid") or 0)
    if state in {"starting", "running"} and not process_is_running(pid):
        state = "interrupted"
        message = "The last worker appears to have stopped. Start/resume the scan to continue missing levels."
    elif state in {"starting", "running"}:
        message = f"Current level: {status.get('current_level') or 'initializing'} · Strategy: {status.get('current_strategy') or 'starting'}"
    elif state == "complete":
        message = f"Ranking scan complete for {status.get('difficulty', 'unknown')} difficulty."
    elif state == "failed":
        message = str(status.get("error") or "Worker failed. Check the worker log.")
    else:
        message = f"Worker state: {state}"
    total = max(1, int(status.get("total_levels", 10) or 10))
    progress = min(1.0, max(0.0, int(status.get("completed_levels", 0) or 0) / total))
    return state.title(), message, progress


def page_rankings(levels: dict[str, dict[str, Any]], difficulty: str) -> None:
    render_hero("Strategy lab / Exhaustive search", "Find the strongest qubit set.", "The worker evaluates every subset of unblocked qubits for each Clifford level. It includes low-retention strategies and candidates beyond normal gameplay check/energy limits.")
    status = read_job_status()
    state_label, message, progress = ranking_status_summary(status)
    with st.container(border=True):
        c1, c2 = st.columns([1, 3])
        c1.metric("Worker status", state_label)
        c2.write(message)
        st.progress(progress, text=f"Campaign progress: {int(progress * 100)}% of levels")
        if status:
            cols = st.columns(4)
            cols[0].metric("Completed levels", f"{status.get('completed_levels', 0)}/{status.get('total_levels', 10)}")
            cols[1].metric("Current level", status.get("current_level") or "—")
            cols[2].metric("Strategies this level", f"{status.get('strategies_done_in_level', 0)}/{status.get('strategies_total_in_level', 0)}")
            cols[3].metric("Cached levels skipped", status.get("skipped_cached_levels", 0))
        b1, b2 = st.columns([1, 1])
        with b1:
            if st.button("Start / resume ranking", type="primary", use_container_width=True):
                try:
                    start_state, pid = launch_background_ranking(difficulty)
                    if start_state == "started":
                        st.success(f"Background ranking started (PID {pid}).")
                    elif start_state == "already_running":
                        st.info(f"A ranking worker is already running (PID {pid or 'unknown'}).")
                    else:
                        st.error("Could not start the ranking worker. Check the status file and worker log.")
                except Exception as exc:
                    st.error(f"Could not start ranking: {type(exc).__name__}: {exc}")
        with b2:
            if st.button("Refresh status and saved results", use_container_width=True):
                st.rerun()
        st.caption(f"Status file: {JOB_STATUS_PATH.relative_to(JOB_STATUS_PATH.parents[1])} · Rankings: {RANKINGS_PATH.relative_to(RANKINGS_PATH.parents[1])}")
        if status and status.get("status") == "failed":
            st.error(f"Worker error: {status.get('error', 'unknown error')}")
            st.caption(f"Worker log: {WORKER_LOG_PATH}")

    level_ids = sorted(levels, key=level_number)
    labels = {key: f"{level_number(key):02d} · {levels[key].get('title', key)}" for key in level_ids}
    selected_id = st.selectbox("View saved ranking for", level_ids, format_func=lambda key: labels[key], key="ranking_level")
    cfg = apply_difficulty(levels[selected_id], difficulty)
    shots = DEFAULT_SHOTS[difficulty]
    try:
        entry = load_completed_level_ranking(selected_id, difficulty, cfg, shots)
    except Exception as exc:
        st.error(f"Ranking database could not be read: {exc}")
        return
    if not entry:
        st.info("No compatible completed ranking is available for this level and difficulty yet. The ranking worker saves each level after evaluating its candidate strategies.")
        return
    if entry.get("payload_type") == "tdoped":
        st.info("This level is a T-doped fidelity-only study. Its saved record is not a strategy ranking.")
        rows = entry.get("strategies", [])
        if rows:
            row = rows[0]
            metrics = [
                {"Metric": "Raw fidelity (%)", "Value": 100 * safe_float(row.get("raw_fidelity"))},
                {"Metric": "Post-selected fidelity (%)", "Value": 100 * safe_float(row.get("post_selected_fidelity"))},
                {"Metric": "Retention (%)", "Value": 100 * safe_float(row.get("retention_rate"))},
            ]
            st.dataframe(pd.DataFrame(metrics), use_container_width=True, hide_index=True)
        return

    st.markdown(f"### {entry.get('level_title', selected_id)}")
    st.caption(f"Completed {entry.get('ranked_at', 'at an unknown time')} · {entry.get('strategies_attempted', 0)} candidates attempted · noise: {', '.join(entry.get('noise_sources', [])) or 'unknown'}")
    valid = [row for row in entry.get("strategies", []) if row.get("status") == "valid" and isinstance(row.get("score"), (int, float))]
    if valid:
        best = entry.get("best_strategy") or valid[0]
        a, b, c = st.columns(3)
        a.metric("Best score", f"{safe_float(best.get('score')):.2f}/100")
        b.metric("Best strategy", format_targets(best.get("targets")))
        c.metric("Valid strategies", len(valid))
        valid = sorted(valid, key=lambda row: (int(row.get("rank") or 10**9), tuple(row.get("targets", []))))
        frame = pd.DataFrame([{
            "Rank": row.get("rank"),
            "Qubits": format_targets(row.get("targets")),
            "Checks": len(row.get("targets", [])),
            "Score": safe_float(row.get("score")),
            "Post-selected fidelity (%)": 100 * safe_float(row.get("post_selected_fidelity")),
            "Raw fidelity (%)": 100 * safe_float(row.get("raw_fidelity")),
            "Retention (%)": 100 * safe_float(row.get("retention_rate")),
            "Accepted shots": row.get("retained_shots", 0),
            "Two-qubit gates added": (row.get("cost") or {}).get("added_two_qubit_gates", "N/A"),
            "Depth added": (row.get("cost") or {}).get("added_depth", "N/A"),
        } for row in valid])
        st.dataframe(frame, use_container_width=True, hide_index=True)
        if len(frame) > 1:
            st.bar_chart(frame.set_index("Qubits")[["Score", "Post-selected fidelity (%)", "Retention (%)"]])
    invalid = [row for row in entry.get("strategies", []) if row.get("status") == "invalid"]
    if invalid:
        with st.expander(f"Failed / invalid candidates ({len(invalid)})"):
            st.dataframe(pd.DataFrame([{
                "Qubits": format_targets(row.get("targets")),
                "Reason": row.get("error", "No error detail saved"),
            } for row in invalid]), use_container_width=True, hide_index=True)


def page_baseline(levels: dict[str, dict[str, Any]], difficulty: str) -> None:
    render_hero("Reference runs / Baseline study", "Measure without checks.", "Run the no-check configuration across the campaign to establish a reference for fidelity and retention. Each level is simulated only when you start the study.")
    st.warning("This runs a simulation for every configured level and may take several minutes. No results are fabricated or filled in before the runs finish.")
    shots = st.number_input("Shots per level", min_value=128, max_value=16384, value=1024, step=128)
    if st.button("Run baseline study", type="primary"):
        rows = []
        progress = st.progress(0.0, text="Preparing baseline study…")
        status = st.empty()
        level_ids = sorted(levels, key=level_number)
        for i, level_id in enumerate(level_ids, start=1):
            cfg = apply_difficulty(levels[level_id], difficulty)
            status.write(f"Running {cfg.get('title', level_id)} ({i}/{len(level_ids)})…")
            try:
                with st.spinner(f"Simulating {level_id}…"):
                    result = play_level(level_id, num_checks=0, target_qubits=[], shots=int(shots), config_overrides=cfg)
                metrics = result.get("metrics", {})
                rows.append({
                    "Level": level_number(level_id),
                    "Mission": cfg.get("title", level_id),
                    "Family": str(cfg.get("payload_type", "clifford")).upper(),
                    "Qubits": cfg.get("num_qubits"),
                    "Depth layers": cfg.get("depth_layers"),
                    "T gates": cfg.get("t_count", 0),
                    "Raw fidelity (%)": 100 * safe_float(metrics.get("raw_fidelity")),
                    "Post-selected fidelity (%)": 100 * safe_float(metrics.get("post_selected_fidelity")),
                    "Retention (%)": 100 * safe_float(metrics.get("retention_rate")),
                    "Shots": metrics.get("raw_shots", int(shots)),
                    "Noise source": result.get("noise_source", "unknown"),
                })
            except Exception as exc:
                rows.append({"Level": level_number(level_id), "Mission": cfg.get("title", level_id), "Family": str(cfg.get("payload_type", "clifford")).upper(), "Error": f"{type(exc).__name__}: {exc}"})
            progress.progress(i / len(level_ids), text=f"Completed {i}/{len(level_ids)} levels")
        st.session_state["baseline_rows"] = rows
        status.success("Baseline study finished. Any failed levels are marked in the table.")
    rows = st.session_state.get("baseline_rows")
    if rows:
        frame = pd.DataFrame(rows)
        st.dataframe(frame, use_container_width=True, hide_index=True)
        if "Raw fidelity (%)" in frame.columns:
            valid_frame = frame[frame["Raw fidelity (%)"].notna()]
            if not valid_frame.empty:
                chart = valid_frame.set_index("Mission")[["Raw fidelity (%)", "Post-selected fidelity (%)"]]
                st.markdown("#### Output fidelity by mission")
                st.bar_chart(chart)
        st.caption("Campaign levels differ in width, depth, and payload family; this is a reference comparison, not a controlled single-variable experiment.")


def page_how_it_works() -> None:
    render_hero("Field guide / Method", "Understand the trade-off.", "Syndrome Siege explores error detection through noisy circuit simulation and post-selection. It rejects flagged shots rather than correcting the quantum state.")
    with st.container(border=True):
        st.markdown("### 1 · Payload circuit")
        st.write("The payload is the quantum computation being evaluated. Clifford levels use the existing Clifford circuit generator; T-doped levels provide non-Clifford fidelity studies.")
        st.markdown("### 2 · Noise")
        st.write("The simulator uses a uniform two-qubit depolarizing and readout-noise model. It tries to derive the rates from FakeTorino calibration data and explicitly labels a synthetic 1% fallback if that path is unavailable.")
        st.markdown("### 3 · Spacetime Pauli checks")
        st.write("On supported Clifford levels, you select target data qubits. `qiskit-paulice` constructs spacetime checks. The game uses the checked circuit's post-selection method to reject shots with a nonzero syndrome.")
        st.markdown("### 4 · Fidelity and retention")
        st.write("Fidelity is the squared Bhattacharyya overlap between the ideal output distribution and the measured data-output distribution. Raw fidelity is measured before filtering; post-selected fidelity uses accepted shots. Retention is the accepted-shot fraction.")
        st.markdown("### 5 · Score")
        st.latex(r"H_\beta = \frac{(1+\beta^2)FR}{F+\beta^2R},\quad S=100H_\beta,\quad \beta=2")
        st.write("This is the formula currently implemented by `src/metrics.py`. Its denominator weights retention by β², so it does not match a fidelity-weighted interpretation of β=2. The frontend preserves the current backend calculation; the discrepancy should be resolved explicitly before changing scores.")
        st.markdown("### 6 · T-doped levels")
        st.write("T-doped circuits are run without Pauli checks because the official `qiskit-paulice` workflow supports Clifford circuits only. They report fidelity results and are not assigned check-placement scores.")
        st.markdown("### 7 · Rankings")
        st.write("The detached ranking worker evaluates every subset of unblocked qubits for Clifford levels, including strategies below the mission minimum-retention threshold and beyond normal gameplay check/energy limits. Blocked qubits are never selected. Saved ranks are only compared when the configuration and environment match.")


def main() -> None:
    levels = get_levels()
    with st.sidebar:
        st.markdown("## ◈ SYNDROME SIEGE")
        st.caption("Quantum error-detection strategy game")
        page = st.radio("Navigation", ["Campaign", "Strategy Rankings", "Baseline Study", "How It Works"], label_visibility="collapsed")
        st.divider()
        difficulty = st.selectbox("Difficulty", DIFFICULTIES, index=1, key="global_difficulty")
        st.caption(f"Shots per gameplay run: {DEFAULT_SHOTS[difficulty]:,}")
        st.divider()
        st.markdown("**Method**")
        st.caption("Clifford checks · syndrome post-selection · noisy Aer simulation")
        st.caption("No IBM Quantum account required")
    if page == "Campaign":
        page_campaign(levels, difficulty)
    elif page == "Strategy Rankings":
        page_rankings(levels, difficulty)
    elif page == "Baseline Study":
        page_baseline(levels, difficulty)
    else:
        page_how_it_works()


if __name__ == "__main__":
    main()
