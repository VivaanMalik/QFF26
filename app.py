"""Streamlit UI for Syndrome Siege.

A separate process precomputes and persists strategy rankings under data/.
Individual runs and campaign progress remain in the current Streamlit session.
"""
from __future__ import annotations

import os
import subprocess
import sys
from typing import Any

import pandas as pd
import streamlit as st

from src.game import load_level_configs, play_level
from src.strategy_ranking import (
    DATA_DIR, DEFAULT_SHOTS, DIFFICULTIES, PROJECT_ROOT, WORKER_LOG_PATH,
    PASS_RELATIVE_SCORE_THRESHOLD, apply_difficulty, classify_relative_score,
    calculate_campaign_progress, enumerate_strategy_sets, maximum_affordable_checks,
    format_targets, load_job_status, load_ranking, process_is_alive,
    safe_float, save_job_status,
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
    .hero-copy { color: #9fb0ca; max-width: 800px; font-size: 1.02rem; }
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

@st.cache_data(show_spinner=False)
def get_levels() -> dict[str, dict[str, Any]]:
    """Load campaign level definitions relative to the project, not the shell CWD."""
    levels = load_level_configs()
    # The campaign contains only Clifford missions. T-doped experiments are a
    # separate research page that derives variants from these same missions.
    return {
        key: dict(cfg)
        for key, cfg in levels.items()
        if str(cfg.get("payload_type", "clifford")).lower() == "clifford"
    }


def level_number(level_id: str) -> int:
    try:
        return int(level_id.rsplit("_", 1)[-1])
    except (ValueError, IndexError):
        return 999


def render_hero(kicker: str, title: str, copy: str) -> None:
    st.markdown(
        f'<div class="hero-kicker">{kicker}</div><div class="hero-title">{title}</div>'
        f'<div class="hero-copy">{copy}</div><br>',
        unsafe_allow_html=True,
    )


def render_metrics(result: dict[str, Any], include_score: bool = True) -> None:
    metrics = result.get("metrics", {})
    columns = st.columns(4 if include_score else 3)
    if include_score:
        score = result.get("game_score")
        columns[0].metric("Strategy score", f"{score:.2f}/100" if score is not None else "N/A")
        offset = 1
    else:
        offset = 0
    columns[offset].metric("Post-selected fidelity", f"{100 * safe_float(metrics.get('post_selected_fidelity')):.2f}%")
    columns[offset + 1].metric("Raw fidelity", f"{100 * safe_float(metrics.get('raw_fidelity')):.2f}%")
    columns[offset + 2].metric("Shot retention", f"{100 * safe_float(metrics.get('retention_rate')):.2f}%")
    total = int(metrics.get("raw_shots", 0) or 0)
    retained = int(metrics.get("retained_shots", 0) or 0)
    st.caption(f"Shots: {total:,} total · {retained:,} accepted · {max(0, total-retained):,} rejected")
    if include_score and metrics.get("check_overhead_penalty") is not None:
        st.caption(
            f"Score components: harmonic score {100*safe_float(metrics.get('harmonic_score_H')):.2f} "
            f"− check-resource penalty {safe_float(metrics.get('check_overhead_penalty')):.2f}"
        )


def start_ranking_worker() -> bool:
    """Launch one detached worker to precompute every level × difficulty."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    status = load_job_status()
    status_name = str(status.get("status", ""))
    pid = status.get("pid")
    if status_name in {"starting", "running"} and process_is_alive(int(pid) if pid else None):
        return False
    if status_name == "starting" and status.get("updated_at"):
        # Avoid launching a duplicate during the brief parent/worker startup gap.
        try:
            updated = status["updated_at"]
            from datetime import datetime
            age = (datetime.now().astimezone() - datetime.fromisoformat(updated).astimezone()).total_seconds()
            if age < 20:
                return False
        except (TypeError, ValueError):
            pass

    save_job_status({
        "status": "starting", "pid": None, "completed_levels": 0,
        "total_levels": 0, "completed_strategies": 0, "total_strategies": 0,
        "current_level_id": None, "current_level_title": None,
        "current_difficulty": None, "current_strategy": None,
        "error": None, "cancel_requested": False,
    })
    try:
        with WORKER_LOG_PATH.open("a", encoding="utf-8") as log_handle:
            worker = subprocess.Popen(
                [sys.executable, "-m", "src.ranking_worker"],
                cwd=str(PROJECT_ROOT),
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                close_fds=(os.name != "nt"),
            )
        # The worker writes its own PID on startup; this also closes the small
        # startup window where a second click could otherwise launch a duplicate.
        save_job_status({"pid": worker.pid})
        return True
    except Exception as exc:
        save_job_status({"status": "failed", "error": f"Could not start worker: {type(exc).__name__}: {exc}"})
        raise


@st.fragment(run_every=2)
def render_ranking_job_status() -> None:
    """Display persisted worker status; this fragment refreshes while work runs."""
    status = load_job_status()
    state = status.get("status")
    if state in {"starting", "running"}:
        pid = status.get("pid")
        if pid and not process_is_alive(int(pid)):
            save_job_status({
                "status": "failed",
                "error": "The ranking worker exited unexpectedly. Completed level rankings remain saved in data/.",
            })
            status = load_job_status()
            state = status.get("status")
        else:
            done = int(status.get("completed_strategies", 0) or 0)
            total = int(status.get("total_strategies", 0) or 0)
            fraction = min(1.0, done / total) if total else 0.0
            st.progress(fraction, text=f"Background ranking sweep: {done:,}/{total:,} strategies")
            st.caption(
                f"Level {status.get('completed_levels', 0)}/{status.get('total_levels', '?')} complete · "
                f"Current: {status.get('current_level_title') or status.get('current_level_id') or 'Starting'} · "
                f"{status.get('current_difficulty') or ''} · {status.get('current_strategy') or ''}"
            )
            return
    if state == "complete":
        st.success(
            f"Ranking sweep complete: {status.get('completed_levels', 0)} level/difficulty combinations "
            f"and {status.get('completed_strategies', 0):,} strategy simulations saved in data/."
        )
    elif state == "failed":
        st.error(f"Ranking worker failed: {status.get('error', 'Unknown error')}")
        st.caption("Any level rankings saved before the failure remain available in campaign mode.")
    elif state == "cancelled":
        st.warning("Ranking sweep was cancelled. Rankings completed before cancellation remain available.")
    else:
        st.info("No background ranking sweep has been run yet.")


def render_ranking_table(ranking: dict[str, Any], chosen_targets: list[int] | None = None) -> None:
    """Render every valid strategy and label only the metrics that are maxed."""
    rows = ranking.get("rows", [])
    if not rows:
        st.warning("No valid strategy results are available for this ranking yet.")
        return
    top = rows[0]
    st.markdown(f"**Highest-scoring strategy:** {top['strategy']} · **{top['score']:.2f}/100**")
    st.caption(
        f"Score range: {ranking.get('worst_score', 0):.2f} to {ranking.get('best_score', 0):.2f} · "
        f"{len(rows)} valid strategies ranked. PASS threshold: relative score ≥ {PASS_RELATIVE_SCORE_THRESHOLD:.0f}/100."
    )
    if ranking.get("baseline_score") is not None:
        st.info(
            f"No-check baseline: {ranking['baseline_score']:.2f}/100. "
            f"{ranking.get('strategies_beating_no_check', 0)} of "
            f"{ranking.get('non_baseline_strategy_count', 0)} checked strategies score higher."
        )
    table = []
    selected = sorted(int(q) for q in (chosen_targets or []))
    for row in rows:
        tags = []
        if row.get("highest_score"):
            tags.append("Max score")
        if row.get("highest_fidelity"):
            tags.append("Max fidelity")
        if row.get("highest_retention"):
            tags.append("Max retention")
        table.append({
            "Rank": row["rank"],
            "Strategy": row["strategy"],
            "Selected strategy": "Yes" if sorted(row.get("targets", [])) == selected and chosen_targets is not None else "",
            "Checks": row["checks"],
            "Score": round(row["score"], 2),
            "vs no-check": "Better" if row.get("beats_no_check") else ("Baseline" if not row.get("targets") else "Worse / tied"),
            "Relative score": "TIE" if row["relative_score"] is None else round(row["relative_score"], 1),
            "Result": row.get("pass_fail", "FAIL"),
            "Fidelity (%)": round(100 * row["fidelity"], 2),
            "Retention (%)": round(100 * row["retention"], 2),
            "Accepted shots": row["retained_shots"],
            "Maxed values": ", ".join(tags),
        })
    st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True, height=650)
    if ranking.get("noise_sources"):
        st.caption("Noise model: " + "; ".join(ranking["noise_sources"]))
    best_f = max(rows, key=lambda row: row["fidelity"])
    best_r = max(rows, key=lambda row: row["retention"])
    st.write(f"- **Maximum fidelity:** {best_f['strategy']} at {100*best_f['fidelity']:.2f}%. ")
    st.write(f"- **Maximum retention:** {best_r['strategy']} at {100*best_r['retention']:.2f}%. ")
    chart = pd.DataFrame({
        "Fidelity (%)": [100 * row["fidelity"] for row in rows],
        "Retention (%)": [100 * row["retention"] for row in rows],
        "Strategy": [row["strategy"] for row in rows],
    }).set_index("Strategy")
    st.markdown("#### Fidelity versus retention")
    st.scatter_chart(chart, x="Retention (%)", y="Fidelity (%)", height=350)


def campaign_ranking_for(level_id: str, difficulty: str) -> dict[str, Any] | None:
    return load_ranking(level_id, difficulty, DEFAULT_SHOTS[difficulty])


def rank_selected_strategy(ranking: dict[str, Any] | None, targets: list[int] | None) -> dict[str, Any] | None:
    if not ranking:
        return None
    target_list = sorted(int(q) for q in (targets or []))
    return next((row for row in ranking.get("rows", []) if sorted(row.get("targets", [])) == target_list), None)


@st.fragment(run_every=2)
def render_campaign_ranking_panel(level_id: str, difficulty: str, targets: list[int]) -> None:
    """Refresh the campaign's ranking panel as background results are saved."""
    ranking = campaign_ranking_for(level_id, difficulty)
    if ranking:
        render_ranking_table(ranking, targets)
        return
    status = load_job_status()
    if status.get("status") in {"starting", "running"}:
        done = int(status.get("completed_strategies", 0) or 0)
        total = int(status.get("total_strategies", 0) or 0)
        fraction = min(1.0, done / total) if total else 0.0
        st.progress(fraction, text=f"Background sweep: {done:,}/{total:,} strategies")
        st.caption(f"Current task: {status.get('current_level_title') or status.get('current_level_id') or 'Starting'} · {status.get('current_difficulty') or ''}")
        st.info("This level's ranking will appear here automatically when the worker saves it.")
    elif status.get("status") == "failed":
        st.error(f"Ranking worker failed: {status.get('error', 'Unknown error')}")
        st.info("Restart the sweep from Strategy Rankings. Rankings saved before the failure remain available.")
    else:
        st.info("No saved ranking is available yet. Start the background ranking sweep above or from the Strategy Rankings page.")


def render_campaign_rank_summary(result: dict[str, Any], ranking: dict[str, Any] | None, targets: list[int]) -> dict[str, Any] | None:
    row = rank_selected_strategy(ranking, targets)
    if row is None:
        st.metric("Strategy rank", "Ranking pending")
        st.caption("Start the background ranking sweep from Strategy Rankings to calculate the relative score and PASS/FAIL result.")
        return None
    relative = row.get("relative_score")
    pass_fail = row.get("pass_fail", classify_relative_score(relative))
    a, b, c, d = st.columns(4)
    a.metric("Strategy rank", f"#{row['rank']} / {len(ranking.get('rows', []))}")
    b.metric("Ranked strategy score", f"{row['score']:.2f}/100")
    c.metric("Relative score", "TIE" if relative is None else f"{relative:.1f}/100")
    d.metric("Campaign result", pass_fail)
    st.caption("PASS means relative score ≥ 50/100; FAIL means below 50. This classification compares strategies for the same level and difficulty.")
    return row


def render_qubit_picker(level_id: str, difficulty: str, num_qubits: int, max_checks: int, research_mode: bool = False) -> list[int]:
    """Circuit-style, click-to-toggle qubit selection."""
    key = f"qubit_selection::{'lab' if research_mode else 'campaign'}::{level_id}::{difficulty}"
    selected = set(st.session_state.get(key, []))
    st.markdown("#### Select qubits to check")
    st.caption("Click a qubit node to toggle it. Every qubit is available; no channels are blocked.")
    cols = st.columns(num_qubits)
    for q, col in enumerate(cols):
        with col:
            active = q in selected
            label = f"✓  q{q}" if active else f"＋  q{q}"
            if st.button(label, key=f"toggle::{key}::{q}", use_container_width=True, type="primary" if active else "secondary"):
                if active:
                    selected.remove(q)
                else:
                    selected.add(q)
                st.session_state[key] = sorted(selected)
                st.rerun()
    selected = sorted(int(q) for q in st.session_state.get(key, []))
    st.caption("Each button is a data-qubit node in the payload register; highlighted nodes are selected targets.")
    a, b, c = st.columns(3)
    a.metric("Selected qubits", len(selected))
    b.metric("Check budget", f"{max_checks - len(selected)} remaining" if len(selected) <= max_checks else f"{len(selected) - max_checks} over")
    c.write("**Strategy**\n\n" + format_targets(selected))
    clear_col, _ = st.columns([1, 4])
    with clear_col:
        if st.button("Clear selection", key=f"clear::{key}"):
            st.session_state[key] = []
            st.rerun()
    return selected


def render_cost(result: dict[str, Any]) -> None:
    cost = result.get("cost") or {}
    with st.expander("Circuit overhead and check details", expanded=False):
        a, b, c = st.columns(3)
        a.metric("Ancilla qubits", cost.get("ancilla_qubits", "N/A"))
        before, after = cost.get("two_qubit_gates_before"), cost.get("two_qubit_gates_after")
        b.metric("Two-qubit gates", f"{before} → {after}" if before is not None and after is not None else "N/A")
        d0, d1 = cost.get("depth_before"), cost.get("depth_after")
        c.metric("Circuit depth", f"{d0} → {d1}" if d0 is not None and d1 is not None else result.get("circuit_depth", "N/A"))
        if result.get("paulice_cost") is not None:
            st.write(f"**Paulice gamma cost:** {result['paulice_cost']:.6g}")
        supports = result.get("check_supports") or []
        if supports:
            st.write("**Generated check supports**")
            for i, support in enumerate(supports, start=1):
                st.write(f"Check {i}: {{{', '.join(f'q{q}' for q in support)}}}")
        st.caption(f"Noise source: {result.get('noise_source', 'unknown')}")


def campaign_progress(level_ids: list[str], difficulty: str) -> tuple[list[str], set[str], bool]:
    """Progress is keyed by difficulty; a pass never carries across presets."""
    completed_by_difficulty = st.session_state.get("campaign_completed_by_difficulty", {})
    return calculate_campaign_progress(level_ids, completed_by_difficulty, difficulty)


def render_strategy_runner(
    levels: dict[str, dict[str, Any]],
    difficulty: str,
    level_ids: list[str],
    selected_id: str,
    research_mode: bool = False,
) -> None:
    base_cfg = levels[selected_id]
    cfg = apply_difficulty(base_cfg, difficulty)
    n = int(cfg.get("num_qubits", 3))
    shots = DEFAULT_SHOTS[difficulty]
    max_checks = maximum_affordable_checks(cfg, n)
    st.markdown(f"### {cfg.get('title', selected_id)}")
    st.write(cfg.get("description", "No description available."))
    st.markdown(
        f'<span class="tag">CLIFFORD</span><span class="tag">{n} data qubits</span>'
        f'<span class="tag">{cfg.get("depth_layers", "?")} layers</span>'
        f'<span class="tag">{shots:,} shots</span>',
        unsafe_allow_html=True,
    )
    st.caption(f"Mission target: score ≥ {safe_float(cfg.get('target_score', 60)):.1f}/100 · minimum retention {100*safe_float(cfg.get('min_retention', .15)):.1f}%")
    selected = render_qubit_picker(selected_id, difficulty, n, max_checks, research_mode=research_mode)
    if len(selected) > max_checks:
        st.warning(f"This mission permits at most {max_checks} checks under its normal budget. Remove {len(selected) - max_checks} selected qubit(s) to run it.")
    run_col, rank_col = st.columns([1, 1])
    with run_col:
        run_clicked = st.button("Run strategy", type="primary", use_container_width=True, key=f"run::{research_mode}::{selected_id}::{difficulty}")
    with rank_col:
        rank_clicked = st.button("Start background ranking sweep", use_container_width=True, key=f"rank-sweep::{research_mode}::{selected_id}::{difficulty}")

    if rank_clicked:
        try:
            started = start_ranking_worker()
            if started:
                st.success("Background worker started. It will rank every campaign level in Guided, Standard, and Full Challenge.")
            else:
                st.info("The background ranking worker is already starting or running.")
        except Exception as exc:
            st.error(f"Could not start ranking worker: {type(exc).__name__}: {exc}")

    if run_clicked:
        if len(selected) > max_checks:
            st.error("Reduce the selected qubits to fit this mission's check budget.")
        else:
            try:
                with st.spinner("Simulating ideal reference, noise, and post-selection…"):
                    result = play_level(
                        selected_id,
                        num_checks=len(selected),
                        target_qubits=selected,
                        shots=shots,
                        config_overrides=cfg,
                    )
                st.session_state["latest_strategy_run"] = {
                    "level_id": selected_id,
                    "difficulty": difficulty,
                    "targets": selected,
                    "result": {key: value for key, value in result.items() if key != "circuit_obj"},
                    "research_mode": research_mode,
                }
                if not research_mode:
                    ranking = campaign_ranking_for(selected_id, difficulty)
                    matched = rank_selected_strategy(ranking, selected)
                    if matched and matched.get("pass_fail") == "PASS":
                        completed_by_difficulty = dict(st.session_state.get("campaign_completed_by_difficulty", {}))
                        completed = set(completed_by_difficulty.get(difficulty, []))
                        completed.add(selected_id)
                        completed_by_difficulty[difficulty] = sorted(completed)
                        st.session_state["campaign_completed_by_difficulty"] = completed_by_difficulty
                        st.rerun()
            except Exception as exc:
                st.error(f"Simulation failed: {type(exc).__name__}: {exc}")
                st.exception(exc)

    saved = st.session_state.get("latest_strategy_run")
    if saved and saved.get("level_id") == selected_id and saved.get("difficulty") == difficulty and saved.get("research_mode") == research_mode:
        result = saved["result"]
        st.divider()
        st.subheader("Mission debrief")
        st.caption(f"Noise source: {result.get('noise_source', 'unknown')} · Strategy: {format_targets(saved.get('targets'))}")
        render_metrics(result)
        ranking = campaign_ranking_for(selected_id, difficulty)
        rank_row = render_campaign_rank_summary(result, ranking, saved.get("targets", []))
        if not research_mode:
            if rank_row is None:
                st.info("Campaign PASS/FAIL is available after the background ranking sweep finishes for this level and difficulty.")
            elif rank_row.get("pass_fail") == "PASS":
                st.success("PASS. This strategy's relative score meets the campaign threshold, and the level is unlocked for progression.")
            elif rank_row.get("pass_fail") == "FAIL":
                st.warning("FAIL. Try another qubit strategy with a relative score of at least 50/100.")
            else:
                st.info("All valid strategies have the same score, so this level is tied rather than marked PASS or FAIL.")
        st.write(result.get("feedback", "Simulation complete."))
        render_cost(result)
        view_key = f"show_rankings::{research_mode}::{selected_id}::{difficulty}"
        if st.button("View strategy rankings", key=f"view-rankings::{research_mode}::{selected_id}::{difficulty}", use_container_width=True):
            st.session_state[view_key] = True
        if st.session_state.get(view_key):
            st.markdown("#### Strategy ranking for this level and difficulty")
            render_campaign_ranking_panel(selected_id, difficulty, saved.get("targets", []))


def page_campaign(levels: dict[str, dict[str, Any]], difficulty: str) -> None:
    level_ids = sorted(levels, key=level_number)
    unlocked, completed, lab_unlocked = campaign_progress(level_ids, difficulty)
    render_hero("Campaign / Mission control", "Choose your checks.", "Pass missions using a strategy's relative score. Open the main menu to choose an unlocked mission, or move between levels with Previous and Next.")
    progress_count = len(completed.intersection(level_ids))
    st.progress(progress_count / max(1, len(level_ids)), text=f"Campaign progress: {progress_count}/{len(level_ids)} missions passed")

    if st.session_state.get("campaign_screen", "menu") == "menu":
        st.markdown("### Campaign main menu")
        st.caption(f"Current difficulty: {difficulty}. Rankings are precomputed for every level and difficulty by the background worker.")
        render_ranking_job_status()
        if st.button("Start background ranking sweep", type="primary", key="campaign-start-ranking"):
            try:
                started = start_ranking_worker()
                st.success("Ranking worker started for all levels and difficulties." if started else "Ranking worker is already starting or running.")
            except Exception as exc:
                st.error(f"Could not start ranking worker: {type(exc).__name__}: {exc}")
        st.divider()
        for level_id in level_ids:
            cfg = levels[level_id]
            available = level_id in unlocked
            passed = level_id in completed
            ranking = campaign_ranking_for(level_id, difficulty)
            best = ranking["rows"][0] if ranking and ranking.get("rows") else None
            status_label = "PASSED" if passed else ("AVAILABLE" if available else "LOCKED")
            info_col, action_col = st.columns([5, 1])
            with info_col:
                st.markdown(f"**{level_number(level_id):02d} · {cfg.get('title', level_id)}**  ")
                st.caption(f"{status_label} · {cfg.get('num_qubits', '?')} data qubits · {cfg.get('depth_layers', '?')} layers")
                if best:
                    st.caption(f"Best strategy: rank #{best['rank']} · score {best['score']:.2f}/100")
                else:
                    st.caption("Best strategy rank and score: ranking not generated yet")
            with action_col:
                if st.button("Open level" if available else "Locked", key=f"campaign-open::{difficulty}::{level_id}", disabled=not available, use_container_width=True):
                    st.session_state[f"campaign_level_id::{difficulty}"] = level_id
                    st.session_state["campaign_screen"] = "level"
                    st.rerun()
            st.divider()
        if lab_unlocked:
            st.success("Campaign complete. The open-ended Research Lab is unlocked in the navigation.")
        return

    level_state_key = f"campaign_level_id::{difficulty}"
    selected_id = st.session_state.get(level_state_key)
    if selected_id not in unlocked:
        selected_id = unlocked[-1]
        st.session_state[level_state_key] = selected_id
    selected_index = unlocked.index(selected_id)
    previous_id = unlocked[selected_index - 1] if selected_index > 0 else None
    next_id = unlocked[selected_index + 1] if selected_index + 1 < len(unlocked) else None
    prev_col, menu_col, next_col = st.columns([1, 1, 1])
    with prev_col:
        if st.button("← Previous level", key=f"campaign-prev::{difficulty}::{selected_id}", disabled=previous_id is None, use_container_width=True):
            st.session_state[level_state_key] = previous_id
            st.rerun()
    with menu_col:
        if st.button("Campaign main menu", key="campaign-back-menu", use_container_width=True):
            st.session_state["campaign_screen"] = "menu"
            st.rerun()
    with next_col:
        if st.button("Next level →", key=f"campaign-next::{difficulty}::{selected_id}", disabled=next_id is None, use_container_width=True):
            st.session_state[level_state_key] = next_id
            st.rerun()
    st.caption(f"Mission {level_number(selected_id)} of {len(level_ids)} · {difficulty}")
    render_ranking_job_status()
    render_strategy_runner(levels, difficulty, unlocked, selected_id, research_mode=False)
    if lab_unlocked:
        st.success("Campaign complete. The open-ended Research Lab is unlocked in the navigation.")
    elif next_id is None:
        st.caption("Pass this level to unlock the next mission.")


def page_strategy_rankings(levels: dict[str, dict[str, Any]], difficulty: str) -> None:
    render_hero(
        "Strategy rankings / Background computation",
        "Rank every strategy across the campaign.",
        "Start one background worker to evaluate every budget-valid qubit subset for every campaign level in Guided, Standard, and Full Challenge. Completed rankings are saved under data/ and reused by campaign mode.",
    )
    if st.button("Start ranking sweep for all levels and difficulties", type="primary", use_container_width=True, key="rank-page-start-all"):
        try:
            started = start_ranking_worker()
            if started:
                st.success("Background worker started. You can navigate to Campaign while it runs.")
            else:
                st.info("The background ranking worker is already starting or running.")
        except Exception as exc:
            st.error(f"Could not start ranking worker: {type(exc).__name__}: {exc}")
    render_ranking_job_status()
    st.divider()
    level_ids = sorted(levels, key=level_number)
    selected_id = st.selectbox(
        "Ranking to view", level_ids,
        format_func=lambda key: f"{level_number(key):02d} · {levels[key].get('title', key)}",
        key="ranking_level",
    )
    cfg = apply_difficulty(levels[selected_id], difficulty)
    n = int(cfg.get("num_qubits", 3))
    shots = DEFAULT_SHOTS[difficulty]
    candidate_count = len(enumerate_strategy_sets(range(n), maximum_affordable_checks(cfg, n)))
    a, b, c = st.columns(3)
    a.metric("Difficulty", difficulty)
    b.metric("Budget-valid strategies", candidate_count)
    c.metric("Shots per strategy", f"{shots:,}")
    ranking = campaign_ranking_for(selected_id, difficulty)
    if not ranking:
        st.info("No saved ranking exists for this level and difficulty yet. Start the background sweep above; it processes all levels and difficulties.")
        return
    render_ranking_table(ranking)
    if ranking.get("failed_rows"):
        with st.expander(f"Failed candidate simulations ({len(ranking['failed_rows'])})"):
            st.dataframe(pd.DataFrame([
                {"Strategy": row["strategy"], "Error": row["error"]}
                for row in ranking["failed_rows"]
            ]), use_container_width=True, hide_index=True, height=260)


def run_t_experiment(level_id: str, cfg: dict[str, Any], t_count: int, shots: int) -> dict[str, Any]:
    tdoped_cfg = dict(cfg)
    tdoped_cfg.update({"title": "T-Gate Research Experiment", "payload_type": "tdoped", "t_count": int(t_count), "max_checks": 0})
    return play_level(level_id, num_checks=0, target_qubits=[], shots=shots, config_overrides=tdoped_cfg)


def t_gate_insight(rows: list[dict[str, Any]]) -> str:
    usable = [row for row in rows if row.get("fidelity") is not None]
    if len(usable) < 2:
        return "Run at least two T-count settings to compare how the measured fidelity changes."
    usable.sort(key=lambda row: row["t_count"])
    first, last = usable[0], usable[-1]
    delta = last["fidelity"] - first["fidelity"]
    direction = "increased" if delta > 0.005 else "decreased" if delta < -0.005 else "changed very little"
    return (
        f"Across the tested settings, noisy output-distribution fidelity {direction} "
        f"from {100*first['fidelity']:.2f}% at {first['t_count']} T gate(s) to "
        f"{100*last['fidelity']:.2f}% at {last['t_count']} T gate(s) "
        f"({100*delta:+.2f} percentage points). This is an observation for this circuit, "
        "depth, and noise model, not a universal rule that more T gates must lower fidelity."
    )


def page_t_gate_research(levels: dict[str, dict[str, Any]], difficulty: str) -> None:
    render_hero("T-gate research / Non-Clifford study", "Investigate the cost of T gates.", "Explore T-doped variants of the same Clifford payload. These experiments use an ideal statevector reference and noisy simulation, without Paulice checks, because the official checking workflow accepts Clifford circuits only.")
    level_ids = sorted(levels, key=level_number)
    selected_id = st.selectbox(
        "Base Clifford mission", level_ids,
        format_func=lambda key: f"{level_number(key):02d} · {levels[key].get('title', key)}",
        key="tdoped_base_level",
    )
    cfg = apply_difficulty(levels[selected_id], difficulty)
    n = int(cfg.get("num_qubits", 3))
    depth = int(cfg.get("depth_layers", 2))
    max_t = n * depth
    shots = DEFAULT_SHOTS[difficulty]
    st.markdown(f"**Controlled setup:** {n} qubits · {depth} layers · {shots:,} noisy shots · same noise model across the sweep.")
    st.caption("This is a separate T-gate study, not a campaign mission. Its matched Clifford control includes fixed Hadamard basis changes so added T phases can affect measured probabilities. The study also applies an explicit 1% depolarizing error to each T/Tdg gate; this is a transparent modeling assumption, not backend-calibrated T-gate data. T count 0 is the study control, and results never change campaign ranking or unlocks.")
    t_count = st.slider("Number of T gates", min_value=0, max_value=max_t, value=min(4, max_t), step=1, key=f"t_count::{selected_id}::{difficulty}")
    c1, c2 = st.columns(2)
    run_one = c1.button("Run selected T count", type="primary", use_container_width=True)
    run_sweep = c2.button("Run T-count sweep", use_container_width=True)
    key = f"tdoped_results::{selected_id}::{difficulty}"
    if run_one:
        try:
            with st.spinner(f"Simulating the circuit with {t_count} T gate(s)…"):
                result = run_t_experiment(selected_id, cfg, t_count, shots)
            rows = st.session_state.get(key, [])
            rows = [row for row in rows if row["t_count"] != t_count]
            rows.append({
                "t_count": t_count,
                "fidelity": safe_float(result.get("metrics", {}).get("raw_fidelity")),
                "depth": int(result.get("circuit_depth", 0)),
                "noise_source": result.get("noise_source", "unknown"),
                "raw_shots": int(result.get("metrics", {}).get("raw_shots", shots)),
            })
            st.session_state[key] = sorted(rows, key=lambda row: row["t_count"])
            st.session_state[f"tdoped_latest::{selected_id}::{difficulty}"] = {
                "t_count": t_count,
                "result": {name: value for name, value in result.items() if name != "circuit_obj"},
            }
        except Exception as exc:
            st.error(f"T-gate simulation failed: {type(exc).__name__}: {exc}")
            st.exception(exc)
    if run_sweep:
        sweep_counts = sorted(set(x for x in [0, 1, 2, 4, 8, 16, 32, max_t, t_count] if 0 <= x <= max_t))
        rows = []
        progress = st.progress(0.0, text="Starting T-count sweep…")
        try:
            for index, count in enumerate(sweep_counts, start=1):
                progress.progress((index - 1) / len(sweep_counts), text=f"Simulating {count} T gate(s) ({index}/{len(sweep_counts)})")
                result = run_t_experiment(selected_id, cfg, count, shots)
                rows.append({
                    "t_count": count,
                    "fidelity": safe_float(result.get("metrics", {}).get("raw_fidelity")),
                    "depth": int(result.get("circuit_depth", 0)),
                    "noise_source": result.get("noise_source", "unknown"),
                    "raw_shots": int(result.get("metrics", {}).get("raw_shots", shots)),
                })
            progress.progress(1.0, text="Sweep complete")
            st.session_state[key] = rows
        except Exception as exc:
            st.error(f"T-count sweep failed: {type(exc).__name__}: {exc}")
            st.exception(exc)
    rows = st.session_state.get(key, [])
    latest = st.session_state.get(f"tdoped_latest::{selected_id}::{difficulty}")
    if latest:
        st.divider()
        st.subheader(f"Selected experiment: {latest['t_count']} T gate(s)")
        render_metrics(latest["result"], include_score=False)
        st.caption(f"Measured circuit depth: {latest['result'].get('circuit_depth', 'N/A')} · Noise source: {latest['result'].get('noise_source', 'unknown')}")
    if rows:
        st.markdown("### T-count comparison")
        frame = pd.DataFrame([{
            "T gates": row["t_count"],
            "Noisy fidelity (%)": round(100 * row["fidelity"], 3),
            "Circuit depth": row["depth"],
            "Shots": row["raw_shots"],
            "Noise model": row["noise_source"],
        } for row in sorted(rows, key=lambda row: row["t_count"])])
        st.dataframe(frame, use_container_width=True, hide_index=True)
        plot = pd.DataFrame({
            "T gates": [row["t_count"] for row in sorted(rows, key=lambda row: row["t_count"])],
            "Noisy fidelity (%)": [100 * row["fidelity"] for row in sorted(rows, key=lambda row: row["t_count"])],
        }).set_index("T gates")
        st.line_chart(plot, height=320)
        st.markdown("### What these results mean")
        st.write(t_gate_insight(rows))
        st.write("The ideal reference is calculated independently for each T count using the statevector. Fidelity measures how closely noisy execution reproduces that circuit's own ideal output distribution. The T=0 control belongs to this research family and is deliberately separate from campaign payload circuits.")
        st.write("All runs in this tab use no checks and receive no campaign score. Shot retention is not used as a T-gate metric. The campaign's checked-versus-no-check comparison remains in the Clifford game.")


def page_research_lab(levels: dict[str, dict[str, Any]], difficulty: str) -> None:
    render_hero("Research lab / Open exploration", "Test your own hypotheses.", "The campaign has taught the core mechanics. Here you can revisit any Clifford mission and compare any valid strategy; the background ranking worker covers every mission and difficulty.")
    level_ids = sorted(levels, key=level_number)
    selected_id = st.selectbox(
        "Research mission", level_ids,
        format_func=lambda key: f"{level_number(key):02d} · {levels[key].get('title', key)}",
        key="lab_level",
    )
    render_strategy_runner(levels, difficulty, level_ids, selected_id, research_mode=True)


def page_how_it_works() -> None:
    render_hero("Field guide / Method", "Understand the trade-off.", "Syndrome Siege explores error detection through noisy circuit simulation and post-selection. It rejects flagged shots rather than correcting the quantum state.")
    with st.container(border=True):
        st.markdown("### 1 · Payload circuit")
        st.write("Campaign missions use Clifford circuits. In the separate T-Gate Research tab, T gates are inserted into the same Clifford backbone to create a controlled non-Clifford family.")
        st.markdown("### 2 · Noise and ideal references")
        st.write("The simulator uses a uniform two-qubit depolarizing and readout-noise model. It tries to derive rates from FakeTorino calibration data and explicitly labels a synthetic fallback if that path is unavailable. Clifford references use stabilizer simulation; T-doped references use an ideal statevector.")
        st.markdown("### 3 · Spacetime Pauli checks")
        st.write("For Clifford missions, the selected qubit set is passed to `qiskit-paulice`, which constructs spacetime checks. The game uses the check object's post-selection method to reject shots with a nontrivial syndrome. No qubits are blocked.")
        st.markdown("### 4 · Fidelity and retention")
        st.write("Fidelity is the squared Bhattacharyya overlap between the ideal output distribution and the measured data-output distribution. Raw fidelity is measured before filtering; post-selected fidelity uses accepted shots. Retention is the accepted-shot fraction.")
        st.markdown("### 5 · Strategy score and relative score")
        st.latex(r"H_\beta = \frac{(1+\beta^2)FR}{F+\beta^2R},\quad S=100H_\beta-P_{\rm check},\quad \beta=2")
        st.write("The mission score starts with the fidelity-retention harmonic score and subtracts the level's configured check-resource penalty. The penalty is zero for no-check and increases with the energy cost of the selected checks; later missions use larger penalty weights. Relative score maps the resulting mission scores between the worst and best valid strategies for that level and difficulty.")
        st.markdown("### 6 · Strategy rankings")
        st.write("A separate background worker evaluates every budget-valid qubit subset for every campaign level in all three difficulties. Completed rankings are saved in data/ and reused by campaign mode. The table labels only maximum score, maximum fidelity, and maximum retention.")
        st.markdown("### 7 · T-Gate Research")
        st.write("The T-gate tab varies the T count for the same underlying Clifford payload. It reports fidelity without checks because the official `qiskit-paulice` workflow accepts Clifford circuits only.")
        st.markdown("### 8 · Campaign and research lab")
        st.write("Passing a mission unlocks the next mission only on the currently selected difficulty. Each difficulty has independent progress. Completing all five missions on one difficulty unlocks the Research Lab for that difficulty in the current session. T-Gate Research is separate and never changes campaign levels. Only computed strategy rankings and worker status are saved in data/; player progress and individual runs are session-only.")


def main() -> None:
    levels = get_levels()
    level_ids = sorted(levels, key=level_number)
    difficulty = st.session_state.get("global_difficulty", "Standard")
    _, completed, lab_unlocked = campaign_progress(level_ids, difficulty)
    with st.sidebar:
        st.markdown("## ◈ SYNDROME SIEGE")
        st.caption("Quantum error-detection strategy game")
        pages = ["Campaign", "Strategy Rankings", "T-Gate Research"]
        if lab_unlocked:
            pages.append("Research Lab")
        pages.append("How It Works")
        page = st.radio("Navigation", pages, label_visibility="collapsed")
        st.divider()
        difficulty = st.radio("Difficulty", DIFFICULTIES, index=1, key="global_difficulty")
        st.caption(f"Shots per simulation: {DEFAULT_SHOTS[difficulty]:,}")
        st.divider()
        st.markdown("**Method**")
        st.caption("Clifford checks · syndrome post-selection · noisy Aer simulation")
        st.caption("Rankings persist in data/. Player progress remains session-only.")
    if page == "Campaign":
        page_campaign(levels, difficulty)
    elif page == "Strategy Rankings":
        page_strategy_rankings(levels, difficulty)
    elif page == "T-Gate Research":
        page_t_gate_research(levels, difficulty)
    elif page == "Research Lab":
        page_research_lab(levels, difficulty)
    else:
        page_how_it_works()


if __name__ == "__main__":
    main()