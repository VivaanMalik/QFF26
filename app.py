"""Qubit Crown — quantum error-detection Streamlit interface.

Frontend redesign only: simulation, difficulty presets, campaign progression,
strategy rankings, and T-gate experiment calculations stay in the existing
src.game / src.strategy_ranking backend.
"""
from __future__ import annotations

import html
import math
import os
import subprocess
import sys
from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st

from src.game import load_level_configs, play_level
from src.analysis_lab import persist_baseline_result, persist_t_gate_result, render_analysis_lab
from src.player_profiles import (
    ProfileError,
    ProfileExistsError,
    authenticate_profile,
    create_profile,
    get_profile,
    init_profile_db,
    list_player_runs,
    load_player_state,
    player_run_count,
    record_player_run,
    save_player_state,
)
from src.strategy_ranking import (
    DATA_DIR,
    DEFAULT_SHOTS,
    DIFFICULTIES,
    PROJECT_ROOT,
    WORKER_LOG_PATH,
    PASS_RELATIVE_SCORE_THRESHOLD,
    apply_difficulty,
    classify_relative_score,
    calculate_campaign_progress,
    enumerate_strategy_sets,
    maximum_affordable_checks,
    format_targets,
    load_job_status,
    load_ranking,
    process_is_alive,
    safe_float,
    save_job_status,
)

# ---------------------------------------------------------------------------
# Global page setup and responsive theme
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Qubit Crown | Quantum Error Detection",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    r"""
    <style>
    :root { color-scheme: dark; }
    .stApp {
      background: radial-gradient(ellipse at 18% -8%, rgba(10,109,143,.20), transparent 35%),
                  radial-gradient(ellipse at 95% 4%, rgba(117,54,161,.13), transparent 30%), #060b14;
      color: #dce8f7;
    }
    [data-testid="stHeader"] { background: rgba(6,11,20,.88); }
    [data-testid="stSidebar"] { background: linear-gradient(180deg,#0b1423,#080e18); border-right: 1px solid #21334a; }
    [data-testid="stSidebar"] * { color: #dce8f7; }
    h1,h2,h3 { color:#f0f6ff; letter-spacing:-.025em; }
    p,li,label,.stMarkdown { color:#c5d3e6; }
    .hero-kicker { color:#62e7ef; text-transform:uppercase; letter-spacing:.19em; font-size:.72rem; font-weight:800; }
    .hero-title { font-size:clamp(2rem,4.2vw,3.45rem); line-height:1.02; font-weight:850; color:#f4f8ff; margin:.28rem 0 .6rem; }
    .hero-copy { color:#9eb1cc; max-width:900px; font-size:1rem; line-height:1.6; }
    .panel { background:linear-gradient(145deg,rgba(19,32,52,.96),rgba(9,17,30,.96)); border:1px solid #293b55; border-radius:17px; padding:1rem 1.15rem; margin-bottom:.7rem; box-shadow:0 10px 32px rgba(0,0,0,.13); }
    .panel-hot { border-color:#1aa5bd; box-shadow:0 0 0 1px rgba(32,202,226,.12),0 12px 35px rgba(0,0,0,.2); }
    .smallcaps { color:#76e3ef; font-size:.7rem; font-weight:800; letter-spacing:.15em; text-transform:uppercase; }
    .tag { display:inline-block; border:1px solid #2b536b; background:#102b3a; color:#83eaf4; border-radius:999px; padding:.23rem .62rem; font-size:.75rem; margin:0 .28rem .3rem 0; }
    .tag-red { border-color:#733746; background:#351a29; color:#ff9eaf; }
    .tag-gold { border-color:#71572a; background:#302516; color:#f5d287; }
    .muted { color:#91a4bf; font-size:.88rem; }
    .lane-card { background:linear-gradient(90deg,#111f32,#0a1422); border:1px solid #263a54; border-radius:13px; padding:.65rem .75rem; }
    .lane-card.selected { border-color:#42d9e8; box-shadow:0 0 0 1px rgba(66,217,232,.2),inset 0 0 25px rgba(16,166,191,.08); }
    .resource-card { background:linear-gradient(135deg,#14233b,#0d1626); border:1px solid #2a3c59; border-radius:13px; padding:.75rem .9rem; min-height:92px; }
    .resource-label { color:#89a2c3; font-size:.72rem; letter-spacing:.08em; text-transform:uppercase; font-weight:750; }
    .resource-value { color:#f4f8ff; font-size:1.5rem; font-weight:850; line-height:1.25; }
    .phase-chip { color:#9eafc7; font-size:.74rem; letter-spacing:.1em; text-transform:uppercase; }
    .stButton > button { border:1px solid #2b3d58; border-radius:10px; min-height:2.55rem; transition:all .15s ease; }
    .stButton > button[kind="primary"] { background:linear-gradient(110deg,#16a5bb,#52d7df); border:1px solid #64e9ef; color:#04111a; font-weight:850; box-shadow:0 0 22px rgba(30,192,210,.10); }
    .stButton > button:hover { border-color:#5ad9e7; color:#fff; transform:translateY(-1px); }
    div[data-testid="stMetric"] { background:linear-gradient(150deg,#111e31,#0b1422); border:1px solid #263853; border-radius:13px; padding:13px 15px; }
    div[data-testid="stMetricLabel"] { color:#95abc8; }
    div[data-testid="stMetricValue"] { color:#f2f7ff; }
    div[data-baseweb="select"] > div,div[data-baseweb="input"] > div { background:#101b2d; border-color:#293d58; }
    div[data-testid="stProgress"] > div > div { background:linear-gradient(90deg,#15a5bc,#5fe8e7); }
    hr { border-color:#23344b; }
    .stDataFrame { border:1px solid #263853; border-radius:12px; overflow:hidden; }
    @keyframes pulseGlow { 0%,100%{opacity:.62} 50%{opacity:1} }
    @keyframes dashShift { to { stroke-dashoffset: -48; } }
    .pulse-path { animation: dashShift 2.4s linear infinite; }
    .signal-dot { display:inline-block; width:8px; height:8px; background:#49e9c2; border-radius:50%; margin-right:.45rem; box-shadow:0 0 10px #49e9c2; animation:pulseGlow 1.8s infinite; }
    .battle-foot { color:#6f87a8; font-size:.73rem; letter-spacing:.05em; }
    .landing-shell { max-width:1180px; margin:0 auto; }
    .landing-title { font-size:clamp(3rem,7vw,6.8rem); line-height:.88; letter-spacing:-.065em; font-weight:950; color:#f5f9ff; margin:.55rem 0 1rem; text-shadow:0 0 45px rgba(44,209,240,.18); }
    .landing-title span { color:#54e4ef; }
    .landing-subtitle { max-width:760px; font-size:1.06rem; line-height:1.7; color:#a7bbd5; }
    .war-frame { border:1px solid rgba(67,203,231,.32); border-radius:22px; padding:8px; background:linear-gradient(145deg,rgba(12,30,49,.95),rgba(5,11,21,.98)); box-shadow:0 20px 80px rgba(0,0,0,.38),0 0 45px rgba(20,162,200,.08); overflow:hidden; }
    .war-svg { width:100%; height:auto; display:block; border-radius:16px; }
    .scanline { animation: scanDown 5s linear infinite; }
    .orbit { transform-origin:460px 285px; animation: orbitSpin 30s linear infinite; }
    .enemy-hover { animation: enemyHover 3.4s ease-in-out infinite; }
    .enemy-hover.delay { animation-delay:-1.7s; }
    .pulse-core { transform-origin:460px 280px; animation: corePulse 2.2s ease-in-out infinite; }
    .beam { stroke-dasharray:7 13; animation: beamRun 1.1s linear infinite; }
    @keyframes scanDown { 0%{transform:translateY(-300px);opacity:0} 20%{opacity:.36} 80%{opacity:.22} 100%{transform:translateY(330px);opacity:0} }
    @keyframes orbitSpin { to { transform:rotate(360deg); } }
    @keyframes enemyHover { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-7px)} }
    @keyframes corePulse { 0%,100%{opacity:.55;transform:scale(.96)} 50%{opacity:1;transform:scale(1.04)} }
    @keyframes beamRun { to { stroke-dashoffset:-40; } }
    .mode-card { border:1px solid #243b55; border-radius:16px; background:linear-gradient(155deg,rgba(18,32,52,.95),rgba(9,16,28,.96)); padding:1rem; min-height:136px; }
    .mode-card .mode-icon { font-size:1.65rem; margin-bottom:.35rem; }
    .mode-card .mode-name { font-weight:850; color:#eef7ff; font-size:1rem; }
    .mode-card .mode-desc { color:#91a7c4; font-size:.82rem; line-height:1.5; margin-top:.3rem; }
    .landing-foot { color:#6d86a6; font-size:.72rem; letter-spacing:.12em; text-transform:uppercase; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Player accounts and per-player persistence
# ---------------------------------------------------------------------------
def current_player() -> dict[str, Any] | None:
    player = st.session_state.get("player_profile")
    return player if isinstance(player, dict) and player.get("id") else None


def _clear_player_scoped_session() -> None:
    exact_keys = {
        "campaign_completed_by_difficulty", "latest_strategy_run", "game_started",
        "main_navigation", "main_navigation_choice", "global_difficulty", "shot_count_choice", "campaign_screen",
        "_player_context_id", "_player_preferences_loaded",
    }
    prefixes = (
        "campaign_level_id::", "qubit_selection::", "show_rankings::",
        "tdoped_results::", "t_count::", "analysis_",
    )
    for key in list(st.session_state.keys()):
        if key in exact_keys or any(str(key).startswith(prefix) for prefix in prefixes):
            st.session_state.pop(key, None)


def logout_current_player() -> None:
    """Clear per-player UI state and return to the sign-in screen."""
    _clear_player_scoped_session()
    st.session_state.pop("player_profile", None)
    st.session_state.pop("_player_context_id", None)
    st.rerun()


def _ensure_player_context() -> None:
    player = current_player()
    if not player:
        return
    player_id = int(player["id"])
    if st.session_state.get("_player_context_id") == player_id:
        return
    _clear_player_scoped_session()
    completed = load_player_state(player_id, "campaign_completed_by_difficulty", {})
    st.session_state["campaign_completed_by_difficulty"] = completed if isinstance(completed, dict) else {}
    preferences = load_player_state(player_id, "preferences", {})
    if isinstance(preferences, dict):
        if preferences.get("difficulty") in DIFFICULTIES:
            st.session_state["global_difficulty"] = preferences["difficulty"]
        if preferences.get("shots") in {512, 1024, 2048, 4096, 8192}:
            # This is a widget value, so initialise it only before the sidebar
            # selectbox is created on the next rerun.
            st.session_state["shot_count_choice"] = int(preferences["shots"])
    st.session_state["_player_context_id"] = player_id
    st.session_state["_player_preferences_loaded"] = True


def _record_current_player_run(
    *, run_type: str, difficulty: str, level_id: str, level_title: str,
    shots: int, result: dict[str, Any], targets: list[int] | None = None,
    extra_details: dict[str, Any] | None = None,
) -> None:
    player = current_player()
    if not player:
        return
    try:
        record_player_run(
            int(player["id"]), run_type=run_type, difficulty=difficulty,
            level_id=level_id, level_title=level_title, shots=int(shots),
            result=result, targets=targets or [], extra_details=extra_details,
        )
    except Exception as exc:
        # A persistence fault must not turn a successful simulation into a failed mission.
        st.warning(f"This run completed, but its personal history could not be saved: {type(exc).__name__}: {exc}")


def render_player_gate() -> None:
    """Local/demo login gate with hashed passwords and SQLite-backed profiles."""
    st.markdown(
        '<div class="landing-shell"><div class="hero-kicker">PLAYER IDENTITY / LOCAL PROFILE</div>'
        '<div class="landing-title">QUBIT<br><span>CROWN</span></div>'
        '<div class="landing-subtitle">Create a player profile to save campaign progression, mission history, baseline runs and T-Gate experiments separately for each player.</div></div>',
        unsafe_allow_html=True,
    )
    sign_in, create = st.tabs(["Sign in", "Create profile"])
    with sign_in:
        with st.form("player_signin_form"):
            username = st.text_input("Username", max_chars=24, key="signin_username")
            password = st.text_input("Password", type="password", key="signin_password")
            submitted = st.form_submit_button("Sign in to command", type="primary", width="stretch")
        if submitted:
            try:
                profile = authenticate_profile(username, password)
            except Exception as exc:
                st.error(f"Could not access the local profile database: {type(exc).__name__}: {exc}")
            else:
                if profile:
                    st.session_state["player_profile"] = profile
                    st.session_state.pop("_player_context_id", None)
                    st.session_state["game_started"] = False
                    st.rerun()
                else:
                    st.error("Username or password was not recognised.")
    with create:
        with st.form("player_create_form"):
            display_name = st.text_input("Display name", max_chars=40, key="create_display_name")
            username_new = st.text_input("Username", max_chars=24, key="create_username", help="3–24 characters: letters, numbers, underscore or hyphen.")
            password_new = st.text_input("Password", type="password", key="create_password", help="Use at least 8 characters.")
            password_confirm = st.text_input("Confirm password", type="password", key="create_password_confirm")
            submitted_new = st.form_submit_button("Create player profile", type="primary", width="stretch")
        if submitted_new:
            if password_new != password_confirm:
                st.error("The passwords do not match.")
            else:
                try:
                    profile = create_profile(username_new, password_new, display_name)
                except (ProfileError, ProfileExistsError) as exc:
                    st.error(str(exc))
                except Exception as exc:
                    st.error(f"Could not create profile: {type(exc).__name__}: {exc}")
                else:
                    st.session_state["player_profile"] = profile
                    st.session_state.pop("_player_context_id", None)
                    st.session_state["game_started"] = False
                    st.success("Profile created. Your campaign progress and personal run history will be saved locally.")
                    st.rerun()
    st.caption("Local demo accounts are stored in data/player_profiles.sqlite3. Passwords are salted and hashed. This local profile gate is not a substitute for production authentication on a publicly hosted service.")


def page_my_profile() -> None:
    player = current_player()
    if not player:
        st.info("Sign in to view your profile.")
        return
    fresh = get_profile(int(player["id"])) or player
    records = list_player_runs(int(player["id"]), limit=1000)
    completed = st.session_state.get("campaign_completed_by_difficulty", {})
    render_hero("PLAYER PROFILE / FIELD RECORD", f"Welcome, {fresh['display_name']}.", "Your campaign progress and saved runs. Rankings are shared research data; your own executions and progression are stored under this profile.")
    a, b, c = st.columns(3)
    a.metric("Saved personal runs", f"{player_run_count(int(player['id'])):,}")
    b.metric("Campaign passes", f"{sum(len(v) for v in completed.values() if isinstance(v, list)):,}")
    c.metric("T-Gate experiments", f"{sum(1 for row in records if row.get('run_type') == 'tgate'):,}")
    st.caption(f"Username: @{fresh['username']} · Profile created: {fresh['created_at_utc']}")
    st.markdown("### Campaign progression")
    progress_rows = [{"Difficulty": difficulty, "Passed levels": ", ".join(completed.get(difficulty, [])) or "None yet", "Count": len(completed.get(difficulty, [])) if isinstance(completed.get(difficulty, []), list) else 0} for difficulty in DIFFICULTIES]
    st.dataframe(pd.DataFrame(progress_rows), width="stretch", hide_index=True)
    st.markdown("### Personal run history")
    if not records:
        st.info("No saved runs yet. Complete a Campaign mission or run an experiment in Analysis Lab to start your history.")
        return
    frame_rows = []
    for row in records:
        frame_rows.append({
            "When (UTC)": row.get("timestamp_utc"),
            "Type": row.get("run_type"),
            "Difficulty": row.get("difficulty"),
            "Level": row.get("level_title", row.get("level_id")),
            "Strategy": row.get("strategy"),
            "Shots": row.get("shots_requested"),
            "Raw fidelity (%)": None if row.get("raw_fidelity") is None else round(100*float(row["raw_fidelity"]), 3),
            "Post-selected fidelity (%)": None if row.get("post_selected_fidelity") is None else round(100*float(row["post_selected_fidelity"]), 3),
            "Retention (%)": None if row.get("retention_rate") is None else round(100*float(row["retention_rate"]), 3),
            "Score": row.get("game_score"),
            "Accepted": row.get("accepted_shots"),
            "Rejected": row.get("rejected_shots"),
            "Noise model": row.get("noise_source"),
        })
    personal_frame = pd.DataFrame(frame_rows)
    st.dataframe(personal_frame, width="stretch", hide_index=True, height=430)
    st.download_button(
        "Export my run history (CSV)",
        data=personal_frame.to_csv(index=False).encode("utf-8"),
        file_name=f"qubit_crown_{fresh['username']}_history.csv",
        mime="text/csv", width="stretch",
    )


# ---------------------------------------------------------------------------
# Data loading and shared helpers
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_levels() -> dict[str, dict[str, Any]]:
    """Load configured levels; Campaign and Clifford strategy studies use Clifford payloads only."""
    levels = load_level_configs()
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


def render_numbered_level_buttons(
    level_ids: list[str],
    levels: dict[str, dict[str, Any]],
    *,
    state_key: str,
    columns_per_row: int = 3,
) -> str | None:
    """Select a level with numbered buttons instead of a dropdown."""
    if not level_ids:
        return None
    level_ids = sorted(level_ids, key=level_number)
    selected = st.session_state.get(state_key)
    if selected not in level_ids:
        selected = level_ids[0]
        st.session_state[state_key] = selected
    for start in range(0, len(level_ids), columns_per_row):
        row_ids = level_ids[start:start + columns_per_row]
        cols = st.columns(columns_per_row)
        for offset, level_id in enumerate(row_ids):
            label = str(levels[level_id].get("title", level_id))
            with cols[offset]:
                if st.button(
                    f"{level_number(level_id)} · {label}",
                    key=f"{state_key}::button::{level_id}",
                    type="primary" if selected == level_id else "secondary",
                    width="stretch",
                ):
                    st.session_state[state_key] = level_id
                    st.rerun()
    return str(st.session_state.get(state_key, selected))


def difficulty_copy(difficulty: str) -> tuple[str, str, str]:
    """Frontend narration only; backend presets remain authoritative."""
    text = difficulty.lower()
    if "guided" in text:
        return (
            "TRAINING DOCTRINE",
            "Guided · Learn the defence grid",
            "A lower-pressure introduction with helpful explanations. Experiment with target selection and observe why a check can improve retained-shot fidelity while discarding some shots.",
        )
    if "standard" in text:
        return (
            "ACTIVE THEATRE",
            "Standard · Command the operation",
            "The normal mission experience. Your actual difficulty preset determines payload size, depth, budget and mission thresholds. Choose checks deliberately and judge the measured result.",
        )
    return (
        "FULL THREAT PROFILE",
        "Challenge · Hold the Crown",
        "The full configured challenge. Mission constraints matter: check budgets, blocked channels, circuit complexity and acceptance thresholds come from the effective backend configuration.",
    )


def render_hero(kicker: str, title: str, copy: str) -> None:
    st.markdown(
        f'<div class="hero-kicker">{html.escape(kicker)}</div>'
        f'<div class="hero-title">{html.escape(title)}</div>'
        f'<div class="hero-copy">{html.escape(copy)}</div><br>',
        unsafe_allow_html=True,
    )



def set_launch_destination(page: str) -> None:
    """Navigate from the main menu without mutating a live widget's state key."""
    st.session_state["game_started"] = True
    # `main_navigation` is application state. The sidebar radio has its own
    # key (`main_navigation_choice`) and is synchronised before it is created.
    st.session_state["main_navigation"] = page
    if page == "Campaign":
        st.session_state["campaign_screen"] = "menu"
    st.session_state.setdefault("campaign_screen", "menu")
    st.rerun()


def _sync_navigation_choice() -> None:
    """Radio callback: transfer the widget value to independent route state."""
    page = st.session_state.get("main_navigation_choice")
    if page in {"Main Menu", "Profile", "Campaign", "Strategy Rankings", "Analysis Lab", "How To"}:
        st.session_state["main_navigation"] = page


def render_landing_page() -> None:
    """Cinematic title screen built from inline SVG/CSS; works on Windows and Linux."""
    st.markdown(
        '<style>[data-testid="stSidebar"]{display:none!important;}'
        '[data-testid="stHeader"]{background:transparent;}'
        '[data-testid="stMainBlockContainer"]{padding-top:1.5rem;max-width:1280px;}'
        '</style>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="landing-shell">'
        '<div class="hero-kicker"><span class="signal-dot"></span> INDEPENDENT QUANTUM DEFENCE COMMAND · LOCAL SIMULATION</div>'
        '<div class="landing-title">QUBIT<br><span>CROWN</span></div>'
        '<div class="landing-subtitle">The frontier is collapsing. Hostile noise is breaching the Qubit Crown. '
        'Command your circuit lanes, deploy spacetime checks, sacrifice flagged shots when needed, '
        'and hold the island through measured fidelity—not firepower alone.</div><br>'
        '</div>',
        unsafe_allow_html=True,
    )
    profile = current_player()
    if profile:
        welcome_col, logout_col = st.columns([5, 1])
        with welcome_col:
            st.caption(f"Signed in as **{profile.get('display_name', profile.get('username', 'Player'))}** · personal progress is saved to this profile")
        with logout_col:
            if st.button("Log out", key="player-logout-landing", width="stretch"):
                logout_current_player()
    # Original inline art: no network images, WebGL, or platform-specific dependencies.
    svg = r"""<div class="war-frame"><svg class="war-svg" viewBox="0 0 920 490" role="img" aria-label="Qubit Crown title screen showing the quantum defence island under attack">
    <defs>
      <linearGradient id="sky" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#102b43"/><stop offset=".48" stop-color="#0a1627"/><stop offset="1" stop-color="#130e26"/></linearGradient>
      <linearGradient id="sea" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#0d3448"/><stop offset="1" stop-color="#07121f"/></linearGradient>
      <linearGradient id="land" x1="0" y1="0" x2=".8" y2="1"><stop offset="0" stop-color="#1b706e"/><stop offset=".5" stop-color="#17494f"/><stop offset="1" stop-color="#0d2535"/></linearGradient>
      <linearGradient id="fort" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#d4faff"/><stop offset=".25" stop-color="#3cc7dc"/><stop offset="1" stop-color="#165574"/></linearGradient>
      <radialGradient id="core"><stop offset="0" stop-color="#d8ffff"/><stop offset=".3" stop-color="#45e5ed"/><stop offset="1" stop-color="#1783be" stop-opacity=".1"/></radialGradient>
      <filter id="bloom" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="5" result="blur"/><feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
      <pattern id="grid" width="32" height="32" patternUnits="userSpaceOnUse"><path d="M32 0H0V32" fill="none" stroke="#49b9cf" stroke-opacity=".11" stroke-width="1"/></pattern>
    </defs>
    <rect width="920" height="490" rx="15" fill="url(#sky)"/><rect width="920" height="490" rx="15" fill="url(#grid)"/>
    <g fill="#a4e9ff" opacity=".65"><circle cx="82" cy="57" r="1.5"/><circle cx="167" cy="92" r="1"/><circle cx="285" cy="45" r="1.4"/><circle cx="727" cy="67" r="1.3"/><circle cx="824" cy="102" r="1.1"/><circle cx="687" cy="156" r="1"/><circle cx="372" cy="90" r="1"/><circle cx="113" cy="207" r="1"/><circle cx="794" cy="237" r="1.4"/></g>
    <path d="M0 250 Q100 224 205 250T420 248T650 252T920 238V490H0Z" fill="url(#sea)"/>
    <g fill="none" stroke="#1d8ba8" stroke-opacity=".45"><path d="M-20 287 Q100 269 225 290T470 286T720 290T950 275"/><path d="M-20 330 Q120 310 250 335T490 327T750 330T950 316"/><path d="M-20 384 Q130 365 260 387T520 378T760 385T950 369"/><path d="M-20 434 Q110 416 245 436T510 431T740 435T950 417"/></g>
    <g class="orbit" fill="none" stroke="#2cbfd1" stroke-opacity=".3"><ellipse cx="460" cy="282" rx="320" ry="130" stroke-dasharray="5 12"/><ellipse cx="460" cy="282" rx="274" ry="104" stroke-dasharray="2 11"/></g>
    <path d="M180 286 L222 240 286 211 369 195 454 205 530 189 617 219 690 253 733 292 685 335 601 357 507 367 411 356 325 369 245 336Z" fill="#06121f" opacity=".65"/>
    <path d="M166 274 L206 231 277 205 355 190 437 197 511 185 597 207 672 245 714 280 680 321 610 349 514 361 421 352 329 365 244 334 190 307Z" fill="url(#land)" stroke="#55e4de" stroke-width="2" stroke-opacity=".75" filter="url(#bloom)"/>
    <path d="M216 274L287 240 352 253 419 220 481 244 540 220 609 260 655 281M244 313L310 288 370 310 435 286 501 315 570 287 634 308M355 190L363 235 352 253 370 310 329 365M511 185L506 244 540 220 501 315 514 361" fill="none" stroke="#5bd8e5" stroke-width="2" stroke-opacity=".48"/>
    <g filter="url(#bloom)"><circle class="pulse-core" cx="460" cy="278" r="70" fill="none" stroke="#66e9f4" stroke-width="1.5" stroke-dasharray="4 7"/><circle cx="460" cy="278" r="49" fill="url(#core)" opacity=".7"/><circle cx="460" cy="278" r="28" fill="none" stroke="#bafaff" stroke-width="2"/><circle cx="460" cy="278" r="10" fill="#caffff"/></g>
    <g fill="url(#fort)" stroke="#b4f8ff" stroke-width="1.5"><path d="M420 267L433 245 447 267V298H420Z"/><path d="M446 253L460 226 474 253V304H446Z"/><path d="M473 267L487 245 500 267V298H473Z"/><path d="M433 276H486V309H433Z"/></g>
    <g fill="#0c2238" stroke="#8ef7ff" stroke-width="1"><rect x="438" y="281" width="11" height="15" rx="2"/><rect x="455" y="267" width="11" height="17" rx="2"/><rect x="472" y="281" width="11" height="15" rx="2"/></g>
    <g fill="none" stroke="#48e1ed" stroke-width="2" opacity=".88"><path d="M101 126L233 233" class="beam"/><path d="M816 115L662 235" class="beam"/><path d="M812 392L678 321" class="beam"/></g>
    <g class="enemy-hover" transform="translate(75 102)"><path d="M0 17L25 0 56 17 34 27 17 26Z" fill="#762e4b" stroke="#ff819f" stroke-width="1.5"/><path d="M22 17L31 9 40 17 31 23Z" fill="#ffb2c0"/><path d="M9 23L0 35 22 28M45 23L56 35 34 28" stroke="#fb5f83" fill="none"/></g>
    <g class="enemy-hover delay" transform="translate(801 87)"><path d="M0 17L25 0 56 17 34 27 17 26Z" fill="#762e4b" stroke="#ff819f" stroke-width="1.5"/><path d="M22 17L31 9 40 17 31 23Z" fill="#ffb2c0"/><path d="M9 23L0 35 22 28M45 23L56 35 34 28" stroke="#fb5f83" fill="none"/></g>
    <g class="enemy-hover" transform="translate(807 368)"><path d="M0 17L25 0 56 17 34 27 17 26Z" fill="#762e4b" stroke="#ff819f" stroke-width="1.5"/><path d="M22 17L31 9 40 17 31 23Z" fill="#ffb2c0"/><path d="M9 23L0 35 22 28M45 23L56 35 34 28" stroke="#fb5f83" fill="none"/></g>
    <g font-family="monospace" font-size="11" letter-spacing="1.5"><text x="28" y="34" fill="#6fe7f2">SECTOR 01 // QUBIT CROWN</text><text x="892" y="34" text-anchor="end" fill="#ff8ca8">HOSTILE SIGNATURES DETECTED</text><text x="27" y="462" fill="#6c9cb8">DEFENCE GRID ONLINE</text><text x="893" y="462" text-anchor="end" fill="#6c9cb8">NOISE THREAT // LEVEL UNKNOWN</text></g>
    <path class="scanline" d="M0 70H920" stroke="#8cf6ff" stroke-opacity=".45" stroke-width="1"/>
    </svg></div>"""
    st.markdown(svg, unsafe_allow_html=True)
    st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)

    st.markdown("### Command deck")
    st.caption("Choose any destination below. Opening an analysis view never launches a simulation; every run requires an explicit button.")
    nav_items = [
        ("Campaign", "Campaign", "Play five Clifford missions. Unlocks are tracked separately for each difficulty."),
        ("Profile", "Player records", "Review personal run history and difficulty-specific campaign progress."),
        ("Strategy Rankings", "Strategy intelligence", "Start or inspect the background sweep, including strategies that failed to simulate."),
        ("Analysis Lab", "Analyze quantum experiments", "Select individual target qubits in Clifford Analysis; compare fidelity, shot retention, baseline controls, T-gate sweeps, and circuit overhead."),
        ("How To", "Controls and metrics", "Understand the sections, experimental settings, and what each metric means."),
    ]
    for start in range(0, len(nav_items), 3):
        cols = st.columns(3)
        for col, (page_name, title, desc) in zip(cols, nav_items[start:start + 3]):
            with col:
                st.markdown(
                    '<div class="mode-card"><div class="smallcaps">' + html.escape(page_name.upper()) + '</div>'
                    '<div class="mode-name">' + html.escape(title) + '</div>'
                    '<div class="mode-desc">' + html.escape(desc) + '</div></div>',
                    unsafe_allow_html=True,
                )
                if st.button(f"Open {page_name}", key=f"landing-nav::{page_name}", width="stretch", type="primary" if page_name == "Campaign" else "secondary"):
                    set_launch_destination(page_name)
    st.markdown('<div class="landing-foot">OFFLINE SIMULATION · QISKIT AER · NO CLOUD BACKEND REQUIRED</div>', unsafe_allow_html=True)


def render_campaign_sector_map(
    level_ids: list[str],
    levels: dict[str, dict[str, Any]],
    unlocked: list[str],
    completed: set[str],
) -> None:
    """Render the campaign as a chain of island sectors with clear lock states."""
    width = 920
    height = 218
    count = len(level_ids)
    gap = (width - 150) / max(1, count - 1) if count > 1 else 0
    nodes = []
    route = []
    for i, level_id in enumerate(level_ids):
        x = 75 + i * gap if count > 1 else width / 2
        y = 108 + (18 if i % 2 else -10)
        route.append((x, y))
        status = "PASSED" if level_id in completed else ("OPEN" if level_id in unlocked else "LOCKED")
        color = "#59f0c5" if level_id in completed else ("#58dce9" if level_id in unlocked else "#52657c")
        fill = "#123d3b" if level_id in completed else ("#103445" if level_id in unlocked else "#121d2d")
        num = level_number(level_id)
        nodes.append(
            f'<g><ellipse cx="{x:.1f}" cy="{y+9:.1f}" rx="48" ry="18" fill="#020813" opacity=".7"/>'
            f'<path d="M{x-42:.1f} {y:.1f}l12 -18 25 -8 29 9 18 18 -14 18 -36 8 -27 -7Z" fill="{fill}" stroke="{color}" stroke-width="1.5"/>'
            f'<circle cx="{x:.1f}" cy="{y-1:.1f}" r="16" fill="#091725" stroke="{color}" stroke-width="2"/>'
            f'<text x="{x:.1f}" y="{y+4:.1f}" text-anchor="middle" font-size="12" font-weight="800" fill="#f0faff">{num:02d}</text>'
            f'<text x="{x:.1f}" y="{y+43:.1f}" text-anchor="middle" font-size="10" letter-spacing="1" fill="{color}">{status}</text></g>'
        )
    path = " ".join(f"{x:.1f},{y:.1f}" for x, y in route)
    svg = (
        '<div class="war-frame"><svg class="war-svg" viewBox="0 0 920 218" role="img" aria-label="Campaign island progression map">'
        '<defs><linearGradient id="mapsea" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#0c2c40"/><stop offset="1" stop-color="#07121e"/></linearGradient></defs>'
        '<rect width="920" height="218" rx="14" fill="url(#mapsea)"/>'
        '<g fill="none" stroke="#1c6782" stroke-opacity=".42"><path d="M0 40 Q110 18 220 41T440 38T660 39T920 28"/><path d="M0 180 Q110 157 220 183T440 179T660 182T920 169"/><path d="M0 75 Q110 57 220 76T440 72T660 76T920 66"/></g>'
        '<polyline points="' + path + '" fill="none" stroke="#2abbd0" stroke-width="3" stroke-dasharray="5 8" opacity=".7"/>'
        + ''.join(nodes) +
        '<text x="20" y="24" fill="#79cfe0" font-family="monospace" font-size="10" letter-spacing="2">FRONTIER MAP / CAMPAIGN ROUTE</text>'
        '<text x="900" y="204" text-anchor="end" fill="#6b8ba7" font-family="monospace" font-size="9">UNLOCKS FOLLOW CURRENT DIFFICULTY PASS RULES</text>'
        '</svg></div>'
    )
    st.markdown(svg, unsafe_allow_html=True)

def render_resource(label: str, value: str, foot: str = "") -> None:
    st.markdown(
        '<div class="resource-card">'
        f'<div class="resource-label">{html.escape(label)}</div>'
        f'<div class="resource-value">{html.escape(value)}</div>'
        f'<div class="muted">{html.escape(foot)}</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def render_island_battlefield(
    title: str,
    num_qubits: int,
    selected: list[int],
    blocked: list[int] | None = None,
    completed: bool = False,
) -> None:
    """Render a tactical island map; the check circuit itself is built by Qiskit."""
    blocked_set = {int(q) for q in (blocked or [])}
    selected_set = {int(q) for q in selected}
    n = max(1, min(int(num_qubits), 12))
    cx, cy = 460.0, 270.0
    node_parts = []
    cable_parts = []
    for q in range(n):
        angle = (-math.pi / 2) + (2 * math.pi * q / n)
        rx, ry = 335.0, 166.0
        x, y = cx + rx * math.cos(angle), cy + ry * math.sin(angle)
        is_selected = q in selected_set
        is_blocked = q in blocked_set
        color = "#ff7592" if is_blocked else ("#69f2ec" if is_selected else "#7091b5")
        fill = "#3b1e2d" if is_blocked else ("#11414a" if is_selected else "#10263a")
        label = "BLOCKED" if is_blocked else ("CHECK ARMED" if is_selected else "DATA LANE")
        glow = ' filter="url(#nodeGlow)"' if is_selected else ""
        cable_parts.append(
            f'<path d="M {cx:.1f} {cy+3:.1f} L {x:.1f} {y:.1f}" stroke="{color}" stroke-opacity="{0.92 if is_selected else 0.42}" stroke-width="{2.6 if is_selected else 1.2}" stroke-dasharray="6 8" fill="none" class="pulse-path"/>'
        )
        node_parts.append(
            f'<g{glow}><circle cx="{x:.1f}" cy="{y:.1f}" r="24" fill="{fill}" stroke="{color}" stroke-width="2"/>'
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="17" fill="#071321" stroke="{color}" stroke-opacity=".65"/>'
            f'<text x="{x:.1f}" y="{y+4:.1f}" text-anchor="middle" fill="#f0f8ff" font-family="monospace" font-size="12" font-weight="800">q{q}</text>'
            f'<text x="{x:.1f}" y="{y+39:.1f}" text-anchor="middle" fill="{color}" font-family="monospace" font-size="8" letter-spacing="1">{label}</text></g>'
        )

    status = "DEFENCE GRID EXECUTED" if completed else "HOSTILE SIGNALS DETECTED · ARM YOUR CHECKS"
    safe_title = html.escape(str(title))
    svg = (
        '<div class="war-frame"><svg class="war-svg" viewBox="0 0 920 510" role="img" aria-label="Interactive quantum island tactical map">'
        '<defs>'
        '<linearGradient id="islandOcean" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#0d3045"/><stop offset="1" stop-color="#07111e"/></linearGradient>'
        '<linearGradient id="islandLand" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#22766e"/><stop offset=".55" stop-color="#17464d"/><stop offset="1" stop-color="#102638"/></linearGradient>'
        '<radialGradient id="islandReactor"><stop offset="0" stop-color="#c9ffff"/><stop offset=".25" stop-color="#4de7eb"/><stop offset="1" stop-color="#1383ad" stop-opacity=".05"/></radialGradient>'
        '<filter id="nodeGlow" x="-80%" y="-80%" width="260%" height="260%"><feGaussianBlur stdDeviation="5" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>'
        '<pattern id="islandGrid" width="26" height="26" patternUnits="userSpaceOnUse"><path d="M26 0H0V26" stroke="#7addeb" stroke-opacity=".1" fill="none"/></pattern>'
        '</defs>'
        '<rect width="920" height="510" rx="16" fill="url(#islandOcean)"/><rect width="920" height="510" rx="16" fill="url(#islandGrid)"/>'
        '<g fill="none" stroke="#2a83a0" stroke-opacity=".35"><path d="M0 100Q120 76 240 103T480 98T720 102T920 84"/><path d="M0 415Q120 392 240 419T480 411T720 416T920 398"/><path d="M20 210Q150 191 275 213T520 209T760 215T920 198"/></g>'
        '<g class="orbit" fill="none" stroke="#52cddd" stroke-opacity=".27"><ellipse cx="460" cy="270" rx="390" ry="190" stroke-dasharray="4 12"/><ellipse cx="460" cy="270" rx="345" ry="155" stroke-dasharray="2 10"/></g>'
        '<path d="M121 269L158 211 224 177 299 153 385 151 452 161 523 149 605 168 686 199 751 242 772 281 734 324 668 352 595 371 520 389 441 382 361 393 284 373 219 347 164 318Z" fill="#030a14" opacity=".66" transform="translate(0 7)"/>'
        '<path d="M121 261L158 203 224 169 299 145 385 143 452 153 523 141 605 160 686 191 751 234 772 273 734 316 668 344 595 363 520 381 441 374 361 385 284 365 219 339 164 310Z" fill="url(#islandLand)" stroke="#4bced0" stroke-width="2" stroke-opacity=".82"/>'
        '<path d="M160 270L245 225 326 236 395 188 470 224 536 188 616 226 710 270M181 309L270 281 344 319 421 280 491 323 574 287 686 315M299 145L326 236 344 319 361 385M452 153L470 224 421 280 441 374M605 160L536 188 491 323 520 381" fill="none" stroke="#74dfe7" stroke-width="2" stroke-opacity=".34"/>'
        + ''.join(cable_parts) +
        '<g class="pulse-core"><circle cx="460" cy="270" r="74" fill="none" stroke="#58e5ed" stroke-width="1.5" stroke-dasharray="5 8"/><circle cx="460" cy="270" r="55" fill="url(#islandReactor)" opacity=".7"/><circle cx="460" cy="270" r="34" fill="#0a293c" stroke="#9cf7fa" stroke-width="2"/><path d="M436 270L449 247H471L484 270L471 293H449Z" fill="#2aa5bf" stroke="#b8ffff" stroke-width="1.5"/><circle cx="460" cy="270" r="8" fill="#e3ffff"/></g>'
        '<g fill="#d8fbff" font-family="monospace" font-size="10" letter-spacing="1.2"><text x="460" y="326" text-anchor="middle">QUBIT CROWN CORE</text></g>'
        '<g class="enemy-hover" transform="translate(42 90)"><path d="M0 18L24 0 53 18 34 29 15 27Z" fill="#6d2748" stroke="#ff819f" stroke-width="1.5"/><path d="M21 18L27 10 36 18 29 23Z" fill="#ffc1cf"/></g>'
        '<g class="enemy-hover delay" transform="translate(820 115)"><path d="M0 18L24 0 53 18 34 29 15 27Z" fill="#6d2748" stroke="#ff819f" stroke-width="1.5"/><path d="M21 18L27 10 36 18 29 23Z" fill="#ffc1cf"/></g>'
        '<g class="enemy-hover" transform="translate(824 365)"><path d="M0 18L24 0 53 18 34 29 15 27Z" fill="#6d2748" stroke="#ff819f" stroke-width="1.5"/><path d="M21 18L27 10 36 18 29 23Z" fill="#ffc1cf"/></g>'
        '<g stroke="#fa6a8f" stroke-width="2" stroke-dasharray="6 9" class="beam" fill="none"><path d="M83 123L166 190"/><path d="M837 147L755 216"/><path d="M840 382L751 335"/></g>'
        + ''.join(node_parts) +
        f'<text x="22" y="28" fill="#78eaf1" font-family="monospace" font-size="11" letter-spacing="2">QUANTUM ISLAND / {safe_title.upper()}</text>'
        f'<text x="898" y="486" text-anchor="end" fill="#9be7ef" font-family="monospace" font-size="10" letter-spacing="1.5">{status}</text>'
        '</svg></div>'
        '<div class="battle-foot">ISLAND VISUALS ARE GAME PRESENTATION. QISKIT GENERATES THE REAL PAYLOAD AND SPACETIME CHECK CIRCUIT.</div>'
    )
    st.markdown(svg, unsafe_allow_html=True)


def _metric_number(value: Any) -> float | None:
    """Return only finite numeric values; do not turn unavailable metrics into zero."""
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _metric_percent(value: Any, *, missing: str = "N/A — run strategy check") -> str:
    number = _metric_number(value)
    return missing if number is None else f"{100 * number:.2f}%"


def render_metrics(result: dict[str, Any], include_score: bool = True) -> None:
    metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
    has_checks = int(result.get("num_checks_used", 0) or 0) > 0
    columns = st.columns(4 if include_score else 3)
    if include_score:
        score = _metric_number(result.get("game_score"))
        columns[0].metric("Battle score", f"{score:.2f}/100" if score is not None else "N/A — run strategy check")
        offset = 1
    else:
        offset = 0
    fidelity_label = "Post-selected fidelity" if has_checks else "No-check fidelity"
    fidelity_key = "post_selected_fidelity" if has_checks else "raw_fidelity"
    columns[offset].metric(fidelity_label, _metric_percent(metrics.get(fidelity_key)))
    columns[offset + 1].metric("Raw fidelity", _metric_percent(metrics.get("raw_fidelity")))
    columns[offset + 2].metric("Shot retention", _metric_percent(metrics.get("retention_rate")))

    total_value = _metric_number(metrics.get("raw_shots"))
    retained_value = _metric_number(metrics.get("retained_shots"))
    if total_value is not None and retained_value is not None:
        total, retained = int(total_value), int(retained_value)
        st.caption(f"Shots: {total:,} total · {retained:,} accepted · {max(0, total-retained):,} rejected")
    else:
        st.caption("Shot counts: N/A — run strategy check to see accepted and rejected shots.")
    harmonic = _metric_number(metrics.get("harmonic_score_H"))
    penalty = _metric_number(metrics.get("check_overhead_penalty"))
    if include_score and harmonic is not None and penalty is not None:
        st.caption(f"Score components: harmonic score {100*harmonic:.2f} − check-resource penalty {penalty:.2f}")


def start_ranking_worker() -> bool:
    """Launch the project's ranking worker without tying it to a Streamlit rerun."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    status = load_job_status()
    status_name = str(status.get("status", ""))
    pid = status.get("pid")
    if status_name in {"starting", "running"} and process_is_alive(int(pid) if pid else None):
        return False
    if status_name == "starting" and status.get("updated_at"):
        try:
            age = (datetime.now().astimezone() - datetime.fromisoformat(status["updated_at"]).astimezone()).total_seconds()
            if age < 20:
                return False
        except (TypeError, ValueError):
            pass
    save_job_status({
        "status": "starting", "pid": None, "completed_levels": 0, "total_levels": 0,
        "completed_strategies": 0, "total_strategies": 0, "current_level_id": None,
        "current_level_title": None, "current_difficulty": None, "current_strategy": None,
        "error": None, "cancel_requested": False,
    })
    # Run the worker through a tiny bootstrap that retries Windows atomic file
    # replacements. Antivirus/indexing/file readers can briefly deny os.replace
    # when strategy_ranking_job.json is being published. The retry is deliberately
    # bounded; a persistent conflict still produces a visible worker error.
    worker_bootstrap = r"""
import os
import runpy
import time

_original_replace = os.replace
def _replace_with_retry(src, dst, *args, **kwargs):
    last_error = None
    for attempt in range(8):
        try:
            return _original_replace(src, dst, *args, **kwargs)
        except PermissionError as exc:
            last_error = exc
            time.sleep(min(0.8, 0.05 * (attempt + 1)))
    raise last_error

os.replace = _replace_with_retry
runpy.run_module("src.ranking_worker", run_name="__main__")
"""
    command = [sys.executable, "-c", worker_bootstrap]
    try:
        with WORKER_LOG_PATH.open("a", encoding="utf-8") as log_handle:
            child_env = os.environ.copy()
            # Ensure UTF-8 stdio and import path work the same on Windows and Linux.
            child_env.setdefault("PYTHONUTF8", "1")
            child_env.setdefault("PYTHONIOENCODING", "utf-8")
            # Project root on sys.path so `src.*` imports resolve when cwd differs.
            existing_pythonpath = child_env.get("PYTHONPATH", "")
            root_str = str(PROJECT_ROOT)
            if root_str not in existing_pythonpath.split(os.pathsep):
                child_env["PYTHONPATH"] = (
                    root_str if not existing_pythonpath
                    else root_str + os.pathsep + existing_pythonpath
                )
            popen_kwargs: dict[str, Any] = {
                "args": command,
                "cwd": str(PROJECT_ROOT),
                "stdin": subprocess.DEVNULL,
                "stdout": log_handle,
                "stderr": subprocess.STDOUT,
                "shell": False,
                "env": child_env,
            }
            # Cross-platform process isolation. Streamlit reruns should not own
            # the worker's lifetime.
            if os.name == "nt":
                popen_kwargs["creationflags"] = (
                    getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
                    | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
                )
                # Windows cannot inherit open file handles when close_fds=True.
                popen_kwargs["close_fds"] = False
            else:
                popen_kwargs["start_new_session"] = True
                popen_kwargs["close_fds"] = True
            worker = subprocess.Popen(**popen_kwargs)
        # Do not write the PID from the UI process after launch. The worker
        # publishes its own PID/status; a second writer here can race with it
        # on Windows when it atomically replaces strategy_ranking_job.json.
        return True
    except Exception as exc:
        save_job_status({"status": "failed", "error": f"Could not start worker: {type(exc).__name__}: {exc}"})
        raise


@st.fragment(run_every=2)
def render_ranking_job_status() -> None:
    status = load_job_status()
    state = status.get("status")
    if state in {"starting", "running"}:
        pid = status.get("pid")
        if pid and not process_is_alive(int(pid)):
            # UI is a reader, not a writer, while a ranking job is active.
            # Do not mutate the shared status JSON here: the worker owns its
            # lifecycle and may be writing the same file at this exact time.
            st.warning(
                "The ranking worker PID is not currently detectable. "
                "The UI has not overwritten the worker status. Inspect the "
                "worker log and refresh before starting another scan."
            )
            st.caption(f"Worker log: {WORKER_LOG_PATH}")
            st.caption(
                f"Last published status: {state} · "
                f"Completed levels: {status.get('completed_levels', 0)} · "
                f"Completed strategies: {status.get('completed_strategies', 0)}"
            )
            return
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
        st.success(f"Ranking sweep complete: {status.get('completed_levels', 0)} level/difficulty combinations and {status.get('completed_strategies', 0):,} strategies saved in data/.")
    elif state == "failed":
        worker_error = str(status.get("error", "Unknown error"))
        if "PermissionError" in worker_error and ("strategy_ranking_job.json" in worker_error or "WinError 5" in worker_error):
            st.error(f"Ranking worker hit a Windows file-lock error: {worker_error}")
            st.caption(
                "This app build retries transient Windows file-replacement locks when launching the worker. "
                "If this status is from an earlier run, start the sweep again after confirming no other worker is running. "
                "Persistent errors can indicate two worker processes or another program locking the status file."
            )
        else:
            st.error(f"Ranking worker failed: {worker_error}")
        st.caption("Completed rankings saved before failure remain available.")
    elif state == "cancelled":
        st.warning("Ranking sweep was cancelled. Completed rankings remain available.")
    else:
        st.info("No background ranking sweep has been run yet.")


def render_ranking_table(ranking: dict[str, Any], chosen_targets: list[int] | None = None) -> None:
    rows = ranking.get("rows", [])
    if not rows:
        st.warning("No valid strategy results are available for this ranking yet.")
        if ranking.get("failed_rows"):
            failed = pd.DataFrame([
                {
                    "Strategy": row.get("strategy", format_targets(row.get("targets", []))),
                    "Checks requested": row.get("checks", len(row.get("targets", []))),
                    "Score": "N/A — run strategy check",
                    "Fidelity": "N/A — run strategy check",
                    "Retention": "N/A — run strategy check",
                    "Error": row.get("error", "No metric returned"),
                }
                for row in ranking["failed_rows"]
            ])
            with st.expander(f"Unavailable strategies ({len(failed)})", expanded=True):
                st.dataframe(failed, width="stretch", hide_index=True)
        return

    top = rows[0]
    top_score = _metric_number(top.get("score"))
    st.markdown(
        f"**Highest-scoring strategy:** {top.get('strategy', 'Unknown')} · "
        + (f"**{top_score:.2f}/100**" if top_score is not None else "**N/A — run strategy check**")
    )
    worst_score = _metric_number(ranking.get("worst_score"))
    best_score = _metric_number(ranking.get("best_score"))
    score_range = (
        f"{worst_score:.2f}–{best_score:.2f}" if worst_score is not None and best_score is not None
        else "N/A — run strategy check"
    )
    st.caption(
        f"Score range: {score_range} · {len(rows)} valid strategies ranked. "
        f"PASS threshold: relative score ≥ {PASS_RELATIVE_SCORE_THRESHOLD:.0f}/100."
    )
    baseline_score = _metric_number(ranking.get("baseline_score"))
    if baseline_score is not None:
        st.info(
            f"No-check baseline: {baseline_score:.2f}/100. "
            f"{ranking.get('strategies_beating_no_check', 0)} of "
            f"{ranking.get('non_baseline_strategy_count', 0)} checked strategies score higher."
        )

    selected = sorted(int(q) for q in (chosen_targets or []))
    table = []
    for row in rows:
        targets = row.get("targets", []) or []
        score = _metric_number(row.get("score"))
        fidelity = _metric_number(row.get("fidelity"))
        retention = _metric_number(row.get("retention"))
        relative = _metric_number(row.get("relative_score"))
        tags = []
        if row.get("highest_score"): tags.append("Max score")
        if row.get("highest_fidelity"): tags.append("Max fidelity")
        if row.get("highest_retention"): tags.append("Max retention")
        table.append({
            "Rank": row.get("rank", "—"), "Strategy": row.get("strategy", format_targets(targets)),
            "Selected": "YES" if sorted(targets) == selected and chosen_targets is not None else "",
            "Checks": row.get("checks", len(targets)),
            "Score": f"{score:.2f}" if score is not None else "N/A — run strategy check",
            "vs baseline": "Better" if row.get("beats_no_check") else ("Baseline" if not targets else "Worse / tied"),
            "Relative score": "TIE" if relative is None else f"{relative:.1f}",
            "Outcome": row.get("pass_fail", "N/A"),
            "Fidelity (%)": f"{100*fidelity:.2f}" if fidelity is not None else "N/A — run strategy check",
            "Retention (%)": f"{100*retention:.2f}" if retention is not None else "N/A — run strategy check",
            "Accepted shots": row.get("retained_shots") if row.get("retained_shots") is not None else "N/A — run strategy check",
            "Highlights": ", ".join(tags),
        })
    st.dataframe(pd.DataFrame(table), width="stretch", hide_index=True, height=540)
    if ranking.get("noise_sources"):
        st.caption("Noise model: " + "; ".join(ranking["noise_sources"]))

    known_fidelity = [(row, _metric_number(row.get("fidelity"))) for row in rows]
    known_fidelity = [(row, value) for row, value in known_fidelity if value is not None]
    known_retention = [(row, _metric_number(row.get("retention"))) for row in rows]
    known_retention = [(row, value) for row, value in known_retention if value is not None]
    if known_fidelity:
        strategy, value = max(known_fidelity, key=lambda item: item[1])
        st.write(f"**Maximum measured fidelity:** {strategy.get('strategy', 'Unknown')} at {100*value:.2f}%. ")
    else:
        st.caption("Maximum measured fidelity: N/A — run strategy check.")
    if known_retention:
        strategy, value = max(known_retention, key=lambda item: item[1])
        st.write(f"**Maximum retained-shot fraction:** {strategy.get('strategy', 'Unknown')} at {100*value:.2f}%. ")
    else:
        st.caption("Maximum retained-shot fraction: N/A — run strategy check.")

    chart_rows = [
        {"Strategy": row.get("strategy", format_targets(row.get("targets", []))),
         "Fidelity (%)": 100 * float(row["fidelity"]), "Retention (%)": 100 * float(row["retention"])}
        for row in rows
        if _metric_number(row.get("fidelity")) is not None and _metric_number(row.get("retention")) is not None
    ]
    if chart_rows:
        chart = pd.DataFrame(chart_rows).set_index("Strategy")
        st.markdown("#### Fidelity versus retention")
        st.scatter_chart(chart, x="Retention (%)", y="Fidelity (%)", height=350)
        st.caption("Top-right is desirable: better measured fidelity while keeping more shots. Only strategies with both metrics are plotted.")
    else:
        st.info("N/A — no strategy has both fidelity and retention recorded. Run strategy check to populate this graph.")

    failed_rows = ranking.get("failed_rows", []) or []
    if failed_rows:
        failed_table = pd.DataFrame([
            {
                "Strategy": row.get("strategy", format_targets(row.get("targets", []))),
                "Checks requested": row.get("checks", len(row.get("targets", []))),
                "Score": "N/A — run strategy check",
                "Fidelity": "N/A — run strategy check",
                "Retention": "N/A — run strategy check",
                "Error": row.get("error", "No metric returned"),
            }
            for row in failed_rows
        ])
        with st.expander(f"Unavailable strategies ({len(failed_rows)})", expanded=False):
            st.caption("A failed/native-crashed strategy does not stop the sweep. Metrics remain N/A until that subset is retried successfully.")
            st.dataframe(failed_table, width="stretch", hide_index=True, height=300)


def campaign_ranking_for(level_id: str, difficulty: str) -> dict[str, Any] | None:
    # Ranking cache identity uses the backend's canonical shot count per difficulty.
    return load_ranking(level_id, difficulty, DEFAULT_SHOTS[difficulty])


def rank_selected_strategy(ranking: dict[str, Any] | None, targets: list[int] | None) -> dict[str, Any] | None:
    if not ranking:
        return None
    target_list = sorted(int(q) for q in (targets or []))
    return next((row for row in ranking.get("rows", []) if sorted(row.get("targets", [])) == target_list), None)


@st.fragment(run_every=2)
def render_campaign_ranking_panel(level_id: str, difficulty: str, targets: list[int]) -> None:
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
    else:
        st.info("No saved ranking is available yet. Start the background ranking sweep from Strategy Rankings.")


def render_campaign_rank_summary(ranking: dict[str, Any] | None, targets: list[int]) -> dict[str, Any] | None:
    row = rank_selected_strategy(ranking, targets)
    if row is None:
        st.metric("Strategy rank", "Ranking pending")
        st.caption("Campaign PASS/FAIL uses the saved strategy ranking for this level and difficulty. Start the ranking sweep to calculate it.")
        return None
    relative = row.get("relative_score")
    pass_fail = row.get("pass_fail", classify_relative_score(relative))
    a, b, c, d = st.columns(4)
    a.metric("Strategy rank", f"#{row['rank']} / {len(ranking.get('rows', []))}")
    b.metric("Ranked score", f"{row['score']:.2f}/100")
    c.metric("Relative score", "TIE" if relative is None else f"{relative:.1f}/100")
    d.metric("Mission condition", pass_fail)
    st.caption("A campaign pass uses the existing backend ranking classification: relative score meets the configured threshold. A tie is not silently converted into a pass.")
    return row


def render_qubit_picker(level_id: str, difficulty: str, num_qubits: int, max_checks: int, research_mode: bool = False) -> list[int]:
    """Clickable qubit lanes styled as check-charge targeting controls."""
    key = f"qubit_selection::{'lab' if research_mode else 'campaign'}::{level_id}::{difficulty}"
    selected = set(int(q) for q in st.session_state.get(key, []))
    st.markdown("#### Deploy spacetime checks")
    st.caption("Select the data-qubit lanes where you want the backend to attempt coherent Pauli checks. All payload qubits remain selectable; the backend constructs the actual spacetime-check circuit.")
    render_island_battlefield(level_id.replace("_", " ").upper(), num_qubits, sorted(selected))
    # Lane buttons are keyboard-accessible Streamlit controls, not JavaScript-only controls.
    cols = st.columns(min(max(1, num_qubits), 6))
    for q in range(num_qubits):
        col = cols[q % len(cols)]
        with col:
            active = q in selected
            label = f"✓ q{q}" if active else f"＋ q{q}"
            if st.button(label, key=f"toggle::{key}::{q}", width="stretch", type="primary" if active else "secondary"):
                if active:
                    selected.remove(q)
                else:
                    selected.add(q)
                st.session_state[key] = sorted(selected)
                st.rerun()
    selected_nonblocked = sorted(int(q) for q in st.session_state.get(key, []))
    st.markdown("##### Defensive budget")
    a, b, c = st.columns(3)
    a.metric("Checks deployed", len(selected_nonblocked))
    b.metric("Check capacity", max_checks)
    b.caption(f"{max(0, max_checks-len(selected_nonblocked))} remaining")
    c.metric("Available shots", "set in sidebar")
    charge_slots = min(12, max_checks)
    charge_visual = "".join("🟦 " if i < len(selected_nonblocked) else "⬛ " for i in range(charge_slots)) or "No check charges available"
    st.markdown(f'<div class="panel"><div class="smallcaps">CHECK-CHARGE MAGAZINE</div><div style="font-size:1.15rem;margin:.45rem 0">{charge_visual}</div><div class="muted">Each selected target is submitted as one requested check, subject to backend validation.</div></div>', unsafe_allow_html=True)
    if selected_nonblocked:
        st.caption("Armed targets: " + format_targets(selected_nonblocked))
    else:
        st.caption("No checks deployed — this is the raw/no-check baseline strategy.")
    clear_col, _ = st.columns([1, 4])
    with clear_col:
        if st.button("Recall all checks", key=f"clear::{key}", width="stretch"):
            st.session_state[key] = []
            st.rerun()
    return selected_nonblocked


def render_cost(result: dict[str, Any]) -> None:
    cost = result.get("cost") or {}
    with st.expander("Battle report · circuit overhead and generated checks", expanded=False):
        a, b, c = st.columns(3)
        a.metric("Ancilla qubits", cost.get("ancilla_qubits", "N/A"))
        before, after = cost.get("two_qubit_gates_before"), cost.get("two_qubit_gates_after")
        b.metric("Two-qubit gates", f"{before} → {after}" if before is not None and after is not None else "N/A")
        d0, d1 = cost.get("depth_before"), cost.get("depth_after")
        c.metric("Circuit depth", f"{d0} → {d1}" if d0 is not None and d1 is not None else result.get("circuit_depth", "N/A"))
        if result.get("paulice_cost") is not None:
            st.write(f"**Paulice gamma cost:** {result['paulice_cost']:.6g} (lower is better during check search)")
        supports = result.get("check_supports") or []
        if supports:
            st.write("**Generated check supports**")
            for i, support in enumerate(supports, start=1):
                st.write(f"Check {i}: {{{', '.join(f'q{q}' for q in support)}}}")
        st.caption(f"Noise source: {result.get('noise_source', 'unknown')}")
        circuit = result.get("_circuit_obj")
        if circuit is not None:
            st.markdown("**Actual executed circuit (Qiskit text drawing)**")
            try:
                drawing = circuit.draw(output="text", fold=120)
                st.code(str(drawing), language="text")
            except Exception as exc:
                st.caption(f"Text circuit drawing unavailable: {type(exc).__name__}: {exc}")


def campaign_progress(level_ids: list[str], difficulty: str) -> tuple[list[str], set[str], bool]:
    completed_by_difficulty = st.session_state.get("campaign_completed_by_difficulty", {})
    return calculate_campaign_progress(level_ids, completed_by_difficulty, difficulty)


def render_strategy_runner(
    levels: dict[str, dict[str, Any]],
    difficulty: str,
    level_ids: list[str],
    selected_id: str,
    shots: int,
    research_mode: bool = False,
) -> None:
    base_cfg = levels[selected_id]
    cfg = apply_difficulty(base_cfg, difficulty)
    n = int(cfg.get("num_qubits", 3))
    max_checks = maximum_affordable_checks(cfg, n)
    blocked = [int(q) for q in (cfg.get("blocked_channels", []) or [])]
    kicker, difficulty_title, difficulty_description = difficulty_copy(difficulty)
    st.markdown(f'<div class="smallcaps">{html.escape(kicker)}</div><h2>{html.escape(cfg.get("title", selected_id))}</h2>', unsafe_allow_html=True)
    st.write(cfg.get("description", "No mission briefing available."))
    tags = (
        f'<span class="tag">CLIFFORD PAYLOAD</span><span class="tag">{n} DATA QUBITS</span>'
        f'<span class="tag">{int(cfg.get("depth_layers", 0))} LAYERS</span><span class="tag">{shots:,} SHOTS</span>'
    )
    st.markdown(tags, unsafe_allow_html=True)
    with st.expander(f"Rules of engagement · {difficulty_title}", expanded=("Guided" in difficulty)):
        st.write(difficulty_description)
        st.write(f"**Mission target:** score ≥ {safe_float(cfg.get('target_score', 60)):.1f}/100")
        st.write(f"**Minimum retention:** {100*safe_float(cfg.get('min_retention', .15)):.1f}%")
        st.write(f"**Effective check capacity:** {max_checks} · **Energy budget:** {cfg.get('energy_budget', 'N/A')} · **Cost per check:** {cfg.get('check_cost', 'N/A')}")
        if blocked:
            st.warning("Blocked channels from the effective configuration: " + ", ".join(f"q{q}" for q in blocked))
        if "Guided" in difficulty:
            st.info("Guided hint: start with no checks, run the simulation, then add a small number of targets and compare fidelity against retention.")
        elif "Challenge" in difficulty or "Full" in difficulty:
            st.warning("Challenge rule: respect all effective mission constraints. Invalid target choices are rejected before the simulation is run.")
    selected = render_qubit_picker(selected_id, difficulty, n, max_checks, research_mode=research_mode)
    if len(selected) > max_checks:
        st.warning(f"This configuration permits at most {max_checks} checks. Recall {len(selected)-max_checks} target(s) before executing.")
    left, right = st.columns([1.25, 1])
    with left:
        run_clicked = st.button("⚡ EXECUTE DEFENCE", type="primary", width="stretch", key=f"run::{research_mode}::{selected_id}::{difficulty}")
    with right:
        rank_clicked = st.button("◈ Start ranking sweep", width="stretch", key=f"rank-sweep::{research_mode}::{selected_id}::{difficulty}")
    if rank_clicked:
        try:
            started = start_ranking_worker()
            if started:
                st.success("Ranking worker launched. You can keep playing while rankings compute.")
            else:
                st.info("The worker is already starting or running.")
        except Exception as exc:
            st.error(f"Could not start ranking worker: {type(exc).__name__}: {exc}")
    if run_clicked:
        if len(selected) > max_checks:
            st.error("Recall some checks to fit the effective mission budget.")
        else:
            try:
                with st.spinner("Defence in progress · building circuit, simulating noise and measuring outcomes…"):
                    result = play_level(
                        selected_id,
                        num_checks=len(selected),
                        target_qubits=selected,
                        shots=int(shots),
                        config_overrides=cfg,
                    )
                # Keep the circuit object separately for real circuit rendering.
                stored = {key: value for key, value in result.items() if key != "circuit_obj"}
                stored["_circuit_obj"] = result.get("circuit_obj")
                st.session_state["latest_strategy_run"] = {
                    "level_id": selected_id,
                    "difficulty": difficulty,
                    "targets": selected,
                    "shots": int(shots),
                    "result": stored,
                    "research_mode": research_mode,
                }
                _record_current_player_run(
                    run_type="research" if research_mode else "campaign",
                    difficulty=difficulty,
                    level_id=selected_id,
                    level_title=str(result.get("level_title", cfg.get("title", selected_id))),
                    shots=int(shots),
                    result=result,
                    targets=selected,
                    extra_details={"config": cfg},
                )
                if not research_mode:
                    ranking = campaign_ranking_for(selected_id, difficulty)
                    matched = rank_selected_strategy(ranking, selected)
                    if matched and matched.get("pass_fail") == "PASS":
                        completed_by_difficulty = dict(st.session_state.get("campaign_completed_by_difficulty", {}))
                        completed = set(completed_by_difficulty.get(difficulty, []))
                        completed.add(selected_id)
                        completed_by_difficulty[difficulty] = sorted(completed)
                        st.session_state["campaign_completed_by_difficulty"] = completed_by_difficulty
                        player = current_player()
                        if player:
                            try:
                                save_player_state(int(player["id"]), "campaign_completed_by_difficulty", completed_by_difficulty)
                            except Exception as exc:
                                st.warning(f"Campaign pass was registered for this session but could not be persisted to your profile: {type(exc).__name__}: {exc}")
                        st.rerun()
            except Exception as exc:
                st.error(f"Simulation failed: {type(exc).__name__}: {exc}")
                st.exception(exc)
    saved = st.session_state.get("latest_strategy_run")
    if saved and saved.get("level_id") == selected_id and saved.get("difficulty") == difficulty and saved.get("research_mode") == research_mode:
        result = saved["result"]
        st.divider()
        st.markdown('<div class="smallcaps">AFTER-ACTION REPORT</div><h2>Mission debrief</h2>', unsafe_allow_html=True)
        st.caption(f"Noise source: {result.get('noise_source', 'unknown')} · Strategy: {format_targets(saved.get('targets'))} · Shots: {saved.get('shots', shots):,}")
        render_metrics(result)
        ranking = campaign_ranking_for(selected_id, difficulty)
        rank_row = render_campaign_rank_summary(ranking, saved.get("targets", []))
        if not research_mode:
            if rank_row is None:
                st.info("Campaign progression is tied to the saved ranking classification. This mission remains pending until its strategy ranking is available.")
            elif rank_row.get("pass_fail") == "PASS":
                st.success("VICTORY · This strategy meets the campaign's relative-score pass condition. The mission is recorded as passed for this difficulty.")
            elif rank_row.get("pass_fail") == "FAIL":
                st.warning("DEFENCE FAILED · Try another target selection with a stronger relative score. The next campaign mission remains locked.")
            else:
                st.info("STRATEGY TIED · All valid strategies tie, so the ranking does not automatically award a pass.")
        st.write(result.get("feedback", "Simulation complete."))
        render_cost(result)
        view_key = f"show_rankings::{research_mode}::{selected_id}::{difficulty}"
        if st.button("View strategy rankings for this mission", key=f"view-rankings::{research_mode}::{selected_id}::{difficulty}", width="stretch"):
            st.session_state[view_key] = True
        if st.session_state.get(view_key):
            st.markdown("#### Strategy ranking for this mission")
            render_campaign_ranking_panel(selected_id, difficulty, saved.get("targets", []))


def page_campaign(levels: dict[str, dict[str, Any]], difficulty: str, shots: int) -> None:
    level_ids = sorted(levels, key=level_number)
    unlocked, completed, lab_unlocked = campaign_progress(level_ids, difficulty)
    kicker, difficulty_title, difficulty_description = difficulty_copy(difficulty)
    render_hero("QUANTUM WARFARE / CAMPAIGN", "Defend the Qubit Crown.", "Command a quantum island under attack. Choose data-qubit lanes for spacetime checks, execute the noisy circuit, and win through measured strategy performance—not click speed.")
    with st.container(border=True):
        a, b, c, d = st.columns(4)
        a.metric("Theatre", difficulty_title.split("·")[0].strip())
        b.metric("Missions passed", f"{len(completed.intersection(level_ids))}/{len(level_ids)}")
        c.metric("Current simulation", f"{shots:,} shots")
        d.metric("Defence budget", "Mission-specific")
        st.progress(len(completed.intersection(level_ids)) / max(1, len(level_ids)), text=f"Campaign integrity · {len(completed.intersection(level_ids))}/{len(level_ids)} missions passed")
        st.caption(difficulty_description)
    if st.session_state.get("campaign_screen", "menu") == "menu":
        st.markdown("### Operations map")
        st.caption("Unlocked missions can be entered. Later sectors remain locked until the current difficulty's existing campaign pass rules are satisfied.")
        render_campaign_sector_map(level_ids, levels, unlocked, completed)
        render_ranking_job_status()
        if st.button("Start background strategy-intelligence sweep", type="primary", key="campaign-start-ranking", width="stretch"):
            try:
                started = start_ranking_worker()
                if started:
                    st.success("Ranking worker launched. Its results will populate the map as they are saved.")
                else:
                    st.info("The ranking worker is already starting or running.")
            except Exception as exc:
                st.error(f"Could not start ranking worker: {type(exc).__name__}: {exc}")
        for level_id in level_ids:
            cfg = apply_difficulty(levels[level_id], difficulty)
            available = level_id in unlocked
            passed = level_id in completed
            ranking = campaign_ranking_for(level_id, difficulty)
            best = ranking["rows"][0] if ranking and ranking.get("rows") else None
            status = "PASSED" if passed else ("DEPLOYABLE" if available else "LOCKED")
            icon = "🟢" if passed else ("🟦" if available else "🔒")
            with st.container(border=True):
                info, action = st.columns([5, 1.3])
                with info:
                    st.markdown(f"**{icon} SECTOR {level_number(level_id):02d} · {cfg.get('title', level_id)}**")
                    st.caption(f"{status} · {cfg.get('num_qubits', '?')} data qubits · {cfg.get('depth_layers', '?')} depth layers · target {safe_float(cfg.get('target_score', 60)):.1f}")
                    if best:
                        st.caption(f"Best strategy #{best['rank']} · {best['score']:.2f}/100")
                    else:
                        st.caption("Strategy intelligence not generated yet")
                with action:
                    if st.button("Enter sector" if available else "Locked", key=f"campaign-open::{difficulty}::{level_id}", disabled=not available, width="stretch"):
                        st.session_state[f"campaign_level_id::{difficulty}"] = level_id
                        st.session_state["campaign_screen"] = "level"
                        st.rerun()
        if lab_unlocked:
            st.success("Campaign complete on this difficulty. You can continue exploring strategies in Analysis Lab.")
        st.markdown("#### Rules of engagement")
        st.write("**Win:** the selected strategy is classified PASS by the saved ranking for this mission and difficulty. **Defeat:** the selected strategy is classified FAIL. **Pending:** ranking is missing or the strategy is tied. A software exception is an execution error, not a gameplay defeat.")
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
        if st.button("← Previous sector", key=f"campaign-prev::{difficulty}::{selected_id}", disabled=previous_id is None, width="stretch"):
            st.session_state[level_state_key] = previous_id
            st.rerun()
    with menu_col:
        if st.button("Operations map", key="campaign-back-menu", width="stretch"):
            st.session_state["campaign_screen"] = "menu"
            st.rerun()
    with next_col:
        if st.button("Next sector →", key=f"campaign-next::{difficulty}::{selected_id}", disabled=next_id is None, width="stretch"):
            st.session_state[level_state_key] = next_id
            st.rerun()
    st.caption(f"Sector {level_number(selected_id)} of {len(level_ids)} · {difficulty}")
    render_ranking_job_status()
    render_strategy_runner(levels, difficulty, level_ids, selected_id, shots, research_mode=False)
    if lab_unlocked:
        st.success("Campaign complete on this difficulty. You can continue exploring strategies in Analysis Lab.")


def page_strategy_rankings(levels: dict[str, dict[str, Any]], difficulty: str) -> None:
    render_hero("STRATEGY INTELLIGENCE / BACKGROUND COMPUTATION", "Find the strongest defence.", "Rank every budget-valid target subset for every Campaign mission and difficulty. The rankings worker runs independently; changing selectors should not stop it.")
    if st.button("Start ranking sweep for all missions and difficulties", type="primary", width="stretch", key="rank-page-start-all"):
        try:
            started = start_ranking_worker()
            if started:
                st.success("Background worker started. You can navigate elsewhere while it runs.")
            else:
                st.info("The worker is already starting or running.")
        except Exception as exc:
            st.error(f"Could not start ranking worker: {type(exc).__name__}: {exc}")
    render_ranking_job_status()
    st.divider()
    level_ids = sorted(levels, key=level_number)
    st.markdown("#### Select a level")
    selected_id = render_numbered_level_buttons(level_ids, levels, state_key="ranking_level")
    if selected_id is None:
        st.info("No campaign levels are configured.")
        return
    cfg = apply_difficulty(levels[selected_id], difficulty)
    n = int(cfg.get("num_qubits", 3))
    shots = DEFAULT_SHOTS[difficulty]
    candidate_count = len(enumerate_strategy_sets(range(n), maximum_affordable_checks(cfg, n)))
    a, b, c = st.columns(3)
    a.metric("Independent difficulty", difficulty)
    b.metric("Budget-valid strategies", candidate_count)
    c.metric("Shots per ranked strategy", f"{shots:,}")
    st.caption("Ranking results are keyed by the canonical backend shot count per difficulty; the Campaign shot selector does not rewrite saved ranking data.")
    ranking = campaign_ranking_for(selected_id, difficulty)
    if not ranking:
        st.info("No saved ranking exists for this mission and difficulty yet. Start the background sweep above.")
        return
    render_ranking_table(ranking)
    if ranking.get("failed_rows"):
        with st.expander(f"Failed candidate simulations ({len(ranking['failed_rows'])})"):
            st.dataframe(pd.DataFrame([{"Strategy": row.get("strategy"), "Error": row.get("error")} for row in ranking["failed_rows"]]), width="stretch", hide_index=True, height=260)


def run_t_experiment(level_id: str, cfg: dict[str, Any], t_count: int, shots: int) -> dict[str, Any]:
    tdoped_cfg = dict(cfg)
    tdoped_cfg.update({"title": "T-Gate Research Experiment", "payload_type": "tdoped", "t_count": int(t_count), "max_checks": 0})
    return play_level(level_id, num_checks=0, target_qubits=[], shots=shots, config_overrides=tdoped_cfg)


def _render_t_gate_experiment_controls(
    levels: dict[str, dict[str, Any]],
    difficulty: str,
    shots: int,
    *,
    initial_level: str | None = None,
) -> None:
    """Render only the T-gate run controls; charts live beside controls in Analysis Lab."""
    st.markdown("#### T-gate experiment controls")
    st.caption(
        "This is a separate no-check study on T-doped circuits. It does not use the Clifford-only "
        "spacetime-check pipeline or affect campaign progression."
    )
    level_ids = sorted(
        [level_id for level_id, cfg in levels.items() if str(cfg.get("payload_type", "clifford")).lower() == "clifford"],
        key=level_number,
    )
    state_key = "analysis_tgate_base_level"
    if initial_level in level_ids and st.session_state.get(state_key) not in level_ids:
        st.session_state[state_key] = initial_level
    selected_id = render_numbered_level_buttons(level_ids, levels, state_key=state_key)
    if selected_id is None:
        st.info("No Clifford base levels are configured for the T-gate study.")
        return

    cfg = apply_difficulty(levels[selected_id], difficulty)
    num_qubits = int(cfg.get("num_qubits", 3))
    depth_layers = int(cfg.get("depth_layers", 2))
    max_t = max(1, num_qubits * depth_layers)
    st.caption(
        f"Base level: {levels[selected_id].get('title', selected_id)} · {num_qubits} data qubits · "
        f"{depth_layers} payload layers · {shots:,} shots · {difficulty}. The level fixes the Clifford "
        "backbone and circuit scale; difficulty changes the effective circuit configuration."
    )
    st.caption(
        "Shots are noisy measurement samples. More shots reduce sampling fluctuation but take longer. "
        "The T count controls how many T/Tdg gates are inserted; T=0 is the matched family control."
    )
    t_count = st.slider(
        "T / Tdg gate count", min_value=0, max_value=max_t,
        value=min(4, max_t), step=1, key=f"t_count::{selected_id}::{difficulty}",
        help="Number of non-Clifford T or T-dagger gates in this separate experiment. A larger count does not guarantee lower fidelity; read the measured curve for this circuit and noise model.",
    )
    run_one_col, sweep_col = st.columns(2)
    run_one = run_one_col.button(
        "Run selected T count", type="primary", width="stretch",
        key=f"tgate-run-one::{selected_id}::{difficulty}::{shots}",
    )
    run_sweep = sweep_col.button(
        "Run T-count sweep", width="stretch",
        key=f"tgate-run-sweep::{selected_id}::{difficulty}::{shots}",
        help="Runs a small set of T counts, including T=0. Each point is saved separately; a failed point does not discard successful points.",
    )
    results_key = f"tdoped_results::{selected_id}::{difficulty}::{shots}"
    if results_key not in st.session_state:
        st.session_state[results_key] = []

    def run_count(count: int) -> dict[str, Any]:
        result = run_t_experiment(selected_id, cfg, int(count), int(shots))
        row = {
            "t_count": int(count),
            "fidelity": safe_float(result.get("metrics", {}).get("raw_fidelity")) if result.get("metrics", {}).get("raw_fidelity") is not None else None,
            "depth": result.get("circuit_depth"),
            "noise_source": result.get("noise_source", "unknown"),
            "raw_shots": result.get("metrics", {}).get("raw_shots", shots),
        }
        persist_t_gate_result(
            level_id=selected_id,
            level_title=str(result.get("level_title", levels[selected_id].get("title", selected_id))),
            difficulty=difficulty, shots=int(shots), config=cfg,
            t_count=int(count), result=result,
        )
        _record_current_player_run(
            run_type="tgate", difficulty=difficulty, level_id=selected_id,
            level_title=str(result.get("level_title", levels[selected_id].get("title", selected_id))),
            shots=int(shots), result=result, targets=[],
            extra_details={"t_count": int(count), "config": cfg, "experiment_type": "tdoped_no_check"},
        )
        return row

    if run_one:
        try:
            with st.spinner(f"Simulating {t_count} T/Tdg gate(s)…"):
                row = run_count(int(t_count))
            rows = [row for row in st.session_state[results_key] if int(row.get("t_count", -1)) != int(t_count)]
            rows.append(row)
            st.session_state[results_key] = sorted(rows, key=lambda item: int(item["t_count"]))
            st.success(f"Saved the T={t_count} run. The comparison graph below has been refreshed.")
        except Exception as exc:
            st.error(f"T-gate run failed: {type(exc).__name__}: {exc}")

    if run_sweep:
        sweep_counts = sorted(set(count for count in (0, 1, 2, 4, 8, 16, 32, max_t, int(t_count)) if 0 <= count <= max_t))
        rows_by_count = {int(row.get("t_count")): row for row in st.session_state[results_key] if row.get("t_count") is not None}
        failures: list[dict[str, str]] = []
        progress = st.progress(0.0, text="Preparing T-count sweep…")
        for index, count in enumerate(sweep_counts, start=1):
            progress.progress((index - 1) / max(1, len(sweep_counts)), text=f"Running T={count} ({index}/{len(sweep_counts)})")
            try:
                with st.spinner(f"Simulating T={count}…"):
                    rows_by_count[count] = run_count(count)
            except Exception as exc:
                failures.append({"T count": str(count), "Error": f"{type(exc).__name__}: {exc}"})
        progress.progress(1.0, text="T-count sweep finished")
        st.session_state[results_key] = sorted(rows_by_count.values(), key=lambda item: int(item["t_count"]))
        success_count = len(rows_by_count)
        st.success(f"Sweep finished: {success_count} saved T-count setting(s).")
        if failures:
            st.warning("Some settings were unavailable. Their missing values will remain N/A; successful points were kept.")
            st.dataframe(pd.DataFrame(failures), hide_index=True, width="stretch")

    st.caption("The graph below compares successful runs with the same circuit configuration. Failed settings remain N/A.")

def _render_baseline_experiment_controls(
    levels: dict[str, dict[str, Any]], difficulty: str, shots: int,
) -> None:
    """Run no-check Clifford controls; results are charted by the parent Analysis Lab view."""
    st.markdown("#### No-check baseline controls")
    st.caption(
        "A baseline runs each Clifford campaign payload with zero checks. No syndrome post-selection is requested, "
        "so retention should be 100% unless the backend reports otherwise. Baselines do not unlock campaign levels."
    )
    st.caption(
        f"Current run settings: {difficulty} · {shots:,} shots per level · 0 checks. "
        "Difficulty selects the effective qubit count/depth and available budget; shots control the number of measurement samples."
    )
    level_ids = sorted(
        [level_id for level_id, cfg in levels.items() if str(cfg.get("payload_type", "clifford")).lower() == "clifford"],
        key=level_number,
    )
    if not level_ids:
        st.info("No Clifford levels are configured.")
        return
    if not st.button(
        "Run no-check baseline across all Clifford levels", type="primary", width="stretch",
        key=f"baseline-start::{difficulty}::{shots}",
        help="Runs one zero-check control for every configured Clifford level using the displayed difficulty and shot count.",
    ):
        return

    failures: list[dict[str, str]] = []
    successes = 0
    progress = st.progress(0.0, text="Preparing no-check baseline…")
    for index, level_id in enumerate(level_ids, start=1):
        progress.progress((index - 1) / len(level_ids), text=f"Running {level_id} ({index}/{len(level_ids)})")
        cfg = apply_difficulty(levels[level_id], difficulty)
        try:
            result = play_level(
                level_id, num_checks=0, target_qubits=[], shots=int(shots),
                config_overrides=cfg,
            )
            persist_baseline_result(
                level_id=level_id,
                level_title=str(result.get("level_title", levels[level_id].get("title", level_id))),
                difficulty=difficulty, shots=int(shots), config=cfg, result=result,
            )
            _record_current_player_run(
                run_type="baseline", difficulty=difficulty, level_id=level_id,
                level_title=str(result.get("level_title", levels[level_id].get("title", level_id))),
                shots=int(shots), result=result, targets=[],
                extra_details={"config": cfg, "study": "zero_check_baseline"},
            )
            successes += 1
        except Exception as exc:
            failures.append({"Level": level_id, "Error": f"{type(exc).__name__}: {exc}"})
    progress.progress(1.0, text="No-check baseline finished")
    st.success(f"Completed {successes}/{len(level_ids)} baseline control(s). Saved observations are displayed below.")
    if failures:
        st.warning(f"{len(failures)} level(s) failed. Their values remain unavailable; the successful controls were preserved.")
        st.dataframe(pd.DataFrame(failures), hide_index=True, width="stretch")


def page_how_it_works() -> None:
    render_hero(
        "HOW TO / FIELD GUIDE", "Know where to go.",
        "Six top-level destinations, three focused Analysis Lab views, and no automatic simulations when a page is opened.",
    )
    with st.container(border=True):
        st.markdown("### 1 · Main Menu")
        st.write("The command deck links to Campaign, Profile, Strategy Rankings, Analysis Lab, and How To. The sidebar navigation remains available after you enter the app.")
        st.markdown("### 2 · Profile")
        st.write("Inspect your own run history and campaign progress. Level unlocks are stored separately for Guided, Standard, and Full Challenge.")
        st.markdown("### 3 · Campaign")
        st.write("Choose an unlocked Clifford level, select data-qubit targets as requested checks, and run the noisy simulation. qiskit-paulice builds the spacetime-check variants; passing a level only advances that same difficulty.")
        st.markdown("### 4 · Strategy Rankings")
        st.write("Start or inspect a background sweep of budget-valid target subsets. Each subset runs in its own process. If a native simulator crash or timeout occurs, the sweep continues and that strategy remains N/A until retried.")
        st.markdown("### 5 · Analysis Lab")
        st.write("Use the numbered buttons to switch between Clifford Analysis, No-check Baseline, and T-Gate Sweep. Select individual target qubits and run a Clifford strategy directly inside Clifford Analysis.")
        st.markdown("### 6 · How To")
        st.write("Use this field guide for navigation and metric definitions. Difficulty buttons are vertical so all names remain readable. Levels use numbered buttons; Clifford strategies use individual target-qubit buttons.")
    st.markdown("### Meaning of each setting")
    st.write("**Difficulty** selects the configured circuit size/depth and affordable check capacity. It does not change campaign progress on another difficulty. **Level** selects the Clifford payload family and its base configuration. **Shots** is the number of noisy measurement samples; more shots take longer but reduce sampling fluctuation. **Target qubits** choose where checks are requested, not the final spacetime supports, which are generated by qiskit-paulice.")
    st.write("**Target qubits** are the individual data-qubit buttons inside Clifford Analysis. Each selected target requests one check, subject to the current difficulty budget; qiskit-paulice builds the resulting spacetime checks. **T/Tdg count** changes the number of non-Clifford gates in the separate T-Gate Sweep. T=0 is its control. Baseline and T-gate experiments never affect Campaign unlocks.")
    st.markdown("### Reading the metrics")
    st.write("**Raw fidelity** compares the noisy output distribution with the ideal reference before filtering. **Post-selected fidelity** uses accepted shots only. **Retention** is accepted shots divided by total shots. Read post-selected fidelity and retention together: a higher conditional fidelity can come at the cost of rejecting more shots.")
    st.write("**N/A — run strategy check** means the requested subset has not produced a usable saved result. It is not equivalent to a measured zero. A failed sweep candidate remains unavailable until it is retried successfully.")
    st.write("**Matched baseline comparisons** require compatible level, difficulty, shots, noise source, and configuration signature where available. Missing controls are left unreported rather than estimated.")


def main() -> None:
    # A player must sign in or create a profile before entering the game.
    try:
        init_profile_db()
    except Exception as exc:
        st.error(f"The local player database could not be initialized: {type(exc).__name__}: {exc}")
        st.stop()
    if not current_player():
        render_player_gate()
        return
    _ensure_player_context()

    # The splash/title scene is the true opening screen. It does not initialise
    # page navigation until the player presses Start or chooses a launch shortcut.
    if (not st.session_state.get("game_started", False)
            or st.session_state.get("main_navigation") == "Main Menu"):
        render_landing_page()
        return

    levels = get_levels()
    if not levels:
        st.error("No Clifford Campaign levels were found. Check config/levels.json and the payload_type fields.")
        st.stop()
    level_ids = sorted(levels, key=level_number)
    difficulty = st.session_state.get("global_difficulty", "Standard")
    with st.sidebar:
        st.markdown("# ◈ QUBIT CROWN")
        st.caption("QUANTUM ERROR DETECTION · QISKIT AER")
        profile = current_player() or {}
        st.markdown(f"**Player:** {html.escape(str(profile.get('display_name', 'Unknown')))}")
        st.caption(f"@{html.escape(str(profile.get('username', 'unknown')))}")
        if st.button("Log out", key="player-logout", width="stretch"):
            logout_current_player()
        st.divider()
        st.markdown("**Navigation**")
        page_options = ["Main Menu", "Profile", "Campaign", "Strategy Rankings", "Analysis Lab", "How To"]
        # Initialise the radio key *before* creating the widget. Route state is
        # intentionally stored separately so landing-page buttons never mutate
        # a widget-owned key after Streamlit has instantiated it.
        route = st.session_state.get("main_navigation", "Campaign")
        if route not in page_options:
            route = "Campaign"
            st.session_state["main_navigation"] = route
        if st.session_state.get("main_navigation_choice") not in page_options or st.session_state.get("main_navigation_choice") != route:
            st.session_state["main_navigation_choice"] = route
        page = st.radio(
            "Navigation", page_options, label_visibility="collapsed",
            key="main_navigation_choice", on_change=_sync_navigation_choice,
        )
        st.session_state["main_navigation"] = page
        st.divider()
        st.markdown("**Difficulty · choose one**")
        current_difficulty = st.session_state.get("global_difficulty", "Standard")
        if current_difficulty not in DIFFICULTIES:
            current_difficulty = "Standard" if "Standard" in DIFFICULTIES else DIFFICULTIES[0]
            st.session_state["global_difficulty"] = current_difficulty
        st.caption("Difficulty changes the circuit size, depth and allowed check budget. It never shares campaign unlocks with another difficulty.")
        # Full-width vertical buttons keep 'Full Challenge' readable in the
        # narrow sidebar and behave consistently on desktop and mobile.
        for index, value in enumerate(DIFFICULTIES, start=1):
            if st.button(
                f"{index} · {value}",
                key=f"sidebar-difficulty::{value}",
                type="primary" if current_difficulty == value else "secondary",
                width="stretch",
            ):
                st.session_state["global_difficulty"] = value
                st.rerun()
        difficulty = st.session_state.get("global_difficulty", current_difficulty)
        st.caption(difficulty_copy(difficulty)[2])
        default_shots = int(DEFAULT_SHOTS[difficulty])
        shot_options = sorted(set([512, 1024, default_shots, 2048, 4096, 8192]))
        if st.session_state.get("shot_count_choice") not in shot_options:
            st.session_state["shot_count_choice"] = default_shots
        shots = st.selectbox(
            "Shots per simulation", shot_options, index=shot_options.index(st.session_state["shot_count_choice"]),
            format_func=lambda v: f"{v:,} shots", key="shot_count_choice",
        )
        st.caption("Shots are the number of noisy measurement samples. More shots reduce sampling noise but take longer; ranking sweeps use their own preset count.")
        player = current_player()
        if player:
            try:
                save_player_state(int(player["id"]), "preferences", {"difficulty": difficulty, "shots": int(shots)})
            except Exception:
                pass
        st.divider()
        completed_by_difficulty = st.session_state.get("campaign_completed_by_difficulty", {})
        completed_now = len(set(completed_by_difficulty.get(difficulty, [])))
        st.markdown("**Island status**")
        st.progress(completed_now / max(1, len(level_ids)), text=f"{completed_now}/{len(level_ids)} sectors passed")
        st.caption("Your campaign progress and personal runs are saved to your profile. Strategy rankings are shared reference data under data/.")
    if page == "Main Menu":
        render_landing_page()
    elif page == "Profile":
        page_my_profile()
    elif page == "Campaign":
        page_campaign(levels, difficulty, int(shots))
    elif page == "Strategy Rankings":
        page_strategy_rankings(levels, difficulty)
    elif page == "Analysis Lab":
        def render_analysis_experiment(
            experiment: str,
            selected_level: str | None = None,
            selected_difficulty: str | None = None,
        ) -> None:
            experiment_difficulty = selected_difficulty or difficulty
            level_ids_for_lab = sorted(levels, key=level_number)
            if experiment == "Clifford Analysis":
                if selected_level not in levels:
                    selected_level = level_ids_for_lab[0] if level_ids_for_lab else None
                if selected_level is None:
                    st.info("No Clifford levels are configured.")
                    return
                render_strategy_runner(
                    levels, experiment_difficulty, level_ids_for_lab, selected_level,
                    int(shots), research_mode=True,
                )
            elif experiment == "No-check Baseline":
                _render_baseline_experiment_controls(levels, experiment_difficulty, int(shots))
            elif experiment == "T-Gate Sweep":
                _render_t_gate_experiment_controls(
                    levels, experiment_difficulty, int(shots), initial_level=selected_level,
                )

        render_analysis_lab(
            levels=levels,
            current_difficulty=difficulty,
            current_shots=int(shots),
            difficulties=DIFFICULTIES,
            canonical_shots=DEFAULT_SHOTS,
            ranking_loader=load_ranking,
            difficulty_applier=apply_difficulty,
            experiment_renderer=render_analysis_experiment,
        )
    else:
        page_how_it_works()


if __name__ == "__main__":
    main()