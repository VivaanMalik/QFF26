"""Local player accounts and per-player persistence for Qubit Crown.

Accounts use PBKDF2-HMAC-SHA256 password hashes and a local SQLite database.
This is intended for a local/demo Streamlit deployment, not public multi-tenant
production authentication.
"""
from __future__ import annotations

import hashlib
from contextlib import contextmanager
import hmac
import json
import os
import re
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "player_profiles.sqlite3"
PBKDF2_ITERATIONS = 310_000
_USERNAME_RE = re.compile(r"^[A-Za-z0-9_-]{3,24}$")
_INITIALIZED_DATABASES: set[str] = set()
_INIT_LOCK = threading.Lock()


class ProfileError(ValueError):
    """Expected profile validation/authentication error."""


class ProfileExistsError(ProfileError):
    """A username is already registered."""


def _connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 15000")
    return conn


@contextmanager
def _database(db_path: str | Path | None = None):
    """Connection scope that commits successful work and always closes SQLite."""
    conn = _connect(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def init_profile_db(db_path: str | Path | None = None) -> None:
    """Create the local profile database/schema if it does not exist."""
    target = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
    key = str(target.resolve())
    if key in _INITIALIZED_DATABASES and target.exists():
        return
    with _INIT_LOCK:
        if key in _INITIALIZED_DATABASES and target.exists():
            return
        with _database(target) as conn:
            conn.executescript(
                """
            CREATE TABLE IF NOT EXISTS players (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL COLLATE NOCASE UNIQUE,
                display_name TEXT NOT NULL,
                password_salt BLOB NOT NULL,
                password_hash BLOB NOT NULL,
                created_at_utc TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS player_state (
                player_id INTEGER NOT NULL REFERENCES players(id) ON DELETE CASCADE,
                state_key TEXT NOT NULL,
                value_json TEXT NOT NULL,
                updated_at_utc TEXT NOT NULL,
                PRIMARY KEY (player_id, state_key)
            );
            CREATE TABLE IF NOT EXISTS player_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL UNIQUE,
                player_id INTEGER NOT NULL REFERENCES players(id) ON DELETE CASCADE,
                timestamp_utc TEXT NOT NULL,
                run_type TEXT NOT NULL,
                difficulty TEXT NOT NULL,
                level_id TEXT NOT NULL,
                level_title TEXT NOT NULL,
                shots_requested INTEGER,
                strategy TEXT NOT NULL,
                checks INTEGER,
                targets_json TEXT NOT NULL,
                game_score REAL,
                raw_fidelity REAL,
                post_selected_fidelity REAL,
                retention_rate REAL,
                raw_shots INTEGER,
                accepted_shots INTEGER,
                rejected_shots INTEGER,
                noise_source TEXT,
                circuit_depth INTEGER,
                ancilla_qubits INTEGER,
                two_qubit_gates_before INTEGER,
                two_qubit_gates_after INTEGER,
                depth_before INTEGER,
                depth_after INTEGER,
                status TEXT NOT NULL,
                details_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_player_runs_player_time
                ON player_runs(player_id, timestamp_utc DESC);
            CREATE INDEX IF NOT EXISTS idx_player_runs_player_kind
                ON player_runs(player_id, run_type, difficulty, level_id);
            """
            )
        _INITIALIZED_DATABASES.add(key)


def _safe_player(row: sqlite3.Row | dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "id": int(row["id"]),
        "username": str(row["username"]),
        "display_name": str(row["display_name"]),
        "created_at_utc": str(row["created_at_utc"]),
    }


def _password_digest(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS, dklen=32)


def create_profile(
    username: str,
    password: str,
    display_name: str = "",
    db_path: str | Path | None = None,
) -> dict[str, Any]:
    """Create a profile and return its public, non-secret fields."""
    init_profile_db(db_path)
    normalized_username = str(username or "").strip()
    if not _USERNAME_RE.fullmatch(normalized_username):
        raise ProfileError("Username must be 3–24 characters using letters, numbers, underscores or hyphens.")
    if not isinstance(password, str) or len(password) < 8:
        raise ProfileError("Password must be at least 8 characters long.")
    shown_name = str(display_name or normalized_username).strip()
    if len(shown_name) < 2 or len(shown_name) > 40:
        raise ProfileError("Display name must be 2–40 characters long.")

    salt = os.urandom(16)
    digest = _password_digest(password, salt)
    created_at = utc_now()
    try:
        with _database(db_path) as conn:
            cursor = conn.execute(
                "INSERT INTO players(username, display_name, password_salt, password_hash, created_at_utc) VALUES (?, ?, ?, ?, ?)",
                (normalized_username, shown_name, salt, digest, created_at),
            )
            player_id = int(cursor.lastrowid)
    except sqlite3.IntegrityError as exc:
        raise ProfileExistsError("That username is already registered. Choose another username or sign in.") from exc
    return {"id": player_id, "username": normalized_username, "display_name": shown_name, "created_at_utc": created_at}


def authenticate_profile(
    username: str,
    password: str,
    db_path: str | Path | None = None,
) -> dict[str, Any] | None:
    """Return a public profile on successful authentication, otherwise None."""
    init_profile_db(db_path)
    normalized_username = str(username or "").strip()
    if not normalized_username or not isinstance(password, str):
        return None
    with _database(db_path) as conn:
        row = conn.execute("SELECT * FROM players WHERE username = ?", (normalized_username,)).fetchone()
    if row is None:
        # Keep unknown-user attempts reasonably similar in cost to known-user attempts.
        _password_digest(password, b"syndrome-siege-auth-salt")
        return None
    expected = bytes(row["password_hash"])
    supplied = _password_digest(password, bytes(row["password_salt"]))
    if not hmac.compare_digest(expected, supplied):
        return None
    return _safe_player(row)


def get_profile(player_id: int, db_path: str | Path | None = None) -> dict[str, Any] | None:
    init_profile_db(db_path)
    with _database(db_path) as conn:
        row = conn.execute("SELECT id, username, display_name, created_at_utc FROM players WHERE id = ?", (int(player_id),)).fetchone()
    return _safe_player(row)


def save_player_state(
    player_id: int,
    state_key: str,
    value: Any,
    db_path: str | Path | None = None,
) -> None:
    """Persist small profile-specific state such as campaign progression/baseline tables."""
    init_profile_db(db_path)
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    with _database(db_path) as conn:
        conn.execute(
            "INSERT INTO player_state(player_id, state_key, value_json, updated_at_utc) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(player_id, state_key) DO UPDATE SET value_json=excluded.value_json, updated_at_utc=excluded.updated_at_utc",
            (int(player_id), str(state_key), encoded, utc_now()),
        )


def load_player_state(
    player_id: int,
    state_key: str,
    default: Any = None,
    db_path: str | Path | None = None,
) -> Any:
    init_profile_db(db_path)
    with _database(db_path) as conn:
        row = conn.execute(
            "SELECT value_json FROM player_state WHERE player_id = ? AND state_key = ?",
            (int(player_id), str(state_key)),
        ).fetchone()
    if row is None:
        return default
    try:
        return json.loads(row["value_json"])
    except (TypeError, json.JSONDecodeError):
        return default


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
        if number != number or number in (float("inf"), float("-inf")):
            return None
        return number
    except (TypeError, ValueError, OverflowError):
        return None


def record_player_run(
    player_id: int,
    *,
    run_type: str,
    difficulty: str,
    level_id: str,
    level_title: str,
    shots: int,
    result: dict[str, Any],
    targets: list[int] | tuple[int, ...] | None = None,
    extra_details: dict[str, Any] | None = None,
    db_path: str | Path | None = None,
) -> str:
    """Record one real completed run in this player's history; excludes circuit objects."""
    import uuid

    init_profile_db(db_path)
    metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
    cost = result.get("cost") if isinstance(result.get("cost"), dict) else {}
    effective = result.get("effective_config") if isinstance(result.get("effective_config"), dict) else {}
    requested = [int(q) for q in (targets or [])]
    raw_shots = metrics.get("raw_shots")
    accepted = metrics.get("retained_shots")
    try:
        raw_shots = int(raw_shots) if raw_shots is not None else int(shots)
    except (TypeError, ValueError):
        raw_shots = int(shots)
    try:
        accepted = int(accepted) if accepted is not None else None
    except (TypeError, ValueError):
        accepted = None
    rejected = max(0, raw_shots - accepted) if accepted is not None else None
    details: dict[str, Any] = {
        "effective_config": effective,
        "payload_type": result.get("payload_type", effective.get("payload_type", "clifford")),
        "official_mode": result.get("official_mode"),
        "num_checks_requested": result.get("num_checks_requested"),
        "num_checks_used": result.get("num_checks_used"),
        "check_supports": result.get("check_supports", []),
        "paulice_cost": result.get("paulice_cost"),
        "metrics": metrics,
        "cost": cost,
    }
    if extra_details:
        details["extra"] = extra_details
    run_id = uuid.uuid4().hex
    fields = (
        run_id, int(player_id), utc_now(), str(run_type), str(difficulty), str(level_id), str(level_title),
        int(shots), "No checks" if not requested else "{" + ", ".join(f"q{q}" for q in requested) + "}",
        len(requested), json.dumps(requested), _finite_float(result.get("game_score")),
        _finite_float(metrics.get("raw_fidelity")), _finite_float(metrics.get("post_selected_fidelity")),
        _finite_float(metrics.get("retention_rate")), raw_shots, accepted, rejected,
        str(result.get("noise_source", "unknown")),
        _try_int(result.get("circuit_depth")), _try_int(cost.get("ancilla_qubits")),
        _try_int(cost.get("two_qubit_gates_before")), _try_int(cost.get("two_qubit_gates_after")),
        _try_int(cost.get("depth_before")), _try_int(cost.get("depth_after")), "complete",
        json.dumps(details, ensure_ascii=False, default=str, allow_nan=False),
    )
    with _database(db_path) as conn:
        conn.execute(
            """INSERT INTO player_runs(
                run_id, player_id, timestamp_utc, run_type, difficulty, level_id, level_title,
                shots_requested, strategy, checks, targets_json, game_score, raw_fidelity,
                post_selected_fidelity, retention_rate, raw_shots, accepted_shots, rejected_shots,
                noise_source, circuit_depth, ancilla_qubits, two_qubit_gates_before,
                two_qubit_gates_after, depth_before, depth_after, status, details_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            fields,
        )
    return run_id


def _try_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError, OverflowError):
        return None


def list_player_runs(
    player_id: int,
    limit: int = 500,
    run_type: str | None = None,
    db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    init_profile_db(db_path)
    count = max(1, min(5000, int(limit)))
    query = "SELECT * FROM player_runs WHERE player_id = ?"
    params: list[Any] = [int(player_id)]
    if run_type:
        query += " AND run_type = ?"
        params.append(str(run_type))
    query += " ORDER BY timestamp_utc DESC, id DESC LIMIT ?"
    params.append(count)
    with _database(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        try:
            item["targets"] = json.loads(item.pop("targets_json"))
        except (KeyError, TypeError, json.JSONDecodeError):
            item["targets"] = []
        try:
            item["details"] = json.loads(item.pop("details_json"))
        except (KeyError, TypeError, json.JSONDecodeError):
            item["details"] = {}
        out.append(item)
    return out


def player_run_count(player_id: int, db_path: str | Path | None = None) -> int:
    init_profile_db(db_path)
    with _database(db_path) as conn:
        row = conn.execute("SELECT COUNT(*) AS n FROM player_runs WHERE player_id = ?", (int(player_id),)).fetchone()
    return int(row["n"] if row else 0)
