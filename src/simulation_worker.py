"""Run one Qiskit simulation in a disposable child process.

Qiskit Aer and qiskit-paulice contain native code. A native failure cannot be
caught by Python, so all app-facing simulations are executed in a subprocess.
Only JSON-safe metrics and a text circuit drawing are returned to the UI.
"""
from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SIMULATION_TIMEOUT_SECONDS = 240
_WINDOWS_NATIVE_CRASH_CODES = {0xC0000005, 0xC0000409, 3221225477, 3221226505}


def _json_safe(value: Any) -> Any:
    """Convert simulation output to strict JSON primitives."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except Exception:
            pass
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError, OverflowError):
        return str(value)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a complete response atomically so the parent never reads a partial file."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def _run_level_child(input_path: str, output_path: str) -> int:
    """Child entry point. The process is allowed to die without harming the UI."""
    destination = Path(output_path)
    try:
        payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
        # This is deliberately the implementation, not the public wrapper, so
        # the child does not recursively spawn another simulation process.
        from src.game import _play_level_impl

        result = _play_level_impl(
            str(payload["level_id"]),
            num_checks=int(payload["num_checks"]),
            target_qubits=[int(q) for q in payload["targets"]],
            shots=int(payload["shots"]),
            config_path=str(payload["config_path"]),
            config_overrides=dict(payload["config"]),
        )
        circuit = result.pop("circuit_obj", None)
        drawing = ""
        if circuit is not None:
            try:
                drawing = str(circuit.draw(output="text", fold=120))
            except Exception:
                # Circuit rendering is supplementary. A successful simulation
                # should still be returnable if the text drawer is unavailable.
                drawing = ""
        result["_circuit_text"] = drawing
        response = {"status": "complete", "result": _json_safe(result)}
    except BaseException as exc:
        response = {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=12),
        }
    try:
        _atomic_write_json(destination, response)
        return 0
    except BaseException:
        return 2


def _crash_message(returncode: int, stderr: str = "") -> str:
    """Describe child crashes without turning them into fabricated metrics."""
    if returncode < 0:
        signal_number = -returncode
        try:
            signal_name = signal.Signals(signal_number).name
        except (ValueError, AttributeError):
            signal_name = f"signal {signal_number}"
        if signal_name == "SIGSEGV":
            return (
                "Native quantum simulator segmentation fault (SIGSEGV). "
                "The app survived; this run has no metrics. Retry with fewer shots/checks."
            )
        return f"Isolated simulation exited via {signal_name}. This run has no metrics."
    if returncode in _WINDOWS_NATIVE_CRASH_CODES:
        return (
            "Native quantum simulator process crashed on Windows. "
            "The app survived; this run has no metrics. Retry with fewer shots/checks."
        )
    detail_lines = [line.strip() for line in stderr.splitlines() if line.strip()][-3:]
    details = " | ".join(detail_lines)
    return (
        f"Isolated simulation exited with code {returncode}. This run has no metrics."
        + (f" Details: {details}" if details else "")
    )


def run_level_isolated(
    *,
    level_id: str,
    targets: list[int],
    config: dict[str, Any],
    shots: int,
    config_path: str = "config/levels.json",
    num_checks: int = 0,
    timeout_seconds: int = SIMULATION_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Run one mission safely and return its JSON-compatible result."""
    payload = {
        "level_id": str(level_id),
        "targets": [int(q) for q in targets],
        "config": dict(config),
        "shots": int(shots),
        "config_path": str(config_path),
        "num_checks": int(num_checks),
    }
    with tempfile.TemporaryDirectory(prefix="qubit_crown_simulation_") as temp_dir:
        root = Path(temp_dir)
        input_path = root / "request.json"
        output_path = root / "result.json"
        try:
            input_path.write_text(
                json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding="utf-8"
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Simulation settings must be JSON-compatible: {exc}") from exc

        command = [
            sys.executable, "-m", "src.simulation_worker",
            "--level-probe", str(input_path), str(output_path),
        ]
        env = os.environ.copy()
        # Avoid uncontrolled native thread pools in nested numerical libraries.
        for variable in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
        ):
            env[variable] = "1"
        env.setdefault("PYTHONUTF8", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")
        existing_pythonpath = env.get("PYTHONPATH", "")
        root_string = str(PROJECT_ROOT)
        if root_string not in existing_pythonpath.split(os.pathsep):
            env["PYTHONPATH"] = (
                root_string if not existing_pythonpath
                else root_string + os.pathsep + existing_pythonpath
            )
        try:
            completed = subprocess.run(
                command,
                cwd=str(PROJECT_ROOT),
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=max(1, int(timeout_seconds)),
                env=env,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Isolated simulation timed out after {timeout_seconds}s. "
                "The app survived; this run has no metrics. Try fewer shots or a smaller circuit."
            ) from exc

        if not output_path.is_file():
            raise RuntimeError(_crash_message(completed.returncode, completed.stderr))
        try:
            response = json.loads(output_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"The isolated simulation returned an unreadable result ({type(exc).__name__})."
            ) from exc
        if completed.returncode != 0:
            raise RuntimeError(_crash_message(completed.returncode, completed.stderr))
        if not isinstance(response, dict) or response.get("status") != "complete":
            error = response.get("error", "The simulation child did not return a complete result.")
            detail = response.get("traceback")
            suffix = f"\n{detail}" if detail else ""
            raise RuntimeError(f"{error}{suffix}")
        result = response.get("result")
        if not isinstance(result, dict):
            raise RuntimeError("The simulation child returned a malformed result object.")
        return result


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) == 3 and argv[0] == "--level-probe":
        return _run_level_child(argv[1], argv[2])
    print("Usage: python -m src.simulation_worker --level-probe INPUT.json OUTPUT.json", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())