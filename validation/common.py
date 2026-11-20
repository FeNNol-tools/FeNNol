"""Shared runner for the quick validation suites.

Each suite builds analytical FENNIX models, runs the ordinary ``fennol_md``
CLI in subprocesses, scores the outputs and reports pass/fail per check.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import zlib

import numpy as np
import yaml

REPOSITORY = Path(__file__).resolve().parents[1]
SEED = 0


def scientific_environment(repository: Path, modules: Path | None) -> dict[str, str]:
    """Ensure CLI subprocesses import this checkout and discover its modules."""
    env = os.environ.copy()
    python_paths = [str(Path(repository).resolve() / "src")]
    python_paths += [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p and p not in python_paths]
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    if modules is not None:
        module_paths = [str(Path(modules).resolve())]
        module_paths += [p for p in env.get("FENNOL_MODULES_PATH", "").split(os.pathsep) if p and p not in module_paths]
        env["FENNOL_MODULES_PATH"] = os.pathsep.join(module_paths)
    return env


def use_modules(modules: Path) -> None:
    """Make custom FENNIX modules discoverable in this process (before importing fennol)."""
    paths = [str(modules)] + [p for p in os.environ.get("FENNOL_MODULES_PATH", "").split(os.pathsep) if p]
    os.environ["FENNOL_MODULES_PATH"] = os.pathsep.join(dict.fromkeys(paths))


def run_cli(config: dict, directory: Path, modules: Path, label: str = "md") -> None:
    """Write ``config`` and run it with ``fennol_md``, keeping the full log."""
    path = directory / f"{label}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    log = directory / f"{label}.log"
    completed = subprocess.run(
        [sys.executable, "-m", "fennol.md.dynamic", str(path)],
        cwd=str(directory),
        env=scientific_environment(REPOSITORY, modules),
        capture_output=True,
        text=True,
    )
    log.write_text(completed.stdout + completed.stderr)
    if completed.returncode:
        raise RuntimeError(f"fennol_md failed; inspect {log}")


def fresh_directory(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def seed_for(name: str) -> int:
    """Deterministic per-check seed; ``--seed`` draws a fresh independent set."""
    return 1000 + zlib.crc32(f"{SEED}:{name}".encode()) % 100000


def effective_samples(values) -> float | None:
    """ESS from Geyer's initial-positive-sequence estimate of the IAT."""
    series = np.asarray(values, dtype=float).ravel()
    n = len(series)
    if n < 2 or not np.all(np.isfinite(series)):
        return None
    centered = series - series.mean()
    variance = float(centered @ centered) / n
    if variance == 0.0:
        return None
    size = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(centered, n=size)
    rho = np.fft.irfft(spectrum * np.conjugate(spectrum), n=size)[:n] / (n * variance)
    rho_sum = 0.0
    for lag in range(0, n - 1, 2):
        pair = rho[lag] + rho[lag + 1]
        if pair <= 0.0:
            break
        rho_sum += pair
    tau = max(0.5, rho_sum - 0.5)
    return min(float(n), n / (2.0 * tau))


def block_standard_error(values, blocks: int = 20) -> float:
    """Standard error of the mean from ``blocks`` contiguous block averages."""
    values = np.asarray(values, dtype=float)
    usable = len(values) // blocks * blocks
    means = values[:usable].reshape(blocks, -1).mean(axis=1)
    return float(np.std(means, ddof=1) / np.sqrt(blocks))


def compare_states(a, b, path: str = "") -> list[str]:
    """Differences between two restart states; keys and values must match exactly."""
    if isinstance(a, dict) and isinstance(b, dict):
        issues = [f"{path}/{k}: missing" for k in set(a) ^ set(b)]
        for key in set(a) & set(b):
            issues += compare_states(a[key], b[key], f"{path}/{key}")
        return issues
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} != {len(b)}"]
        return [i for k, (x, y) in enumerate(zip(a, b)) for i in compare_states(x, y, f"{path}[{k}]")]
    if isinstance(a, str) or a is None:
        return [] if a == b else [f"{path}: {a!r} != {b!r}"]
    try:
        x, y = np.asarray(a), np.asarray(b)
    except Exception:
        return []
    if x.dtype == object or y.dtype == object:
        return []
    if x.shape != y.shape:
        return [f"{path}: shape {x.shape} != {y.shape}"]
    if np.issubdtype(x.dtype, np.floating) or np.issubdtype(y.dtype, np.floating):
        # A restart must reproduce the uninterrupted run bit for bit.
        if not np.array_equal(x, y, equal_nan=True):
            return [f"{path}: max deviation {float(np.nanmax(np.abs(x - y))):.3g}"]
        return []
    return [] if np.array_equal(x, y) else [f"{path}: values differ"]


def run_restart(directory: Path, configure, steps: int, modules: Path,
                expected=(".dyn.restart",)) -> dict:
    """Run ``steps`` uninterrupted and as two halves; compare every restart file.

    ``configure(run_dir, nsteps)`` returns ``(config, system_name)``; both runs
    must use the same seed so that the split run can reproduce the continuous one.
    Each suffix in ``expected`` must be written by both runs, so that a method
    that stops saving its bias state fails instead of being skipped.
    """
    if steps % 2:
        raise ValueError("restart steps must be even")
    names = {}
    for label, segments in (("continuous", [steps]), ("split", [steps // 2] * 2)):
        run_dir = fresh_directory(directory / label)
        for index, segment in enumerate(segments):
            config, names[label] = configure(run_dir, segment)
            # Dumps can wrap coordinates and rebuild neighbor lists, so both runs
            # dump at the split point and at the end.
            config["tdump[fs]"] = config["dt[fs]"] * (steps // 2)
            run_cli(config, run_dir, modules, f"segment{index}")
    continuous = names["continuous"]
    files = sorted(continuous.parent.glob(continuous.name + ".*restart"))
    issues = {}
    for suffix in expected:
        if not Path(f"{continuous}{suffix}").exists():
            issues[suffix] = ["missing after continuous run"]
    for path in files:
        suffix = path.name[len(continuous.name):]
        other = Path(f"{names['split']}{suffix}")
        if not other.exists():
            issues[suffix] = ["missing after split run"]
            continue
        with path.open("rb") as one, other.open("rb") as two:
            found = compare_states(pickle.load(one), pickle.load(two))
        if found:
            issues[suffix] = found
    compared = [p.name[len(continuous.name):] for p in files]
    return {"compared_files": compared, "differences": issues,
            "status": "pass" if compared and not issues else "fail"}


def argument_parser(description: str, sections) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--only", nargs="+", choices=list(sections), default=list(sections))
    parser.add_argument("--methods", nargs="+", default=None,
                        help="restrict to these case labels")
    parser.add_argument("--jobs", type=int, default=4, help="concurrent fennol_md runs")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed set; rerun a failed check with another one before investigating")
    parser.add_argument("--config", type=Path, default=None, help="alternative config.yaml")
    return parser


def _summary(result: dict) -> str:
    details = {k: v for k, v in result.items()
               if isinstance(v, (int, float, str)) and k not in ("status", "unit")}
    for key, sub in result.items():
        if isinstance(sub, dict) and "rmse_kbt" in sub:
            details[f"{key}_rmse"] = sub["rmse_kbt"]
    return ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
                     for k, v in details.items())


def run_suite(jobs, n_jobs: int, report_path: Path, post=None) -> int:
    """Run ``(key, callable)`` jobs concurrently, print a summary, return an exit code.

    ``post(results)`` may return extra checks that compare completed runs.
    """
    def execute(item):
        key, job = item
        try:
            result = job()
        except Exception as error:  # report and keep running the other checks
            result = {"status": "error", "error": f"{type(error).__name__}: {error}"}
        print(f"{result['status']:>5}  {key}", flush=True)
        return key, result

    with ThreadPoolExecutor(max_workers=max(1, n_jobs)) as pool:
        results = dict(pool.map(execute, jobs))
    if post is not None:
        for key, job in post(results):
            results.update([execute((key, job))])

    previous = json.loads(report_path.read_text()) if report_path.exists() else {}
    previous.update(results)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(previous, indent=1, default=float))
    print()
    for key in sorted(results):
        print(f"{results[key]['status']:>5}  {key:28s} {_summary(results[key])}")
    failed = [k for k, r in results.items() if r["status"] != "pass"]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed; report: {report_path}")
    return 1 if failed else 0
