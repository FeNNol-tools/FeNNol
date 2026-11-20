from collections import defaultdict
from pathlib import Path

EPSILON = 1.e-7


def symmetric_absolute_relative_delta(expected: float, actual: float) -> float:
    """Compute the symmetric absolute relative delta."""
    if (expected + actual) < EPSILON:
        return 0.

    delta = 2 * abs(expected - actual) / abs(expected + actual)

    return delta


def parse_metric_traj_file(path_metric_traj: Path):
    """Return a metrics trajectories as a dictionary."""
    metrics_traj = defaultdict(lambda: [])
    with open(path_metric_traj, "r") as f:
        header_line = f.readline().strip().lstrip("# ")
        keys = [item.split(":")[1] for item in header_line.split()]

        for line in f:
            values = line.split()
            for i, value in enumerate(values):
                metrics_traj[keys[i]].append(float(value))

    return metrics_traj
