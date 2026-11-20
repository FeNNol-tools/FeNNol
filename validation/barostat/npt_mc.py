"""Independent Monte Carlo references for the barostat checks.

``NPTMonteCarlo`` samples the force-shifted LJ model in reduced units
(epsilon = sigma = kBT = 1). The isotropic cell is a general 3x3 matrix, so
every proposed volume is reevaluated with a fresh explicit lattice sum.
Ordered particle/image pairs, including nonzero self-images, are counted with
a factor of one half.

``sample_cell_spring`` samples the cell of an ideal gas held by a cell spring.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math

import numpy as np
from numba import njit


@njit
def _lattice_shift_array(cell: np.ndarray, cutoff: float) -> np.ndarray:
    """Enumerate images in compiled loops for the independent reference."""
    inverse = np.linalg.inv(cell)
    bounds = np.empty(3, dtype=np.int64)
    count = 1
    for axis in range(3):
        squared_norm = 0.0
        for row in range(3):
            squared_norm += inverse[row, axis] * inverse[row, axis]
        bounds[axis] = int(math.ceil(cutoff * math.sqrt(squared_norm))) + 1
        count *= 2 * bounds[axis] + 1
    shifts = np.empty((count, 3), dtype=np.int64)
    index = 0
    for a in range(-bounds[0], bounds[0] + 1):
        for b in range(-bounds[1], bounds[1] + 1):
            for c in range(-bounds[2], bounds[2] + 1):
                shifts[index, 0] = a
                shifts[index, 1] = b
                shifts[index, 2] = c
                index += 1
    return shifts


@njit
def _force_shifted_lj_energy_compiled(
    distance: float, epsilon: float, sigma: float, cutoff: float
) -> float:
    if distance >= cutoff:
        return 0.0
    safe_distance = max(distance, 1.0e-12)
    ratio6 = (sigma / safe_distance) ** 6
    pair_energy = 4.0 * epsilon * (ratio6 * ratio6 - ratio6)
    ratio6_cutoff = (sigma / cutoff) ** 6
    energy_cutoff = 4.0 * epsilon * (ratio6_cutoff * ratio6_cutoff - ratio6_cutoff)
    derivative_cutoff = (
        24.0 * epsilon / cutoff * (ratio6_cutoff - 2.0 * ratio6_cutoff * ratio6_cutoff)
    )
    return pair_energy - energy_cutoff - (safe_distance - cutoff) * derivative_cutoff


@njit
def _total_energy_compiled(
    xyz: np.ndarray,
    cell: np.ndarray,
    epsilon: float,
    sigma: float,
    cutoff: float,
) -> float:
    shifts = _lattice_shift_array(cell, cutoff)
    n_atoms = len(xyz)
    total = 0.0
    for i in range(n_atoms):
        for j in range(n_atoms):
            for image in range(len(shifts)):
                sx, sy, sz = shifts[image, 0], shifts[image, 1], shifts[image, 2]
                if i == j and sx == 0 and sy == 0 and sz == 0:
                    continue
                dx = xyz[j, 0] - xyz[i, 0] + sx * cell[0, 0] + sy * cell[1, 0] + sz * cell[2, 0]
                dy = xyz[j, 1] - xyz[i, 1] + sx * cell[0, 1] + sy * cell[1, 1] + sz * cell[2, 1]
                dz = xyz[j, 2] - xyz[i, 2] + sx * cell[0, 2] + sy * cell[1, 2] + sz * cell[2, 2]
                distance_squared = dx * dx + dy * dy + dz * dz
                if distance_squared < cutoff * cutoff:
                    total += _force_shifted_lj_energy_compiled(
                        math.sqrt(distance_squared), epsilon, sigma, cutoff
                    )
    return 0.5 * total


@njit
def _particle_interaction_energy_compiled(
    xyz: np.ndarray,
    cell: np.ndarray,
    atom_index: int,
    epsilon: float,
    sigma: float,
    cutoff: float,
) -> float:
    shifts = _lattice_shift_array(cell, cutoff)
    total = 0.0
    for j in range(len(xyz)):
        if j == atom_index:
            continue
        for image in range(len(shifts)):
            sx, sy, sz = shifts[image, 0], shifts[image, 1], shifts[image, 2]
            dx = xyz[j, 0] - xyz[atom_index, 0] + sx * cell[0, 0] + sy * cell[1, 0] + sz * cell[2, 0]
            dy = xyz[j, 1] - xyz[atom_index, 1] + sx * cell[0, 1] + sy * cell[1, 1] + sz * cell[2, 1]
            dz = xyz[j, 2] - xyz[atom_index, 2] + sx * cell[0, 2] + sy * cell[1, 2] + sz * cell[2, 2]
            distance_squared = dx * dx + dy * dy + dz * dz
            if distance_squared < cutoff * cutoff:
                total += _force_shifted_lj_energy_compiled(
                    math.sqrt(distance_squared), epsilon, sigma, cutoff
                )
    return total


def force_shifted_lj_energy(
    distance: float, epsilon: float, sigma: float, cutoff: float
) -> float:
    """Return the 3D force-shifted 12-6 pair energy, without tail correction."""
    if distance <= 0:
        return math.inf
    return float(_force_shifted_lj_energy_compiled(distance, epsilon, sigma, cutoff))


def lattice_shifts(cell: np.ndarray, cutoff: float) -> tuple[np.ndarray, ...]:
    """Enumerate a conservative set of integer images for a triclinic cell."""
    inverse = np.linalg.inv(np.asarray(cell, dtype=np.float64))
    # For any vector shorter than cutoff, each fractional component is bounded
    # by cutoff times the corresponding inverse-cell column norm. A pair's
    # fractional offset can add at most one in either direction.
    bounds = np.ceil(cutoff * np.linalg.norm(inverse, axis=0)).astype(int) + 1
    return tuple(
        np.asarray(index, dtype=np.int64)
        for index in itertools.product(
            *(range(-int(bound), int(bound) + 1) for bound in bounds)
        )
    )


def total_energy(
    coordinates: np.ndarray,
    cell: np.ndarray,
    epsilon: float,
    sigma: float,
    cutoff: float,
) -> float:
    """Compute an independent explicit periodic lattice-sum energy."""
    xyz = np.asarray(coordinates, dtype=np.float64)
    lattice = np.asarray(cell, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("coordinates must have shape (natoms, 3)")
    if lattice.shape != (3, 3) or abs(np.linalg.det(lattice)) <= 0:
        raise ValueError("cell must be a nonsingular 3x3 matrix")
    return float(_total_energy_compiled(xyz, lattice, epsilon, sigma, cutoff))


def particle_interaction_energy(
    coordinates: np.ndarray,
    cell: np.ndarray,
    atom_index: int,
    epsilon: float,
    sigma: float,
    cutoff: float,
) -> float:
    """Return all pair contributions involving one atom for a trial move."""
    xyz = np.asarray(coordinates, dtype=np.float64)
    lattice = np.asarray(cell, dtype=np.float64)
    if atom_index < 0 or atom_index >= len(xyz):
        raise IndexError("atom_index is outside the coordinate array")
    return float(
        _particle_interaction_energy_compiled(
            xyz, lattice, atom_index, epsilon, sigma, cutoff
        )
    )


@dataclass(frozen=True)
class VolumeProposal:
    coordinates: np.ndarray
    cell: np.ndarray
    volume: float
    delta_energy: float
    log_volume_jacobian: float
    log_acceptance: float


class NPTMonteCarlo:
    """Metropolis NPT sampler using single-particle and isotropic log-V moves.

    No center-of-mass constraint is applied, matching the unconstrained
    particle coordinates used by the CLI cases. A symmetric proposal in
    ``ln(V)`` gives the Jacobian ``(N+1) ln(V'/V)``: ``N`` from scaling the
    particle coordinates and one from the proposal-measure transformation.
    """

    def __init__(
        self,
        coordinates: np.ndarray,
        cell: np.ndarray,
        beta: float,
        pressure: float,
        epsilon: float,
        sigma: float,
        cutoff: float,
        seed: int,
    ):
        self.coordinates = np.asarray(coordinates, dtype=np.float64).copy()
        self.cell = np.asarray(cell, dtype=np.float64).copy()
        self.beta = float(beta)
        self.pressure = float(pressure)
        self.epsilon = float(epsilon)
        self.sigma = float(sigma)
        self.cutoff = float(cutoff)
        self.rng = np.random.default_rng(seed)
        self.energy = total_energy(
            self.coordinates, self.cell, self.epsilon, self.sigma, self.cutoff
        )
        self.accepted_translations = 0
        self.attempted_translations = 0
        self.accepted_volumes = 0
        self.attempted_volumes = 0

    @property
    def volume(self) -> float:
        return float(abs(np.linalg.det(self.cell)))

    def propose_log_volume(self, delta_log_volume: float) -> VolumeProposal:
        """Create a fully evaluated volume proposal, including self-images."""
        scale = math.exp(float(delta_log_volume) / 3.0)
        proposed_cell = self.cell * scale
        proposed_coordinates = self.coordinates * scale
        proposed_volume = self.volume * math.exp(float(delta_log_volume))
        proposed_energy = total_energy(
            proposed_coordinates,
            proposed_cell,
            self.epsilon,
            self.sigma,
            self.cutoff,
        )
        delta_energy = proposed_energy - self.energy
        log_jacobian = (len(self.coordinates) + 1) * math.log(
            proposed_volume / self.volume
        )
        log_acceptance = (
            -self.beta
            * (delta_energy + self.pressure * (proposed_volume - self.volume))
            + log_jacobian
        )
        return VolumeProposal(
            coordinates=proposed_coordinates,
            cell=proposed_cell,
            volume=proposed_volume,
            delta_energy=delta_energy,
            log_volume_jacobian=log_jacobian,
            log_acceptance=log_acceptance,
        )

    def attempt_translation(self, max_displacement: float) -> bool:
        self.attempted_translations += 1
        atom = int(self.rng.integers(len(self.coordinates)))
        trial = self.coordinates.copy()
        trial[atom] += self.rng.uniform(-max_displacement, max_displacement, 3)
        fractional = trial[atom] @ np.linalg.inv(self.cell)
        trial[atom] = (fractional % 1.0) @ self.cell
        old_interaction = particle_interaction_energy(
            self.coordinates,
            self.cell,
            atom,
            self.epsilon,
            self.sigma,
            self.cutoff,
        )
        new_interaction = particle_interaction_energy(
            trial, self.cell, atom, self.epsilon, self.sigma, self.cutoff
        )
        delta_energy = new_interaction - old_interaction
        log_acceptance = -self.beta * delta_energy
        if math.log(self.rng.random()) < min(0.0, log_acceptance):
            self.coordinates = trial
            self.energy += delta_energy
            self.accepted_translations += 1
            return True
        return False

    def attempt_volume(self, max_log_volume: float) -> bool:
        self.attempted_volumes += 1
        delta = self.rng.uniform(-max_log_volume, max_log_volume)
        proposal = self.propose_log_volume(delta)
        if math.log(self.rng.random()) < min(0.0, proposal.log_acceptance):
            self.coordinates = proposal.coordinates
            self.cell = proposal.cell
            self.energy += proposal.delta_energy
            self.accepted_volumes += 1
            return True
        return False

    def sweep(self, max_displacement: float, max_log_volume: float) -> None:
        for _ in range(len(self.coordinates)):
            self.attempt_translation(max_displacement)
        self.attempt_volume(max_log_volume)

    def run(
        self,
        burn_in: int,
        production: int,
        stride: int,
        max_displacement: float,
        max_log_volume: float,
    ) -> dict[str, np.ndarray | float | int]:
        for _ in range(burn_in):
            self.sweep(max_displacement, max_log_volume)
        volumes: list[float] = []
        energies: list[float] = []
        for step in range(production):
            self.sweep(max_displacement, max_log_volume)
            if (step + 1) % stride == 0:
                volumes.append(self.volume)
                energies.append(self.energy)
        return {
            "volume": np.asarray(volumes),
            "energy": np.asarray(energies),
            "translation_acceptance": self.accepted_translations
            / max(1, self.attempted_translations),
            "volume_acceptance": self.accepted_volumes / max(1, self.attempted_volumes),
            "attempted_translations": self.attempted_translations,
            "attempted_volumes": self.attempted_volumes,
        }


# Lower-triangular cell entries in the order of the anisotropic barostat mask.
TRIL_ENTRIES = ((0, 0), (1, 1), (2, 2), (1, 0), (2, 0), (2, 1))


def _cell_spring_log_density(entries, natoms, beta, pressure, stiffness, length):
    """log p(h) = N ln V - beta (PV + E(h)), uniform in the lower-triangular entries."""
    cell = np.zeros(entries.shape[:-1] + (3, 3))
    for column, (row, col) in enumerate(TRIL_ENTRIES):
        cell[..., row, col] = entries[..., column]
    diagonal = entries[..., :3]
    volume = np.prod(np.clip(diagonal, 1e-12, None), axis=-1)
    metric = np.einsum("...ij,...kj->...ik", cell, cell) - length**2 * np.eye(3)
    energy = 0.5 * stiffness * np.sum(metric**2, axis=(-2, -1))
    log_density = natoms * np.log(volume) - beta * (pressure * volume + energy)
    return np.where(np.all(diagonal > 0, axis=-1), log_density, -np.inf)


def sample_cell_spring(natoms, beta, pressure, stiffness, length, mask,
                       chains=4000, sweeps=6000, burn_in=1000, seed=0):
    """Mean and variance of the six lower-triangular cell entries.

    The atoms are an ideal gas, so integrating them out leaves the six cell
    entries with density V**N exp(-beta (PV + E(h))). Entries outside ``mask``
    stay zero. Many short vectorized Metropolis chains start from the cubic
    cell of side ``length``.
    """
    rng = np.random.default_rng(seed)
    active = np.asarray(mask, dtype=float)
    entries = np.zeros((chains, 6))
    entries[:, :3] = length
    args = (natoms, beta, pressure, stiffness, length)
    log_density = _cell_spring_log_density(entries, *args)
    # Steps of about 0.6 and 0.9 standard deviations of the spring alone.
    step = length * np.array([0.06, 0.06, 0.06, 0.09, 0.09, 0.09]) * active
    total = np.zeros(6)
    total_squared = np.zeros(6)
    count = 0
    for sweep in range(sweeps):
        trial = entries + rng.normal(size=entries.shape) * step
        trial_density = _cell_spring_log_density(trial, *args)
        accept = np.log(rng.random(chains)) < trial_density - log_density
        entries[accept] = trial[accept]
        log_density[accept] = trial_density[accept]
        if sweep >= burn_in:
            total += entries.sum(axis=0)
            total_squared += (entries**2).sum(axis=0)
            count += chains
    mean = total / count
    return mean, total_squared / count - mean**2
