"""Centre-of-mass treatment and degree-of-freedom count of the thermostats (remove_com, fix_com)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fennol.md.thermostats import get_thermostat

NAT = 8


def _simulation_setup():
    mass = np.linspace(1.0, 4.0, NAT)
    return {
        "kT": 1.0,
        "mass": mass,
        "species": np.ones(NAT, dtype=np.int32),
        "mass_Da": mass,
        "totmass_Da": float(mass.sum()),
        "nat": NAT,
        "nfree": NAT,
        "nreplicas": 1,
        "remove_com_velocity": True,
        "ndof": 3 * NAT - 3,
    }


def _com_velocity(vel, mass):
    return np.sum(mass[:, None] * np.asarray(vel), axis=0) / mass.sum()


@pytest.mark.parametrize("name", ["BUSSI", "NOSE"])
def test_global_thermostat_targets_internal_degrees_of_freedom_under_external_kicks(name):
    simulation_setup = _simulation_setup()
    thermostat, _, state, vel, _, _ = get_thermostat(
        {"thermostat": name, "gamma": 2.0}, dt=0.25, simulation_setup=simulation_setup,
        fprec=np.float64, rng_key=jax.random.PRNGKey(1),
    )
    mass = jnp.asarray(simulation_setup["mass"])
    kick = jnp.array([0.3, -0.2, 0.1])  # uniform external force on every atom

    def step(carry, _):
        vel, state = carry
        vel, state = thermostat(vel + kick, state)
        return (vel, state), (0.5 * jnp.sum(mass[:, None] * vel**2), vel)

    (_, _), (kinetic, velocities) = jax.lax.scan(step, (vel, state), None, length=4000)
    np.testing.assert_allclose(_com_velocity(velocities[-1], simulation_setup["mass"]), 0.0, atol=1e-6)
    if name == "BUSSI":
        # Bussi samples the canonical kinetic energy of the ndof internal modes.
        assert float(np.mean(kinetic[500:])) == pytest.approx(0.5 * simulation_setup["ndof"], rel=0.05)


def test_nve_removes_com_velocity():
    simulation_setup = _simulation_setup()
    thermostat, _, state, vel, _, _ = get_thermostat(
        {"thermostat": "NVE"}, dt=0.1, simulation_setup=simulation_setup,
        fprec=np.float64, rng_key=jax.random.PRNGKey(0),
    )
    drifted, _ = thermostat(vel + 1.0, state)
    np.testing.assert_allclose(_com_velocity(drifted, simulation_setup["mass"]), 0.0, atol=1e-6)


def _load(tmp_path, nat=3, **params):
    from fennol.md.initial import load_simulation_setup

    xyz = tmp_path / "system.xyz"
    xyz.write_text(f"{nat}\n\n" + "".join(f"Ar {2.0 * i} 0.0 0.0\n" for i in range(nat)))
    simulation_setup, _ = load_simulation_setup({"xyz_input/file": str(xyz), "temperature": 300.0, **params}, np.float64)
    return simulation_setup


@pytest.mark.parametrize(
    "params, removed, ndof",
    [
        ({"thermostat": "BUSSI"}, True, 6),
        ({"thermostat": "BUSSI", "remove_com": False}, False, 9),
        ({"thermostat": "LGV"}, False, 9),
        ({"thermostat": "NVE", "frozen_atoms": [1]}, False, 6),
    ],
)
def test_remove_com_keyword_sets_degrees_of_freedom(tmp_path, params, removed, ndof):
    simulation_setup = _load(tmp_path, **params)
    assert simulation_setup["remove_com_velocity"] is removed
    assert simulation_setup["ndof"] == ndof


def test_remove_com_refuses_a_single_atom(tmp_path):
    with pytest.raises(ValueError, match="remove_com"):
        _load(tmp_path, nat=1, thermostat="BUSSI")
    assert _load(tmp_path, nat=1, thermostat="BUSSI", remove_com=False)["ndof"] == 3


def test_nve_without_com_removal_keeps_external_acceleration():
    simulation_setup = {**_simulation_setup(), "remove_com_velocity": False, "thermal_com": True, "ndof": 3 * NAT}
    thermostat, _, state, vel, _, _ = get_thermostat(
        {"thermostat": "NVE", "temperature": 300.0}, dt=0.1, simulation_setup=simulation_setup,
        fprec=np.float64, rng_key=jax.random.PRNGKey(0),
    )
    # The initial centre-of-mass velocity is thermal, not removed.
    assert np.linalg.norm(_com_velocity(vel, simulation_setup["mass"])) > 0.0
    drifted, _ = thermostat(vel + 1.0, state)
    np.testing.assert_allclose(_com_velocity(drifted - vel, simulation_setup["mass"]), 1.0, atol=1e-6)


def test_bussi_without_com_removal_thermalizes_all_modes_in_external_wells():
    # Independent harmonic wells: total momentum is not conserved and the
    # centre of mass is a thermal degree of freedom (3N).
    simulation_setup = {**_simulation_setup(), "remove_com_velocity": False, "thermal_com": True, "ndof": 3 * NAT}
    dt = 0.05
    thermostat, _, state, vel, _, _ = get_thermostat(
        {"thermostat": "BUSSI", "gamma": 2.0}, dt=dt, simulation_setup=simulation_setup,
        fprec=np.float64, rng_key=jax.random.PRNGKey(3),
    )
    mass = jnp.asarray(simulation_setup["mass"])[:, None]
    k = jnp.linspace(0.5, 2.0, 3 * NAT).reshape(NAT, 3)

    def step(carry, _):
        x, vel, state = carry
        vel = vel - 0.5 * dt * k * x / mass
        x = x + dt * vel
        vel = vel - 0.5 * dt * k * x / mass
        vel, state = thermostat(vel, state)
        return (x, vel, state), (0.5 * jnp.sum(mass * vel**2), 0.5 * jnp.sum(k * x**2))

    _, (kinetic, potential) = jax.lax.scan(step, (jnp.zeros_like(vel), vel, state), None, length=20000)
    assert float(np.mean(kinetic[2000:])) == pytest.approx(0.5 * 3 * NAT, rel=0.05)
    assert float(np.mean(potential[2000:])) == pytest.approx(0.5 * 3 * NAT, rel=0.1)


def test_fix_com_refuses_frozen_atoms(monkeypatch, tmp_path):
    from fennol.md import integrate as integrate_module

    class Model:
        energy_unit = "KCALPERMOL"
        preproc_state = None

    simulation_setup = {"nat": NAT, "nreplicas": 1, "frozen_atoms": np.array([0]), "name": str(tmp_path / "md")}
    monkeypatch.setattr(integrate_module, "load_model", lambda _params: Model())
    monkeypatch.setattr(integrate_module, "load_simulation_setup", lambda _params, _fprec: (simulation_setup, {}))
    monkeypatch.setattr(integrate_module, "get_restart_file", lambda _data: str(tmp_path / "no-restart"))
    with pytest.raises(ValueError, match="fix_com"):
        integrate_module.initialize_dynamics({"fix_com": True}, "float32", jax.random.PRNGKey(0))
