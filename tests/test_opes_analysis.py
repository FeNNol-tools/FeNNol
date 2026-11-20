"""Free energies reconstructed from OPES kernels by `scripts/analyze_opes.py`."""

import numpy as np
import pytest
from scripts import analyze_opes


def test_analyze_opes_accepts_one_kernel_row(tmp_path, monkeypatch, capsys):
    from scripts import analyze_opes

    kernel_file = tmp_path / "single_kernel.dat"
    kernel_file.write_text("1.0 0.5 0.2\n")
    output_file = tmp_path / "free_energy.dat"
    monkeypatch.setattr(
        "sys.argv",
        ["analyze_opes.py", str(kernel_file), "--outfile", str(output_file)],
    )

    analyze_opes.main()

    result = output_file.read_text()
    assert "# CV0" in result
    assert "nan" not in result.lower()


@pytest.mark.parametrize("selected", [0, 1])
def test_marginal_gaussian_free_energy(tmp_path, monkeypatch, selected):
    # Integrating either coordinate of a separable Gaussian leaves a Gaussian.
    kernels = tmp_path / "kernels.dat"
    np.savetxt(kernels, [[2.0, 0.0, 0.0, 1.0, 2.0]])
    output = tmp_path / "fes.dat"
    monkeypatch.setattr(
        "sys.argv",
        ["analyze_opes", str(kernels), "--cvs", str(selected),
         "--bounds", "-2", "2", "--dx", "0.8", "-o", str(output)],
    )
    analyze_opes.main()
    data = np.loadtxt(output)
    # At 300 K, F(x)-F(0) = kT x²/(2 sigma²), independently of height.
    kT = 300.0 / analyze_opes.KELVIN * analyze_opes.KCALPERMOL
    sigma = [1.0, 2.0][selected]
    np.testing.assert_allclose(data[:, 1], kT * data[:, 0] ** 2 / (2 * sigma**2), atol=6e-7)


def test_two_dimensional_gaussian_respects_requested_axis_order(tmp_path, monkeypatch):
    kernels = tmp_path / "kernels.dat"
    np.savetxt(kernels, [[1.0, 0.0, 0.0, 1.0, 2.0]])
    output = tmp_path / "fes.dat"
    monkeypatch.setattr(
        "sys.argv",
        ["analyze_opes", str(kernels), "--cvs", "1", "0",
         "--bounds", "-2", "2", "-2", "2", "--dx", "0.8", "0.8",
         "-o", str(output)],
    )
    analyze_opes.main()
    data = np.loadtxt(output)
    kT = 300.0 / analyze_opes.KELVIN * analyze_opes.KCALPERMOL
    # The first output coordinate is CV1, whose standard deviation is two.
    expected = kT * (data[:, 0] ** 2 / 8.0 + data[:, 1] ** 2 / 2.0)
    np.testing.assert_allclose(data[:, 2], expected, atol=6e-7)


def _run(monkeypatch, kernels, output, *extra):
    monkeypatch.setattr(
        "sys.argv",
        ["analyze_opes", str(kernels), "--bounds", "-2", "2", "--dx", "0.8", "-o", str(output), *extra],
    )
    analyze_opes.main()
    return np.loadtxt(output)


def test_explore_kernels_from_header_scale_by_gamma(tmp_path, monkeypatch):
    # Explore kernels estimate p**(1/gamma): F = -gamma kT log(sum of kernels).
    kernels = tmp_path / "kernels.dat"
    np.savetxt(kernels, [[1.0, 0.0, 1.0]], header="mode=explore gamma=4 temperature=400\nh x sigma_x")
    data = _run(monkeypatch, kernels, tmp_path / "fes.dat")
    kT = 400.0 / analyze_opes.KELVIN * analyze_opes.KCALPERMOL
    np.testing.assert_allclose(data[:, 1], 4 * kT * data[:, 0] ** 2 / 2, atol=6e-7)
    # Kernel files without the mode line need the bias factor on the command line.
    np.savetxt(kernels, [[1.0, 0.0, 1.0]])
    legacy = _run(monkeypatch, kernels, tmp_path / "legacy.dat", "--explore", "4", "-T", "400")
    np.testing.assert_allclose(legacy[:, 1], data[:, 1], atol=6e-7)


@pytest.mark.parametrize("mode", ["explore", "metadyn"])
def test_marginals_are_refused_for_explore_and_metadynamics_kernels(tmp_path, monkeypatch, mode):
    kernels = tmp_path / "kernels.dat"
    np.savetxt(kernels, [[1.0, 0.0, 0.0, 1.0, 2.0]], header=f"mode={mode} gamma=4 temperature=300")
    with pytest.raises(ValueError, match="Marginal"):
        _run(monkeypatch, kernels, tmp_path / "fes.dat", "--cvs", "0")
