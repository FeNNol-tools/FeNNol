#!/usr/bin/env python3
import numpy as np
import argparse
from pathlib import Path

KCALPERMOL = 627.5096080305927
KELVIN = 3.15774e5


def main():
    parser = argparse.ArgumentParser(description="analyze opes kernel file")
    parser.add_argument(
        "kernel_file", type=Path, help="file containing the OPES kernels."
    )
    parser.add_argument(
        "--temperature",
        "-T",
        type=float,
        default=None,
        help="temperature in Kelvin (default: from the kernel file, else 300.0)",
    )
    parser.add_argument(
        "--cvs",
        type=int,
        nargs="+",
        default=[0],
        help="zero-based index of the CVs to analyze (default: [0])",
    )
    parser.add_argument(
        "--bounds",
        type=float,
        nargs="+",
        default=None,
        help="bounds for the CVs. If provided, should be a sequence of 2*n floats, where n is the number of CVs. Format: [min1, max1, min2, max2, ...]. If not provided, the bounds are determined from the kernel file.",
    )
    parser.add_argument(
        "--dx",
        type=float,
        nargs="+",
        default=None,
        help="grid spacing for the CVs. If provided, should be a sequence of n floats, where n is the number of CVs. If not provided, the grid spacing is determined from the kernel file.",
    )
    # parser.add_argument(
    #     "--unit",
    #     type=str,
    #     default="kcal/mol",
    #     choices=["kcal/mol", "kJ/mol", "eV", "Ha"],
    #     help="energy unit for the output (default: kcal/mol)",
    # )
    parser.add_argument(
        "--emax",
        type=float,
        default=100.,
        help="Free energy in kcal/mol corresponding to zero probability (default: 100.0). This is used to avoid numerical issues when computing the free energy.",
    )
    parser.add_argument(
        "--outfile",
        "-o",
        type=Path,
        default=None,
        help="output file for the free energy data. If not provided, the output will be printed to stdout.",
    )
    parser.add_argument(
        "--metadyn",
        action="store_true",
        help="metadynamics kernels (only needed for kernel files without a mode line)",
    )
    parser.add_argument(
        "--explore",
        type=float,
        default=None,
        metavar="GAMMA",
        help="OPES-explore kernels with bias factor GAMMA (only needed for kernel files without a mode line)",
    )
    args = parser.parse_args()

    # check the input file
    kernel_file = args.kernel_file.resolve()
    assert kernel_file.exists(), f"Input file {kernel_file} does not exist"
    kernel_params = np.loadtxt(kernel_file, ndmin=2)

    # fennol_md writes "# mode=... gamma=... temperature=..." as the first line
    metadata = {}
    with open(kernel_file) as f:
        first = f.readline()
    if first.startswith("#") and "mode=" in first:
        metadata = dict(item.split("=", 1) for item in first[1:].split())
    mode = metadata.get("mode", "opes")
    gamma = float(metadata["gamma"]) if "gamma" in metadata else None
    if args.metadyn:
        mode = "metadyn"
    if args.explore is not None:
        mode, gamma = "explore", args.explore
    if mode == "explore" and gamma is None:
        raise ValueError("OPES-explore kernels need the bias factor: use --explore GAMMA")
    temperature = args.temperature
    if temperature is None:
        temperature = float(metadata.get("temperature", 300.0))

    ncvs = (kernel_params.shape[1] - 1) // 2
    all_h = kernel_params[:, 0]
    all_cvs = kernel_params[:, 1 : ncvs + 1]
    all_sigma = kernel_params[:, ncvs + 1 :]

    icvs = []
    for i in args.cvs:
        if i < 0 or i >= ncvs:
            raise ValueError(f"CV index {i} is out of bounds for {ncvs} CVs.")
        icvs.append(i)
    icvs = np.array(icvs, dtype=int)
    if len(icvs) == 0:
        raise ValueError(
            "No CVs specified for analysis. Use --cvs to specify CV indices."
        )
    if len(icvs) > ncvs:
        raise ValueError(
            f"Too many CVs specified for analysis. Maximum is {ncvs}, but {len(icvs)} were provided."
        )
    if len(icvs) > 2:
        raise ValueError(
            f"Max 2 CVs are supported for analysis, but {len(icvs)} were provided."
        )

    if len(icvs) < ncvs and mode != "opes":
        # Only plain OPES kernels estimate the unbiased distribution, whose
        # marginal is the integral of the kernels over the other CVs.
        raise ValueError(
            f"Marginal free energies are not available for {mode} kernels: "
            "analyze all CVs and integrate exp(-F/kT) over the others."
        )
    if len(icvs) == ncvs:
        h = all_h
        cvs = all_cvs[:, icvs]
        sigma = all_sigma[:, icvs]
    else:
        # get indices of the other cvs
        other_cvs = np.setdiff1d(np.arange(ncvs), icvs)
        other_sigma = all_sigma[:, other_cvs]
        marginal = np.prod(other_sigma, axis=1)
        h = all_h * marginal
        cvs = all_cvs[:, icvs]
        sigma = all_sigma[:, icvs]

    ### FIND BOUNDS
    if args.bounds is None:
        min_bounds = np.min(cvs - 3 * sigma, axis=0)
        max_bounds = np.max(cvs + 3 * sigma, axis=0)
    else:
        if len(args.bounds) != 2 * len(icvs):
            raise ValueError(
                f"Bounds must be a sequence of 2*n floats, where n is the number of CVs ({len(icvs)})."
            )
        min_bounds = np.array(args.bounds[0::2], dtype=float)
        max_bounds = np.array(args.bounds[1::2], dtype=float)

    min_bounds, max_bounds = np.minimum(min_bounds, max_bounds), np.maximum(
        min_bounds, max_bounds
    )
    cv_space = max_bounds - min_bounds
    

    ### GRID SPACING
    if args.dx is None:
        dx = np.min(sigma, axis=0) / 5.0
    else:
        if len(args.dx) != len(icvs):
            raise ValueError(
                f"Grid spacing must be a sequence of n floats, where n is the number of CVs ({len(icvs)})."
            )
        dx = np.array(args.dx, dtype=float)

    ngrids = np.ceil(cv_space / dx).astype(int)

    alpha = sigma**-2
    ncvs = len(icvs)
    if ncvs == 1:
        # 1D grid
        grid = np.linspace(min_bounds[0], max_bounds[0], ngrids[0])[:, None]
    elif ncvs == 2:
        # 2D grid
        x = np.linspace(min_bounds[0], max_bounds[0], ngrids[0])
        y = np.linspace(min_bounds[1], max_bounds[1], ngrids[1])
        x, y = np.meshgrid(x, y, indexing="ij")
        grid = np.column_stack((x.ravel(), y.ravel()))

    d2 = np.sum(alpha[None, :, :] * (grid[:, None, :] - cvs[None, :, :]) ** 2, axis=-1)
    prob = np.sum(h[None, :] * np.exp(-0.5*d2), axis=1)

    energy_unit = KCALPERMOL
    kT = temperature / KELVIN * energy_unit

    Emax = args.emax

    if mode == "metadyn":
        # fennol_md writes metadynamics kernel heights in kcal/mol.
        F = -prob
    elif mode == "explore":
        # Explore kernels estimate the well-tempered distribution exp(-F/(gamma kT)).
        eps = np.exp(-Emax / (gamma * kT))
        F = -gamma * kT * np.log(prob + eps)
    else:
        eps = np.exp(-Emax / kT)
        F = -kT * np.log(prob + eps)  

    F -= np.min(F)  # shift the free energy to zero at the minimum

    out_str = ""
    out_str+=f"# min bounds = {min_bounds}\n"
    out_str+=f"# max bounds = {max_bounds}\n"
    out_str+=f"# dx ={dx}\n"
    if ncvs == 1:
        # 1D grid
        grid = grid[:, 0]
        F = F.reshape(ngrids[0])
        out_str+=f"# CV{icvs[0]}    F[kcal/mol]\n"
        for i in range(ngrids[0]):
            out_str+=f"{grid[i]:.6f} {F[i]:.6f}\n"

    elif ncvs == 2:
        # 2D grid
        x = grid[:, 0].reshape(ngrids[0], ngrids[1])
        y = grid[:, 1].reshape(ngrids[0], ngrids[1])
        F = F.reshape(ngrids[0], ngrids[1])
        out_str+=f"# CV{icvs[0]}    CV{icvs[1]}    F[kcal/mol]\n"
        for i in range(ngrids[0]):
            for j in range(ngrids[1]):
                out_str+=f"{x[i,j]:.6f} {y[i,j]:.6f} {F[i,j]:.6f}\n"
            out_str+="\n"
    
    if args.outfile is not None:
        with open(args.outfile, "w") as f:
            f.write(out_str)
        print(f"Free energy data written to {args.outfile}")
    else:
        print(out_str)


if __name__ == "__main__":
    main()
