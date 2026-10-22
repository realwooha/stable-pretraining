"""Plot every registered Eq2 alignment seed and the Euler step-size check."""

import argparse
import csv
import gzip
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np


def main() -> None:
    """Render scalar curves and all six finest-step endpoint matrices."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.data_dir / "trajectories.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    with gzip.open(args.data_dir / "results.json.gz", "rt") as stream:
        runs = json.load(stream)
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    for noise, color in ((0.0, "#327ba8"), (0.001, "#d69236")):
        for step, style in ((0.01, ":"), (0.005, "--"), (0.0025, "-")):
            seeds = [
                [
                    r
                    for r in rows
                    if int(r["seed"]) == seed
                    and float(r["noise_std"]) == noise
                    and float(r["step_size"]) == step
                ]
                for seed in (0, 1, 2)
            ]
            times = np.array([float(r["flow_time"]) for r in seeds[0]])
            for axis, key in (
                (axes[0, 0], "alignment_diagonal_mean"),
                (axes[0, 1], "alignment_max_identity_error"),
            ):
                values = np.array([[float(r[key]) for r in seed] for seed in seeds])
                axis.plot(
                    times,
                    values.mean(0),
                    linestyle=style,
                    color=color,
                    label=f"noise {noise:g}, h={step:g}",
                )
                if step == 0.0025:
                    axis.fill_between(
                        times, values.min(0), values.max(0), alpha=0.15, color=color
                    )
            if step == 0.0025:
                values = np.array(
                    [[float(r["x_min_eigenvalue"]) for r in seed] for seed in seeds]
                )
                axes[1, 0].plot(
                    times, values.mean(0), color=color, label=f"noise {noise:g}"
                )
                axes[1, 0].fill_between(
                    times, values.min(0), values.max(0), alpha=0.15, color=color
                )
        differences = []
        for seed in (0, 1, 2):
            pair = {
                r["step_size"]: r
                for r in runs
                if r["seed"] == seed and r["noise_std"] == noise
            }
            errors = []
            for coarse, fine in zip(pair[0.005]["records"], pair[0.0025]["records"]):
                first = np.array([coarse["w1"], coarse["w2"]])
                second = np.array([fine["w1"], fine["w2"]])
                errors.append(
                    np.linalg.norm(first - second) / max(1, np.linalg.norm(second))
                )
            differences.append(errors)
        differences = np.array(differences)
        axes[1, 1].plot(
            times, differences.mean(0), color=color, label=f"noise {noise:g}"
        )
        axes[1, 1].fill_between(
            times, differences.min(0), differences.max(0), color=color, alpha=0.15
        )
    axes[0, 0].set(
        title="A. Original sorted-SVD alignment",
        ylabel="Mean absolute diagonal",
        ylim=(0, 1.03),
    )
    axes[0, 1].set(
        title="B. Full matrix difference from identity",
        ylabel="Maximum elementwise absolute error",
        ylim=(0, 1.03),
    )
    axes[1, 0].set(
        title="C. X positive-definiteness at recorded times",
        ylabel="Minimum eigenvalue of X",
    )
    axes[1, 0].set_yscale("symlog", linthresh=1e-8)
    axes[1, 1].set(
        title="D. Same-time step-size comparison",
        ylabel="Relative factor difference: h=.005 vs .0025",
    )
    for axis in axes.flat:
        axis.set_xlabel("Mean-loss gradient-flow time")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    fig.supxlabel(
        "Fixed batch, FP64, three seeds. Shading: observed min/max at finest step; not confidence intervals."
    )
    fig.savefig(args.output / "eq2-alignment-trajectories.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(11, 9), layout="constrained")
    fig.set_constrained_layout_pads(h_pad=0.12, hspace=0.12)
    for row, noise in enumerate((0.0, 0.001)):
        for seed in (0, 1, 2):
            run = next(
                r
                for r in runs
                if r["seed"] == seed
                and r["noise_std"] == noise
                and r["step_size"] == 0.0025
            )
            final = run["records"][-1]
            u1, _, _ = np.linalg.svd(final["w1"])
            _, _, v2 = np.linalg.svd(final["w2"])
            im = axes[row, seed].imshow(np.abs(v2 @ u1), cmap="viridis", vmin=0, vmax=1)
            axes[row, seed].set(
                title=f"Seed {seed}, noise {noise:g}\ndiagonal mean {final['alignment_diagonal_mean']:.4f}",
                xlabel="W1 left singular direction",
                ylabel="W2 right singular direction",
            )
    fig.colorbar(im, ax=axes, shrink=0.8, label="Absolute alignment")
    fig.supxlabel(
        "All registered seeds at time 100, h=.0025; no reordering or best-seed selection."
    )
    fig.savefig(args.output / "eq2-alignment-matrices.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
