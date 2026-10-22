"""Plot the registered CPU mechanism runs and recorded real-encoder summaries."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np


def read_csv(path: Path) -> list[dict]:
    """Read scalar plotting inputs.

    Args:
        path: CSV containing previously generated numerical observations.

    Returns:
        Rows with their original column names.
    """
    with path.open() as stream:
        return list(csv.DictReader(stream))


def band(axis, values: np.ndarray, color: str, label: str) -> None:
    """Plot a three-seed mean and the observed min/max range.

    Args:
        axis: Matplotlib axis.
        values: Array with x values first and three seed trajectories after.
        color: Matplotlib color.
        label: Legend label.
    """
    axis.plot(values[0], values[1:].mean(0), color=color, label=label)
    axis.fill_between(
        values[0], values[1:].min(0), values[1:].max(0), color=color, alpha=0.18
    )


def main() -> None:
    """Render the two synthetic experiments and a historical-spectrum summary."""
    default = Path(__file__).parent / "results/directclr"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=default)
    parser.add_argument("--output", type=Path, default=default / "figures")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    curves = read_csv(args.data_dir / "trajectories.csv")
    spectra = read_csv(args.data_dir / "final-spectra.csv")
    alignments = read_csv(args.data_dir / "alignment.csv")
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "DejaVu Sans",
        }
    )

    def values(rows, mechanism, arm, xkey, ykey, relative=False):
        selected = [r for r in rows if r["mechanism"] == mechanism and r["arm"] == arm]
        per_seed = [
            [r for r in selected if int(r["seed"]) == seed] for seed in (0, 1, 2)
        ]
        x = np.array([float(r[xkey]) for r in per_seed[0]])
        y = np.array([[float(r[ykey]) for r in seed] for seed in per_seed])
        if relative:
            y = y / y[:, :1]
        return np.vstack((x, y))

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), layout="constrained")
    for arm, color, label in [
        ("weak", "#327ba8", "Weak noise (std 0.01)"),
        ("strong", "#d69236", "Strong noise (std 1.5)"),
    ]:
        band(
            axes[0],
            values(
                spectra, "augmentation", arm, "component", "covariance_eigenvalue", True
            ),
            color,
            label,
        )
        band(
            axes[1],
            values(curves, "augmentation", arm, "update", "effective_rank"),
            color,
            label,
        )
    axes[0].set(
        yscale="log",
        xlabel="Spectral component",
        ylabel="Covariance eigenvalue / largest",
        title="A. Final centered feature spectra",
    )
    axes[1].set(
        xlabel="Optimizer updates",
        ylabel="Entropy effective rank (out of 16)",
        ylim=(0, 16.5),
        title="B. Directional suppression with stronger noise",
    )
    for axis in axes:
        axis.legend(fontsize=9)
        axis.grid(alpha=0.2)
    fig.supxlabel(
        "Three fixed seeds; bands show observed min/max, not confidence intervals."
    )
    fig.savefig(args.output / "strong-augmentation.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), layout="constrained")
    for arm, color, label in [
        ("one-layer", "#327ba8", "Single layer"),
        ("two-layer", "#7865a8", "Two layers"),
    ]:
        band(
            axes[0, 0],
            values(
                spectra, "implicit", arm, "component", "covariance_eigenvalue", True
            ),
            color,
            label,
        )
        band(
            axes[0, 1],
            values(curves, "implicit", arm, "update", "effective_rank"),
            color,
            label,
        )
    axes[0, 0].set(
        yscale="log",
        xlabel="Spectral component",
        ylabel="Covariance eigenvalue / largest",
        title="A. Small modes remain weak after factorization",
    )
    axes[0, 1].set(
        xlabel="Optimizer updates",
        ylabel="Entropy effective rank (out of 16)",
        ylim=(0, 16.5),
        title="B. Matched initial encoder products",
    )
    for axis in axes[0]:
        axis.legend()
        axis.grid(alpha=0.2)
    for axis, update, title in zip(
        axes[1], [0, 10000], ["C. Initial layer alignment", "D. Final layer alignment"]
    ):
        matrix = np.zeros((16, 16))
        for row in alignments:
            if int(row["update"]) == update:
                matrix[int(row["row"]), int(row["column"])] += (
                    float(row["absolute_alignment"]) / 3
                )
        im = axis.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
        axis.set(
            title=title + f" (mean diagonal {np.diag(matrix).mean():.3f})",
            xlabel="Singular direction",
            ylabel="Singular direction",
        )
        fig.colorbar(im, ax=axis, shrink=0.75)
    fig.supxlabel(
        "Three seeds; weak isotropic noise std 0.001. Perfect diagonal alignment is not reproduced."
    )
    fig.savefig(args.output / "implicit-regularization.png", dpi=160)
    plt.close(fig)

    historical = default / "historical-spectrum-summary.csv"
    if historical.exists():
        rows = read_csv(historical)
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
        labels = ["DirectCLR", "Linear projector", "No projector"]
        for axis, key, title in [
            (
                axes[0],
                "centered_effective_rank",
                "Entropy rank of centered singular values",
            ),
            (
                axes[1],
                "components_for_95_percent_variance",
                "Components retaining 95% of variance",
            ),
        ]:
            bars = axis.bar(
                labels,
                [float(r[key]) for r in rows],
                color=["#327ba8", "#248977", "#d69236"],
            )
            axis.bar_label(bars, fmt="%.1f", padding=3)
            axis.set(
                title=title,
                ylabel="Dimensions",
                ylim=(0, 1100 if key == "centered_effective_rank" else 550),
            )
            axis.grid(axis="y", alpha=0.2)
            axis.set_axisbelow(True)
        fig.supxlabel(
            "Historical 100-epoch encoders; same 4,096 training rows and full 2,048-D center-crop features.\nSummary metrics measure different spectral properties; no new feature extraction."
        )
        fig.savefig(args.output / "historical-encoder-spectra.png", dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    main()
