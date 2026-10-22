"""CPU mechanisms from Jing et al., ICLR 2022, using the released toy recipe.

This independent implementation follows the dot-product objective and settings
in facebookresearch/directclr/toy_task, rather than the distance-form objective
in the paper's theoretical Equation 2. No image data or learned checkpoint is
used. Run ``python benchmarks/toy/directclr_mechanisms.py --help``.
"""

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn


def objective(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Compute unnormalized, one-way cross-view toy InfoNCE.

    Args:
        first: Anchor embeddings with shape [batch, dimension].
        second: Aligned candidate embeddings with the same shape.

    Returns:
        Mean negative log probability of aligned cross-view pairs.
    """
    scores = first @ second.T / 0.1
    return (scores.logsumexp(1) - scores.diagonal()).mean()


def make_model(seed: int, mechanism: str, depth: int) -> nn.Module:
    """Construct paired models with controlled initialization.

    Args:
        seed: Fixed initialization seed.
        mechanism: Augmentation or implicit-regularization setting.
        depth: Number of linear layers, one or two.

    Returns:
        CPU model; the implicit single-layer control starts at W2 @ W1.
    """
    if mechanism == "augmentation":
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            return nn.Linear(16, 16, bias=True)
    generator = np.random.default_rng(seed)
    weights = []
    for _ in range(2):
        left, _, right = np.linalg.svd(generator.random((16, 16)))
        weight = (left * np.exp(-0.35 * np.arange(16))) @ right
        weights.append(torch.tensor(weight, dtype=torch.float32))
    with torch.random.fork_rng():
        model = nn.Sequential(*(nn.Linear(16, 16, bias=False) for _ in range(depth)))
    with torch.no_grad():
        if depth == 1:
            model[0].weight.copy_(weights[1] @ weights[0])
        else:
            for layer, weight in zip(model, weights):
                layer.weight.copy_(weight)
    return model


def views(
    images: torch.Tensor, noise_std: float, mechanism: str, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply paired Gaussian perturbations.

    Args:
        images: Synthetic inputs with 16 coordinates.
        noise_std: Standard deviation of the perturbation.
        mechanism: Selects the affected input coordinates.
        generator: Dedicated augmentation RNG stream.

    Returns:
        Two independently perturbed views of each input.
    """
    noise = torch.randn((2, *images.shape), generator=generator) * noise_std
    if mechanism == "augmentation":
        noise[:, :, :8] = 0
    return images + noise[0], images + noise[1]


@torch.no_grad()
def diagnostics(model: nn.Module, probe: torch.Tensor) -> dict:
    """Measure centered spectra and layer alignment without consuming RNG.

    Args:
        model: One- or two-layer affine/linear CPU network.
        probe: Fixed synthetic evaluation inputs, independent of training.

    Returns:
        Scalar diagnostics, centered covariance eigenvalues and alignment.
    """
    weights = [p.double() for p in model.parameters() if p.ndim == 2]
    product = weights[0]
    for weight in weights[1:]:
        product = weight @ product
    centered = (probe.double() - probe.double().mean(0)) @ product.T
    singular = torch.linalg.svdvals(centered)
    eigenvalues = singular.square() / (len(probe) - 1)
    probability = singular / singular.sum().clamp_min(1e-30)
    rank = torch.exp(-(probability * probability.clamp_min(1e-30).log()).sum())
    alignment = None
    if len(weights) == 2:
        left = torch.linalg.svd(weights[0]).U
        right = torch.linalg.svd(weights[1]).Vh
        alignment = (right @ left).abs()
    return dict(
        effective_rank=float(rank),
        tail8_energy_fraction=float(eigenvalues[8:].sum() / eigenvalues.sum()),
        min_max_singular_ratio=float(singular[-1] / singular[0]),
        augmented_column_energy_ratio=float(
            product[:, 8:].square().sum() / product[:, :8].square().sum()
        ),
        alignment_diagonal_mean=float(alignment.diagonal().mean())
        if alignment is not None
        else None,
        covariance_eigenvalues=eigenvalues.tolist(),
        weight_singular_values=[torch.linalg.svdvals(w).tolist() for w in weights],
        product_singular_values=torch.linalg.svdvals(product).tolist(),
        alignment=alignment.tolist() if alignment is not None else None,
    )


def simulate(seed: int, mechanism: str, arm: str, updates: int) -> dict:
    """Run one fixed-budget arm and retain scheduled diagnostics.

    Args:
        seed: One of the predeclared independent training seeds.
        mechanism: Augmentation or implicit-regularization setting.
        arm: Predeclared weak/strong or one-layer/two-layer condition.
        updates: Successful optimizer-update budget.

    Returns:
        Complete scheduled numerical observations for this run.
    """
    depth = 2 if arm == "two-layer" else 1
    noise = (1.5 if arm == "strong" else 0.01) if mechanism == "augmentation" else 0.001
    lr = 0.001 if mechanism == "augmentation" else 0.01
    data = torch.randn(10000, 16, generator=torch.Generator().manual_seed(1000 + seed))
    probe = torch.randn(1000, 16, generator=torch.Generator().manual_seed(4000 + seed))
    noise_generator = torch.Generator().manual_seed(3000 + seed)
    index_generator = np.random.default_rng(2000 + seed)
    model = make_model(seed, mechanism, depth)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0, weight_decay=0)
    records = [dict(update=0, loss=None, **diagnostics(model, probe))]
    started = time.perf_counter()
    for update in range(1, updates + 1):
        indices = index_generator.choice(len(data), 128, replace=False)
        first, second = views(data[indices], noise, mechanism, noise_generator)
        loss = objective(model(first), model(second))
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"nonfinite loss: {seed}/{mechanism}/{arm}/{update}"
            )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if update % 100 == 0 or update == updates:
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise FloatingPointError("nonfinite model state")
            records.append(
                dict(
                    update=update,
                    loss=float(loss.detach()),
                    **diagnostics(model, probe),
                )
            )
    return dict(
        seed=seed,
        mechanism=mechanism,
        arm=arm,
        noise_std=noise,
        learning_rate=lr,
        updates=updates,
        seconds=time.perf_counter() - started,
        records=records,
    )


def write_outputs(output: Path, runs: list[dict]) -> None:
    """Write scalar plot inputs and a complete numerical result record.

    Args:
        output: Existing dedicated output directory.
        runs: Completed experimental runs, including every declared seed.
    """
    (output / "results.json").write_text(
        json.dumps(runs, indent=2, allow_nan=False) + "\n"
    )
    trajectories, spectra, alignments = [], [], []
    for run in runs:
        identity = {k: run[k] for k in ("mechanism", "arm", "seed")}
        for record in run["records"]:
            trajectories.append(
                {
                    **identity,
                    **{
                        k: record[k]
                        for k in (
                            "update",
                            "loss",
                            "effective_rank",
                            "tail8_energy_fraction",
                            "min_max_singular_ratio",
                            "augmented_column_energy_ratio",
                            "alignment_diagonal_mean",
                        )
                    },
                }
            )
        final = run["records"][-1]
        for index, value in enumerate(final["covariance_eigenvalues"]):
            spectra.append(
                {
                    **identity,
                    "component": index + 1,
                    "covariance_eigenvalue": value,
                    "weight1_singular": final["weight_singular_values"][0][index],
                    "product_singular": final["product_singular_values"][index],
                }
            )
        if run["arm"] == "two-layer":
            for endpoint in (run["records"][0], final):
                for row in range(16):
                    for column in range(16):
                        alignments.append(
                            {
                                **identity,
                                "update": endpoint["update"],
                                "row": row,
                                "column": column,
                                "absolute_alignment": endpoint["alignment"][row][
                                    column
                                ],
                            }
                        )
    for name, rows in [
        ("trajectories", trajectories),
        ("final-spectra", spectra),
        ("alignment", alignments),
    ]:
        with (output / (name + ".csv")).open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=list(rows[0]), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    """Execute the fixed three-seed CPU experiment or a labeled timing pilot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--pilot", action="store_true", help="Timing only: 100 updates, one arm"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    runs = []
    arms = [
        ("augmentation", "weak"),
        ("augmentation", "strong"),
        ("implicit", "one-layer"),
        ("implicit", "two-layer"),
    ]
    for seed in [0] if args.pilot else [0, 1, 2]:
        for mechanism, arm in arms[:1] if args.pilot else arms:
            result = simulate(seed, mechanism, arm, 100 if args.pilot else 10000)
            runs.append(result)
            print(
                json.dumps({k: v for k, v in result.items() if k != "records"}),
                flush=True,
            )
            (args.output / "progress.json").write_text(
                json.dumps(
                    dict(
                        completed=len(runs),
                        expected=1 if args.pilot else 12,
                        pilot=args.pilot,
                    )
                )
            )
    if args.pilot:
        (args.output / "pilot.json").write_text(json.dumps(runs, indent=2) + "\n")
    else:
        write_outputs(args.output, runs)


if __name__ == "__main__":
    main()
