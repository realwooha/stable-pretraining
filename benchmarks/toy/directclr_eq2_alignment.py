"""Fixed-budget CPU alignment diagnostics for Equation 2 of Jing et al. (2022).

Uses a fixed synthetic batch and mean loss, so the step-size comparisons
approximate the same deterministic gradient flow. This differs from the
stochastic dot-product experiment in ``directclr_mechanisms.py``.
"""

import argparse
import csv
import gzip
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits


def initial_state(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create unaligned factors, a Gaussian batch and fixed augmentation noise.

    Args:
        seed: One of the declared independent seeds.

    Returns:
        W1, W2, inputs and unit Gaussian perturbations, all FP64 arrays.
    """
    rng = np.random.default_rng(seed)
    weights = []
    for _ in range(2):
        left, _, right = np.linalg.svd(rng.random((16, 16)))
        weights.append((left * np.exp(-0.35 * np.arange(16))) @ right)
    inputs = np.random.default_rng(1000 + seed).normal(size=(128, 16))
    noise = np.random.default_rng(3000 + seed).normal(size=inputs.shape)
    return *weights, inputs, noise


def equation2(
    product: np.ndarray, inputs: np.ndarray, positive: np.ndarray
) -> tuple[float, np.ndarray]:
    """Evaluate mean Eq2 and its symmetric input-space matrix X.

    The diagonal candidate is the paired positive. Off-diagonal candidates
    are other anchors, rather than the other view's examples. For the mean
    objective, dL/dW = -W @ X. No normalization or temperature is added.

    Args:
        product: Effective linear map W2 @ W1.
        inputs: Fixed anchor rows.
        positive: Fixed positive-view rows.

    Returns:
        Mean loss and symmetric X, including the 1/batch scaling.
    """
    embedding = inputs @ product.T
    squared = np.sum(embedding**2, axis=1)
    scores = embedding @ embedding.T - (squared[:, None] + squared[None, :]) / 2
    delta = positive - inputs
    positive_scores = -np.sum((delta @ product.T) ** 2, axis=1) / 2
    np.fill_diagonal(scores, positive_scores)
    maximum = scores.max(axis=1)
    exponent = np.exp(scores - maximum[:, None])
    denominator = exponent.sum(axis=1)
    loss = np.mean(maximum + np.log(denominator) - positive_scores)
    probabilities = exponent / denominator[:, None]
    np.fill_diagonal(probabilities, 0)
    mass = probabilities.sum(axis=1)
    laplacian = np.diag(mass + probabilities.sum(axis=0))
    laplacian -= probabilities + probabilities.T
    matrix = inputs.T @ laplacian @ inputs - delta.T @ (mass[:, None] * delta)
    matrix /= len(inputs)
    return float(loss), (matrix + matrix.T) / 2


def observe(
    first: np.ndarray,
    second: np.ndarray,
    inputs: np.ndarray,
    positive: np.ndarray,
    invariant: np.ndarray,
) -> dict:
    """Record alignment, local theorem-condition checks and numerical state.

    Args:
        first: First linear factor.
        second: Second linear factor.
        inputs: Anchor rows.
        positive: Positive-view rows.
        invariant: Initial Gram difference W1 W1.T - W2.T W2.

    Returns:
        Scalars and factor matrices for independent post-run verification.
    """
    product = second @ first
    loss, matrix = equation2(product, inputs, positive)
    left, singular1, _ = np.linalg.svd(first)
    _, singular2, right = np.linalg.svd(second)
    alignment = np.abs(right @ left)
    eigenvalues = np.linalg.eigvalsh(matrix)
    gram_difference = first @ first.T - second.T @ second
    gaps = [np.min((s[:-1] - s[1:]) / s[:-1]) for s in (singular1, singular2)]
    return dict(
        loss=loss,
        alignment_diagonal_mean=float(np.trace(alignment) / len(alignment)),
        alignment_max_identity_error=float(np.max(np.abs(alignment - np.eye(16)))),
        x_min_eigenvalue=float(eigenvalues[0]),
        x_max_eigenvalue=float(eigenvalues[-1]),
        observed_positive_definite=bool(
            eigenvalues[0] > 1e-12 * max(1, np.max(np.abs(eigenvalues)))
        ),
        product_frobenius_norm=float(np.linalg.norm(product)),
        minimum_relative_gap_w1=float(gaps[0]),
        minimum_relative_gap_w2=float(gaps[1]),
        gram_invariant_relative_drift=float(
            np.linalg.norm(gram_difference - invariant)
            / max(1, np.linalg.norm(invariant))
        ),
        w1=first.tolist(),
        w2=second.tolist(),
    )


def simulate(
    seed: int, noise_std: float, step_size: float, horizon: float = 100
) -> dict:
    """Integrate both factors with simultaneous explicit Euler updates.

    Args:
        seed: Initialization and data seed.
        noise_std: Fixed positive-view perturbation scale, zero or 0.001.
        step_size: Fixed Euler step size.
        horizon: Declared gradient-flow time; the loss is averaged over rows.

    Returns:
        Every integer-time observation and actual loop duration.
    """
    first, second, inputs, noise = initial_state(seed)
    positive = inputs + noise_std * noise
    invariant = first @ first.T - second.T @ second
    updates = round(horizon / step_size)
    record_every = round(1 / step_size)
    if not np.isclose(updates * step_size, horizon) or not np.isclose(
        record_every * step_size, 1
    ):
        raise ValueError("Horizon and integer record times must fit the step size")
    records = [
        dict(
            update=0, flow_time=0, **observe(first, second, inputs, positive, invariant)
        )
    ]
    started = time.perf_counter()
    for update in range(1, updates + 1):
        _, matrix = equation2(second @ first, inputs, positive)
        product_gradient = -(second @ first) @ matrix
        gradient1 = second.T @ product_gradient
        gradient2 = product_gradient @ first.T
        first = first - step_size * gradient1
        second = second - step_size * gradient2
        if not np.isfinite(first).all() or not np.isfinite(second).all():
            raise FloatingPointError(f"Nonfinite factors at update {update}")
        if update % record_every == 0 or update == updates:
            records.append(
                dict(
                    update=update,
                    flow_time=float(update * step_size),
                    **observe(first, second, inputs, positive, invariant),
                )
            )
    return dict(
        seed=seed,
        noise_std=noise_std,
        step_size=step_size,
        horizon=horizon,
        updates=updates,
        seconds=time.perf_counter() - started,
        records=records,
    )


def main() -> None:
    """Run all registered arms, or a labeled 100-update timing pilot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with threadpool_limits(limits=1):
        if args.pilot:
            run = simulate(0, 0.001, 0.01, horizon=1)
            (args.output / "pilot.json").write_text(json.dumps(run, indent=2) + "\n")
            print(
                json.dumps(dict(pilot=True, updates=100, seconds=run["seconds"])),
                flush=True,
            )
            return
        runs = []
        for seed in (0, 1, 2):
            for noise in (0.0, 0.001):
                for step in (0.01, 0.005, 0.0025):
                    run = simulate(seed, noise, step)
                    runs.append(run)
                    with gzip.open(args.output / "results.json.gz", "wt") as stream:
                        json.dump(runs, stream, allow_nan=False)
                    print(
                        json.dumps(
                            dict(
                                completed=len(runs),
                                seed=seed,
                                noise=noise,
                                step=step,
                                seconds=run["seconds"],
                            )
                        ),
                        flush=True,
                    )
        rows = [
            dict(
                seed=run["seed"],
                noise_std=run["noise_std"],
                step_size=run["step_size"],
                **{k: v for k, v in rec.items() if k not in ("w1", "w2")},
            )
            for run in runs
            for rec in run["records"]
        ]
        with (args.output / "trajectories.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=list(rows[0]), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
