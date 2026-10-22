"""Independent numerical checks for the Eq2 alignment experiment."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.special import logsumexp
import torch
from threadpoolctl import threadpool_limits

pytestmark = pytest.mark.unit
SCRIPT = (
    Path(__file__).resolve().parents[3] / "benchmarks/toy/directclr_eq2_alignment.py"
)
spec = importlib.util.spec_from_file_location("directclr_eq2_alignment", SCRIPT)
toy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(toy)


def dense_torch_objective(first, positive):
    negatives = -(first[:, None, :] - first[None, :, :]).square().sum(-1) / 2
    pair = -(first - positive).square().sum(-1) / 2
    scores = torch.where(
        torch.eye(len(first), dtype=torch.bool), pair[:, None], negatives
    )
    return (torch.logsumexp(scores, 1) - pair).mean()


@pytest.mark.parametrize("noise", [0, 0.001, 0.8])
def test_equation2_value_and_x_match_independent_pair_sum(noise):
    rng = np.random.default_rng(9)
    inputs = rng.normal(size=(7, 4))
    positive = inputs + noise * rng.normal(size=inputs.shape)
    weight = rng.normal(size=(3, 4))
    z, p = inputs @ weight.T, positive @ weight.T
    scores = -np.sum((z[:, None] - z[None]) ** 2, axis=-1) / 2
    np.fill_diagonal(scores, -np.sum((z - p) ** 2, axis=1) / 2)
    expected_loss = np.mean(logsumexp(scores, axis=1) - np.diag(scores))
    alpha = np.exp(scores - logsumexp(scores, axis=1, keepdims=True))
    matrix = np.zeros((4, 4))
    for i in range(len(inputs)):
        for j in range(len(inputs)):
            if i != j:
                delta = inputs[i] - inputs[j]
                matrix += alpha[i, j] * np.outer(delta, delta)
        delta = positive[i] - inputs[i]
        matrix -= (1 - alpha[i, i]) * np.outer(delta, delta)
    loss, actual = toy.equation2(weight, inputs, positive)
    assert loss == pytest.approx(expected_loss, abs=1e-12)
    np.testing.assert_allclose(actual, matrix / len(inputs), atol=1e-12, rtol=1e-12)


def test_product_and_factor_gradients_match_autograd():
    rng = np.random.default_rng(7)
    inputs, positive = rng.normal(size=(2, 6, 4))
    first, second = rng.normal(size=(2, 4, 4)) * 0.2
    w1 = torch.tensor(first, requires_grad=True)
    w2 = torch.tensor(second, requires_grad=True)
    x, p = torch.tensor(inputs), torch.tensor(positive)
    product = (w2 @ w1).detach().requires_grad_()
    grad = torch.autograd.grad(
        dense_torch_objective(x @ product.T, p @ product.T), product
    )[0]
    _, matrix = toy.equation2(second @ first, inputs, positive)
    expected = -(second @ first) @ matrix
    np.testing.assert_allclose(grad, expected, atol=1e-12, rtol=1e-12)
    grad1, grad2 = torch.autograd.grad(
        dense_torch_objective(x @ w1.T @ w2.T, p @ w1.T @ w2.T), (w1, w2)
    )
    np.testing.assert_allclose(grad1, second.T @ expected, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(grad2, expected @ first.T, atol=1e-12, rtol=1e-12)


def test_torch_reference_gradcheck():
    rng = torch.Generator().manual_seed(27)
    x = torch.randn(4, 3, generator=rng, dtype=torch.float64, requires_grad=True)
    p = torch.randn(4, 3, generator=rng, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(dense_torch_objective, (x, p))


def test_zero_noise_x_is_positive_semidefinite():
    rng = np.random.default_rng(29)
    x = rng.normal(size=(64, 5))
    _, matrix = toy.equation2(rng.normal(size=(5, 5)) * 0.1, x, x)
    assert np.linalg.eigvalsh(matrix)[0] > 0


def test_factorization_gram_invariant_has_zero_first_derivative():
    first, second, x, noise = toy.initial_state(1)
    _, matrix = toy.equation2(second @ first, x, x + 0.001 * noise)
    grad = -(second @ first) @ matrix
    d1, d2 = -second.T @ grad, -grad @ first.T
    derivative = d1 @ first.T + first @ d1.T - d2.T @ second - second.T @ d2
    np.testing.assert_allclose(derivative, 0, atol=1e-14)


def test_simultaneous_update_and_initial_unaligned_factors():
    first, second, x, noise = toy.initial_state(0)
    left, _, _ = np.linalg.svd(first)
    _, _, right = np.linalg.svd(second)
    assert np.diag(np.abs(right @ left)).mean() < 0.5
    _, matrix = toy.equation2(second @ first, x, x + 0.001 * noise)
    grad = -(second @ first) @ matrix
    expected1 = first - second.T @ grad
    expected2 = second - grad @ first.T
    with threadpool_limits(limits=1):
        run = toy.simulate(0, 0.001, 1, horizon=1)
    np.testing.assert_allclose(run["records"][-1]["w1"], expected1, atol=1e-14)
    np.testing.assert_allclose(run["records"][-1]["w2"], expected2, atol=1e-14)
