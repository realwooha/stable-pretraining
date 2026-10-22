"""Numerical contracts for the independent CPU mechanism experiments."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.unit
SCRIPT = Path(__file__).resolve().parents[3] / "benchmarks/toy/directclr_mechanisms.py"
spec = importlib.util.spec_from_file_location("directclr_mechanisms", SCRIPT)
toy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(toy)


def test_objective_matches_independent_dense_probability():
    rng = np.random.default_rng(14)
    first, second = rng.normal(size=(2, 4, 3)) * 0.1
    scores = first @ second.T / 0.1
    probabilities = np.exp(scores) / np.exp(scores).sum(axis=1, keepdims=True)
    expected = -np.log(np.diag(probabilities)).mean()
    actual = toy.objective(torch.tensor(first), torch.tensor(second))
    assert actual.item() == pytest.approx(expected, rel=1e-12)


def test_objective_gradcheck():
    rng = torch.Generator().manual_seed(10)
    first = (
        torch.randn(3, 4, generator=rng, dtype=torch.float64) * 0.1
    ).requires_grad_()
    second = (
        torch.randn(3, 4, generator=rng, dtype=torch.float64) * 0.1
    ).requires_grad_()
    assert torch.autograd.gradcheck(toy.objective, (first, second))


def test_objective_keeps_magnitude_information():
    first = torch.eye(3, dtype=torch.float64) * 0.1
    assert toy.objective(first, first) != toy.objective(first * 2, first * 2)


def test_augmentation_only_perturbs_registered_coordinates():
    data = torch.zeros(128, 16)
    first, second = toy.views(
        data, 1.5, "augmentation", torch.Generator().manual_seed(0)
    )
    assert torch.count_nonzero(first[:, :8]) == 0
    assert torch.count_nonzero(second[:, :8]) == 0
    assert torch.count_nonzero(first[:, 8:]) == 128 * 8
    assert not torch.equal(first[:, 8:], second[:, 8:])


def test_implicit_controls_have_matched_initial_products():
    for seed in (0, 1, 2):
        single = toy.make_model(seed, "implicit", 1)
        double = toy.make_model(seed, "implicit", 2)
        torch.testing.assert_close(
            single[0].weight, double[1].weight @ double[0].weight, rtol=0, atol=0
        )
        assert all(layer.bias is None for layer in double)


def test_diagnostics_match_independent_covariance_and_do_not_consume_rng():
    model = toy.make_model(0, "implicit", 2)
    probe = torch.randn(1000, 16, generator=torch.Generator().manual_seed(11))
    before = torch.random.get_rng_state().clone()
    result = toy.diagnostics(model, probe)
    assert torch.equal(before, torch.random.get_rng_state())
    product = model[1].weight.detach().double() @ model[0].weight.detach().double()
    embedding = probe.double().numpy() @ product.numpy().T
    covariance = np.cov(embedding, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
    np.testing.assert_allclose(
        result["covariance_eigenvalues"], eigenvalues, rtol=1e-8, atol=1e-15
    )
    singular = np.linalg.svd(embedding - embedding.mean(0), compute_uv=False)
    probability = singular / singular.sum()
    assert result["effective_rank"] == pytest.approx(
        np.exp(-(probability * np.log(probability)).sum())
    )


def test_factorized_chain_rule_matches_product_gradient():
    rng = torch.Generator().manual_seed(19)
    first = torch.randn(5, 4, generator=rng, dtype=torch.float64) * 0.1
    second = torch.randn(5, 4, generator=rng, dtype=torch.float64) * 0.1
    w1 = torch.randn(4, 4, generator=rng, dtype=torch.float64, requires_grad=True)
    w2 = torch.randn(4, 4, generator=rng, dtype=torch.float64, requires_grad=True)
    product = (w2 @ w1).detach().requires_grad_()
    product_gradient = torch.autograd.grad(
        toy.objective(first @ product.T, second @ product.T), product
    )[0]
    grad1, grad2 = torch.autograd.grad(
        toy.objective(first @ w1.T @ w2.T, second @ w1.T @ w2.T), (w1, w2)
    )
    torch.testing.assert_close(grad1, w2.T @ product_gradient)
    torch.testing.assert_close(grad2, product_gradient @ w1.T)


def test_short_run_is_reproducible_with_finite_scheduled_records():
    first = toy.simulate(0, "implicit", "two-layer", 3)
    second = toy.simulate(0, "implicit", "two-layer", 3)
    assert first["records"] == second["records"]
    assert first["records"][-1]["update"] == 3
    assert np.isfinite(first["records"][-1]["loss"])
