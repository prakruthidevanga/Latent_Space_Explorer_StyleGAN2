import pytest
import torch

from latent_utils import create_latent_from_seed, interpolate_latents, interpolation_parameters
from diagnostics import calculate_diagnostics


def test_seed_is_reproducible():
    first = create_latent_from_seed(17, 8)
    second = create_latent_from_seed(17, 8)
    assert torch.equal(first, second)


def test_different_seeds_normally_differ():
    assert not torch.equal(create_latent_from_seed(17, 8), create_latent_from_seed(18, 8))


def test_interpolation_endpoints_and_midpoint():
    a = torch.tensor([[0.0, 2.0]])
    b = torch.tensor([[4.0, 6.0]])
    assert torch.equal(interpolate_latents(a, b, 0.0), a)
    assert torch.equal(interpolate_latents(a, b, 1.0), b)
    assert torch.allclose(interpolate_latents(a, b, 0.5), torch.tensor([[2.0, 4.0]]))


@pytest.mark.parametrize("t", [-0.01, 1.01, float("nan"), float("inf")])
def test_invalid_t_is_rejected(t):
    with pytest.raises(ValueError):
        interpolate_latents(torch.zeros((1, 3)), torch.ones((1, 3)), t)


def test_latent_shape_mismatch_is_rejected():
    with pytest.raises(ValueError):
        interpolate_latents(torch.zeros((1, 3)), torch.ones((1, 4)), 0.5)


def test_invalid_seed_and_dimension_are_rejected():
    with pytest.raises(ValueError):
        create_latent_from_seed(-1, 8)
    with pytest.raises(ValueError):
        create_latent_from_seed(1, 0)


def test_interpolation_parameters_include_both_endpoints():
    values = interpolation_parameters(5)
    assert len(values) == 5
    assert float(values[0]) == pytest.approx(0.0)
    assert float(values[-1]) == pytest.approx(1.0)
    assert list(values) == sorted(values.tolist())


def test_interpolation_step_count_validation():
    with pytest.raises(ValueError):
        interpolation_parameters(1)
    with pytest.raises(ValueError):
        interpolation_parameters(100)


def test_distances_follow_linear_path():
    a = torch.tensor([[0.0, 0.0]])
    b = torch.tensor([[4.0, 0.0]])
    mid = interpolate_latents(a, b, 0.5)
    metrics = calculate_diagnostics(a, b, mid)
    assert metrics["endpoint_distance"] == pytest.approx(4.0)
    assert metrics["distance_from_start"] == pytest.approx(2.0)
    assert metrics["distance_from_end"] == pytest.approx(2.0)
