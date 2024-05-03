"""Utility classes representing numpyro models and guides."""

import jax.numpy as jnp
import pytest
from flowjax.bijections import Affine, Exp
from flowjax.distributions import Laplace, LogNormal, Normal, Transformed
from flowjax.experimental.numpyro import sample
from numpyro import plate

from cnpe.models import AbstractNumpyroGuide, AbstractNumpyroModel


@pytest.fixture()
def model():
    class Model(AbstractNumpyroModel):
        reparam_names = {"a", "b"}
        observed_names = {"c"}

        def call_without_reparam(self, obs=None):
            a = sample("a", Normal(1, 2))
            b = sample("b", Laplace(a, a))
            sample("c", Laplace(b, b), obs=obs)

    return Model()


@pytest.fixture()
def guide():
    class Guide(AbstractNumpyroGuide):
        a_base: Normal = Normal(3, 4)
        b_base: Laplace = Laplace(3, 4)

        def __call__(self):
            sample("a_base", self.a_base)
            sample("b_base", self.b_base)

    return Guide()


def test_guide_log_prob_original_space(model, guide):
    data = {"a": jnp.array(1), "b": jnp.array(2)}
    a_log_prob = Transformed(guide.a_base, Affine(1, 2)).log_prob(data["a"])
    b_log_prob = Transformed(guide.b_base, Affine(data["a"], data["a"])).log_prob(
        data["b"],
    )
    log_prob = guide.log_prob_original_space(data, model)
    expected = a_log_prob + b_log_prob
    assert pytest.approx(expected) == log_prob


@pytest.fixture()
def plate_model():
    class PlateModel(AbstractNumpyroModel):
        reparam_names = {"scale", "theta"}
        observed_names = {"x"}

        def call_without_reparam(self, obs=None):
            scale = sample("scale", LogNormal())
            with plate("dim", 2):
                b = sample("theta", Normal(0, scale))

            sample("x", Normal(b, scale))

    return PlateModel()


@pytest.fixture()
def plate_guide():
    class PlateGuide(AbstractNumpyroGuide):
        scale_base: Laplace = Laplace(1)
        theta_base: Laplace = Laplace(1)

        def __call__(self):
            sample("scale_base", self.scale_base)
            with plate("dim", 2):
                sample("theta_base", self.theta_base)

    return PlateGuide()


def test_guide_log_prob_original_space_plate(plate_model, plate_guide):
    data = {"scale": jnp.array(2), "theta": jnp.array([2, 3])}

    expected = [
        Transformed(plate_guide.scale_base, Exp()).log_prob(data["scale"]),
        Transformed(plate_guide.theta_base, Affine(0, data["scale"])).log_prob(
            data["theta"],
        ),
    ]
    expected = sum(arr.sum() for arr in expected)
    realized = plate_guide.log_prob_original_space(data, plate_model)
    assert pytest.approx(expected) == realized
