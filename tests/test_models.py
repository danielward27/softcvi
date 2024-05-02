"""Utility classes representing numpyro models and guides."""

import jax.numpy as jnp
import pytest
from flowjax.bijections import Affine
from flowjax.distributions import Laplace, Normal, Transformed
from flowjax.experimental.numpyro import sample

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

    # TODO test with a more complex example (e.g with plates)

    # TODO add a check somewhere for asserting same latent names in model and guide.
