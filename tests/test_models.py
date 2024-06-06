"""Utility classes representing numpyro models and guides."""

import jax.numpy as jnp
import jax.random as jr
import pytest
from flowjax.bijections import Affine, Exp
from flowjax.distributions import Laplace, LogNormal, Normal, Transformed
from flowjax.experimental.numpyro import sample
from numpyro import plate
from softce.models import AbstractGuide, AbstractModel


def simple_model_and_guide():
    class Model(AbstractModel):
        reparameterized: bool | None = None
        reparam_names = {"a", "b"}
        observed_names = {"c"}

        def call_without_reparam(self, obs=None):
            a = sample("a", Normal(1, 2))
            b = sample("b", Laplace(a, jnp.exp(a)))
            sample("c", Laplace(b, b), obs=obs)

    class Guide(AbstractGuide):
        a_base: Normal = Normal(3, 4)
        b_base: Laplace = Laplace(3, 4)

        def __call__(self):
            sample("a_base", self.a_base)
            sample("b_base", self.b_base)

    return Model(), Guide()


def plate_model_and_guide():
    class PlateModel(AbstractModel):
        reparameterized: bool | None = None
        reparam_names = {"scale", "theta"}
        observed_names = {"x"}

        def call_without_reparam(self, obs=None):
            scale = sample("scale", LogNormal())
            with plate("dim", 2):
                b = sample("theta", Normal(0, scale))

            sample("x", Normal(b, scale))

    class PlateGuide(AbstractGuide):
        scale_base: Laplace = Laplace(1)
        theta_base: Laplace = Laplace(1)

        def __call__(self):
            sample("scale_base", self.scale_base)
            with plate("dim", 2):
                sample("theta_base", self.theta_base)

    return PlateModel(), PlateGuide()


def test_log_prob_original_space():
    # Simple example
    model, guide = simple_model_and_guide()
    data = {"a": jnp.array(1), "b": jnp.array(2)}
    a_log_prob = Transformed(guide.a_base, Affine(1, 2)).log_prob(data["a"])
    b_log_prob = Transformed(
        guide.b_base,
        Affine(data["a"], jnp.exp(data["a"])),
    ).log_prob(
        data["b"],
    )
    log_prob = guide.log_prob_original_space(data, model)
    expected = a_log_prob + b_log_prob
    assert pytest.approx(expected) == log_prob

    # Test example with plate
    model, guide = plate_model_and_guide()
    data = {"scale": jnp.array(2), "theta": jnp.array([2, 3])}

    expected = [
        Transformed(guide.scale_base, Exp()).log_prob(data["scale"]),
        Transformed(guide.theta_base, Affine(0, data["scale"])).log_prob(
            data["theta"],
        ),
    ]
    expected = sum(arr.sum() for arr in expected)
    realized = guide.log_prob_original_space(data, model)
    assert pytest.approx(expected) == realized


def test_prior_log_density():
    model, _ = simple_model_and_guide()

    # Test in original space
    prior_samp = {"a": jnp.array(1), "b": jnp.array(2)}
    expected = sum(
        [
            Normal(1, 2).log_prob(prior_samp["a"]),
            Laplace(prior_samp["a"], jnp.exp(prior_samp["a"])).log_prob(
                prior_samp["b"],
            ),
        ],
    )
    log_prob = model.reparam(set_val=False).prior.log_prob(prior_samp)
    assert pytest.approx(expected) == log_prob

    # Test in reparameterized space
    prior_samp = {"a_base": jnp.array(1), "b_base": jnp.array(2)}

    expected = sum(
        [
            Normal().log_prob(prior_samp["a_base"]),
            Laplace().log_prob(prior_samp["b_base"]),
        ],
    )
    log_prob = model.reparam().prior.log_prob(prior_samp)
    assert pytest.approx(expected) == log_prob

    model, _ = plate_model_and_guide()
    # Test in original space
    prior_samp = {"scale": jnp.array(2), "theta": jnp.array([3, 3])}

    expected = sum(
        [
            LogNormal().log_prob(prior_samp["scale"]),
            Normal(0, prior_samp["scale"]).log_prob(prior_samp["theta"]).sum(),
        ],
    )
    log_prob = model.reparam(set_val=False).prior.log_prob(prior_samp)
    assert pytest.approx(expected) == log_prob

    prior_samp = {"scale_base": jnp.array(2), "theta_base": jnp.array([3, 3])}

    expected = sum(
        [
            Normal().log_prob(prior_samp["scale_base"]),
            Normal().log_prob(prior_samp["theta_base"]).sum(),
        ],
    )
    log_prob = model.reparam().prior.log_prob(prior_samp)
    assert pytest.approx(expected) == log_prob


def test_latents_to_original_space():
    model, _ = simple_model_and_guide()
    latents = model.reparam(set_val=True).prior.sample(jr.PRNGKey(0))
    original_space_1 = model.reparam(set_val=False).prior.sample(jr.PRNGKey(0))
    original_space_2 = model.latents_to_original_space(latents)
    assert original_space_1 == original_space_2
