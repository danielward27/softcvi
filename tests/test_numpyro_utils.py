import jax.numpy as jnp
import jax.random as jr
import numpyro
import pytest
from flowjax.distributions import Normal
from flowjax.experimental.numpyro import sample
from numpyro import handlers

from cnpe.numpyro_utils import (
    ApplyTransformReparam,
    prior_log_density,
    trace_to_distribution_transforms,
    trace_to_log_prob,
)


def model(obs=None):
    with numpyro.plate("plate", 5):
        x = sample("x", numpyro.distributions.Normal())
        sample("y", numpyro.distributions.Normal(x), obs=obs)


def test_prior_log_density():
    prior_samp = {"x": jnp.arange(5)}
    expected = Normal().log_prob(prior_samp["x"]).sum()
    log_prob = prior_log_density(model, data=prior_samp, observed_nodes=["y"])
    assert pytest.approx(expected) == log_prob


def test_trace_to_log_prob():
    obs = jnp.arange(5)
    trace = handlers.trace(handlers.seed(model, jr.PRNGKey(0))).get_trace(obs=obs)
    log_probs = trace_to_log_prob(trace)

    assert log_probs["x"].shape == (5,)
    assert log_probs["y"].shape == (5,)
    assert pytest.approx(trace["y"]["value"]) == obs
    assert pytest.approx(log_probs["x"]) == Normal().log_prob(trace["x"]["value"])
    assert pytest.approx(log_probs["y"]) == Normal().log_prob(obs - trace["x"]["value"])


def test_trace_to_distribution_transforms():

    def model(obs=None):

        with numpyro.plate("plate", 5):
            x = sample("x", Normal())

        sample("y", Normal(x), obs=obs)

    data = {"x": jnp.arange(5)}
    trace = handlers.trace(handlers.condition(model, data)).get_trace(obs=jnp.zeros(5))
    transforms = trace_to_distribution_transforms(trace)
    assert pytest.approx(transforms["x"][0](jnp.zeros(5))) == jnp.zeros(5)
    assert pytest.approx(transforms["y"][0](jnp.zeros(5))) == data["x"]

    # Ensuring nested plates does't lead to nested ExpandedDistributions
    def nested_plate_model():
        with numpyro.plate("plate1", 2):
            with numpyro.plate("plate1", 3):
                sample("x", Normal(0, 2))

    trace = handlers.trace(handlers.condition(nested_plate_model, {"x": 1})).get_trace()
    transforms = trace_to_distribution_transforms(trace)
    assert transforms["x"][0](1) == 2


# To get the guide on the original model space we do a three step procedure.
# 1) get samples on original space (e.g. by tracing the conditioned model).
# 2) Get the transforms used in transform reparam from the original model
# 3) Apply these transforms to the guide


def test_ApplyTransformReparam():
    # Check we can change model_reparam, "back" into unreparameterized model_original

    loc, scale = 3, 4
    affine = numpyro.distributions.transforms.AffineTransform(loc, scale)
    reparam_config = {"x": ApplyTransformReparam(affine)}

    def model_original():
        with numpyro.plate("a", 5):
            sample("x", numpyro.distributions.Normal(loc, scale))

    def model_reparam():
        with numpyro.plate("a", 5):
            sample("x", numpyro.distributions.Normal())

    models = {
        "original": model_original,
        "reparam": handlers.reparam(model_reparam, config=reparam_config),
    }

    def _get_samp_and_log_prob(model):
        model = handlers.seed(model, jr.PRNGKey(0))
        trace = handlers.trace(model).get_trace()
        samp = trace["x"]["value"]
        log_prob = trace["x"]["fn"].log_prob(trace["x"]["value"])
        return {"sample": samp, "log_prob": log_prob}

    samp_and_log_probs = {
        k: _get_samp_and_log_prob(model) for k, model in models.items()
    }

    for key in ["sample", "log_prob"]:
        assert (
            pytest.approx(samp_and_log_probs["original"][key])
            == samp_and_log_probs["reparam"][key]
        )
