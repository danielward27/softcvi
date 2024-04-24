import jax.numpy as jnp
import jax.random as jr
import numpyro
import pytest
from flowjax.distributions import Normal
from flowjax.experimental.numpyro import sample
from numpyro import handlers

from cnpe.numpyro_utils import (
    get_sample_site_names,
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

    def nested_plate_model():
        with numpyro.plate("plate1", 2):
            with numpyro.plate("plate1", 3):
                sample("x", Normal(0, 2))

    trace = handlers.trace(handlers.condition(nested_plate_model, {"x": 1})).get_trace()
    transforms = trace_to_distribution_transforms(trace)
    assert transforms["x"][0](1) == 2


def test_get_sample_site_names():
    names = get_sample_site_names(model)
    assert names["latent"] == ["x", "y"]
    assert names["observed"] == []

    names = get_sample_site_names(model, obs=jnp.array(0))
    assert names["observed"] == ["y"]
    assert names["latent"] == ["x"]
