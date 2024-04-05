import jax.numpy as jnp
import jax.random as jr
import numpyro
import pytest
from numpyro import handlers, sample
from numpyro.distributions import Normal

from cnpe.numpyro_utils import prior_log_density, trace_to_log_prob


def model(obs=None):
    with numpyro.plate("plate", 5):
        x = sample("x", Normal(0, 1))
        sample("y", Normal(x, 1), obs=obs)


def test_prior_log_density():
    prior_samp = {"x": jnp.arange(5)}
    expected = Normal().log_prob(prior_samp["x"]).sum()
    log_prob = prior_log_density(model, data=prior_samp, observed_nodes=["y"])
    assert pytest.approx(expected) == log_prob


def test_trace_to_log_prob():
    obs = jnp.arange(5)
    trace = handlers.trace(handlers.seed(model, jr.PRNGKey(0))).get_trace(
        obs=obs,
    )
    log_probs = trace_to_log_prob(trace)

    assert log_probs["x"].shape == (5,)
    assert log_probs["y"].shape == (5,)
    assert pytest.approx(trace["y"]["value"]) == obs
    assert pytest.approx(log_probs["x"]) == Normal().log_prob(trace["x"]["value"])
    assert pytest.approx(log_probs["y"]) == Normal().log_prob(obs - trace["x"]["value"])
