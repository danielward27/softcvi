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
    validate_data_and_model_match,
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

    # Check conditioned site treated as observed even if not provided in observed_nodes
    cond_model = handlers.condition(model, {"y": jnp.ones(5)})
    log_prob = prior_log_density(cond_model, data=prior_samp, observed_nodes={})
    assert pytest.approx(expected) == log_prob

    # Check errors if name in observed nodes in samples
    prior_samp["y"] = jnp.ones(5)
    with pytest.raises(
        ValueError,
        match="does not match model latents",
    ):
        prior_log_density(model, data=prior_samp, observed_nodes={"y"})


def test_trace_to_log_prob():
    obs = jnp.arange(5)
    trace = handlers.trace(handlers.seed(model, jr.PRNGKey(0))).get_trace(obs=obs)
    log_probs = trace_to_log_prob(trace, reduce=False)

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
    assert names.observed == set()
    assert names.latent == {"x", "y"}
    assert names.all == {"x", "y"}

    names = get_sample_site_names(model, obs=jnp.array(0))
    assert names.observed == {"y"}
    assert names.latent == {"x"}
    assert names.all == {"x", "y"}


def test_validate_data_and_model_match():

    assert (
        validate_data_and_model_match(
            data={"x": jnp.ones(5), "y": jnp.ones(5)},
            model=model,
        )
        is None
    )  # Doesn't raise

    with pytest.raises(ValueError, match="not in model"):
        validate_data_and_model_match({"z": jnp.ones(5), "x": jnp.ones(5)}, model)

    with pytest.raises(ValueError, match="shape"):
        validate_data_and_model_match({"x": jnp.ones(5), "y": jnp.ones(())}, model)
