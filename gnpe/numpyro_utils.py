from collections.abc import Iterable
from functools import partial

import jax.numpy as jnp
from numpyro import handlers
from numpyro.distributions.util import is_identically_one
from numpyro.infer import util


def log_density(model, data, *args, **kwargs):
    """numpyro.infer.util.log_density, with (arguably) a better signature."""
    return util.log_density(model, args, kwargs, params=data)


def prior_log_density(
    model,
    data: dict,
    observed_nodes: Iterable[str],
):
    """Given a model and data, evalutate the prior log probability."""
    # To skip sampling observed nodes we provide dummy samples.
    # We could use block, but that does not generally avoid sampling the observed node.
    data = data | {name: jnp.empty(()) for name in observed_nodes}
    model = handlers.condition(model, data)
    model_trace = handlers.trace(model).get_trace()

    log_prob = jnp.zeros(())
    for site in model_trace.values():
        if site["type"] == "sample" and site["name"] not in observed_nodes:
            log_prob += eval_site_log_prob(site).sum()
    return log_prob


def eval_site_log_prob(site):
    """Evaluate the log probability of a site."""
    log_prob_fn = site["fn"].log_prob

    if site["intermediates"]:
        log_prob_fn = partial(log_prob_fn, intermediates=site["intermediates"])

    log_prob = log_prob_fn(site["value"])

    if site["scale"] is not None and not is_identically_one(site["scale"]):
        log_prob = site["scale"] * log_prob

    return log_prob


def trace_to_log_prob(trace, *, reduce=False):
    """Computes dictionary of log probabilities, or a scalar value for a trace.

    Args:
        trace: The numpyro trace.
        reduce: Whether to reduce the result to a scalar value.
    """
    log_prob = {
        k: eval_site_log_prob(site)
        for k, site in trace.items()
        if site["type"] == "sample"
    }
    if reduce:
        log_prob = sum(v.sum() for v in log_prob.values())
    return log_prob
