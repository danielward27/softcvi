"""Numpyro utility functions."""

from collections.abc import Callable, Iterable, Sequence
from functools import partial

import jax.numpy as jnp
import jax.random as jr
from jax import eval_shape
from jax.tree_util import Partial, tree_map
from numpyro import distributions as ndist
from numpyro import handlers
from numpyro.distributions.util import is_identically_one
from numpyro.infer import util
from numpyro.ops.pytree import PytreeTrace


# TODO this description I believe is incorrect, I assume it is the joint?
def log_density(model, data, *args, **kwargs):
    """Compute log density of data under the model.

    We assert that all sample sites are either observed in the trace
    We assert that all latents for the model are passed in data.
    """
    names = get_sample_site_names(model, *args, **kwargs)

    if (d_names := sorted(data.keys())) != (l_names := sorted(names["latent"])):
        raise ValueError(f"Data keys {d_names} does not match model latents {l_names}.")

    return util.log_density(model, args, kwargs, params=data)


def shape_only_trace(model: Callable, *args, **kwargs):
    """Trace the numpyro model using ``jax.eval_shape``, avoiding array flops.

    Note callables are wrapped to  are also removed from the output (not)

    Args:
       model: The numpyro model.
       args: Arguments passed to model.
       kwargs: Key word arguments passed to model.
    """

    def get_trace(fn):
        fn = handlers.seed(fn, jr.PRNGKey(0))
        trace = handlers.trace(fn).get_trace(*args, **kwargs)

        # We wrap all callables to ensure return value are all valid jax types
        trace = tree_map(
            lambda leaf: Partial(leaf) if callable(leaf) else leaf,
            tree=trace,
            is_leaf=callable,
        )
        return PytreeTrace(trace)

    trace = eval_shape(lambda: get_trace(model)).trace

    # unwrap the result
    return tree_map(
        lambda leaf: leaf.func if isinstance(leaf, Partial) else leaf,
        tree=trace,
        is_leaf=lambda leaf: isinstance(leaf, Partial),
    )


def get_sample_site_names(model, *args, **kwargs) -> dict[str, list[str]]:
    """Infer the names of the latents of a model given args and kwargs.

    Args:
        model: Model from which to infer the latents.
        *args: Arguments passed to model.
        obs_nodes: Observed nodes to exclude from result (if they are observed in the
            trace, the observed node will be excluded autmatically).

    Returns:
        A dictionary of lists with keys "observed", "latents".
    """
    trace = shape_only_trace(model, *args, **kwargs)

    result = {"observed": [], "latent": []}
    for k, v in trace.items():
        if v["type"] != "sample":
            continue
        if v["is_observed"]:
            result["observed"].append(k)
        else:
            result["latent"].append(k)
    return result


def trace_except_obs(model, observed_nodes: Iterable[str], *args, **kwargs):
    """Trace a model, excluding the observed nodes.

    This assumes no nodes are decscendents of the observed nodes (often the case
    if the model describes the assumed data generating process).
    """
    # TODO Provide dummy to obs to avoid sampling - is there a better way?
    data = {k: jnp.empty(()) for k in observed_nodes}
    model = handlers.condition(model, data)
    model = handlers.block(model, hide=observed_nodes)
    return handlers.trace(model).get_trace(*args, **kwargs)


def trace_to_distribution_transforms(trace):
    """Get the numpyro transforms any transformed distributions in trace.

    Note if TransformReparam is used for a distribution, then the transform will not be
    extracted from the trace as the transform is instead treated as a deterministic
    function.
    """
    transforms = {}
    for k, site in trace.items():

        if site["type"] == "sample":
            dist = site["fn"]

            while isinstance(dist, ndist.ExpandedDistribution | ndist.Independent):
                dist = dist.base_dist

            if isinstance(dist, ndist.TransformedDistribution):
                transforms[k] = dist.transforms

    return transforms


def prior_log_density(
    model,
    data: dict,
    observed_nodes: Sequence[str],
    *args,
    **kwargs,
):
    """Given a model and data, evalutate the prior log probability."""
    latent_names = get_sample_site_names(model, *args, **kwargs)["latent"]
    latent_names = [name for name in latent_names if name not in observed_nodes]
    if (d_names := sorted(data.keys())) != (l_names := sorted(latent_names)):
        raise ValueError(f"Data keys {d_names} does not match model latents {l_names}.")

    model = handlers.condition(model, data)
    model_trace = trace_except_obs(model, observed_nodes, *args, **kwargs)
    log_prob = jnp.zeros(())
    for site in model_trace.values():
        if site["type"] == "sample":
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