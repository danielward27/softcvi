"""Numpyro utility functions."""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from functools import partial

import jax.numpy as jnp
import jax.random as jr
from jax import ShapeDtypeStruct, eval_shape
from jax.tree_util import Partial, tree_map
from jaxtyping import Array
from numpyro import distributions as ndist
from numpyro import handlers
from numpyro.distributions.util import is_identically_one
from numpyro.infer import util
from numpyro.ops.pytree import PytreeTrace


# TODO data first for easier vmap?
# TODO this description I believe is incorrect, I assume it is the joint?
def log_density(model, data, *args, **kwargs):
    """Compute log density of data under the model.

    We check that all sample sites are either observed in the trace or passed in data.
    """
    names = get_sample_site_names(model, *args, **kwargs)

    if data.keys() != names.latent:
        raise ValueError(
            f"Data keys {data.keys()} do not match model latents {names.latent}.",
        )

    validate_data_and_model_match(data, model, *args, **kwargs)
    return util.log_density(model, args, kwargs, params=data)


# TODO data first for more natural vmap?
def prior_log_density(
    model,
    data: dict[str, Array],
    obs_nodes: Iterable[str],
    *args,
    **kwargs,
):
    """Given a model and data, evalutate the prior log probability.

    If the model contains observed sites in the trace, these are not included in the
    prior density. Note however, we expect values to be explicitly passed e.g. for
    sample sites set by handlers.substitute.

    Args:
        model: Numpyro model.
        data: Dictionary of arrays of (unbatched) data.
        obs_nodes: Names of observed variables. These only need to be provided if not
            observed in the model trace.
    """
    names = get_sample_site_names(model, *args, **kwargs)
    latent_names = names.latent - set(obs_nodes)
    if data.keys() != latent_names:
        raise ValueError(
            f"Data keys {data.keys()} does not match model latents {latent_names}.",
        )
    validate_data_and_model_match(data, model, *args, **kwargs)
    model = handlers.substitute(model, data)  # substitute to avoid converting to obs
    model_trace = trace_except_obs(model, obs_nodes, *args, **kwargs)
    return trace_to_log_prob(model_trace)


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


def get_sample_site_names(model, *args, **kwargs):
    """Infer the names of the sample sites of a model given args and kwargs.

    Args:
        model: Model from which to infer the latents.
        *args: Arguments passed to model.
        **kwargs: Key word arguments passed to the model.

    Returns:
        A dataclass with ``observed``, ``latent`` and ``all`` field/property names.
    """
    trace = shape_only_trace(model, *args, **kwargs)

    observed, latent = [], []
    for name, site in trace.items():
        if site["type"] != "sample":
            continue
        if site["is_observed"]:
            observed.append(name)
        else:
            latent.append(name)

    @dataclass
    class _Names:
        observed: set[str]
        latent: set[str]

        @property
        def all(self) -> set[str]:
            return observed + latent

    return _Names(set(observed), set(latent))


def trace_except_obs(model, observed_nodes: Iterable[str], *args, **kwargs):
    """Trace a model, excluding the observed nodes.

    This assumes no nodes are decscendents of the observed nodes (often the case
    if the model describes the assumed data generating process).
    """
    # Just do a shape only trace,
    dummy_obs = {k: ShapeDtypeStruct((), float) for k in observed_nodes}
    model = handlers.condition(model, dummy_obs)
    obs_names = get_sample_site_names(model).observed  # obs in original model or dummy
    model = handlers.block(model, hide=obs_names)
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


def eval_site_log_prob(site):
    """Evaluate the log probability of a site."""
    log_prob_fn = site["fn"].log_prob

    if site["intermediates"]:
        log_prob_fn = partial(log_prob_fn, intermediates=site["intermediates"])

    log_prob = log_prob_fn(site["value"])

    if site["scale"] is not None and not is_identically_one(site["scale"]):
        log_prob = site["scale"] * log_prob

    return log_prob


def trace_to_log_prob(trace, *, reduce=True):
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


def validate_data_and_model_match(
    data: dict[str, Array],
    model: Callable,
    *args,
    **kwargs,
):
    """Validate the data and model match (names and shapes).

    For each site in data, validate that the shapes match what is produced by the
    model. Note, if you have batch dimensions in data, this function must be vectorized,
    e.g. using eqx.filter_vmap.

    Args:
        data: The data.
        model: The model.
        *args: Args passed to model when tracing to infer shapes.
        **kwargs: kwargs passed to model when tracing to infer shapes.
    """
    trace = shape_only_trace(model, *args, **kwargs)
    for name, samples in data.items():
        if name not in trace:  # TODO Check with names function?
            raise ValueError(f"Data with name {name} is not in model trace.")

        trace_shape = trace[name]["value"].shape

        if trace[name]["type"] == "sample" and trace_shape != data[name].shape:
            raise ValueError(
                f"{name} had shape {trace_shape} in model, but shape "
                f"{samples.shape} in data.",
            )
