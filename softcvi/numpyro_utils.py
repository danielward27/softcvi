"""Numpyro utility functions."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import partial

import jax.random as jr
from jax import eval_shape
from jax.tree_util import Partial, tree_map
from jaxtyping import Array
from numpyro import distributions as ndist
from numpyro import handlers
from numpyro.distributions.util import is_identically_one
from numpyro.infer.initialization import init_to_sample
from numpyro.ops.pytree import PytreeTrace


def shape_only_trace(model: Callable, *args, **kwargs):
    """Trace the numpyro model using ``jax.eval_shape``, avoiding array flops.

    Note callables are wrapped to  are also removed from the output (not)

    Args:
       model: The numpyro model.
       args: Arguments passed to model.
       kwargs: Key word arguments passed to model.
    """

    # Adapted from https://github.com/pyro-ppl/numpyro/blob/5af9ebda72bd7aeb08c61e4248ecd0d982473224/numpyro/infer/inspect.py#L39
    def get_trace(fn):
        fn = handlers.substitute(
            handlers.seed(model, 0),
            substitute_fn=init_to_sample,
        )  # Support improper uniform
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


def get_sample_site_names(model: Callable, *args, **kwargs):
    """Infer the names of the sample sites of a model given args and kwargs.

    Args:
        model: Model from which to infer the latents.
        *args: Arguments passed to model.
        **kwargs: Key word arguments passed to the model.

    Returns:
        A dataclass with ``observed``, ``latent`` and ``all`` field/property names.
    """
    trace = shape_only_trace(model, *args, **kwargs)

    observed, latent = set(), set()
    for name, site in trace.items():
        if site["type"] != "sample":
            continue
        if site["is_observed"]:
            observed.add(name)
        else:
            latent.add(name)

    @dataclass
    class _Names:
        observed: set[str]
        latent: set[str]

        @property
        def all(self) -> set[str]:
            return self.observed | self.latent

    return _Names(set(observed), set(latent))


def trace_to_distribution_transforms(trace: dict):
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


def eval_site_log_prob(site: dict):
    """Evaluate the log probability of a site."""
    log_prob_fn = site["fn"].log_prob

    if site["intermediates"]:
        log_prob_fn = partial(log_prob_fn, intermediates=site["intermediates"])

    log_prob = log_prob_fn(site["value"])

    if site["scale"] is not None and not is_identically_one(site["scale"]):
        log_prob = site["scale"] * log_prob

    return log_prob


def trace_to_log_prob(trace: dict, *, reduce: bool = True):
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
    assert_present: Iterable[str] | None = None,
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
        assert_present: An iterable of site names to check are provided in data.
            Defaults to None.
        **kwargs: kwargs passed to model when tracing to infer shapes.
    """
    # TODO allow auxilary sites in guide?
    if assert_present is not None:
        for site in assert_present:
            if site not in data:
                raise ValueError(f"Expected {site} to be provided in data.")

    trace = shape_only_trace(model, *args, **kwargs)
    for name, samples in data.items():
        if name not in trace:
            raise ValueError(f"Got {name} which does not exist in trace.")

        trace_shape = trace[name]["value"].shape

        if trace[name]["type"] == "sample" and trace_shape != data[name].shape:
            raise ValueError(
                f"{name} had shape {trace_shape} in model, but shape "
                f"{samples.shape} in data.",
            )
