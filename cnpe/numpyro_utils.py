"""Numpyro utility functions."""

from collections.abc import Iterable
from functools import partial

import jax.numpy as jnp
from numpyro import distributions as ndist
from numpyro import handlers
from numpyro.distributions.util import is_identically_one
from numpyro.infer import util
from numpyro.infer.reparam import Reparam


def log_density(model, data, *args, **kwargs):
    """numpyro.infer.util.log_density, with (arguably) a nicer signature."""
    return util.log_density(model, args, kwargs, params=data)


def trace_except_obs(model, observed_nodes: Iterable[str]):
    """Trace a model, excluding the observed nodes."""
    # Provide dummy to obs to avoid sampling
    data = {k: jnp.empty(()) for k in observed_nodes}
    model = handlers.condition(model, data)
    model = handlers.block(model, hide=observed_nodes)
    return handlers.trace(model).get_trace()


def trace_to_distribution_transforms(trace):
    """Get the numpyro transforms from the transformed distributions in trace.

    Note if TransformReparam is used for a distribution, then the transform will not be
    extracted from the trace as the transform is instead treated as a deterministic
    function.
    """
    transforms = {}
    for k, site in trace.items():
        dist = site["fn"]

        while isinstance(dist, ndist.ExpandedDistribution | ndist.Independent):
            dist = dist.base_dist

        if isinstance(dist, ndist.TransformedDistribution):
            transforms[k] = dist.transforms

    return transforms


class ApplyTransformReparam(Reparam):
    """Apply a transform to reparameterize.

    Note this is different to TransformReparam in numpyro, which will reparameterize by
    avoiding applying a transformation in a transformed distribution. In contrast, this
    reparameterizes by applying a transform (which will impact the underlying model).

    Args:
        transform: numpyro transform.
    """

    def __init__(
        self,
        transform: ndist.transforms.Transform,
    ):
        self.transform = transform

    def __call__(self, name, fn, obs):
        assert obs is None, "ApplyTransformReparam does not support observe statements"
        fn, expand_shape, event_dim = self._unwrap(fn)
        fn = ndist.TransformedDistribution(fn, self.transform)
        return self._wrap(fn, expand_shape, event_dim), None


def prior_log_density(
    model,
    data: dict,
    observed_nodes: Iterable[str],
):
    """Given a model and data, evalutate the prior log probability."""
    model = handlers.condition(model, data)
    model_trace = trace_except_obs(model, observed_nodes)
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
