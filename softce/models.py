"""Abstract model and guide class containing reparameterization logic.

Numpyro can be a bit clunky to work with, these classes provides some convenient methods
for sampling, log density evaluation and reparameterization. In general, all methods
should work in the single sample case, and require explicit vectorization otherwise,
for example using ``equinox.filter_vmap`` or ``jax.vmap``.
"""

from abc import abstractmethod
from collections.abc import Callable, Iterable

import equinox as eqx
from jax import ShapeDtypeStruct, tree_util
from jaxtyping import Array, PRNGKeyArray
from numpyro import handlers
from numpyro.distributions.transforms import ComposeTransform
from numpyro.infer import reparam
from softce.numpyro_utils import (
    get_sample_site_names,
    trace_except_obs,
    trace_to_distribution_transforms,
    trace_to_log_prob,
    validate_data_and_model_match,
)

# TODO avoid *args, **kwargs?


class AbstractModelOrGuide(eqx.Module):
    """Abstract class representing shared model and guide components."""

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    @property
    def site_names(self) -> set:
        return get_sample_site_names(self).all

    def sample(self, key: PRNGKeyArray) -> dict[str, Array]:
        """Sample the joint distribution, returning a tuple, (latents, observed)."""
        trace = handlers.trace(handlers.seed(self, key)).get_trace()
        return {k: v["value"] for k, v in trace.items() if v["type"] == "sample"}

    def log_prob(self, data: dict[str, Array], *, reduce: bool = True):
        """The joint probability under the model.

        Data should include all nodes (latents and observed).
        """
        validate_data_and_model_match(data, self, assert_present=self.site_names)
        trace = handlers.trace(handlers.substitute(self, data)).get_trace()
        return trace_to_log_prob(trace, reduce=reduce)


class AbstractModel(AbstractModelOrGuide):
    """Abstract class used for numpyro models.

    The reparameterized flag changes whether the original, or reparameterized model
    is used.

    Attributes:
        observed_names: names for the observed nodes.
        reparam_names: tuple of latent names to which TransformReparam is applied.
        reparameterized: A flag denoting whether to use the reparameterized model, or
            the model on the original space. None is used to represent that it must be
            explicitly set before the model is called.
    """

    observed_names: eqx.AbstractVar[set[str] | frozenset[str]]
    reparam_names: eqx.AbstractVar[set[str] | frozenset[str]]
    reparameterized: eqx.AbstractVar[bool | None]

    @abstractmethod
    def call_without_reparam(self, *args, **kwargs):
        """An implentation of the numpyro model, without applying reparameterization.

        Generally, do not directly use this method, instead calling the class directly,
        using model.reparam(set_val=False) if required.
        """
        pass

    def __call__(self, *args, **kwargs):
        """The numpyro model, applying reparameterizations."""
        if self.reparameterized is None:
            raise ValueError(
                "Reparameterized flag was None. Set to True/False using model.reparam.",
            )
        if self.reparameterized:
            config = {name: reparam.TransformReparam() for name in self.reparam_names}
            with handlers.reparam(config=config):
                self.call_without_reparam(*args, **kwargs)
        else:
            self.call_without_reparam(*args, **kwargs)

    @property
    def latent_names(self):
        return self.site_names - self.observed_names

    def reparam(self, *, set_val: bool | None = True):
        """Returns a copy of the model, with the reparameterized flag changed."""
        return eqx.tree_at(
            where=lambda model: model.reparameterized,
            pytree=self,
            replace=set_val,
            is_leaf=lambda leaf: leaf is None,
        )

    def get_reparam_transforms(self, latents: dict[str, Array], *args, **kwargs):
        """Infer the deterministic transforms applied under reparameterization.

        Note this only applies for TransformReparam (not other deterministic sites).

        Args:
            latents: latent variables from the data space (not the base space).
            *args: Positional arguments for model.
            **kwargs: Key word arguments for model.
        """
        model = self.reparam(set_val=False)
        validate_data_and_model_match(latents, model, assert_present=model.latent_names)
        model_trace = trace_except_obs(
            handlers.substitute(model, latents),
            self.observed_names,
            *args,
            **kwargs,
        )
        transforms = trace_to_distribution_transforms(model_trace)
        return {k: t for k, t in transforms.items() if k in self.reparam_names}

    def sample_predictive(
        self,
        key: PRNGKeyArray,
        latents: dict[str, Array],
    ):
        """Sample a single sample from the predictive/likelihood distribution.

        To generate mutiple samples, use e.g. jax.vmap over a set of keys.
        """
        validate_data_and_model_match(latents, self, assert_present=self.latent_names)
        predictive = handlers.trace(
            handlers.seed(
                handlers.condition(self, data=latents),
                key,
            ),
        ).get_trace()
        return {name: predictive[name]["value"] for name in self.observed_names}

    def log_likelihood(
        self,
        latents: dict[str, Array],
        obs: dict[str, Array],
        *,
        reduce: bool = True,
    ):
        validate_data_and_model_match(latents, self, assert_present=self.latent_names)
        validate_data_and_model_match(obs, self, assert_present=self.observed_names)
        trace = handlers.trace(handlers.substitute(self, latents)).get_trace(obs=obs)
        obs_trace = {k: v for k, v in trace.items() if k in self.observed_names}
        return trace_to_log_prob(obs_trace, reduce=reduce)

    def latents_to_original_space(
        self,
        latents: dict[str, Array],
        *args,
        **kwargs,
    ) -> dict[str, Array]:
        """Convert a set of latents from the reparameterized space to original space.

        Args:
            latents: The set of latents from the reparameterized space.
            *args: Positional arguments passed when tracing.
            **kwargs: Key word arguments passed when tracing.
        """
        # TODO Again we assume we can trace except obs reliably
        latents = {k: v for k, v in latents.items()}  # Avoid mutating
        model = self.reparam(set_val=True)
        validate_data_and_model_match(latents, model, assert_present=model.latent_names)
        model = handlers.condition(model, latents)
        trace = trace_except_obs(model, self.observed_names, *args, **kwargs)

        for name in self.reparam_names:
            latents.pop(f"{name}_base")
            latents[name] = trace[name]["value"]

        return latents

    @property
    def prior(self):
        dummy_obs = {k: ShapeDtypeStruct((), float) for k in self.observed_names}
        model = handlers.condition(self, dummy_obs)  # To avoid sampling observed nodes
        return NumpyroModelToModel(
            handlers.block(model, hide=self.observed_names),
            observed_names=frozenset({}),
            reparam_names=self.reparam_names,
            reparameterized=self.reparameterized,
        )


class NumpyroModelToModel(AbstractModel):
    # Wrapper to wrap a standard numpyro model to a numpyro guide
    model: Callable
    observed_names: frozenset[str]
    reparam_names: frozenset[str]
    reparameterized: bool | None

    def __init__(
        self,
        model: Callable,
        observed_names: Iterable,
        reparam_names: Iterable,
        reparameterized: bool | None = None,
    ):

        self.model = model
        self.observed_names = frozenset(observed_names)
        self.reparam_names = frozenset(reparam_names)
        self.reparameterized = reparameterized

    def call_without_reparam(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class AbstractGuide(AbstractModelOrGuide):
    """Abstract class used for numpyro guides."""

    @abstractmethod
    def __call__(self, obs: dict[str, Array]):
        """Numpyro model over the latent variables.

        Note, these should have a "_base" postfix when transform reparam is used.
        """
        pass

    def log_prob_original_space(
        self,
        latents: dict,
        model: AbstractModel,
        *args,
        reduce: bool = True,
        **kwargs,
    ):
        """Compute the log probability in the original space, inferred from model.

        Guides are usually defined in a reparameterized space. This utility allows
        evaluating the log density for samples from the original space. Currently
        only reparameterization with TransformReparam is supported.

        Args:
            latents: Latents from the original space (not the base space).
            model: Model from which to infer the reparameterization used.
            *args: Positional arguments passed to the model and guide.
            reduce: Whether to reduce the result to a scalar or return a dictionary
                of log probabilities for each site.
            **kwargs: Key word arguments passed to the model and guide.
        """
        model = model.reparam(set_val=False)
        validate_data_and_model_match(
            data=latents,
            model=model,
            assert_present=model.latent_names,
        )
        transforms = model.get_reparam_transforms(latents, *args, **kwargs)

        base_samples = {}
        log_dets = {}
        for k, latent in latents.items():
            if k in transforms:
                transform = ComposeTransform(transforms[k])
                base_val = transform.inv(latent)
                log_dets[k] = transform.log_abs_det_jacobian(base_val, None)
                base_samples[f"{k}_base"] = base_val
            else:
                base_samples[k] = latent

        trace = handlers.trace(handlers.condition(self, base_samples)).get_trace()
        log_probs = trace_to_log_prob(trace, reduce=False)

        for k in transforms.keys():
            assert log_probs[f"{k}_base"].shape == log_dets[k].shape
            log_probs[k] = log_probs[f"{k}_base"] - log_dets[k]
            log_probs.pop(f"{k}_base")

        if reduce:
            return sum(v.sum() for v in log_probs.values())

        return log_probs

    def sample(
        self,
        key: PRNGKeyArray,
        *,
        log_prob: bool = False,
        reduce: bool = True,
    ):
        """Sample from the guide, and optionally return its log probability.

        To draw multiple samples, vectorize (jax.vmap) over a set of keys.
        """
        trace = handlers.trace(fn=handlers.seed(self, key)).get_trace()
        samples = {k: v["value"] for k, v in trace.items() if v["type"] == "sample"}
        if not log_prob:
            return samples
        return samples, trace_to_log_prob(trace, reduce=reduce)
