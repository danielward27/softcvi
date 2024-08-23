"""Abstract model and guide class containing reparameterization logic.

Numpyro handlers can be a bit clunky to work with, these classes provides some
convenient methods for sampling, log density evaluation and reparameterization.
In general, all methods should work in the single sample case, and require explicit
vectorization otherwise, for example using ``equinox.filter_vmap`` or ``jax.vmap``.

In general, we currently assume that the only argument/key word argument that can be
passed to a model obs (the observations), and a guide takes no arguments in its call
method.

"""

from abc import abstractmethod
from collections.abc import Callable, Iterable

import equinox as eqx
from flowjax.wrappers import unwrap
from jax import ShapeDtypeStruct
from jaxtyping import Array, PRNGKeyArray
from numpyro import handlers
from numpyro.distributions.transforms import ComposeTransform
from numpyro.infer import reparam

from softcvi.numpyro_utils import (
    get_sample_site_names,
    trace_to_distribution_transforms,
    trace_to_log_prob,
    validate_data_and_model_match,
)


class AbstractModelOrGuide(eqx.Module):
    """Abstract class representing shared model and guide components."""

    @abstractmethod
    def __call__(self, obs: dict[str, Array] | None = None):
        pass

    @property
    def site_names(self) -> set:
        return get_sample_site_names(unwrap(self)).all

    def sample(self, key: PRNGKeyArray) -> dict[str, Array]:
        """Sample the joint distribution, returning a tuple, (latents, observed)."""
        trace = handlers.trace(handlers.seed(unwrap(self), key)).get_trace()
        return {k: v["value"] for k, v in trace.items() if v["type"] == "sample"}

    def log_prob(self, data: dict[str, Array], *, reduce: bool = True):
        """The joint probability under the model.

        Data should include all nodes (latents and observed).
        """
        self = unwrap(self)
        validate_data_and_model_match(data, self, assert_present=self.site_names)
        trace = handlers.trace(handlers.substitute(self, data)).get_trace()
        return trace_to_log_prob(trace, reduce=reduce)


class AbstractModel(AbstractModelOrGuide):
    """Abstract class used for numpyro models.

    The class serves two purposes:
        1) Facilitating easy and togglable reparameterization.
        2) Giving access to a more readable methods for sampling and density evaluation.

    The main purpose of this class is to facilitate togglable reparameterization.

    Attributes:
        observed_names: Names for the observed nodes.
        reparam_names:  Names of latents to which TransformReparam is applied,
            if the reparameterized flag is set to True.
        reparameterized: A flag denoting whether to use the reparameterized model, or
            the model on the original space. We set this to None on intialization,
            meaning it must be explicitly set for methods that may refer to either
            the reparameterized or original model.
    """

    observed_names: eqx.AbstractVar[set[str] | frozenset[str]]
    reparam_names: eqx.AbstractVar[set[str] | frozenset[str]]
    reparameterized: eqx.AbstractVar[bool | None]

    @abstractmethod
    def call_without_reparam(self, obs: dict[str, Array] | None = None):
        """An implentation of the numpyro model, without applying reparameterization.

        Generally, do not directly use this method, instead calling the class directly,
        using model.reparam(set_val=False) if required.
        """
        pass

    def __call__(self, obs: dict[str, Array] | None = None):
        """The numpyro model, applying reparameterizations."""
        self = unwrap(self)
        if self.reparameterized is None:
            raise ValueError(
                "Reparameterized flag was None. Set to True/False using model.reparam.",
            )
        if self.reparameterized:
            config = {name: reparam.TransformReparam() for name in self.reparam_names}
            with handlers.reparam(config=config):
                self.call_without_reparam(obs=obs)
        else:
            self.call_without_reparam(obs=obs)

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

    def get_reparam_transforms(
        self,
        latents: dict[str, Array],
        obs: dict[str, Array] | None,
    ):
        """Infer the deterministic transforms applied under reparameterization.

        Note this only applies for TransformReparam (not other deterministic sites).

        Args:
            latents: latent variables from the data space (not the base space).
            obs: Observations. If this is None, we assume that we can infer the
                reparameterization from the prior, independent from the observations.
        """
        model = unwrap(self).reparam(set_val=False)
        model = model.prior if obs is None else model
        validate_data_and_model_match(latents, model, assert_present=model.latent_names)
        model = handlers.substitute(model, latents)
        model_trace = handlers.trace(model).get_trace(obs=obs)
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
        self = unwrap(self)
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
        self = unwrap(self)
        validate_data_and_model_match(latents, self, assert_present=self.latent_names)
        validate_data_and_model_match(obs, self, assert_present=self.observed_names)
        trace = handlers.trace(handlers.substitute(self, latents)).get_trace(obs=obs)
        obs_trace = {k: v for k, v in trace.items() if k in self.observed_names}
        return trace_to_log_prob(obs_trace, reduce=reduce)

    def latents_to_original_space(
        self,
        latents: dict[str, Array],
        obs: dict[str, Array] | None,
    ) -> dict[str, Array]:
        """Convert a set of latents from the reparameterized space to original space.

        Args:
            latents: The set of latents from the reparameterized space.
            obs: Observations. If this is None, we assume we can infer
                reparameterizations from the prior, with no dependency on the
                observations.
        """
        self = unwrap(self)
        latents = {k: v for k, v in latents.items()}  # Avoid mutating
        model = self.reparam(set_val=True)
        model = model.prior if obs is None else model
        validate_data_and_model_match(latents, model, assert_present=model.latent_names)
        model = handlers.condition(model, latents)
        trace = handlers.trace(model).get_trace(obs=obs)

        for name in self.reparam_names:
            latents.pop(f"{name}_base")
            latents[name] = trace[name]["value"]

        return latents

    @property
    def prior(self):
        """Get the prior distribution.

        This assumes that no latent variables are children of the observed nodes.
        """
        dummy_obs = {k: ShapeDtypeStruct((), float) for k in self.observed_names}
        model = handlers.condition(self, dummy_obs)  # To avoid sampling observed nodes
        return NumpyroModelToModel(
            handlers.block(model, hide=self.observed_names),
            observed_names=frozenset({}),
            reparam_names=self.reparam_names,
            reparameterized=self.reparameterized,
        )


class NumpyroModelToModel(AbstractModel):
    """Wrap a numpyro model to an AbstractModel isntance.

    Currently assumes no trainable model parameters.
    """

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

    def call_without_reparam(self, obs: dict[str, Array] | None = None):
        return self.model(obs=obs)


class AbstractGuide(AbstractModelOrGuide):
    """Abstract class used for numpyro guides."""

    @abstractmethod
    def __call__(self):
        """Numpyro model over the latent variables.

        Note, these should have a "_base" postfix when transform reparam is used.
        """
        pass

    def log_prob_original_space(
        self,
        latents: dict,
        obs: dict[str, Array] | None,
        model: AbstractModel,
        *,
        reduce: bool = True,
    ):
        """Compute the log probability in the original space.

        Guides are usually defined in a reparameterized space. This utility allows
        evaluating the log density for samples from the original space, by inferring
        the reparameterizations from the model. Currently only reparameterization with
        TransformReparam is supported.

        Args:
            latents: Latents from the original space (not the base space).
            model: Model from which to infer the reparameterization used.
            obs: Observations. If this is None, we assume we can infer the
                reparameterizations from the prior distribution.
            reduce: Whether to reduce the result to a scalar or return a dictionary
                of log probabilities for each site.
        """
        model = unwrap(model).reparam(set_val=False)
        validate_data_and_model_match(
            data=latents,
            model=model,
            assert_present=model.latent_names,
        )
        transforms = model.get_reparam_transforms(latents, obs=obs)

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

        trace = handlers.trace(
            handlers.condition(unwrap(self), base_samples),
        ).get_trace()
        log_probs = trace_to_log_prob(trace, reduce=False)

        for k in transforms.keys():
            assert log_probs[f"{k}_base"].shape == log_dets[k].shape
            log_probs[k] = log_probs[f"{k}_base"] - log_dets[k]
            log_probs.pop(f"{k}_base")

        if reduce:
            return sum(v.sum() for v in log_probs.values())

        return log_probs

    def sample_and_log_prob(self, key: PRNGKeyArray, *, reduce: bool = True):
        """Sample from the guide, and optionally return its log probability.

        In some instances, this will be more efficient than calling each methods
        seperately. To draw multiple samples, vectorize (jax.vmap) over a set of keys.

        Args:
            key: Jax random key.
            reduce: Whether to reduce the result to a scalar or return a dictionary
                of log probabilities for each site.
        """
        self = unwrap(self)
        trace = handlers.trace(fn=handlers.seed(self, key)).get_trace()
        samples = {k: v["value"] for k, v in trace.items() if v["type"] == "sample"}
        return samples, trace_to_log_prob(trace, reduce=reduce)
