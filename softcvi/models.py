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
from typing import ClassVar

import equinox as eqx
from flowjax.wrappers import unwrap
from jax import ShapeDtypeStruct
from jaxtyping import Array, PRNGKeyArray
from numpyro import handlers
from numpyro.distributions.transforms import ComposeTransform
from numpyro.infer import reparam

from softcvi.numpyro_utils import (
    get_sample_site_names,
    shape_only_trace,
    trace_to_distribution_transforms,
    trace_to_log_prob,
)


def _check_present(names, data):
    for site in names:
        if site not in data:
            raise ValueError(f"Expected {site} to be provided in data.")


class AbstractProbabilisticProgram(eqx.Module):
    """Abstract class representing a (numpyro) probabilistic program.

    Provides convenient distribution-like methods for common use cases.
    """

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    def sample(self, key: PRNGKeyArray, *args, **kwargs) -> dict[str, Array]:
        """Sample the joint distribution.

        Args:
            key: Jax random key.
            *args: Positional arguments passed to the program.
            **kwargs: Key word arguments passed to the program.
        """
        seeded_model = handlers.seed(unwrap(self), key)
        trace = handlers.trace(seeded_model).get_trace(*args, **kwargs)
        return {k: v["value"] for k, v in trace.items() if v["type"] == "sample"}

    def log_prob(self, data: dict[str, Array], *args, **kwargs):
        """The joint probability under the model.

        Args:
            data: Dictionary of samples, including all sites in the program.
            *args: Positional arguments passed to the program.
            **kwargs: Key word arguments passed to the program.
        """
        """"""
        self = unwrap(self)
        self.validate_data(data, *args, **kwargs)
        _check_present(self.site_names(*args, **kwargs), data)
        sub_model = handlers.substitute(self, data)
        trace = handlers.trace(sub_model).get_trace(*args, **kwargs)
        return trace_to_log_prob(trace, reduce=True)

    def sample_and_log_prob(self, key: PRNGKeyArray, *args, **kwargs):
        """Sample and return its log probability.

        In some instances, this will be more efficient than calling each methods
        seperately. To draw multiple samples, vectorize (jax.vmap) over a set of keys.

        Args:
            key: Jax random key.
            *args: Positional arguments passed to the program.
            **kwargs: Key word arguments passed to the program.
        """
        self = unwrap(self)
        trace = handlers.trace(fn=handlers.seed(self, key)).get_trace(*args, **kwargs)
        samples = {k: v["value"] for k, v in trace.items() if v["type"] == "sample"}
        return samples, trace_to_log_prob(trace)

    def validate_data(
        self,
        data: dict[str, Array],
        *args,
        **kwargs,
    ):
        """Validate the data names and shapes are compatible with the model.

        For each site in data, validate that the shapes match what is produced by the
        model. Note, if you have batch dimensions in data, this function must be
        vectorized, e.g. using eqx.filter_vmap.

        Args:
            data: The data.
            model: The model.
            *args: Args passed to model when tracing to infer shapes.
            **kwargs: kwargs passed to model when tracing to infer shapes.
        """
        trace = shape_only_trace(self, *args, **kwargs)
        for name, samples in data.items():
            if name not in trace:
                raise ValueError(f"Got {name} which does not exist in trace.")

            trace_shape = trace[name]["value"].shape

            if trace[name]["type"] == "sample" and trace_shape != data[name].shape:
                raise ValueError(
                    f"{name} had shape {trace_shape} in model, but shape "
                    f"{samples.shape} in data.",
                )

    def site_names(self, *args, **kwargs) -> set:
        return get_sample_site_names(unwrap(self), *args, **kwargs).all


class AbstractModel(AbstractProbabilisticProgram):
    """Abstract class used for model component."""

    observed_names: eqx.AbstractVar[set[str] | frozenset[str]]

    @property
    def latent_names(self):
        return self.site_names() - self.observed_names

    def sample_predictive(
        self,
        key: PRNGKeyArray,
        latents: dict[str, Array],
    ):
        """Sample a single sample from the predictive/likelihood distribution.

        To generate mutiple samples, use e.g. jax.vmap over a set of keys.
        """
        self = unwrap(self)
        self.validate_data(latents)
        _check_present(self.latent_names, latents)
        conditioned = handlers.seed(handlers.condition(self, data=latents), key)
        predictive = handlers.trace(conditioned).get_trace()
        return {name: predictive[name]["value"] for name in self.observed_names}

    def log_likelihood(
        self,
        latents: dict[str, Array],
        obs: dict[str, Array],
        *,
        reduce: bool = True,
    ):
        self = unwrap(self)
        self.validate_data(latents | obs)
        _check_present(self.latent_names, latents)
        _check_present(self.observed_names, obs)
        trace = handlers.trace(handlers.substitute(self, latents)).get_trace(obs=obs)
        obs_trace = {k: v for k, v in trace.items() if k in self.observed_names}
        return trace_to_log_prob(obs_trace, reduce=reduce)

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


class AbstractReparameterizedModel(AbstractModel):
    """Abstract class used for a reparameterized numpyro.

    Attributes:
        reparam_names:  Names of latents to which TransformReparam is applied,
            if the reparameterized flag is set to True. This will reparameterize
            any numpyro or flowjax transformed distributions.
        reparameterized: A flag denoting whether to use the reparameterized model, or
            the model on the original space. We set this to None on intialization,
            meaning it must be explicitly set for methods that may refer to either
            the reparameterized or original model.
    """

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
        """The numpyro model, applying reparameterizations if self.reparameterized."""
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
        self.validate_data(latents)
        _check_present(model.latent_names, latents)
        model = handlers.substitute(model, latents)
        model_trace = handlers.trace(model).get_trace(obs=obs)
        transforms = trace_to_distribution_transforms(model_trace)
        return {k: t for k, t in transforms.items() if k in self.reparam_names}

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
        model.validate_data(latents)
        _check_present(model.latent_names, latents)
        model = handlers.condition(model, latents)
        trace = handlers.trace(model).get_trace(obs=obs)

        for name in self.reparam_names:
            latents.pop(f"{name}_base")
            latents[name] = trace[name]["value"]
        return latents


class ModelToReparameterized(AbstractReparameterizedModel):
    """Wrapper class to mimic a reparameterized model.

    This does not perform any reparameterization, but provides a wrapper of an
    AbstractModel into an AbstractReparameterizedModel with zero reparameterized sites.
    This prevents e.g. frequent use of ``if AbstractReparameterizedModel: ...``.
    """

    model: AbstractModel
    observed_names: frozenset[str]
    reparameterized: bool | None = False  # No effect if set to true
    reparam_names: ClassVar[frozenset] = frozenset()

    def __init__(self, model: AbstractModel):
        self.model = model
        self.observed_names = frozenset(model.observed_names)
        self.reparameterized = False

    def call_without_reparam(self, obs: dict[str, Array] | None = None):
        return self.model(obs)


class NumpyroModelToModel(AbstractReparameterizedModel):
    """Wrap a numpyro model to an AbstractReparameterizedModel instance.

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


class AbstractGuide(AbstractProbabilisticProgram):
    """Abstract class used for numpyro guides.

    Note that the call method must support passing obs, even if unused, for consistency
    of API.
    """

    @abstractmethod
    def __call__(self, obs: dict[str, Array] | None = None):
        pass

    def log_prob_original_space(
        self,
        latents: dict,
        model: AbstractReparameterizedModel,
        obs: dict[str, Array] | None = None,
        *,
        reduce: bool = True,
    ):
        """Compute the log probability in the original space.

        Guides are usually defined in a reparameterized space. This utility allows
        evaluating the log density for samples from the original space, by inferring
        the reparameterizations from the model. Currently only reparameterization with
        TransformReparam is supported. The guide must support passing of observations,
        even if unused, for consistency of API (allows e.g. amortized VI).

        Args:
            latents: Latents from the original space (not the base space).
            model: Model from which to infer the reparameterization used.
            obs: Observations. If this is None, we assume we can infer the
                reparameterizations from the prior distribution.
            reduce: Whether to reduce the result to a scalar or return a dictionary
                of log probabilities for each site.
        """
        model = unwrap(model).reparam(set_val=False)
        model.validate_data(latents)
        _check_present(model.latent_names, latents)
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

        conditioned_guide = handlers.condition(unwrap(self), base_samples)
        trace = handlers.trace(conditioned_guide).get_trace(obs=obs)
        log_probs = trace_to_log_prob(trace, reduce=False)

        for k in transforms.keys():
            assert log_probs[f"{k}_base"].shape == log_dets[k].shape
            log_probs[k] = log_probs[f"{k}_base"] - log_dets[k]
            log_probs.pop(f"{k}_base")

        if reduce:
            return sum(v.sum() for v in log_probs.values())

        return log_probs
