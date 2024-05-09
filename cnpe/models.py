"""Abstract model and guide class containing reparameterization logic."""

from abc import abstractmethod

import equinox as eqx
from jaxtyping import Array, PRNGKeyArray
from numpyro import handlers
from numpyro.distributions.transforms import ComposeTransform
from numpyro.infer import reparam

from cnpe.numpyro_utils import (
    get_sample_site_names,
    trace_except_obs,
    trace_to_distribution_transforms,
    trace_to_log_prob,
    validate_data_and_model_match,
)

# TODO avoid *args, **kwargs?


class AbstractNumpyroModel(eqx.Module):
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

    observed_names: eqx.AbstractClassVar[set[str]]
    reparam_names: eqx.AbstractVar[set[str]]
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
        """Sample a single sample from the predictive distribution.

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

    @property
    def latent_names(self):
        return get_sample_site_names(self).all - self.observed_names

    def prior_log_prob(
        self,
        latents: dict[str, Array],
        *,
        reduce: bool = True,
    ):
        """Compute the prior log probability under the model.

        Note, this assumes there are no latents that are children of the observed
        nodes in the DAG.

        Args:
            reduce: Whether to reduce the result to a scalar or return a dictionary
                of log probabilities for each site.
        """
        validate_data_and_model_match(latents, self, assert_present=self.latent_names)
        model = handlers.substitute(
            self,
            latents,
        )  # substitute to avoid converting to obs
        model_trace = trace_except_obs(model, self.observed_names)
        return trace_to_log_prob(model_trace, reduce=reduce)


class AbstractNumpyroGuide(eqx.Module):
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
        model: AbstractNumpyroModel,
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

        trace = handlers.trace(handlers.condition(self, base_samples)).get_trace(
            *args,
            **kwargs,
        )
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
        obs: dict[str, Array],
        *,
        log_prob: bool = False,
        reduce: bool = True,
    ):
        """Sample from the guide, and optionally return its log probability.

        To draw multiple samples, vectorize (jax.vmap) over a set of keys.
        """
        trace = handlers.trace(
            fn=handlers.seed(self, key),
        ).get_trace(obs=obs)
        samples = {k: v["value"] for k, v in trace.items() if v["type"] == "sample"}
        if not log_prob:
            return samples
        return samples, trace_to_log_prob(trace, reduce=reduce)
