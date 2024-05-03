"""Abstract model and guide class containing reparameterization logic."""

from abc import abstractmethod

import equinox as eqx
from jaxtyping import Array
from numpyro import handlers
from numpyro.distributions.transforms import ComposeTransform
from numpyro.infer import reparam

from cnpe.numpyro_utils import (
    trace_except_obs,
    trace_to_distribution_transforms,
    trace_to_log_prob,
    validate_data_and_model_match,
)


class AbstractNumpyroModel(eqx.Module):
    """Abstract class used for numpyro models.

    Attributes:
        observed_names: names for the observed nodes.
        reparam_names: tuple of latent names to which TransformReparam is applied.
    """

    observed_names: eqx.AbstractClassVar[set[str]]
    reparam_names: eqx.AbstractVar[set[str]]

    def __call__(self, *args, **kwargs):
        """The numpyro model, applying reparameterizations."""
        config = {name: reparam.TransformReparam() for name in self.reparam_names}
        with handlers.reparam(config=config):
            self.call_without_reparam(*args, **kwargs)

    @abstractmethod
    def call_without_reparam(self, *args, **kwargs):
        """An implentation of the numpyro model, without applying reparameterization."""
        pass

    def get_reparam_transforms(self, latents: dict, *args, **kwargs):
        """Infer the deterministic transforms applied under reparameterization.

        Note this only applies for TransformReparam (not other deterministic sites).

        Args:
            latents: latent variables from the data space (not the base space).
            *args: Positional arguments for model.
            **kwargs: Key word arguments for model.
        """
        model_trace = trace_except_obs(
            handlers.substitute(self.call_without_reparam, latents),
            self.observed_names,
            *args,
            **kwargs,
        )
        transforms = trace_to_distribution_transforms(model_trace)
        return {k: t for k, t in transforms.items() if k in self.reparam_names}


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
        validate_data_and_model_match(
            latents,
            model.call_without_reparam,
            *args,
            **kwargs,
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
            log_probs[k] = log_probs[f"{k}_base"] - log_dets[k]
            log_probs.pop(f"{k}_base")

        if reduce:
            return sum(v.sum() for v in log_probs.values())

        return log_probs
