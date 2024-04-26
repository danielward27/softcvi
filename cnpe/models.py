"""Abstract model and guide class containing reparameterization logic."""

from abc import abstractmethod

import equinox as eqx
from jaxtyping import Array
from numpyro import handlers
from numpyro.distributions.transforms import ComposeTransform
from numpyro.infer import reparam

from cnpe.numpyro_utils import (
    log_density,
    trace_except_obs,
    trace_to_distribution_transforms,
)


class AbstractNumpyroModel(eqx.Module):
    """Abstract class used for numpyro models.

    Attributes:
        obs_names: names for the observed nodes.
        reparam_names: tuple of latent names to which TransformReparam is applied.
    """

    obs_names: eqx.AbstractClassVar[tuple[str]]
    reparam_names: eqx.AbstractVar[tuple[str]]

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

        Args:
            latents: latent variables from the data space (not the base space).
            *args: Positional arguments for model.
            **kwargs: Key word arguments for model.
        """
        model_trace = trace_except_obs(
            handlers.substitute(self.call_without_reparam, latents),
            self.obs_names,
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
        **kwargs,
    ):
        """Compute the log probability in the original space, inferred from model.

        Guides are usually defined in a reparameterized space. This utility allows
        evaluating the log density for samples from the original space. Currently
        only reparameterization with TransformReparam is supported.

        Args:
            latents: Latents from the original space (not the base space).
            model: model from which to infer the reparameterization used.
        """
        transforms = model.get_reparam_transforms(latents, *args, **kwargs)
        log_det = 0

        base_samples = {}
        for k, latent in latents.items():
            if k in transforms:
                transform = ComposeTransform(transforms[k])
                base_val = transform.inv(latent)
                log_det += transform.log_abs_det_jacobian(base_val, None).sum()
                base_samples[f"{k}_base"] = base_val
            else:
                base_samples[k] = latent

        return log_density(self, base_samples, *args, **kwargs)[0] - log_det
