from collections.abc import Callable

import equinox as eqx
import numpyro
from flowjax.distributions import (
    AbstractDistribution,
)
from flowjax.experimental.numpyro import sample
from jax import Array
from jax.numpy import vectorize
from numpyro.infer.reparam import TransformReparam


class LocScaleHierarchicalModel(eqx.Module):
    """Hierarchical model, with a hyperprior on the prior location and scale.

    Args:
        likelihood: The likelihood, a conditional distribution taking in samples
            from prior.
        loc: Hyperprior over the priors location.
        scale: Hyperprior over the priors scale.
        prior: The prior distribution, a callable returning a distribution, taking
            in the location and scale.
        n_obs: The number of observations (i.e. the size of the plate).
        reparam: A dictionary with keys matching site names, and values being a numpyro
            reparameterization strategy. Defaults to TransformReparam() for "loc" and
            "scale". This means we must sample "loc_base" and "scale_base" in the guide.
    """

    likelihood: AbstractDistribution
    loc: AbstractDistribution
    scale: AbstractDistribution
    prior: Callable
    n_obs: int
    reparam: dict

    def __init__(
        self,
        likelihood: AbstractDistribution,
        loc: AbstractDistribution,
        scale: AbstractDistribution,
        prior: Callable,
        n_obs: int,
        reparam: dict = None,
    ):

        self.likelihood = likelihood
        self.loc = loc
        self.scale = scale
        self.n_obs = n_obs
        self.prior = prior
        if reparam is None:
            self.reparam = {
                "loc": TransformReparam(),
                "scale": TransformReparam(),
            }
        else:
            self.reparam = reparam

    def __check_init__(self):
        for k, p in {"loc": self.loc, "scale": self.scale}.items():
            if self.likelihood.cond_shape != p.shape:
                raise ValueError(
                    f"Simulator cond_shape {self.likelihood.cond_shape} does not match "
                    f"prior_{k} shape {p.shape}.",
                )

    def __call__(self, obs: Array | None = None):
        """The numpyro model.

        Args:
            obs: The observations. Defaults to None.
        """
        if obs is not None and obs.shape[0] != self.n_obs:
            raise ValueError(f"Expected obs.shape[0]=={self.n_obs}, got {obs.shape[0]}")

        with numpyro.handlers.reparam(config=self.reparam):
            loc = sample("loc", self.loc)
            scale = sample("scale", self.scale)
            prior = self.prior(loc, scale)

            with numpyro.plate("obs", self.n_obs):
                z = sample("z", prior)
                sample("x", self.likelihood, condition=z, obs=obs)


class LocScaleHierarchicalGuide(eqx.Module):
    loc: AbstractDistribution
    scale: AbstractDistribution
    z: AbstractDistribution
    z_embedding_net: Callable
    n_obs: int

    """Construct a guide for LocScaleHierarchicalModel.

    Args:
        loc: The distribution over the prior location. This is a conditional
            distribution that takes in the embedded z values.
        scale: The distribution over the prior scale. This is a conditional distribution
            taking in the embedded z values.
        z: The distribution over the likelihoods conditioning variable z. This
            is a conditional distribution taking in observations.
        z_embedding_net: The embedding network for z.
        n_obs: The number of observations.

    """

    def __call__(self, obs: Array):
        """The numpyro model.

        Args:
            obs: An array of observations.
        """
        self._argcheck(obs)

        with numpyro.plate("obs", obs.shape[0]):
            z = sample("z", self.z, condition=obs)

        embed = vectorize(self.z_embedding_net, signature="(a)->(b)")(z).mean(0)
        sample("loc_base", self.loc, condition=embed)
        sample("scale_base", self.scale, condition=embed)

    def _argcheck(self, obs):
        if (s := obs.shape[-2]) != self.n_obs:
            raise ValueError(f"Expected obs.shape[-2]=={self.n_obs}, got {s}")
