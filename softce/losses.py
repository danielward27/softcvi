"""Numpyro compatible loss functions."""

from abc import abstractmethod
from collections.abc import Callable
from functools import partial
from typing import ClassVar

import equinox as eqx
import jax
import jax.random as jr
import optax
from flowjax.wrappers import unwrap
from jax import nn
from jax.lax import stop_gradient
from jaxtyping import Array, Float, PRNGKeyArray, PyTree, Scalar
from numpyro.infer import Trace_ELBO
from softce.models import AbstractGuide, AbstractModel


class AbstractLoss(eqx.Module):
    """Abstract class representing a loss function."""

    has_aux: eqx.AbstractVar[bool]

    @abstractmethod
    def __call__(
        self,
        params: AbstractGuide,
        static: AbstractGuide,
        key: PRNGKeyArray,
    ) -> Float[Scalar, ""]:
        pass


class NegativeEvidenceLowerBound(AbstractLoss):
    """The negative evidence lower bound (ELBO) loss function.

    Args:
        model: Numpyro model.
        obs: The observed data.
        n_particals: The number of samples to use in the ELBO approximation.
    """

    model: AbstractModel
    obs: dict[str, Array]
    n_particles: int
    has_aux: ClassVar[bool] = False

    def __init__(
        self,
        *,
        model: AbstractModel,
        obs: dict[str, Array],
        n_particles: int,
    ):
        self.model = model
        self.obs = obs
        self.n_particles = n_particles

    def __call__(
        self,
        params: PyTree,
        static: PyTree,
        key: PRNGKeyArray,
    ) -> Float[Scalar, ""]:
        return Trace_ELBO(self.n_particles).loss(
            key,
            {},
            partial(self.model, obs=self.obs),
            unwrap(eqx.combine(params, static)),
        )


class SoftContrastiveEstimationLoss(AbstractLoss):
    model: AbstractModel
    n_particles: int
    obs: dict[str, Array]
    proposal: Callable | None
    has_aux: ClassVar[bool] = False

    def __init__(
        self,
        *,
        model: AbstractModel,
        n_particles: int,
        obs: dict[str, Array],
        proposal: Callable | None = None,
    ):
        """Contrastive loss function.

        A proposal is sampled, and classficiation labels generated from a comparison
        of joint probabilities. Parameterizing a classifier in terms of q(theta|x)

        Args:
            model: _description_
            n_particles: _description_
            obs: _description_
            proposal: _description_. Defaults to None.

        Raises:
            ValueError: _description_
        """
        if n_particles < 2:
            raise ValueError(
                "Need at least two particles for classification objective.",
            )
        self.model = model
        self.n_particles = n_particles
        self.obs = obs
        self.proposal = proposal

    @eqx.filter_jit
    def __call__(
        self,
        params: AbstractGuide,
        static: AbstractGuide,
        key: PRNGKeyArray,
    ) -> Float[Scalar, ""]:
        if self.proposal is not None:
            proposal = self.proposal
        else:
            proposal = unwrap(eqx.combine(stop_gradient(params), static))

        guide = unwrap(eqx.combine(params, static))

        def get_log_probs(key):
            latents = proposal.sample(key)
            joint_lp = self.model.log_prob(latents | self.obs)
            return (joint_lp, guide.log_prob(latents))

        key, subkey = jr.split(key)
        joint_log_probs, guide_log_probs = jax.vmap(get_log_probs)(
            jr.split(subkey, self.n_particles),
        )
        labels = nn.softmax(joint_log_probs)
        log_predictions = nn.log_softmax(guide_log_probs)
        return optax.losses.softmax_cross_entropy(log_predictions, labels).mean()
