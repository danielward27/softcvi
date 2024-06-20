"""Numpyro compatible loss functions."""

from abc import abstractmethod
from functools import partial
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from flowjax.wrappers import unwrap
from jax import nn
from jax.lax import stop_gradient
from jaxtyping import Array, Float, PRNGKeyArray, PyTree, Scalar
from numpyro.infer import RenyiELBO, Trace_ELBO

from softce.models import AbstractGuide, AbstractModel


class AbstractLoss(eqx.Module):
    """Abstract class representing a loss function."""

    @abstractmethod
    def __call__(
        self,
        params: AbstractGuide,
        static: AbstractGuide,
        key: PRNGKeyArray,
    ) -> Float[Scalar, ""]:
        pass


class EvidenceLowerBoundLoss(AbstractLoss):
    """The negative evidence lower bound (ELBO) loss function.

    Args:
        model: Numpyro model.
        obs: The observed data.
        n_particals: The number of samples to use in the ELBO approximation.
    """

    model: AbstractModel
    obs: dict[str, Array]
    n_particles: int

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


class RenyiLoss(AbstractLoss):
    """The negative evidence lower bound (ELBO) loss function.

    Args:
        model: Numpyro model.
        obs: The observed data.
        n_particals: The number of samples to use in the ELBO approximation.
    """

    alpha: float | int
    model: AbstractModel
    obs: dict[str, Array]
    n_particles: int

    def __init__(
        self,
        *,
        alpha: float | int,
        model: AbstractModel,
        obs: dict[str, Array],
        n_particles: int,
    ):
        self.alpha = alpha
        self.model = model
        self.obs = obs
        self.n_particles = n_particles

    def __call__(
        self,
        params: PyTree,
        static: PyTree,
        key: PRNGKeyArray,
    ) -> Float[Scalar, ""]:
        return RenyiELBO(alpha=self.alpha, num_particles=self.n_particles).loss(
            key,
            {},
            partial(self.model, obs=self.obs),
            unwrap(eqx.combine(params, static)),
        )


class SelfNormalizedForwardKL(AbstractLoss):
    # Following https://arxiv.org/pdf/2203.04176
    model: AbstractModel
    n_particles: int
    obs: dict[str, Array]

    def __init__(
        self,
        *,
        model,
        n_particles,
        obs: dict[str, Array],
    ):
        self.model = model
        self.n_particles = n_particles
        self.obs = obs

    @eqx.filter_jit
    def __call__(
        self,
        params: AbstractGuide,
        static: AbstractGuide,
        key: PRNGKeyArray,
    ) -> Float[Scalar, ""]:

        proposal = unwrap(eqx.combine(stop_gradient(params), static))
        guide = unwrap(eqx.combine(params, static))
        samples, proposal_lps = jax.vmap(proposal.sample_and_log_prob)(
            jr.split(key, self.n_particles),
        )

        joint_lps = jax.vmap(lambda latents: self.model.log_prob(latents | self.obs))(
            samples
        )

        log_weights = joint_lps - proposal_lps
        normalized_weights = nn.softmax(log_weights)
        guide_lps = jax.vmap(guide.log_prob)(samples)
        return jnp.mean(normalized_weights * (joint_lps - guide_lps))


class SoftContrastiveEstimationLoss(AbstractLoss):
    model: AbstractModel
    n_particles: int
    obs: dict[str, Array]
    alpha: int | float
    negative_distribution: Literal["proposal", "posterior"]

    def __init__(
        self,
        *,
        model: AbstractModel,
        n_particles: int,
        obs: dict[str, Array],
        alpha: int | float,
        negative_distribution: Literal["proposal", "posterior"],
    ):
        """Contrastive loss function."""
        if n_particles < 2:
            raise ValueError(
                "Need at least two particles for classification objective.",
            )
        self.model = model
        self.n_particles = n_particles
        self.obs = obs
        self.alpha = alpha
        self.negative_distribution = negative_distribution

    @eqx.filter_jit
    def __call__(
        self,
        params: AbstractGuide,
        static: AbstractGuide,
        key: PRNGKeyArray,
    ) -> Float[Scalar, ""]:
        proposal = unwrap(eqx.combine(stop_gradient(params), static))
        guide = unwrap(eqx.combine(params, static))

        def get_log_probs(key):
            latents, proposal_log_prob = proposal.sample_and_log_prob(key)
            joint_lp = self.model.log_prob(latents | self.obs)
            return (joint_lp, proposal_log_prob, guide.log_prob(latents))

        key, subkey = jr.split(key)

        # proposal_log_probs only used in second case - inefficient?
        joint_log_probs, proposal_log_probs, guide_log_probs = jax.vmap(get_log_probs)(
            jr.split(subkey, self.n_particles),
        )
        if self.negative_distribution == "posterior":
            negative_log_probs = joint_log_probs * self.alpha
        else:
            assert self.negative_distribution == "proposal"
            negative_log_probs = proposal_log_probs * self.alpha

        labels = nn.softmax(joint_log_probs - negative_log_probs)
        log_predictions = nn.log_softmax(guide_log_probs - negative_log_probs)
        return optax.losses.softmax_cross_entropy(log_predictions, labels).mean()
