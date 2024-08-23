"""Loss functions.

The general potern taken here is that the loss functions take a partitioned 
``(model, guide)`` tuple as the first two arguments (see ``equinox.partition``), and a
key as the third. This supports trainable parameters in both the model and guide.
As such, for all inexact arrays in the model and guide, explicitly marking them
as non-trainable is required if they should be considered fixed, for example using
using ``flowjax.wrappers.non_trainable``, or by accessing with a property that
applies  ``jax.lax.stop_gradient``.
"""

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
from jaxtyping import Array, Float, PRNGKeyArray, Scalar
from numpyro.infer import RenyiELBO, Trace_ELBO

from softcvi.models import AbstractGuide, AbstractModel


class AbstractLoss(eqx.Module):
    """Abstract class representing a loss function."""

    @abstractmethod
    def __call__(
        self,
        params: tuple[AbstractModel, AbstractGuide],
        static: tuple[AbstractModel, AbstractGuide],
        key: PRNGKeyArray,
    ) -> Float[Scalar, " "]:
        pass


class EvidenceLowerBoundLoss(AbstractLoss):
    """The negative evidence lower bound (ELBO) loss function.

    Args:
        obs: The observed data.
        n_particals: The number of samples to use in the ELBO approximation.
    """

    obs: dict[str, Array]
    n_particles: int

    def __init__(
        self,
        *,
        obs: dict[str, Array],
        n_particles: int,
    ):
        self.obs = obs
        self.n_particles = n_particles

    def __call__(
        self,
        params: tuple[AbstractModel, AbstractGuide],
        static: tuple[AbstractModel, AbstractGuide],
        key: PRNGKeyArray,
    ) -> Float[Scalar, ""]:
        model, guide = unwrap(eqx.combine(params, static))
        return Trace_ELBO(self.n_particles).loss(
            key,
            {},
            partial(model, obs=self.obs),
            guide,
        )


class RenyiLoss(AbstractLoss):
    """Wraps numpyro Renyi objective. See https://arxiv.org/abs/1602.02311.

    Args:
        alpha: alpha value.
        obs: The observed data.
        n_particles: The number of samples to use in the ELBO approximation.
    """

    obs: dict[str, Array]
    n_particles: int
    alpha: float | int

    def __init__(
        self,
        *,
        obs: dict[str, Array],
        n_particles: int,
        alpha: float | int,
    ):
        self.alpha = alpha
        self.obs = obs
        self.n_particles = n_particles

    def __call__(
        self,
        params: tuple[AbstractModel, AbstractGuide],
        static: tuple[AbstractModel, AbstractGuide],
        key: PRNGKeyArray,
    ) -> Float[Scalar, ""]:
        model, guide = unwrap(eqx.combine(params, static))
        return RenyiELBO(alpha=self.alpha, num_particles=self.n_particles).loss(
            key,
            {},
            partial(model, obs=self.obs),
            guide,
        )


class SelfNormImportanceWeightedForwardKLLoss(AbstractLoss):
    """A self normalized importance weighted estimate of the forward KL divergence.

    We follow the gradient estimator shown in  https://arxiv.org/pdf/2203.04176 by
    default, but provide an option for the lower variance estimator introduced in
    https://arxiv.org/abs/2407.15687.

    Args:
        n_particles: Number of particles to use in loss approximation.
        obs: The dictionary of observations.
        low_variance: Whether to add the gradient of the average variational
            probabilities to the loss, which will reduce the variance when the
            variational distribution is close to the true posterior.
    """

    n_particles: int
    obs: dict[str, Array]
    low_variance: bool

    def __init__(
        self,
        *,
        n_particles,
        obs: dict[str, Array],
        low_variance: bool = False,
    ):
        self.n_particles = n_particles
        self.obs = obs
        self.low_variance = low_variance

    @eqx.filter_jit
    def __call__(
        self,
        params: tuple[AbstractModel, AbstractGuide],
        static: tuple[AbstractModel, AbstractGuide],
        key: PRNGKeyArray,
    ) -> Float[Scalar, ""]:
        model, guide = unwrap(eqx.combine(params, static))
        proposal = unwrap(eqx.combine(stop_gradient(params[1]), static[1]))
        samples, proposal_lps = jax.vmap(proposal.sample_and_log_prob)(
            jr.split(key, self.n_particles),
        )

        joint_lps = jax.vmap(lambda latents: model.log_prob(latents | self.obs))(
            samples,
        )

        log_weights = joint_lps - proposal_lps
        normalized_weights = nn.softmax(log_weights)
        guide_lps = jax.vmap(guide.log_prob)(samples)
        loss = jnp.sum(normalized_weights * (joint_lps - guide_lps))
        if self.low_variance:
            mean_lp = jnp.mean(guide_lps)
            loss += mean_lp - stop_gradient(mean_lp)  # Avoid changing loss val
        return loss


class SoftContrastiveEstimationLoss(AbstractLoss):
    """The SoftCVI loss function.

    Args:
        n_particles: The number of particles used for estimating the loss.
        obs: The dictionary of observations.
        alpha: Tempering parameter on the interval [0, 1] applied to the negative
            distribution, i.e. raising the negative distribution to a power.
        negative_distribution: The negative distribution, either "proposal", in which
            case we use ``stop_gradient(guide)`` as the negative distribution,
            or "posterior", in which case we use the model joint density. Defaults to
            "proposal".
    """

    n_particles: int
    obs: dict[str, Array]
    alpha: int | float
    negative_distribution: Literal["proposal", "posterior"]

    def __init__(
        self,
        *,
        n_particles: int,
        obs: dict[str, Array],
        alpha: int | float,
        negative_distribution: Literal["proposal", "posterior"] = "proposal",
    ):

        if n_particles < 2:
            raise ValueError(
                "Need at least two particles for classification objective.",
            )
        self.n_particles = n_particles
        self.obs = obs
        self.alpha = alpha
        self.negative_distribution = negative_distribution

    @eqx.filter_jit
    def __call__(
        self,
        params: tuple[AbstractModel, AbstractGuide],
        static: tuple[AbstractModel, AbstractGuide],
        key: PRNGKeyArray,
    ) -> Float[Scalar, ""]:
        model, guide = unwrap(eqx.combine(params, static))
        proposal = unwrap(eqx.combine(stop_gradient(params[1]), static[1]))

        def get_log_probs(key):

            if self.negative_distribution == "posterior":
                latents = proposal.sample(key)
                joint_lp = model.log_prob(latents | self.obs)
                negative_lp = joint_lp * self.alpha
            else:
                assert self.negative_distribution == "proposal"
                latents, proposal_lp = proposal.sample_and_log_prob(key)
                joint_lp = model.log_prob(latents | self.obs)
                negative_lp = proposal_lp * self.alpha

            return {
                "joint": joint_lp,
                "negative": negative_lp,
                "guide": guide.log_prob(latents),
            }

        key, subkey = jr.split(key)
        log_probs = jax.vmap(get_log_probs)(jr.split(subkey, self.n_particles))
        labels = nn.softmax(log_probs["joint"] - log_probs["negative"])
        log_predictions = nn.log_softmax(log_probs["guide"] - log_probs["negative"])
        return optax.losses.softmax_cross_entropy(log_predictions, labels).mean()
