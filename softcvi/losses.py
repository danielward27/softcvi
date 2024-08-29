"""Loss functions.

The general potern taken here is that the loss functions take a partitioned 
``(model, guide)`` tuple as the first two arguments (see ``equinox.partition``),
followed by the observations and a key. In general this approach supports trainable
parameters in both the model and guide. As such, for all inexact arrays in the model
and guide, explicitly marking them as non-trainable is required if they should be
considered fixed, for example using using ``flowjax.wrappers.non_trainable``, or by
accessing through a property that applies ``jax.lax.stop_gradient``.
"""

from abc import abstractmethod
from functools import partial
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from flowjax.wrappers import unwrap
from jax import nn
from jax.lax import stop_gradient
from jaxtyping import Array, Float, PRNGKeyArray, Scalar
from numpyro.infer import RenyiELBO, Trace_ELBO
from optax.losses import softmax_cross_entropy

from softcvi.models import AbstractGuide, AbstractReparameterizedModel


class AbstractLoss(eqx.Module):
    """Abstract class representing a loss function."""

    @abstractmethod
    def __call__(
        self,
        params: tuple[AbstractReparameterizedModel, AbstractGuide],
        static: tuple[AbstractReparameterizedModel, AbstractGuide],
        obs: dict[str, Array],
        key: PRNGKeyArray,
    ) -> Float[Scalar, " "]:
        pass


class EvidenceLowerBoundLoss(AbstractLoss):
    """The negative evidence lower bound (ELBO) loss function.

    Args:
        n_particles: The number of samples to use in the ELBO approximation.
    """

    n_particles: int

    def __init__(
        self,
        *,
        n_particles: int,
    ):
        self.n_particles = n_particles

    def __call__(
        self,
        params: tuple[AbstractReparameterizedModel, AbstractGuide],
        static: tuple[AbstractReparameterizedModel, AbstractGuide],
        obs: dict[str, Array],
        key: PRNGKeyArray,
    ) -> Float[Scalar, ""]:
        model, guide = unwrap(eqx.combine(params, static))
        return Trace_ELBO(self.n_particles).loss(
            key,
            {},
            partial(model, obs=obs),
            guide,
        )


class RenyiLoss(AbstractLoss):
    """Wraps numpyro Renyi objective. See https://arxiv.org/abs/1602.02311.

    Args:
        alpha: alpha value.
        obs: The observed data.
        n_particles: The number of samples to use in the ELBO approximation.
    """

    n_particles: int
    alpha: float | int

    def __init__(
        self,
        *,
        n_particles: int,
        alpha: float | int,
    ):
        self.alpha = alpha
        self.n_particles = n_particles

    def __call__(
        self,
        params: tuple[AbstractReparameterizedModel, AbstractGuide],
        static: tuple[AbstractReparameterizedModel, AbstractGuide],
        obs: dict[str, Array],
        key: PRNGKeyArray,
    ) -> Float[Scalar, ""]:
        model, guide = unwrap(eqx.combine(params, static))
        return RenyiELBO(alpha=self.alpha, num_particles=self.n_particles).loss(
            key,
            {},
            partial(model, obs=obs),
            guide,
        )


class SelfNormImportanceWeightedForwardKLLoss(AbstractLoss):
    """A self normalized importance weighted estimate of the forward KL divergence.

    We follow the gradient estimator shown in  https://arxiv.org/pdf/2203.04176 by
    default, but provide an option for the lower variance estimator introduced in
    https://arxiv.org/abs/2407.15687.

    Args:
        n_particles: Number of particles to use in loss approximation.
        low_variance: Whether to add the gradient of the average variational
            probabilities to the loss, which will reduce the variance when the
            variational distribution is close to the true posterior.
    """

    n_particles: int
    low_variance: bool

    def __init__(
        self,
        *,
        n_particles,
        low_variance: bool = False,
    ):
        self.n_particles = n_particles
        self.low_variance = low_variance

    @eqx.filter_jit
    def __call__(
        self,
        params: tuple[AbstractReparameterizedModel, AbstractGuide],
        static: tuple[AbstractReparameterizedModel, AbstractGuide],
        obs: dict[str, Array],
        key: PRNGKeyArray,
    ) -> Float[Scalar, ""]:
        model, guide = unwrap(eqx.combine(params, static))
        proposal = unwrap(eqx.combine(stop_gradient(params[1]), static[1]))
        samples, proposal_lps = jax.vmap(proposal.sample_and_log_prob)(
            jr.split(key, self.n_particles),
        )

        joint_lps = jax.vmap(lambda latents: model.log_prob(latents | obs))(samples)
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

    Note that by default this does not train model parameters, only the variational
    parameters (see elbo_model).

    Args:
        n_particles: The number of particles used for estimating the loss.
        alpha: Tempering parameter on the interval [0, 1] applied to the negative
            distribution, i.e. raising the negative distribution to a power.
        negative_distribution: The negative distribution, either "proposal", in which
            case we use ``stop_gradient(guide)`` as the negative distribution,
            or "posterior", in which case we use ``stop_gradient(model_joint)``.
            Defaults to "proposal".
        elbo_optimize_model: If true, adds ``-jnp.mean(model_joint)`` to the loss
            function, to allow training model parameters by maximizing a lower bound on
            the marginal likelihood (the ELBO). Defaults to False.
    """

    n_particles: int
    alpha: int | float
    negative_distribution: Literal["proposal", "posterior"]
    elbo_optimize_model: bool

    def __init__(
        self,
        *,
        n_particles: int,
        alpha: int | float,
        negative_distribution: Literal["proposal", "posterior"] = "proposal",
        elbo_optimize_model: bool = False,
    ):

        if n_particles < 2:
            raise ValueError(
                "Need at least two particles for classification objective.",
            )
        self.n_particles = n_particles
        self.alpha = alpha
        self.negative_distribution = negative_distribution
        self.elbo_optimize_model = elbo_optimize_model

    @eqx.filter_jit
    def __call__(
        self,
        params: tuple[AbstractReparameterizedModel, AbstractGuide],
        static: tuple[AbstractReparameterizedModel, AbstractGuide],
        obs: dict[str, Array],
        key: PRNGKeyArray,
    ) -> Float[Scalar, ""]:
        model, guide = unwrap(eqx.combine(params, static))
        stop_grad_model, proposal = unwrap(eqx.combine(stop_gradient(params), static))

        def get_log_probs(key):

            if self.negative_distribution == "posterior":
                latents = proposal.sample(key)
                positive_lp = stop_grad_model.log_prob(latents | obs)
                negative_lp = positive_lp * self.alpha
            else:
                assert self.negative_distribution == "proposal"
                latents, proposal_lp = proposal.sample_and_log_prob(key)
                positive_lp = stop_grad_model.log_prob(latents | obs)
                negative_lp = proposal_lp * self.alpha

            log_probs = {
                "positive": positive_lp,
                "negative": negative_lp,
                "guide": guide.log_prob(latents),
            }

            if self.elbo_optimize_model:
                log_probs["joint"] = model.log_prob(latents | obs)

            return log_probs

        key, subkey = jr.split(key)
        log_probs = jax.vmap(get_log_probs)(jr.split(subkey, self.n_particles))
        labels = nn.softmax(log_probs["positive"] - log_probs["negative"])
        log_predictions = nn.log_softmax(log_probs["guide"] - log_probs["negative"])
        loss = softmax_cross_entropy(log_predictions, labels).mean()
        if self.elbo_optimize_model:
            loss -= jnp.mean(log_probs["joint"])
        return loss
