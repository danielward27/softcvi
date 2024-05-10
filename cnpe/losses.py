"""Numpyro compatible loss functions."""

from abc import abstractmethod
from collections.abc import Callable
from functools import partial
from typing import ClassVar

import equinox as eqx
import jax.random as jr
from flowjax.wrappers import unwrap
from jax import vmap
from jax.lax import stop_gradient
from jax.scipy.special import logsumexp
from jaxtyping import Array, Float, PRNGKeyArray, PyTree, Scalar
from numpyro.infer import Trace_ELBO

from cnpe.models import AbstractNumpyroGuide, AbstractNumpyroModel
from cnpe.numpyro_utils import log_density


class AbstractLoss(eqx.Module):
    """Abstract class representing a loss function."""

    has_aux: eqx.AbstractVar[bool]

    @abstractmethod
    def __call__(
        self,
        params: AbstractNumpyroGuide,
        static: AbstractNumpyroGuide,
        key: PRNGKeyArray,
    ) -> Float[Scalar, ""]:
        pass


class AmortizedMaximumLikelihood(AbstractLoss):
    """Loss function for simple amortized maximum likelihood.

    Samples the joint distribution and fits p(latents|observed) in an amortized
    manner by maximizing the probability of the latents over the joint samples.
    The guide must accept a key word array argument "obs".

    Args:
        model: The numpyro probabilistic model. This must allow passing obs.
        num_particles: The number of particles to use in the estimation at each
            step. Defaults to 1.
    """

    model: AbstractNumpyroModel
    num_particles: int
    has_aux: ClassVar[bool] = False

    def __init__(
        self,
        model: Callable,
        num_particles: int = 1,
    ):

        self.model = model
        self.num_particles = num_particles

    @eqx.filter_jit
    def __call__(
        self,
        params: AbstractNumpyroGuide,
        static: AbstractNumpyroGuide,
        key: PRNGKeyArray,
    ) -> Float[Scalar, ""]:
        def single_sample_loss(key):
            guide = unwrap(eqx.combine(params, static))
            latents, obs = self.model.sample_joint(key)
            return -log_density(guide, latents, obs=obs)[0]

        return vmap(single_sample_loss)(jr.split(key, self.num_particles)).mean()


class ContrastiveLoss(AbstractLoss):
    """Create the contrastive loss function.

    Args:
        model: The numpyro probabilistic model.
        obs: An array of observations.
        observed_name: The name of the observed node.
        n_contrastive: The number of contrastive samples to use. Defaults to 20.
        stop_grad_for_contrastive_sampling: Whether to apply stop_gradient to the
            parameters used for contrastive sampling. Defaults to False.
        aux: Whether to return the individual loss components as a dictionary of
            auxiliary values. Defaults to False.
    """

    model: AbstractNumpyroModel
    obs: dict[str, Array]
    n_contrastive: int
    stop_grad_for_contrastive_sampling: bool
    has_aux: ClassVar[bool] = False

    def __init__(
        self,
        *,
        model: Callable,
        obs: dict[str, Array],
        n_contrastive: int = 20,
        stop_grad_for_contrastive_sampling: bool = False,
    ):

        self.model = model
        self.obs = obs
        self.n_contrastive = n_contrastive
        self.stop_grad_for_contrastive_sampling = stop_grad_for_contrastive_sampling

    @eqx.filter_jit
    def __call__(
        self,
        params: AbstractNumpyroGuide,
        static: AbstractNumpyroGuide,
        key: PRNGKeyArray,
    ) -> Float[Scalar, ""]:
        guide = unwrap(eqx.combine(params, static))
        guide_detatched = eqx.combine(stop_gradient(params), static)

        contrastive_key, guide_key, predictive_key = jr.split(key, 3)
        proposal_samp = guide_detatched.sample(guide_key, self.obs)
        x_samp = self.model.sample_predictive(predictive_key, proposal_samp)

        if self.stop_grad_for_contrastive_sampling:
            contrastive_guide = guide_detatched
        else:
            contrastive_guide = guide
        contrastive_samps = vmap(contrastive_guide.sample, in_axes=[0, None])(
            jr.split(contrastive_key, self.n_contrastive),
            self.obs,
        )
        log_prob_given_x = log_density(guide, proposal_samp, obs=x_samp)[0]

        log_prob_contrasative = eqx.filter_vmap(
            partial(log_density, guide, obs=x_samp),
        )(contrastive_samps)[0]
        log_prob_prior_contrastive = vmap(self.model.prior_log_prob)(contrastive_samps)
        normalizer = logsumexp(
            log_prob_contrasative - log_prob_prior_contrastive,
        )
        return -(log_prob_given_x - normalizer)

    def log_sum_exp_normalizer(self, guide, latents, predictive):
        """Computes log sum(q(z|x)/p(z))."""

        @vmap
        def log_proposal_to_prior_ratio(latents):
            proposal_log_prob = log_density(guide, latents, obs=predictive)[0]
            prior_log_prob = self.model.prior_log_prob(latents)
            return proposal_log_prob - prior_log_prob

        return logsumexp(log_proposal_to_prior_ratio(latents))


class NegativeEvidenceLowerBound(AbstractLoss):
    """The negative evidence lower bound (ELBO) loss function.

    Args:
        model: Numpyro model.
        obs: The observed data.
        n_particals: The number of samples to use in the ELBO approximation.
    """

    model: Callable
    obs: dict[str, Array]
    n_particals: int = 1
    has_aux: ClassVar[bool] = False

    def __call__(
        self,
        params: PyTree,
        static: PyTree,
        key: PRNGKeyArray,
    ) -> Float[Scalar, ""]:
        return Trace_ELBO(self.n_particals).loss(
            key,
            {},
            self.model,
            unwrap(eqx.combine(params, static)),
            obs=self.obs,
        )


# TODO If the mlp takes in all observations, then it should be replaced for VI (
# e.g. replacing the MLP with just a bias module ignoring obs.)
