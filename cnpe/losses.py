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
from jaxtyping import Array, PRNGKeyArray, PyTree
from numpyro import handlers
from numpyro.infer import Trace_ELBO

from cnpe.models import AbstractNumpyroGuide, AbstractNumpyroModel
from cnpe.numpyro_utils import log_density, prior_log_density, trace_to_log_prob


class AbstractLoss(eqx.Module):
    """Abstract class representing a loss function."""

    has_aux: eqx.AbstractVar[bool]

    @abstractmethod
    def __call__(
        self,
        params: AbstractNumpyroGuide,
        static: AbstractNumpyroGuide,
        key: PRNGKeyArray,
    ):
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
    ):
        def single_sample_loss(key):
            guide = unwrap(eqx.combine(params, static))
            trace = handlers.trace(handlers.seed(self.model, key)).get_trace()
            latents = {k: v["value"] for k, v in trace.items() if v["type"] == "sample"}
            obs = {name: latents.pop(name) for name in self.model.obs_names}
            return -log_density(guide, latents, obs=obs)[0]

        return vmap(single_sample_loss)(jr.split(key, self.num_particles)).mean()


class ContrastiveLoss(AbstractLoss):
    """Create the contrastive loss function.

    Args:
        model: The numpyro probabilistic model.
        obs: An array of observations.
        obs_name: The name of the observed node.
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
    has_aux: bool

    def __init__(
        self,
        *,
        model: Callable,
        obs: dict[str, Array],
        n_contrastive: int = 20,
        stop_grad_for_contrastive_sampling: bool = False,
        has_aux: bool = False,
    ):

        self.model = model
        self.obs = obs
        self.n_contrastive = n_contrastive
        self.stop_grad_for_contrastive_sampling = stop_grad_for_contrastive_sampling
        self.has_aux = has_aux

    @eqx.filter_jit
    def __call__(
        self,
        params: AbstractNumpyroGuide,
        static: AbstractNumpyroGuide,
        key: PRNGKeyArray,
    ):
        guide = unwrap(eqx.combine(params, static))
        guide_detatched = eqx.combine(stop_gradient(params), static)

        contrastive_key, guide_key, predictive_key = jr.split(key, 3)
        proposal_samp, proposal_log_prob_given_x_obs = self.sample_proposal(
            guide_key,
            guide_detatched,
            log_prob=True,
        )
        x_samp = self.sample_predictive(predictive_key, proposal_samp)

        if self.stop_grad_for_contrastive_sampling:
            contrastive_guide = guide_detatched
        else:
            contrastive_guide = guide
        contrastive_samp = self.sample_proposal(
            contrastive_key,
            contrastive_guide,
            n=self.n_contrastive,
        )
        log_prob_given_x = log_density(guide, proposal_samp, obs=x_samp)[0]
        log_prob_prior = prior_log_density(
            model=self.model,
            data=proposal_samp,
            observed_nodes=self.model.obs_names,
        )

        log_prob_contrasative, log_prob_prior_contrastive = self.log_proposal_and_prior(
            contrastive_samp,
            guide,
            x_samp,
        )
        normalizer = logsumexp(log_prob_contrasative - log_prob_prior_contrastive)
        loss = -(
            log_prob_given_x
            - log_prob_prior
            + proposal_log_prob_given_x_obs
            - normalizer
        )
        if self.has_aux:
            raise NotImplementedError()
        return loss

    def sample_proposal(
        self,
        key: PRNGKeyArray,
        proposal: AbstractNumpyroGuide,
        n: int | None = None,
        *,
        log_prob: bool = False,
    ):

        def sample_single(key):
            trace = handlers.trace(
                fn=handlers.seed(proposal, key),
            ).get_trace(obs=self.obs)
            samples = {k: v["value"] for k, v in trace.items() if v["type"] == "sample"}
            if not log_prob:
                return samples
            return samples, trace_to_log_prob(trace, reduce=True)

        if n is None:
            return sample_single(key)

        return vmap(sample_single)(jr.split(key, n))

    @partial(vmap, in_axes=[None, 0, None, None])
    def log_proposal_and_prior(self, latents, guide, predictive):
        proposal_log_prob = log_density(guide, latents, obs=predictive)[0]
        prior_log_prob = prior_log_density(self.model, latents, self.model.obs_names)
        return proposal_log_prob, prior_log_prob

    def log_sum_exp_normalizer(self, guide, latents, predictive):
        """Computes log sum(q(z|x)/p(z))."""

        @vmap
        def log_proposal_to_prior_ratio(latents):
            proposal_log_prob = log_density(guide, latents, obs=predictive)[0]
            prior_log_prob = prior_log_density(
                self.model,
                latents,
                self.model.obs_names,
            )
            return proposal_log_prob - prior_log_prob

        return logsumexp(log_proposal_to_prior_ratio(latents))

    def sample_predictive(self, key, latents):
        """Sample the observed node from the model, given the latents."""
        # TODO check all latents present?
        predictive = handlers.trace(
            handlers.seed(
                handlers.condition(self.model, data=latents),
                key,
            ),
        ).get_trace()
        return {name: predictive[name]["value"] for name in self.model.obs_names}


class NegativeEvidenceLowerBound(AbstractLoss):
    """The negative evidence lower bound (ELBO) loss function.

    Args:
        model: Numpyro model.
        obs: The observed data.
        n_particals: The number of samples to use in the ELBO approximation.
    """

    model: Callable
    obs: Array
    n_particals: int = 1
    has_aux: ClassVar[bool] = False

    def __call__(
        self,
        params: PyTree,
        static: PyTree,
        key: PRNGKeyArray,
    ):
        return Trace_ELBO(self.n_particals).loss(
            key,
            {},
            self.model,
            unwrap(eqx.combine(params, static)),
            obs=self.obs,
        )
