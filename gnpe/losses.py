from collections.abc import Callable
from functools import partial

import equinox as eqx
import jax.random as jr
from jax import Array, vmap
from jax.lax import stop_gradient
from jax.scipy.special import logsumexp
from numpyro import handlers
from numpyro.infer import Predictive, Trace_ELBO

from gnpe.numpyro_utils import log_density, prior_log_density


class AmortizedMaximumLikelihood(eqx.Module):
    """Loss function for simple amortized maximum likelihood.

    Samples the joint distribution and fits p(latents|observed) in an amortized
    manner by maximizing the probability of the latents over the joint samples.
    The guide must accept a key word array argument "observations".

    Args:
        model: The numpyro probabilistic model. This must allow passing
        observed_name: The name of the observed node in the model
        num_particles: The number of particles to use in the estimation at each
            step. Defaults to 1.
    """

    model: Callable
    observed_name: str
    num_particles: int

    def __init__(
        self,
        model: Callable,
        observed_name: str,
        num_particles: int = 1,
    ):

        self.model = model
        self.observed_name = observed_name
        self.num_particles = num_particles

    @eqx.filter_jit
    def __call__(
        self,
        params,
        static,
        key,
    ):
        def single_sample_loss(key):
            guide = eqx.combine(params, static)
            prior_predictive = Predictive(self.model, num_samples=self.num_particles)
            model_samp = prior_predictive(key)
            predictive_samps = model_samp.pop(self.observed_name)
            guide_log_prob, _ = log_density(guide, model_samp, obs=predictive_samps)
            return -guide_log_prob / self.num_particles

        return vmap(single_sample_loss)(jr.split(key)).mean()


class ContrastiveLoss(eqx.Module):
    """Create the contrastive loss function.

    Args:
        model: The numpyro probabilistic model.
        obs: An array of observations.
        obs_name: The name of the observed node.
        n_contrastive: The number of contrastive samples to use. Defaults to 20.
    """

    model: Callable
    obs: Array
    n_contrastive: int
    obs_name: str
    aux: bool

    def __init__(
        self,
        *,
        model: Callable,
        obs: Array,
        obs_name: str,
        n_contrastive: int = 20,
        aux: bool = False,
    ):

        self.model = model
        self.obs = obs
        self.n_contrastive = n_contrastive
        self.obs_name = obs_name
        self.aux = aux

    @eqx.filter_jit
    def __call__(
        self,
        params,
        static,
        key,
    ):
        guide = eqx.combine(params, static)
        guide_detatched = eqx.combine(stop_gradient(params), static)

        contrastive_key, guide_key, predictive_key = jr.split(key, 3)
        proposal_samp = self.sample_proposal(guide_key, guide_detatched)
        x_samp = self.sample_predictive(predictive_key, proposal_samp)
        contrastive_samp = self.sample_proposal(
            contrastive_key,
            guide,
            n=self.n_contrastive,
        )
        proprosal_log_prob = log_density(guide, proposal_samp, obs=x_samp)[0]

        log_proposal_contrasative, log_prior_contrastive = self.log_proposal_and_prior(
            contrastive_samp,
            guide,
            x_samp,
        )

        normalizer = logsumexp(log_proposal_contrasative - log_prior_contrastive)
        loss = -(proprosal_log_prob - normalizer)
        if self.aux:
            return loss, (
                proprosal_log_prob,
                log_proposal_contrasative,
                log_prior_contrastive,
            )
        return loss

    def sample_proposal(self, key, proposal, n=None):

        def sample_single(key):
            trace = handlers.trace(
                fn=handlers.seed(proposal, key),
            ).get_trace(obs=self.obs)
            return {k: v["value"] for k, v in trace.items() if v["type"] == "sample"}

        if n is None:
            return sample_single(key)

        return vmap(sample_single)(jr.split(key, n))

    @partial(vmap, in_axes=[None, 0, None, None])
    def log_proposal_and_prior(self, latents, guide, predictive):
        proposal_log_prob = log_density(guide, latents, obs=predictive)[0]
        prior_log_prob = prior_log_density(self.model, latents, [self.obs_name])
        return proposal_log_prob, prior_log_prob

    def log_sum_exp_normalizer(self, guide, latents, predictive):
        """Computes log sum(q(z|x)/p(z))."""

        @vmap
        def log_proposal_to_prior_ratio(latents):
            proposal_log_prob = log_density(guide, latents, obs=predictive)[0]
            prior_log_prob = prior_log_density(self.model, latents, [self.obs_name])
            return proposal_log_prob - prior_log_prob

        return logsumexp(log_proposal_to_prior_ratio(latents))

    def sample_predictive(self, key, latents):
        """Sample the observed node from the model, given the latents."""
        predictive = handlers.trace(
            handlers.seed(
                handlers.condition(self.model, data=latents),
                key,
            ),
        ).get_trace()
        return predictive[self.obs_name]["value"]


class NegativeEvidenceLowerBound(eqx.Module):
    """The negative evidence lower bound (ELBO) loss function.

    Args:
        model: Numpyro model.
        obs: The observed data.
        n_particals: The number of samples to use in the ELBO approximation.
    """

    model: Callable
    obs: Array
    n_particals: int = 1

    def __call__(
        self,
        params,
        static,
        key,
    ):
        return Trace_ELBO(self.n_particals).loss(
            key,
            {},
            self.model,
            eqx.combine(params, static),
            obs=self.obs,
        )
