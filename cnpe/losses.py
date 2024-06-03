"""Numpyro compatible loss functions."""

from abc import abstractmethod
from collections.abc import Callable
from functools import partial
from typing import ClassVar

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from flowjax.wrappers import unwrap
from jax import nn, vmap
from jax.lax import stop_gradient
from jax.scipy.special import logsumexp
from jaxtyping import Array, Float, PRNGKeyArray, PyTree, Scalar
from numpyro.infer import Trace_ELBO
from optax.losses import sigmoid_binary_cross_entropy

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


class ClassifierLoss(AbstractLoss):
    """Two class contrastive loss function."""  # TODO update docs

    model: AbstractNumpyroModel
    obs: dict[str, Array]
    num_pairs: int
    has_aux: ClassVar[bool] = False

    def __init__(
        self,
        *,
        model: Callable,
        obs: dict[str, Array],
        num_pairs: int = 1,
    ):

        self.model = model
        self.obs = obs
        self.num_pairs = num_pairs

    @eqx.filter_jit
    def __call__(
        self,
        params: AbstractNumpyroGuide,
        static: AbstractNumpyroGuide,
        key: PRNGKeyArray,
    ) -> Float[Scalar, ""]:
        # TODO consider sample and log_prob combined
        guide = unwrap(eqx.combine(params, static))
        proposal = unwrap(eqx.combine(stop_gradient(params), static))

        key, subkey = jr.split(key)
        latents1 = proposal.sample(subkey, obs=self.obs)
        lp1 = log_density(guide, latents1, obs=self.obs)[0]

        key, subkey = jr.split(key)
        latents2 = proposal.sample(subkey, obs=self.obs)
        lp2 = log_density(guide, latents2, obs=self.obs)[0]

        label = nn.sigmoid(
            log_density(self.model, latents1, obs=self.obs)[0]
            - log_density(self.model, latents2, obs=self.obs)[0],
        )
        logit = lp1 - lp2
        return sigmoid_binary_cross_entropy(logit, label)


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


# TODO test consistency between two implementations
class TwoClassContrastive(AbstractLoss):
    """Two class contrastive loss function.

    If num pairs==1, then the loss computation samples a pair of samples from
    the joint distribution, and the contrastive set is made by flipping the latent
    samples.
    """  # TODO update docs

    model: AbstractNumpyroModel
    obs: dict[str, Array]
    num_pairs: int
    has_aux: ClassVar[bool] = False

    def __init__(
        self,
        *,
        model: Callable,
        obs: dict[str, Array],
        num_pairs: int = 1,
    ):

        self.model = model
        self.obs = obs
        self.num_pairs = num_pairs

    @eqx.filter_jit
    def __call__(
        self,
        params: AbstractNumpyroGuide,
        static: AbstractNumpyroGuide,
        key: PRNGKeyArray,
    ) -> Float[Scalar, ""]:

        guide = unwrap(eqx.combine(params, static))

        def loss_single(key):
            proposal = eqx.combine(stop_gradient(params), static)  # TODO

            key, subkey = jr.split(key)
            proposal_samps = vmap(partial(proposal.sample, obs=self.obs))(
                jr.split(subkey),
            )
            contrastive_samps = {
                k: jnp.flip(arr, axis=0) for k, arr in proposal_samps.items()
            }

            key, subkey = jr.split(key)
            x_samps = vmap(self.model.sample_predictive)(
                jr.split(subkey),
                proposal_samps,
            )

            ratio_fn = vmap(self.log_guide_to_prior_ratio, in_axes=[0, 0, None])
            joint_ratio = ratio_fn(proposal_samps, x_samps, guide)
            contrastive_ratio = ratio_fn(contrastive_samps, x_samps, guide)
            labels = jnp.ones(2) - 1e-5  # TODO smoothing perhaps improves robustness.
            return sigmoid_binary_cross_entropy(joint_ratio - contrastive_ratio, labels)

        return eqx.filter_vmap(loss_single)(jr.split(key, self.num_pairs)).mean()

    def log_guide_to_prior_ratio(self, latents, predictive, guide):
        proposal_log_prob = log_density(guide, latents, obs=predictive)[0]
        prior_log_prob = self.model.prior_log_prob(latents)
        return proposal_log_prob - prior_log_prob


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
    """  # TODO update docs

    model: AbstractNumpyroModel
    obs: dict[str, Array]
    n_contrastive: int
    reparameterized_sampling: bool
    has_aux: ClassVar[bool] = False
    fixed_proposal: AbstractNumpyroGuide | None

    def __init__(
        self,
        *,
        model: Callable,
        obs: dict[str, Array],
        n_contrastive: int,
        reparameterized_sampling: bool = True,
        fixed_proposal: AbstractNumpyroGuide | None = None,
    ):

        self.model = model
        self.obs = obs
        self.n_contrastive = n_contrastive
        self.reparameterized_sampling = reparameterized_sampling
        self.fixed_proposal = fixed_proposal

    @eqx.filter_jit
    def __call__(
        self,
        params: AbstractNumpyroGuide,
        static: AbstractNumpyroGuide,
        key: PRNGKeyArray,
    ) -> Float[Scalar, ""]:
        guide = unwrap(eqx.combine(params, static))

        if self.fixed_proposal is not None:
            proposal = self.fixed_proposal
        elif not self.reparameterized_sampling:
            proposal = eqx.combine(stop_gradient(params), static)
        else:
            proposal = guide

        contrastive_key, guide_key, predictive_key = jr.split(key, 3)
        proposal_samp = proposal.sample(guide_key, self.obs)
        x_samp = self.model.sample_predictive(predictive_key, proposal_samp)

        contrastive_samps = vmap(
            partial(proposal.sample, obs=self.obs),
        )(jr.split(contrastive_key, self.n_contrastive))

        joint_ratio = self.log_guide_to_prior_ratio(proposal_samp, x_samp, guide)

        contrastive_ratios = vmap(
            partial(
                self.log_guide_to_prior_ratio,
                predictive=x_samp,
                guide=guide,
            ),
        )(contrastive_samps)

        # Include joint in contrastive
        contrastive_ratios = jnp.append(contrastive_ratios, joint_ratio)
        normalizer = logsumexp(contrastive_ratios)
        return -(joint_ratio - normalizer)

    def log_guide_to_prior_ratio(self, latents, predictive, guide):
        proposal_log_prob = log_density(guide, latents, obs=predictive)[0]
        prior_log_prob = self.model.prior_log_prob(latents)
        return proposal_log_prob - prior_log_prob


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
