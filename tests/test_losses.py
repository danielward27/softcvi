import equinox as eqx
import jax
import jax.flatten_util
import jax.numpy as jnp
import jax.random as jr
import pytest
from flowjax.distributions import AbstractDistribution, Normal
from flowjax.experimental.numpyro import sample

from softce import losses
from softce.models import AbstractGuide, AbstractModel


class Model(AbstractModel):
    reparameterized: bool | None = None
    observed_names = {"b"}
    reparam_names = set()

    def call_without_reparam(self, obs=None):
        a = sample("a", Normal(jnp.zeros((3,))))
        sample("b", Normal(a), obs=obs["b"] if obs is not None else None)


class Guide(AbstractGuide):
    a_guide = Normal(jnp.ones(3))

    def __call__(self):
        sample("a", Normal(jnp.ones(3)))


test_cases = [
    losses.EvidenceLowerBoundLoss(
        model=Model().reparam(set_val=True),
        obs={"b": jnp.array(jnp.arange(3))},
        n_particles=2,
    ),
    losses.RenyiLoss(
        alpha=0,
        model=Model().reparam(set_val=True),
        obs={"b": jnp.array(jnp.arange(3))},
        n_particles=2,
    ),
    losses.SoftContrastiveEstimationLoss(
        model=Model().reparam(set_val=True),
        obs={"b": jnp.array(jnp.arange(3))},
        n_particles=2,
        alpha=0.5,
        negative_distribution="posterior",
    ),
    losses.SoftContrastiveEstimationLoss(
        model=Model().reparam(set_val=True),
        obs={"b": jnp.array(jnp.arange(3))},
        n_particles=2,
        alpha=0.2,
        negative_distribution="proposal",
    ),
    losses.SelfNormImportanceWeightedForwardKLLoss(
        model=Model().reparam(set_val=True),
        obs={"b": jnp.array(jnp.arange(3))},
        n_particles=2,
    ),
]


@pytest.mark.parametrize("loss", test_cases)
def test_losses_run(loss):
    guide = Guide()
    loss_val = loss(*eqx.partition(guide, eqx.is_inexact_array), key=jr.PRNGKey(0))
    assert loss_val.shape == ()


@pytest.mark.parametrize("negative_distribution", ["posterior", "proposal"])
def test_softce_grad_zero_at_optimum(negative_distribution):
    obs = {"b": jnp.array(jnp.arange(3))}

    class OptimalGuide(AbstractGuide):
        a_guide: AbstractDistribution

        def __init__(self, obs):
            posterior_variance = 1 / 2
            posterior_mean = obs["b"] / 2
            self.a_guide = Normal(jnp.full(3, posterior_mean), posterior_variance**0.5)

        def __call__(self):
            sample("a", Normal(jnp.ones(3)))

    loss = losses.SoftContrastiveEstimationLoss(
        model=Model().reparam(set_val=True),
        obs=obs,
        n_particles=2,
        alpha=0.75,
        negative_distribution=negative_distribution,
    )

    guide = OptimalGuide(obs)
    params, static = eqx.partition(guide, eqx.is_inexact_array)

    grad = jax.grad(loss)(params, static, jr.PRNGKey(0))
    grad = jax.flatten_util.ravel_pytree(grad)[0]

    assert pytest.approx(grad) == 0
