import equinox as eqx
import jax
import jax.flatten_util
import jax.numpy as jnp
import jax.random as jr
import pytest
from flowjax.distributions import AbstractDistribution, Normal
from flowjax.experimental.numpyro import sample

from softcvi import losses
from softcvi.models import AbstractGuide, AbstractModel


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
        obs={"b": jnp.array(jnp.arange(3))},
        n_particles=2,
    ),
    losses.RenyiLoss(
        alpha=0,
        obs={"b": jnp.array(jnp.arange(3))},
        n_particles=2,
    ),
    losses.SoftContrastiveEstimationLoss(
        obs={"b": jnp.array(jnp.arange(3))},
        n_particles=2,
        alpha=0.5,
        negative_distribution="posterior",
    ),
    losses.SoftContrastiveEstimationLoss(
        obs={"b": jnp.array(jnp.arange(3))},
        n_particles=2,
        alpha=0.2,
        negative_distribution="proposal",
    ),
    losses.SelfNormImportanceWeightedForwardKLLoss(
        obs={"b": jnp.array(jnp.arange(3))},
        n_particles=2,
    ),
]


@pytest.mark.parametrize("loss", test_cases)
def test_losses_run(loss):
    model, guide = Model().reparam(set_val=True), Guide()
    loss_val = loss(
        *eqx.partition((model, guide), eqx.is_inexact_array),
        key=jr.PRNGKey(0),
    )
    assert loss_val.shape == ()


obs = {"b": jnp.array(jnp.arange(3))}

test_cases = {
    "SoftCVI-proposal": (
        losses.SoftContrastiveEstimationLoss(
            obs=obs,
            n_particles=2,
            alpha=0.75,
            negative_distribution="proposal",
        ),
        True,
    ),
    "SoftCVI-posterior": (
        losses.SoftContrastiveEstimationLoss(
            obs=obs,
            n_particles=2,
            alpha=0.75,
            negative_distribution="posterior",
        ),
        True,
    ),
    "SNIS-fKL": (
        losses.SelfNormImportanceWeightedForwardKLLoss(
            obs=obs,
            n_particles=2,
        ),
        False,
    ),
    "SNIS-fKL-low-var": (
        losses.SelfNormImportanceWeightedForwardKLLoss(
            obs=obs,
            n_particles=2,
            low_variance=True,
        ),
        True,
    ),
}


@pytest.mark.parametrize(
    ("loss", "expect_zero_grad"),
    test_cases.values(),
    ids=test_cases.keys(),
)
def test_grad_zero_at_optimum(loss, *, expect_zero_grad: bool):

    class OptimalGuide(AbstractGuide):
        a_guide: AbstractDistribution

        def __init__(self, obs):
            posterior_variance = 1 / 2
            posterior_mean = obs["b"] / 2
            self.a_guide = Normal(jnp.full(3, posterior_mean), posterior_variance**0.5)

        def __call__(self):
            sample("a", self.a_guide)

    model = Model().reparam(set_val=True)
    guide = OptimalGuide(loss.obs)
    params, static = eqx.partition((model, guide), eqx.is_inexact_array)
    grad = jax.grad(loss)(params, static, jr.PRNGKey(1))
    grad = jax.flatten_util.ravel_pytree(grad)[0]
    is_zero_grad = pytest.approx(grad, abs=1e-5) == 0
    assert is_zero_grad is expect_zero_grad
