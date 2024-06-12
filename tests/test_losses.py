import equinox as eqx
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
    ),
]


@pytest.mark.parametrize("loss", test_cases)
def test_losses_run(loss):
    guide = Guide()
    loss_val = loss(*eqx.partition(guide, eqx.is_inexact_array), key=jr.PRNGKey(0))
    assert loss_val.shape == ()
