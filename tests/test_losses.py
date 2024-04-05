import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import pytest
from flowjax.distributions import AbstractDistribution, Normal
from flowjax.experimental.numpyro import sample
from flowjax.flows import masked_autoregressive_flow

from cnpe.losses import AmortizedMaximumLikelihood, ContrastiveLoss


def model(obs=None):
    a = sample("a", Normal(jnp.zeros((3,))))
    sample("b", Normal(a), obs=obs)


class Guide(eqx.Module):
    a_guide: AbstractDistribution

    def __call__(self, obs):
        sample("a", self.a_guide, condition=obs)


@pytest.fixture()
def guide():
    return Guide(
        masked_autoregressive_flow(
            key=jr.PRNGKey(0),
            base_dist=Normal(jnp.zeros((3,))),
            cond_dim=3,
        ),
    )


def test_maximum_likelihood_loss(guide):
    loss = AmortizedMaximumLikelihood(model, observed_name="b")
    loss(*eqx.partition(guide, eqx.is_inexact_array), key=jr.PRNGKey(0))


def test_contrastive_loss():
    loss = ContrastiveLoss(
        model=model,
        obs=jnp.array(jnp.arange(3)),
        obs_name="b",
    )

    loss(*eqx.partition(guide, eqx.is_inexact_array), key=jr.PRNGKey(0))

    # TODO likely a more robust test we can add.
