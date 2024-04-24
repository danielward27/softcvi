import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import pytest
from flowjax.distributions import AbstractDistribution, Normal
from flowjax.experimental.numpyro import sample
from flowjax.flows import masked_autoregressive_flow

from cnpe.losses import AmortizedMaximumLikelihood, ContrastiveLoss
from cnpe.models import AbstractNumpyroGuide, AbstractNumpyroModel


@pytest.fixture
def model():
    class Model(AbstractNumpyroModel):
        obs_names = ("b",)
        reparam_names = ()

        def call_without_reparam(self, obs=None):
            a = sample("a", Normal(jnp.zeros((3,))))
            sample("b", Normal(a), obs=obs)

    return Model()


@pytest.fixture
def guide():
    class Guide(AbstractNumpyroGuide):
        a_guide: AbstractDistribution

        def __call__(self, obs):
            sample("a", self.a_guide, condition=obs)

    return Guide(
        masked_autoregressive_flow(
            key=jr.PRNGKey(0),
            base_dist=Normal(jnp.zeros((3,))),
            cond_dim=3,
        ),
    )


def test_maximum_likelihood_loss(model, guide):
    loss = AmortizedMaximumLikelihood(model)
    loss(*eqx.partition(guide, eqx.is_inexact_array), key=jr.PRNGKey(0))


def test_contrastive_loss(model, guide):
    loss = ContrastiveLoss(
        model=model,
        obs=jnp.array(jnp.arange(3)),
    )

    loss(*eqx.partition(guide, eqx.is_inexact_array), key=jr.PRNGKey(0))

    # TODO likely a more robust test we can add.
