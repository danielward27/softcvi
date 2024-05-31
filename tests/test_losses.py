import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import pytest
from flowjax.distributions import AbstractDistribution, Normal
from flowjax.experimental.numpyro import sample
from flowjax.flows import masked_autoregressive_flow

from cnpe.losses import (
    AmortizedMaximumLikelihood,
    ContrastiveLoss,
    NegativeEvidenceLowerBound,
)
from cnpe.models import AbstractNumpyroGuide, AbstractNumpyroModel


@pytest.fixture()
def model():
    class Model(AbstractNumpyroModel):
        reparameterized: bool | None = None
        observed_names = {"b"}
        reparam_names = set()

        def call_without_reparam(self, obs=None):
            a = sample("a", Normal(jnp.zeros((3,))))
            sample("b", Normal(a), obs=obs["b"] if obs is not None else None)

    return Model()


@pytest.fixture()
def guide():
    class Guide(AbstractNumpyroGuide):
        a_guide: AbstractDistribution

        def __call__(self, obs):
            sample("a", self.a_guide, condition=obs["b"])

    return Guide(
        masked_autoregressive_flow(
            key=jr.PRNGKey(0),
            base_dist=Normal(jnp.zeros((3,))),
            cond_dim=3,
        ),
    )


# TODO likely a more robust tests we can add - just check it runs


def test_maximum_likelihood_loss(model, guide):
    loss = AmortizedMaximumLikelihood(model.reparam(set_val=True))
    loss(*eqx.partition(guide, eqx.is_inexact_array), key=jr.PRNGKey(0))


def test_contrastive_loss(model, guide):
    loss = ContrastiveLoss(
        model=model.reparam(set_val=True),
        obs={"b": jnp.array(jnp.arange(3))},
        n_contrastive=5,
    )

    loss(*eqx.partition(guide, eqx.is_inexact_array), key=jr.PRNGKey(0))


def test_negative_elbo_loss(model, guide):
    loss = NegativeEvidenceLowerBound(
        model=model.reparam(set_val=True),
        obs={"b": jnp.array(jnp.arange(3))},
    )
    loss(*eqx.partition(guide, eqx.is_inexact_array), key=jr.PRNGKey(0))
