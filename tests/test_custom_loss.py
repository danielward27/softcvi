import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from flowjax.distributions import AbstractDistribution, Normal
from flowjax.experimental.numpyro import sample
from flowjax.flows import masked_autoregressive_flow

from gnpe.losses import ContrastiveLoss


def test_contrastive_loss():

    def model(observations=None):
        a = sample("a", Normal(jnp.zeros((3,))))
        sample("b", Normal(a), obs=observations)

    class Guide(eqx.Module):
        a_guide: AbstractDistribution

        def __call__(self, observations):
            sample("a", self.a_guide, condition=observations)

    guide = Guide(
        masked_autoregressive_flow(
            key=jr.PRNGKey(0),
            base_dist=Normal(jnp.zeros((3,))),
            cond_dim=3,
        ),
    )

    loss = ContrastiveLoss(
        model=model,
        obs=jnp.array(jnp.arange(3)),
        obs_name="b",
    )

    loss(*eqx.partition(guide, eqx.is_inexact_array), key=jr.PRNGKey(0))

    # TODO test in some way?
