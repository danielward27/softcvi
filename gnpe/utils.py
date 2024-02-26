import warnings
from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
from flowjax.distributions import AbstractDistribution
from jax import Array
from jax.flatten_util import ravel_pytree


# TODO test
def drop_nan_and_warn(*arrays, axis: int):

    shapes_except_last = [a.shape[:-1] for a in arrays]
    assert all(s == shapes_except_last[0] for s in shapes_except_last)

    is_finite = [jnp.isfinite(a).all(axis=-1).squeeze() for a in arrays]
    is_finite = jnp.stack(is_finite, axis=-1).all(axis=-1)

    if not jnp.all(is_finite):
        warnings.warn(
            f"Dropping {(~is_finite).sum()} nan or inf values.",
            stacklevel=1,
        )
        arrays = [jnp.compress(is_finite, a, axis) for a in arrays]
    return arrays


class MLPParameterizedDistribution(AbstractDistribution):
    constructor: Callable
    mlp: eqx.nn.MLP
    shape: tuple[int, ...]
    cond_shape: tuple[int, ...]

    def __init__(
        self,
        key: Array,
        distribution: AbstractDistribution,
        *,
        cond_dim: int,
        width_size,
        depth: int = 1,
        **kwargs,
    ):
        if distribution.cond_shape is not None:
            raise ValueError("Expected unconditional distribution.")

        def get_constructor_and_num_params(distribution):
            params, static = eqx.partition(distribution, eqx.is_inexact_array)
            init, unravel = ravel_pytree(params)

            def constructor(ravelled_params: Array):
                params = unravel(ravelled_params + init)
                return eqx.combine(params, static)

            return constructor, len(init)

        self.constructor, num_params = get_constructor_and_num_params(distribution)
        self.mlp = eqx.nn.MLP(
            cond_dim,
            out_size=num_params,
            width_size=width_size,
            depth=depth,
            key=key,
            **kwargs,
        )
        self.shape = distribution.shape
        self.cond_shape = (cond_dim,)

    def _sample(self, key, condition):
        dist = self.constructor(self.mlp(condition))
        return dist.sample(key)

    def _log_prob(self, x, condition):
        dist = self.constructor(self.mlp(condition))
        return dist.log_prob(x)
