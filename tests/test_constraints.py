import jax.numpy as jnp
import pytest
from jax import vmap

from gnpe.constraints import Interval


def test_interval():
    test_x = jnp.linspace(-0.9, 0.9)
    constraint = Interval(-1, 1)
    result1 = vmap(constraint.bijection.transform)(test_x)
    assert pytest.approx(result1) == jnp.arctanh(test_x)

    # Test invariance to affine transform of samples and interval
    loc, scale = 7, 0.1
    constraint = Interval(-1 * scale + loc, 1 * scale + loc)
    result2 = vmap(constraint.bijection.transform)(test_x * scale + loc)
    assert pytest.approx(result2, rel=1e-3) == result1
