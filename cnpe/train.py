"""Modified from flowjax to allow aux."""

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from flowjax.wrappers import NonTrainable
from jaxtyping import PRNGKeyArray, PyTree
from tqdm import tqdm

from cnpe.losses import AbstractLoss


@eqx.filter_jit
def step(
    params: PyTree,
    static: PyTree,
    *args,
    optimizer: optax.GradientTransformation,
    opt_state: PyTree,
    loss_fn: AbstractLoss,
):
    """Carry out a training step.

    Args:
        params: Parameters for the model
        static: Static components of the model.
        *args: Arguments passed to the loss function.
        optimizer: Optax optimizer.
        opt_state: Optimizer state.
        loss_fn: The loss function. This should take params and static as the first two
            arguments.

    Returns:
        tuple: (params, opt_state, loss_val) or (params, opt_state, (loss_val, aux))
    """
    loss_val, grads = eqx.filter_value_and_grad(
        loss_fn,
        has_aux=loss_fn.has_aux,
    )(  # TODO Already filtered... do we need this?
        params,
        static,
        *args,
    )
    updates, opt_state = optimizer.update(grads, opt_state, params=params)
    params = eqx.apply_updates(params, updates)
    return params, opt_state, loss_val


def train(
    key: PRNGKeyArray,
    guide: Callable,
    loss_fn: AbstractLoss,
    *,
    steps: int,
    optimizer: optax.GradientTransformation,
    show_progress: bool = True,
) -> tuple:
    """Fit the model given the loss function.

    Args:
        key: Jax random key.
        guide: The guide (model over the latents) to fit.
        loss_fn: The loss function.
        steps: Maximum number of optimization steps.
        optimizer: The optax optimizer.
        show_progress: Whether to show a progress bar. Defaults to True.

    Returns:
        Tuple of ``(guide, losses)`` or ``(guide, losses, aux)`` if ``loss.has_aux``.
    """
    params, static = eqx.partition(
        guide,
        eqx.is_inexact_array,
        is_leaf=lambda leaf: isinstance(leaf, NonTrainable),
    )
    opt_state = optimizer.init(params)

    losses = []

    if loss_fn.has_aux:
        auxiliaries = []

    keys = tqdm(jr.split(key, steps), disable=not show_progress)

    for key in keys:
        params, opt_state, loss_val = step(
            params,
            static,
            key,
            optimizer=optimizer,
            opt_state=opt_state,
            loss_fn=loss_fn,
        )

        if loss_fn.has_aux:
            loss_val, aux = loss_val
            auxiliaries.append(aux)

        losses.append(loss_val.item())
        keys.set_postfix({"loss": loss_val.item()})

        if _is_converged(losses):
            keys.set_postfix_str(f"{keys.postfix} (Convergence criteria reached.)")
            break

    if loss_fn.has_aux:
        return eqx.combine(params, static), losses, auxiliaries
    meta_data = {"losses": jnp.array(losses), "converged": _is_converged(losses)}
    return eqx.combine(params, static), meta_data


def _is_converged(losses, window_size=300):
    # Compares median loss from indices [-2*n:-n] and [-n:] as a measure of convergence
    if len(losses) < 2 * window_size:
        return False

    losses = jnp.asarray(losses[-2 * window_size :])

    @jax.jit
    def _jit_is_converged(losses):
        a, b = jnp.split(losses, 2)
        return jnp.median(a) < jnp.median(b)

    return _jit_is_converged(losses)
