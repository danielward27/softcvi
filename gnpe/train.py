"""Modified from flowjax to allow aux"""

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax.random as jr
import optax
from flowjax.distributions import AbstractDistribution
from jax import Array
from tqdm import tqdm

PyTree = Any


@eqx.filter_jit
def step(
    params: PyTree,
    static: PyTree,
    *args,
    optimizer: optax.GradientTransformation,
    opt_state: PyTree,
    loss_fn: Callable,
    has_aux: bool = False,
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
        has_aux: Whether the loss function also returns auxilary data.

    Returns:
        tuple: (params, opt_state, loss_val) or (params, opt_state, (loss_val, aux))
    """
    loss_val, grads = eqx.filter_value_and_grad(loss_fn, has_aux=has_aux)(
        params,
        static,
        *args,
    )
    updates, opt_state = optimizer.update(grads, opt_state, params=params)
    params = eqx.apply_updates(params, updates)
    return params, opt_state, loss_val


def train(
    key: Array,
    dist: AbstractDistribution,
    loss_fn: Callable,
    *,
    steps: int = 100,
    learning_rate: float = 5e-4,
    optimizer: optax.GradientTransformation | None = None,
    filter_spec: Callable | PyTree = eqx.is_inexact_array,
    has_aux: bool = False,
    return_best: bool = True,
    show_progress: bool = True,
) -> tuple[AbstractDistribution, list]:
    """Train a distribution (e.g. a flow) by variational inference.

    Args:
        key: Jax PRNGKey.
        dist: Distribution object, trainable parameters are found using
            equinox.is_inexact_array.
        loss_fn: The loss function to optimize (e.g. the ElboLoss).
        steps: The number of training steps to run. Defaults to 100.
        learning_rate: Learning rate. Defaults to 5e-4.
        optimizer: Optax optimizer. If provided, this overrides the default Adam
            optimizer, and the learning_rate is ignored. Defaults to None.
        filter_spec: Equinox `filter_spec` for specifying trainable parameters. Either
            a callable `leaf -> bool`, or a PyTree with prefix structure matching `dist`
            with True/False values. Defaults to eqx.is_inexact_array.
        has_aux: Whether the loss function returns auxilary data to be stored, in
            addition to the loss value.
        return_best: Whether the result should use the parameters where the minimum loss
            was reached (when True), or the parameters after the last update (when
            False). Defaults to True.
        show_progress: Whether to show progress bar. Defaults to True.

    Returns:
        A tuple containing with (distribution, losses) is has_aux is False, and
            (distribution, losses, aux) if has_aux is True.
    """
    if optimizer is None:
        optimizer = optax.adam(learning_rate)

    params, static = eqx.partition(dist, filter_spec)
    opt_state = optimizer.init(params)

    losses = []

    if has_aux:
        auxiliaries = []

    best_params = params
    keys = tqdm(jr.split(key, steps), disable=not show_progress)

    for key in keys:
        params, opt_state, loss = step(
            params,
            static,
            key,
            optimizer=optimizer,
            opt_state=opt_state,
            loss_fn=loss_fn,
            has_aux=has_aux,
        )

        if has_aux:
            loss, aux = loss
            auxiliaries.append(aux)

        losses.append(loss.item())
        keys.set_postfix({"loss": loss.item()})
        if loss.item() == min(losses):
            best_params = params
    params = best_params if return_best else params

    if has_aux:
        return eqx.combine(params, static), losses, auxiliaries

    return eqx.combine(params, static), losses
