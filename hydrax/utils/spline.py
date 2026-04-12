from functools import partial
from typing import Callable, Literal

import jax
import jax.numpy as jnp
from interpax import Akima1DInterpolator, interp1d
from jax import vmap

InterpMethodType = Literal["zero", "linear", "cubic", "akima"]
InterpFuncType = Callable[[jax.Array, jax.Array, jax.Array], jax.Array]


"""We define the interpolation functions here so they're picklable for async."""


@partial(vmap, in_axes=(None, None, 0))
def interp_zero(tq: jax.Array, tk: jax.Array, knots: jax.Array) -> jax.Array:
    """Zero-order spline interpolation."""
    # for a zero-order spline, take the "next" knot as the control
    # ex: tq = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    #     tk = [0.0, 0.25, 0.5]
    #     inds = [0, 0, 0, 1, 1, 2]  # searchsorted trick does this
    #     interp_func(tq, tk, knots) = knots[:, inds]
    return knots[jnp.searchsorted(tk, tq, side="right") - 1]


@partial(vmap, in_axes=(None, None, 0))
def interp_linear(tq: jax.Array, tk: jax.Array, knots: jax.Array) -> jax.Array:
    """Linear spline interpolation."""
    return interp1d(tq, tk, knots, method="linear", extrap=True)


@partial(vmap, in_axes=(None, None, 0))
def interp_cubic(tq: jax.Array, tk: jax.Array, knots: jax.Array) -> jax.Array:
    """Cubic spline interpolation."""
    return interp1d(tq, tk, knots, method="cubic2", extrap=True)


def get_interp_func(method: InterpMethodType) -> InterpFuncType:
    """Get the 1D interpolation function based on the specified method.

    In particular, the function will have signature
        u_traj = interp_func(tq, tk, knots),
    where
        * tq is a 1D array of query times of shape (H,)
        * tk is a 1D array of knot times of shape (num_knots,),
        * knots is an array of shape (num_rollouts, num_knots), and
        * u_traj is the batch of interpolated trajectories of shape
            (num_rollouts, H).
    Here, we expect H to be the number of control time steps over some horizon T
    in seconds.

    Args:
        method: The interpolation method to use. Can be "zero", "linear",
            "cubic", or "akima".

    Returns:
        interp_func: The interpolation function.
    """
    if method == "zero":
        interp_func = interp_zero
    elif method == "linear":
        interp_func = interp_linear
    elif method == "cubic":
        interp_func = interp_cubic
    elif method == "akima":
        interp_func = interp_akima
    else:
        raise ValueError(
            f"Unknown interpolation method: {method}. "
            "Expected one of ['zero', 'linear', 'cubic', 'akima']."
        )
    return interp_func


# ---------------------------------------------------------------------------
# B-spline utilities for MTP
# ---------------------------------------------------------------------------


def clamped_knot_vector(
    num_ctrl_points: int,
    degree: int,
    dtype: jnp.dtype = jnp.float32,
) -> jnp.ndarray:
    """Create a clamped (open) B-spline knot vector on [0, 1].

    The first and last ``degree + 1`` knots are repeated so the spline
    interpolates the first and last control points exactly.

    Args:
        num_ctrl_points: Number of control points.
        degree: B-spline degree (≥ 1).
        dtype: Array dtype.

    Returns:
        Knot vector of length ``num_ctrl_points + degree + 1``.
    """
    n_internal = num_ctrl_points - degree
    start = jnp.zeros(degree + 1, dtype=dtype)
    end = jnp.ones(degree + 1, dtype=dtype)
    internal = jnp.arange(1, n_internal, dtype=dtype) / n_internal
    return jnp.concatenate([start, internal, end])


@partial(jax.jit, static_argnums=(1, 2, 3))
def compute_bspline_basis(
    knots: jax.Array,
    degree: int,
    num_points: int,
    dtype: jnp.dtype = jnp.float32,
) -> jax.Array:
    """Compute the B-spline basis matrix via Cox–de Boor recursion.

    Args:
        knots: The knot vector of shape ``(num_ctrl_points + degree + 1,)``.
        degree: B-spline degree.
        num_points: Number of uniformly spaced query points in the active
            parameter domain ``[knots[degree], knots[-1-degree]]``.
        dtype: Output dtype.

    Returns:
        Basis matrix ``B`` of shape ``(num_points, num_ctrl_points)`` such that
        ``B @ control_points`` yields the evaluated spline values.
    """
    knots = jnp.asarray(knots, dtype=dtype)
    one = jnp.array(1.0, dtype=dtype)
    zero = jnp.array(0.0, dtype=dtype)

    # Query points in the active domain, excluding the start
    t = jnp.linspace(
        knots[degree], knots[-1 - degree],
        num_points + 1, dtype=dtype,
    )[1:]

    # Degree-0 basis: N_{i,0}(t) = 1 if knots[i] <= t < knots[i+1]
    b = jnp.where(
        (knots[:-1] <= t[:, None]) & (t[:, None] < knots[1:]),
        one,
        zero,
    )

    # Cox–de Boor recursion
    for d in range(1, degree + 1):
        left_hi, left_lo = knots[d:-1], knots[: -d - 1]
        b_left = jnp.where(
            left_hi > left_lo,
            ((t[:, None] - left_lo) / (left_hi - left_lo)) * b[:, :-1],
            zero,
        )
        right_hi, right_lo = knots[d + 1 :], knots[1:-d]
        b_right = jnp.where(
            right_hi > right_lo,
            ((right_hi - t[:, None]) / (right_hi - right_lo)) * b[:, 1:],
            zero,
        )
        b = b_left + b_right

    # Fix the last row: the rightmost basis function should be 1 at t = 1
    last = b.shape[0] - 1
    b = b.at[last, :].set(0.0).at[last, -1].set(1.0)

    return b
