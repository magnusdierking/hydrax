"""Quaternion convention utilities for converting between MuJoCo and SciPy/JAX.

MuJoCo uses [w, x, y, z] ordering, while SciPy and JAX Rotation use [x, y, z, w].
"""

import jax.numpy as jnp


def quat_mj_to_scipy(q: jnp.ndarray) -> jnp.ndarray:
    """Convert a quaternion from MuJoCo [w, x, y, z] to SciPy [x, y, z, w]."""
    return jnp.array([q[1], q[2], q[3], q[0]])


def quat_scipy_to_mj(q: jnp.ndarray) -> jnp.ndarray:
    """Convert a quaternion from SciPy [x, y, z, w] to MuJoCo [w, x, y, z]."""
    return jnp.array([q[3], q[0], q[1], q[2]])
