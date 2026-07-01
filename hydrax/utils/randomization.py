"""Deterministic per-domain parameter overrides for sampling-based MPC.

Companion to ``Task.domain_randomize_model``: that hook samples random
parameters per domain via ``jax.vmap``, which means it can't see a domain
index and so can't apply caller-supplied per-domain values directly. This
module compiles a user-friendly spec into a batched override dict and folds
it into an existing controller's ``model`` / ``randomized_axes``.

Usage::

    ctrl = PredictiveSampling(task, num_randomizations=3, ...)
    apply_randomization_spec(ctrl, {
        "geom": {
            "block_geom": {
                "friction": (0, jnp.array([0.4, 0.6, 0.8])),
            },
        },
    })

Body-mass overrides are deliberately rejected (see ``NotImplementedError``):
writing ``body_mass`` directly leaves derived quantities such as
``body_inertia`` / ``body_invweight0`` inconsistent, and reconstructing them
requires re-running ``MjSpec.compile`` per domain.
"""

from typing import Any, Dict, Tuple, Union

import jax
import jax.numpy as jnp
import mujoco

from hydrax.alg_base import SamplingBasedController

# Friendly name -> MJX model field name, organized by entity kind.
# Extend on demand; intentionally minimal to keep validation strict.
_ALIASES: Dict[str, Dict[str, str]] = {
    "geom": {
        "friction": "geom_friction",
        "solref": "geom_solref",
        "solimp": "geom_solimp",
        "margin": "geom_margin",
    },
    "body": {},  # mass intentionally absent — raised explicitly below
    # Joint fields are DOF-indexed; the joint name resolves to its DOF
    # address before staging (see below). Only valid for joints with a
    # single DOF (slide/hinge).
    "joint": {
        "frictionloss": "dof_frictionloss",
        "damping": "dof_damping",
        "armature": "dof_armature",
    },
}

# Body-level friendly names that resolve to a geom_* field applied across
# every geom attached to the named body. Lets ``body.friction`` express
# "set this on the whole body" without enumerating each child geom.
_BODY_TO_GEOM_FIELDS: Dict[str, str] = {
    "friction": "geom_friction",
    "solref": "geom_solref",
    "solimp": "geom_solimp",
    "margin": "geom_margin",
}

# MuJoCo object kinds that resolve names via mj_name2id.
_MJ_OBJ = {
    "geom": mujoco.mjtObj.mjOBJ_GEOM,
    "body": mujoco.mjtObj.mjOBJ_BODY,
    "joint": mujoco.mjtObj.mjOBJ_JOINT,
}

# Joint types with exactly one DOF, so a joint id maps to a single DOF index.
_SINGLE_DOF_JOINT_TYPES = {
    int(mujoco.mjtJoint.mjJNT_SLIDE),
    int(mujoco.mjtJoint.mjJNT_HINGE),
}

# Entries declared here trigger a clear NotImplementedError rather than a
# silent write that would leave derived quantities stale.
_DERIVED_QUANTITY_BLOCKLIST = {
    ("body", "mass"),
}

Entry = Union[jax.Array, Tuple[int, jax.Array]]


def apply_randomization_spec(
    ctrl: SamplingBasedController,
    spec: Dict[str, Dict[str, Dict[str, Entry]]],
) -> SamplingBasedController:
    """Fold a per-domain parameter spec into an already-built controller.

    Mutates ``ctrl.model`` and ``ctrl.randomized_axes`` in place so the
    randomized fields are picked up by ``rollout_with_randomizations``.

    Args:
        ctrl: A controller built with ``num_randomizations > 1``.
        spec: Three-level dict ``{kind: {entity_name: {field: entry}}}``.
            ``entry`` is either a full ndarray of shape
            ``(num_randomizations, *field_shape[1:])`` or a tuple
            ``(local_idx, values)`` where ``values`` has shape
            ``(num_randomizations,)``.

    Returns:
        The same controller, for chaining.

    Raises:
        ValueError: spec or value shape is invalid, or num_randomizations <= 1.
        KeyError: unknown kind or friendly field name.
        NotImplementedError: ``body.mass`` (derived quantities not recomputed).
    """
    N = ctrl.num_randomizations
    if N <= 1:
        raise ValueError(
            "apply_randomization_spec requires num_randomizations > 1; "
            f"got {N}. The spec defines per-domain values."
        )

    mj_model = ctrl.task.mj_model
    overrides: Dict[str, jax.Array] = {}

    for kind, entities in spec.items():
        if kind not in _ALIASES:
            raise KeyError(
                f"Unknown randomization kind {kind!r}. "
                f"Supported: {sorted(_ALIASES)}."
            )
        alias = _ALIASES[kind]
        for entity_name, fields in entities.items():
            entity_id = mujoco.mj_name2id(mj_model, _MJ_OBJ[kind], entity_name)
            if entity_id < 0:
                raise ValueError(
                    f"{kind} {entity_name!r} not found in the MuJoCo model."
                )
            for friendly_name, entry in fields.items():
                _stage_field(
                    overrides,
                    ctrl,
                    kind,
                    alias,
                    entity_name,
                    entity_id,
                    friendly_name,
                    entry,
                    N,
                )

    if not overrides:
        return ctrl

    ctrl.model = ctrl.model.tree_replace(overrides)
    if ctrl.randomized_axes is None:
        ctrl.randomized_axes = jax.tree.map(lambda x: None, ctrl.task.model)
    ctrl.randomized_axes = ctrl.randomized_axes.tree_replace(
        {f: 0 for f in overrides}
    )
    return ctrl


def _normalize_entry(
    entry: Entry, mjx_field: str, num_randomizations: int
) -> Tuple[Union[int, None], jax.Array]:
    """Return ``(local_idx_or_None, values_as_jax_array)``."""
    if isinstance(entry, tuple):
        if len(entry) != 2:
            raise ValueError(
                f"{mjx_field}: tuple entry must be (local_idx, values), "
                f"got tuple of length {len(entry)}."
            )
        local_idx, values = entry
        if not isinstance(local_idx, (int, jnp.integer)):
            raise ValueError(
                f"{mjx_field}: local_idx must be an int, got {type(local_idx)}."
            )
        values = jnp.asarray(values)
        if values.ndim != 1:
            raise ValueError(
                f"{mjx_field}: values for (local_idx, values) form must be "
                f"1D, got shape {values.shape}."
            )
        local_idx = int(local_idx)
    else:
        local_idx = None
        values = jnp.asarray(entry)

    if values.shape[0] != num_randomizations:
        raise ValueError(
            f"{mjx_field}: leading dim must equal num_randomizations="
            f"{num_randomizations}, got {values.shape[0]}."
        )
    return local_idx, values


def _stage_override(
    overrides: Dict[str, jax.Array],
    ctrl: SamplingBasedController,
    mjx_field: str,
    entity_id: int,
    entry: Entry,
    num_randomizations: int,
) -> None:
    """Normalize ``entry`` and write it into ``overrides[mjx_field]``."""
    local_idx, values = _normalize_entry(entry, mjx_field, num_randomizations)
    if mjx_field in overrides:
        # Already tiled by an earlier entry in this call.
        tiled = overrides[mjx_field]
    else:
        current = getattr(ctrl.model, mjx_field)
        tiled = _tile_if_unbatched(
            current, num_randomizations, ctrl.randomized_axes, mjx_field
        )
    _validate_shape(tiled, mjx_field, entity_id, local_idx, values)
    if local_idx is None:
        tiled = tiled.at[:, entity_id].set(values)
    else:
        tiled = tiled.at[:, entity_id, local_idx].set(values)
    overrides[mjx_field] = tiled


def _stage_field(
    overrides: Dict[str, jax.Array],
    ctrl: SamplingBasedController,
    kind: str,
    alias: Dict[str, str],
    entity_name: str,
    entity_id: int,
    friendly_name: str,
    entry: Entry,
    num_randomizations: int,
) -> None:
    """Resolve one ``field`` of one entity and stage its per-domain override."""
    if (kind, friendly_name) in _DERIVED_QUANTITY_BLOCKLIST:
        raise NotImplementedError(
            f"Randomizing {kind}.{friendly_name} is not supported "
            "yet: writing it directly leaves derived quantities "
            "(e.g. body_inertia, body_invweight0) inconsistent. "
            "Recompiling MjSpec per domain is needed."
        )
    # Body-scoped geom fields expand to one write per child geom.
    if kind == "body" and friendly_name in _BODY_TO_GEOM_FIELDS:
        mjx_field = _BODY_TO_GEOM_FIELDS[friendly_name]
        geom_ids = _child_geom_ids(ctrl.task.mj_model, entity_id)
        if not geom_ids:
            raise ValueError(
                f"body {entity_name!r} has no child geoms to "
                f"randomize {friendly_name!r} on."
            )
        for gid in geom_ids:
            _stage_override(
                overrides, ctrl, mjx_field, gid, entry, num_randomizations
            )
        return
    if friendly_name not in alias:
        raise KeyError(
            f"Unknown {kind} field {friendly_name!r}. "
            f"Known: {sorted(alias)}."
        )
    mjx_field = alias[friendly_name]
    target_id = _resolve_target_id(
        ctrl.task.mj_model, kind, entity_id, entity_name
    )
    _stage_override(
        overrides, ctrl, mjx_field, target_id, entry, num_randomizations
    )


def _resolve_target_id(
    mj_model: mujoco.MjModel,
    kind: str,
    entity_id: int,
    entity_name: str,
) -> int:
    """Map an entity id to the index used by its MJX field.

    For ``joint`` kinds the aliased fields are DOF-indexed, so the joint id is
    resolved to its DOF address; only single-DOF joints (slide/hinge) are
    valid, since multi-DOF joints (ball/free) span several DOFs. All other
    kinds index by the entity id directly.
    """
    if kind != "joint":
        return entity_id
    if int(mj_model.jnt_type[entity_id]) not in _SINGLE_DOF_JOINT_TYPES:
        raise ValueError(
            f"joint {entity_name!r} is not single-DOF (slide/hinge); only "
            "those can be randomized via the 'joint' kind."
        )
    return int(mj_model.jnt_dofadr[entity_id])


def _child_geom_ids(mj_model: mujoco.MjModel, body_id: int) -> list:
    """Return the geom ids attached to ``body_id``."""
    start = int(mj_model.body_geomadr[body_id])
    count = int(mj_model.body_geomnum[body_id])
    return list(range(start, start + count))


def _tile_if_unbatched(
    field: jax.Array,
    num_randomizations: int,
    randomized_axes: Any,
    mjx_field: str,
) -> jax.Array:
    """Add a leading randomization axis if the field lacks one."""
    already_batched = (
        randomized_axes is not None
        and getattr(randomized_axes, mjx_field, None) == 0
    )
    if already_batched:
        return field
    reps = (num_randomizations,) + (1,) * field.ndim
    return jnp.tile(field[None], reps)


def _validate_shape(
    tiled: jax.Array,
    mjx_field: str,
    entity_id: int,
    local_idx: Union[int, None],
    values: jax.Array,
) -> None:
    """Check that the override fits at the target slot."""
    if entity_id >= tiled.shape[1]:
        raise ValueError(
            f"{mjx_field}: entity index {entity_id} out of range "
            f"(field has {tiled.shape[1]} entries)."
        )
    if local_idx is None:
        expected = tiled.shape[2:]
        if values.shape[1:] != expected:
            raise ValueError(
                f"{mjx_field}: trailing shape mismatch, expected {expected}, "
                f"got {values.shape[1:]}."
            )
    else:
        if tiled.ndim < 3:
            raise ValueError(
                f"{mjx_field}: local_idx form requires a per-entity vector "
                f"field, but shape is {tiled.shape}."
            )
        if local_idx >= tiled.shape[2]:
            raise ValueError(
                f"{mjx_field}: local_idx {local_idx} out of range "
                f"(field has {tiled.shape[2]} components per entity)."
            )
