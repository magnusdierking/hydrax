from typing import Dict

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax import ROOT
from hydrax.task_base import Task


class Bugtrap(Task):
    """Extends particle task with a local minimum (the "bugtrap").

    This prevents the mass from reaching the target.
    """

    def __init__(self, impl: str = "jax"):
        """Load the MuJoCo model and set task parameters."""
        mj_model = mujoco.MjModel.from_xml_path(
            ROOT + "/models/bugtrap/scene.xml"
        )

        super().__init__(
            mj_model,
            trace_sites=["pointmass"],
            impl=impl,
        )

        self.pointmass_id = mj_model.site("pointmass").id
        self.pointmass_body_id = mj_model.site("pointmass").bodyid[0]

        self.sid = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "sphere_force"
        )
        self.sid_touch = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "sphere_touch"
        )
        self.adr = mj_model.sensor_adr[self.sid]
        self.adr_touch = mj_model.sensor_adr[self.sid_touch]
        self.goal_id = mj_model.body_mocapid[mj_model.body("goal").id]
        self.actuator_joint_idxs = mj_model.actuator_trnid[:, 0]

    def contact_force_cost(self, state: mjx.Data) -> jax.Array:
        """Binary cost for contact (external forces only)."""
        contact = jnp.linalg.norm(state.cfrc_ext[self.pointmass_body_id])
        contact = jnp.where(contact > 0.0, 1.0, 0.0)
        return contact

    def goal_cost(self, state: mjx.Data) -> jax.Array:
        """Distance to goal (1-norm)."""
        position_cost = jnp.linalg.norm(
            state.site_xpos[self.pointmass_id] - state.mocap_pos[self.goal_id]
        )
        return position_cost

    def running_cost(
        self, state: mjx.Data, control: jax.Array = None
    ) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ) encourages target tracking."""
        contact_cost = self.contact_force_cost(state)
        position_cost = self.goal_cost(state)
        running_cost = 5 * contact_cost + 0.1 * jnp.square(position_cost)

        return running_cost

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T)."""
        return 1 * self.running_cost(state, jnp.zeros(self.mj_model.nu))

    def domain_randomize_model(self, rng: jax.Array) -> Dict[str, jax.Array]:
        """Randomly perturb the actuator gains."""
        multiplier = jax.random.uniform(
            rng,
            self.model.actuator_gainprm[:, 0].shape,
            minval=0.9,
            maxval=1.1,
        )
        new_gains = self.model.actuator_gainprm[:, 0] * multiplier
        new_gains = self.model.actuator_gainprm.at[:, 0].set(new_gains)
        return {"actuator_gainprm": new_gains}

    def domain_randomize_data(
        self, data: mjx.Data, rng: jax.Array
    ) -> Dict[str, jax.Array]:
        """Randomly shift the measured particle position."""
        shift = jax.random.uniform(rng, (2,), minval=-0.0001, maxval=0.0001)
        return {"qpos": data.qpos + shift}


class BugTrap(Task):
    """A planar point mass that must navigate around U-shaped inner walls.

    Uses particle_navigation model with an SDF-based wall proximity cost,
    creating a differentiable local minimum that requires global exploration.
    """

    def __init__(
        self,
        impl: str = "jax",
        wall_weight: float = 5.0,
        wall_sharpness: float = 50.0,
        position_weight: float = 5.0,
        velocity_weight: float = 0.1,
        control_weight: float = 0.1,
    ) -> None:
        """Load the MuJoCo model and set task parameters.

        Args:
            impl: Which backend implementation to use.
            wall_weight: Multiplier on the exponential wall proximity cost.
            wall_sharpness: Decay rate in the ``exp(-sharpness * dist)`` wall cost.
            position_weight: Weight on the squared position tracking error.
            velocity_weight: Weight on the squared velocity penalty.
            control_weight: Weight on the squared control penalty.
        """
        mj_model = mujoco.MjModel.from_xml_path(
            ROOT + "/models/particle_navigation/scene.xml"
        )
        super().__init__(mj_model, trace_sites=["pointmass"], impl=impl)

        self.pointmass_id = mj_model.site("pointmass").id

        self.wall_weight = wall_weight
        self.wall_sharpness = wall_sharpness
        self.position_weight = position_weight
        self.velocity_weight = velocity_weight
        self.control_weight = control_weight

        # Cache planar geometry of the three inner walls for the SDF-style cost.
        self._wall_pos = jnp.array(
            [
                mj_model.geom("wall_ix").pos[:2],
                mj_model.geom("wall_iy").pos[:2],
                mj_model.geom("wall_neg_iy").pos[:2],
            ]
        )
        self._wall_size = jnp.array(
            [
                mj_model.geom("wall_ix").size[:2],
                mj_model.geom("wall_iy").size[:2],
                mj_model.geom("wall_neg_iy").size[:2],
            ]
        )

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ): wall SDF + tracking + control."""
        pos = state.site_xpos[self.pointmass_id][None, :2]
        wall_dist = jnp.abs(pos - self._wall_pos) - self._wall_size
        outside_dist = jnp.maximum(wall_dist, 0.0)
        inside_dist = jnp.minimum(jnp.max(wall_dist, axis=-1), 0.0)
        dist = (
            jnp.sqrt(jnp.sum(jnp.square(outside_dist), axis=-1) + 1e-12)
            + inside_dist
        ).min(axis=-1)
        wall_cost = self.wall_weight * jnp.exp(-self.wall_sharpness * dist)
        control_cost = jnp.sum(jnp.square(control))
        return wall_cost + self.terminal_cost(state) + self.control_weight * control_cost

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T): position tracking + velocity reg."""
        position_cost = jnp.sum(
            jnp.square(state.site_xpos[self.pointmass_id] - state.mocap_pos[0])
        )
        velocity_cost = jnp.sum(jnp.square(state.qvel))
        return self.position_weight * position_cost + self.velocity_weight * velocity_cost

    def domain_randomize_model(self, rng: jax.Array) -> Dict[str, jax.Array]:
        """Randomly perturb the actuator gains by ±10%."""
        multiplier = jax.random.uniform(
            rng,
            self.model.actuator_gainprm[:, 0].shape,
            minval=0.9,
            maxval=1.1,
        )
        new_gains = self.model.actuator_gainprm[:, 0] * multiplier
        new_gains = self.model.actuator_gainprm.at[:, 0].set(new_gains)
        return {"actuator_gainprm": new_gains}

    def domain_randomize_data(
        self, data: mjx.Data, rng: jax.Array
    ) -> Dict[str, jax.Array]:
        """Randomly shift the measured particle position."""
        shift = jax.random.uniform(rng, (2,), minval=-0.01, maxval=0.01)
        return {"qpos": data.qpos + shift}

    def make_data(self) -> mjx.Data:
        """Create a new state with enough contact slots for MjWarp."""
        return super().make_data(nconmax=10, naconmax=100)
