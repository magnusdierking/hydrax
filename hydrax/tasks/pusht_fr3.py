from functools import partial
from typing import Dict, Sequence

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from jax import lax
from jax.scipy.spatial.transform import Rotation as JRotation
from mujoco import mjx
from scipy.spatial.transform import Rotation as R

from hydrax import ROOT
from hydrax.task_base import Task
from hydrax.utils.quaternion import quat_mj_to_scipy


class PushTFr3(Task):
    """Push a T-shaped block to a desired pose."""

    def __init__(
        self,
        impl: str = "warp",  # default warp because mjx is unstable
        trace_sites: Sequence[str] | None = ["T_1", "T_2", "ee_site", "T_3"],
        actuation_type: str = "velocity",  # 'velocity' or 'position'
        sampling_space: str = "task",  # 'task' or 'joint'
        max_lin_vel: float = 0.15,
        manipulation_type: str = "free",  # 'free' or 'joint'
    ):
        """Load the MuJoCo model and set task parameters."""
        self.sampling_space = sampling_space
        self.actuation_type = actuation_type

        if self.sampling_space == "task":
            nu = 2
            ctrl_limits: Dict[str, jax.Array] = {
                "u_min": jnp.array([-max_lin_vel, -max_lin_vel]),
                "u_max": jnp.array([max_lin_vel, max_lin_vel]),
            }
        elif self.sampling_space == "joint":
            nu = None  # default in parent
            ctrl_limits = None
        else:
            raise ValueError("'sampling_space' must be 'task' or 'joint'")

        model_path = ROOT + "/models/pusht_fr3"
        if actuation_type == "position":
            model_path += "/pos_ctrl"
        elif actuation_type == "velocity":
            model_path += "/vel_ctrl"
        else:
            raise ValueError(
                "'actuation_type' must be 'position' or 'velocity'"
            )

        self.manipulation_type = manipulation_type
        if self.manipulation_type == "free":
            model_path += "/free"
        elif self.manipulation_type == "joint":
            model_path += "/joint"
        else:
            raise ValueError("'manipulation_type' must be 'free' or 'joint'")

        model_path += "/scene.xml"
        mj_model = mujoco.MjModel.from_xml_path(model_path)

        super().__init__(
            impl=impl,
            mj_model=mj_model,
            trace_sites=trace_sites,
            nu=nu,
            ctrl_limits=ctrl_limits,
            ncon_per_env=24,
            nac_per_env=24,
            nj_per_env=99,
        )

        # Get sensor ids
        self.block_position_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "position_world"
        )
        self.block_orientation_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "orientation_world"
        )
        self.goal_position_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "goal_position_world"
        )
        self.goal_orientation_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "goal_orientation_world"
        )

        self.ee_position_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "ee_frame_pos"
        )
        self.ee_orientation_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "ee_frame_quat"
        )
        self.block_global_position_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "position_world"
        )
        self.ee_goal_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "safety"
        )
        self.ee_t1_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "ee_t1"
        )
        self.ee_t2_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "ee_t2"
        )
        self.ee_t3_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "ee_t3"
        )
        # Get actuator joint indices
        self.actuator_joint_names = [
            "fr3_joint1",
            "fr3_joint2",
            "fr3_joint3",
            "fr3_joint4",
            "fr3_joint5",
            "fr3_joint6",
            "fr3_joint7",
        ]
        self.actuator_joint_ids = [
            mj_model.joint(name).id for name in self.actuator_joint_names
        ]
        self.actuator_joint_idxs = self.mj_model.jnt_qposadr[
            self.actuator_joint_ids
        ]
        self.dof_adr = self.mj_model.jnt_dofadr[self.actuator_joint_ids]

        # Cache sensor addresses (avoiding mj_name2id in the loop)
        self.ee_pos_adr = self.mj_model.sensor_adr[self.ee_position_sensor]
        self.ee_quat_adr = self.mj_model.sensor_adr[self.ee_orientation_sensor]

        # Cache keyframe qpos for nullspace
        key_name = "home"
        key_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_KEY, key_name
        )
        if key_id != -1:
            # Pre-slice the home position for only relevant joints
            self.q_home = self.mj_model.key_qpos[key_id][
                self.actuator_joint_idxs
            ]
        else:
            self.q_home = None

        # Pre-allocate geometric Jacobians for sim-sim ik
        self.jacp = np.zeros(
            (3, self.model.nv), dtype=np.float64
        )  # translational
        self.jacr = np.zeros((3, self.model.nv), dtype=np.float64)  # rotational

        # Get T id
        self.T_bid = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_BODY, "block"
        )

        # Get block joint indices
        if self.manipulation_type == "joint":
            self.T_joint_names = ["T_x", "T_y", "T_z"]
        elif self.manipulation_type == "free":
            self.T_joint_names = ["T"]

        self.T_joint_idxs = [
            mj_model.joint(name).id for name in self.T_joint_names
        ]

        self.joint_limits = self.mj_model.jnt_range[self.actuator_joint_ids]

        # special to this task
        self.ee_body_id = self.mj_model.body("ee_frame").id
        self.goal_quat_block = jnp.array([1.0, 0.0, 0.0, 0.0])  # [w, x, y, z]
        # initial end effector
        self.goal_quat_ee = jnp.array(
            [0.0, 0.7071, 0.7071, 0.0]
        )  # [w, x, y, z]
        self.goal_pos_ee = jnp.array([0.3, 0.0, 0.045])

    ##################################
    ##       Goal Error Terms       ##
    ##################################

    def _get_ee_position(self, state: mjx.Data) -> jax.Array:
        """Get the end effector position."""
        # get from body position
        ee_pos = state.xpos[self.ee_body_id]
        return ee_pos

    def _get_position_err(self, state: mjx.Data) -> jax.Array:
        """Get the position error of the block relative to a goal position."""
        sensor_adr = self.model.sensor_adr[self.block_position_sensor]
        goal_adr = self.model.sensor_adr[self.goal_position_sensor]
        error = (
            state.sensordata[sensor_adr : sensor_adr + 3]
            - state.sensordata[goal_adr : goal_adr + 3]
        )
        return error

    def _get_orientation_err(self, state: mjx.Data) -> jax.Array:
        """Computes the minimum angle required to rotate from current to goal."""
        # MuJoCo sensors provide [w, x, y, z].
        # JAX Rotation (and SciPy) expects [x, y, z, w].
        s_idx = self.model.sensor_adr[self.block_orientation_sensor]
        g_idx = self.model.sensor_adr[self.goal_orientation_sensor]

        # Slice and reorder from MuJoCo [w,x,y,z] to SciPy [x,y,z,w]
        q_curr = quat_mj_to_scipy(state.sensordata[s_idx : s_idx + 4])
        q_goal = quat_mj_to_scipy(state.sensordata[g_idx : g_idx + 4])

        # Initialize Rotation objects
        r_curr = JRotation.from_quat(q_curr)
        r_goal = JRotation.from_quat(q_goal)

        # Compute relative rotation: ΔR = R_curr⁻¹ * R_goal
        relative_rot = r_curr.inv() * r_goal

        # angle θ in [0, π]
        return relative_rot.magnitude()

    ##################################
    ##      End Effector Terms      ##
    ##################################

    def _get_T_attractor(self, state: mjx.Data) -> jax.Array:
        """Attraction term for end-effector based on sites."""
        ee_t1_adr = self.model.sensor_adr[self.ee_t1_sensor]
        ee_t1_pos = state.sensordata[ee_t1_adr : ee_t1_adr + 3]
        ee_t2_adr = self.model.sensor_adr[self.ee_t2_sensor]
        ee_t2_pos = state.sensordata[ee_t2_adr : ee_t2_adr + 3]
        ee_t3_adr = self.model.sensor_adr[self.ee_t3_sensor]
        ee_t3_pos = state.sensordata[ee_t3_adr : ee_t3_adr + 3]
        error = (
            jnp.linalg.norm(ee_t1_pos, ord=2)
            + jnp.linalg.norm(ee_t2_pos, ord=2)
            + jnp.linalg.norm(ee_t3_pos, ord=2)
        )
        return error

    def _safety_zone_cost(self, state: mjx.Data) -> jax.Array:
        """Get a cost based on the distance between the end effector and the goal."""
        sensor_adr = self.model.sensor_adr[self.ee_goal_sensor]
        distance = state.sensordata[sensor_adr : sensor_adr + 3]
        distance = jnp.linalg.norm(distance)
        cost = jnp.where(distance > 0.45, 1.0, 0.0)
        return cost

    def running_cost(
        self, state: mjx.Data, control: jax.Array = None
    ) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ).

        Args:
            state: The current state xₜ.
            control: The control action uₜ.

        Returns:
            The scalar running cost ℓ(xₜ, uₜ)
        """
        position_err = self._get_position_err(state)
        orientation_err = self._get_orientation_err(state)

        position_cost = jnp.linalg.norm(position_err)
        orientation_cost = jnp.linalg.norm(orientation_err)

        safety_cost = self._safety_zone_cost(state)
        attractor_cost = self._get_T_attractor(state)

        if self.manipulation_type == "joint":
            error = (
                30 * position_cost
                + 3 * orientation_cost
                + 0.005 * attractor_cost
            )
        elif self.manipulation_type == "free":
            error = (
                30 * position_cost
                + 3 * orientation_cost
                + 0.005 * attractor_cost
            )

        error += safety_cost
        return error

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        return 10 * self.running_cost(state, jnp.zeros(self.model.nu))

    def domain_randomize_model(self, rng: jax.Array) -> Dict[str, jax.Array]:
        return {}


    @partial(jax.jit, static_argnums=(0,))
    def control_mapper_mjx(self, state: mjx.Data, u: jax.Array) -> jax.Array:
        """Map controls from sampling space to actuator space (MJX version).

        Maps end effector velocity (x, y) to joint velocity. z is set to keep
        the end effector at the same height. Angular velocity is set to zero.

        NOTE: This method is used inside the JITed rollouts and needs to be
        JIT compatible
        """
        if self.actuation_type == "position":
            raise NotImplementedError("Position control not implemented yet")
        elif self.actuation_type == "velocity":
            sensor_adr_pos = self.model.sensor_adr[self.ee_position_sensor]
            ee_pos = state.sensordata[sensor_adr_pos : sensor_adr_pos + 3]
            sensor_adr = self.model.sensor_adr[self.ee_orientation_sensor]
            ee_quat = state.sensordata[sensor_adr : sensor_adr + 4]

            jacp, jacr = mjx.jac(
                self.model,
                state,
                jnp.zeros_like(
                    ee_pos
                ),  # NOTE centre of the end effector sphere
                jnp.array(self.ee_body_id),
            )  # (NV, 3) each
            J_full = jnp.hstack([jacp, jacr])  # (NV, 6)
            J = J_full[self.dof_adr, :].T  # (6, 7) based on 7 actuated joints

            goal_quat = jnp.array(
                [0.0, 0.7071, 0.7071, 0.0]
            )  # force ee to point down
            e_rot = mjx._src.math.quat_sub(goal_quat, ee_quat)  # (3,)
            twist_err = jnp.array(
                [
                    u[0],
                    u[1],
                    0.045 - ee_pos[2],  # Z-height regulation
                    e_rot[0],
                    e_rot[1],
                    e_rot[2],
                ]
            )  # [ex, ey, ez, ewx, ewy, ewz]
            reg = 1e-3 * jnp.eye(6)
            dq = J.T @ jnp.linalg.solve(J @ J.T + reg, twist_err)
            dq = jnp.clip(dq, self.act_min, self.act_max)

        return dq

    ##################################
    ##      Initial-state setup     ##
    ##################################

    def _solve_ik(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        q_seed: np.ndarray | None = None,
        max_iters: int = 200,
        pos_tol: float = 1e-4,
        rot_tol: float = 1e-3,
        step_scale: float = 0.5,
        damping: float = 1e-3,
    ) -> tuple[np.ndarray, bool]:
        """Iterative damped least-squares IK for the EE pose.

        Args:
            target_pos: Desired EE position in world frame, shape (3,).
            target_quat: Desired EE quaternion in MuJoCo [w, x, y, z].
            q_seed: Initial guess for the 7 FR3 joints. Defaults to q_home.
            max_iters: Maximum solver iterations.
            pos_tol: Position-error convergence threshold (m).
            rot_tol: Rotation-error convergence threshold (rad).
            step_scale: IK update step size; values < 1 trade speed for
                stability.
            damping: Damped least-squares regularizer.

        Returns:
            (q_solution, converged) where q_solution is the 7-vector of joint
            angles and converged indicates whether tolerances were met.
        """
        if q_seed is None:
            if self.q_home is None:
                raise RuntimeError(
                    "No q_seed provided and no 'home' keyframe in the model."
                )
            q_seed = np.asarray(self.q_home, dtype=np.float64)
        q = np.asarray(q_seed, dtype=np.float64).copy()

        scratch = mujoco.MjData(self.mj_model)
        r_target = R.from_quat(
            [target_quat[1], target_quat[2], target_quat[3], target_quat[0]]
        )
        q_min = self.joint_limits[:, 0]
        q_max = self.joint_limits[:, 1]
        reg = damping * np.eye(6)

        converged = False
        for _ in range(max_iters):
            scratch.qpos[self.actuator_joint_idxs] = q
            mujoco.mj_forward(self.mj_model, scratch)

            ee_pos = scratch.xpos[self.ee_body_id]
            ee_quat = scratch.sensordata[
                self.ee_quat_adr : self.ee_quat_adr + 4
            ]
            r_curr = R.from_quat(
                [ee_quat[1], ee_quat[2], ee_quat[3], ee_quat[0]]
            )
            pos_err = target_pos - ee_pos
            rot_err = (r_target * r_curr.inv()).as_rotvec()

            if (
                np.linalg.norm(pos_err) < pos_tol
                and np.linalg.norm(rot_err) < rot_tol
            ):
                converged = True
                break

            mujoco.mj_jacBody(
                self.mj_model, scratch, self.jacp, self.jacr, self.ee_body_id
            )
            J = np.vstack([self.jacp, self.jacr])[:, self.dof_adr]
            twist = np.concatenate([pos_err, rot_err])
            dq = step_scale * J.T @ np.linalg.solve(J @ J.T + reg, twist)
            q = np.clip(q + dq, q_min, q_max)

        return q, converged

    def set_initial_state(
        self,
        mj_data: mujoco.MjData,
        ee_pos: np.ndarray | None = None,
        ee_quat: np.ndarray | None = None,
        T_xy: np.ndarray | None = None,
        T_yaw: float | None = None,
        q_seed: np.ndarray | None = None,
    ) -> None:
        """Override the robot and/or T pose in ``mj_data`` in place.

        Designed as an overlay on top of a keyframe load: pass only the
        elements you want to override.

        Args:
            mj_data: MjData to mutate. Caller is expected to have populated
                qpos (e.g., from the 'home' keyframe) before this call.
            ee_pos: Desired EE position (m, world). Triggers IK on the FR3
                joints when provided.
            ee_quat: Desired EE quaternion in MuJoCo [w, x, y, z]. If only
                ``ee_pos`` is given, defaults to the canonical point-down
                quaternion used by this task.
            T_xy: Desired T-block planar position (x, y) in meters. Z is left
                at the keyframe value.
            T_yaw: Desired T-block yaw around world Z (rad). Supported for
                both ``manipulation_type='free'`` (encoded as a quaternion
                on the free joint) and ``'joint'`` (written to the T_z
                hinge).
            q_seed: Optional IK seed (7,). Defaults to ``self.q_home``.
        """
        if ee_pos is not None:
            ee_pos = np.asarray(ee_pos, dtype=np.float64)
            if ee_quat is None:
                ee_quat = np.asarray(self.goal_quat_ee, dtype=np.float64)
            else:
                ee_quat = np.asarray(ee_quat, dtype=np.float64)
            q_sol, converged = self._solve_ik(
                ee_pos, ee_quat, q_seed=q_seed
            )
            if not converged:
                print(
                    "WARNING: PushTFr3 IK did not converge for "
                    f"ee_pos={ee_pos.tolist()}; using best-effort solution."
                )
            mj_data.qpos[self.actuator_joint_idxs] = q_sol
        elif ee_quat is not None:
            raise ValueError(
                "ee_quat was provided without ee_pos; specify ee_pos as well."
            )

        if T_xy is not None or T_yaw is not None:
            if self.manipulation_type == "free":
                T_adr = int(self.mj_model.jnt_qposadr[self.T_joint_idxs[0]])
                if T_xy is not None:
                    mj_data.qpos[T_adr : T_adr + 2] = np.asarray(
                        T_xy, dtype=np.float64
                    )
                if T_yaw is not None:
                    half = 0.5 * float(T_yaw)
                    mj_data.qpos[T_adr + 3 : T_adr + 7] = np.array(
                        [np.cos(half), 0.0, 0.0, np.sin(half)],
                        dtype=np.float64,
                    )
            else:  # 'joint'
                # T_x, T_y are slides; T_z is a hinge holding the yaw angle.
                if T_xy is not None:
                    T_xy_addrs = self.mj_model.jnt_qposadr[
                        self.T_joint_idxs[:2]
                    ]
                    mj_data.qpos[T_xy_addrs] = np.asarray(
                        T_xy, dtype=np.float64
                    )
                if T_yaw is not None:
                    T_yaw_adr = int(
                        self.mj_model.jnt_qposadr[self.T_joint_idxs[2]]
                    )
                    mj_data.qpos[T_yaw_adr] = float(T_yaw)

        mujoco.mj_forward(self.mj_model, mj_data)

    def control_mapper_mj(
        self,
        state: mujoco.MjData,
        u: jnp.ndarray,
    ) -> jnp.ndarray:
        """Map controls from sampling space to actuator space (MuJoCo version).

        Maps end effector velocity (x, y) to joint velocity. z is set to keep
        the end effector at the same height. Angular velocity is set to zero.

        NOTE: This method is used in the sim to sim scripts
        """
        # 1. Fill pre-allocated Jacobian buffers
        mujoco.mj_jacBody(
            self.mj_model, state, self.jacp, self.jacr, self.ee_body_id
        )

        # 2. Slice Jacobian for actuated DOFs
        # Stack p and r, then slice columns corresponding to fr3 joints
        J = np.vstack([self.jacp, self.jacr])[:, self.dof_adr]

        # 3. Calculate Errors
        ee_pos = state.sensordata[self.ee_pos_adr : self.ee_pos_adr + 3]
        ee_quat = state.sensordata[self.ee_quat_adr : self.ee_quat_adr + 4]

        # Orientation error via SciPy (converting MuJoCo wxyz -> SciPy xyzw)
        r_curr = R.from_quat([ee_quat[1], ee_quat[2], ee_quat[3], ee_quat[0]])
        r_goal = R.from_quat(
            [
                self.goal_quat_ee[1],
                self.goal_quat_ee[2],
                self.goal_quat_ee[3],
                self.goal_quat_ee[0],
            ]
        )

        # Error rotation in world frame: r_goal * inv(r_curr)
        # as_rotvec returns the axis-angle vector (3,)
        error_rotvec = (r_goal * r_curr.inv()).as_rotvec()

        # 4. Construct Twist: [vx, vy, vz, wx, wy, wz]
        twist = np.array(
            [
                u[0],
                u[1],
                0.045 - ee_pos[2],  # Z-height regulation
                error_rotvec[0],
                error_rotvec[1],
                error_rotvec[2],
            ]
        )  # [ex, ey, ez, ewx, ewy, ewz]

        reg = 1e-3 * np.eye(6)
        dq = J.T @ np.linalg.solve(J @ J.T + reg, twist)

        return dq
