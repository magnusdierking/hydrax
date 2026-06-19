import colorsys
import os
import time
from typing import Sequence

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np

from hydrax import ROOT
from hydrax.alg_base import SamplingBasedController
from hydrax.utils.video import VideoRecorder


def _default_domain_palette(
    num_domains: int, alpha: float
) -> np.ndarray:
    """Evenly spaced hues for distinguishing per-domain traces."""
    palette = np.zeros((num_domains, 4), dtype=np.float32)
    for d in range(num_domains):
        r, g, b = colorsys.hsv_to_rgb(d / max(num_domains, 1), 0.8, 0.95)
        palette[d] = (r, g, b, alpha)
    return palette

"""
Tools for deterministic (synchronous) simulation, with the simulator and
controller running one after the other in the same thread.
"""


def run_interactive(  # noqa: PLR0912, PLR0915
    controller: SamplingBasedController,
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    frequency: float,
    initial_knots: jax.Array = None,
    fixed_camera_id: int = None,
    show_traces: bool = True,
    max_traces: int = None,
    trace_idxs: Sequence[int] = None,
    trace_width: float = 5.0,
    trace_color: Sequence = [1.0, 1.0, 1.0, 0.1],
    show_domain_traces: bool = False,
    domain_trace_colors: Sequence[Sequence[float]] = None,
    reference: np.ndarray = None,
    reference_fps: float = 30.0,
    record_video: bool = False,
) -> None:
    """Run an interactive simulation with the MPC controller.

    This is a deterministic simulation, with the controller and simulation
    running in the same thread. This is useful for repeatability, but is less
    realistic than asynchronous simulation.

    Note: the actual control frequency may be slightly different than what is
    requested, because the control period must be an integer multiple of the
    simulation time step.

    Args:
        controller: The controller instance, which includes the task
                    (e.g., model, cost) definition.
        mj_model: The MuJoCo model for the system to use for simulation. Could
                  be slightly different from the model used by the controller.
        mj_data: A MuJoCo data object containing the initial system state.
        frequency: The requested control frequency (Hz) for replanning.
        initial_knots: The initial knot points for the control spline at t=0
        fixed_camera_id: The camera ID to use for the fixed camera view.
        show_traces: Whether to show traces for the site positions.
        max_traces: The maximum number of traces to show at once.
        trace_idxs: The indices of the traces to show.
        trace_width: The width of the trace lines (in pixels).
        trace_color: The RGBA color of the trace lines. When
            ``show_domain_traces`` is False, applied to all traces. When True,
            its alpha is reused for the auto-generated per-domain palette.
        show_domain_traces: If True, draw one trace per (sample, domain) pair
            instead of just the first domain. Each domain gets a distinct
            color. Multiplies the number of rendered lines by
            ``controller.num_randomizations``.
        domain_trace_colors: Optional RGBA palette (one row per domain) used
            when ``show_domain_traces`` is True. Defaults to an HSV-spaced
            palette that reuses the alpha from ``trace_color``.
        reference: The reference trajectory (qs) to visualize.
        reference_fps: The frame rate of the reference trajectory.
        record_video: Whether to record a video of the simulation.
    """
    # Report the planning horizon in seconds for debugging
    print(
        f"Planning with {controller.ctrl_steps} steps "
        f"over a {controller.plan_horizon} second horizon "
        f"with {controller.num_knots} knots."
    )

    # Figure out how many sim steps to run before replanning
    replan_period = 1.0 / frequency
    sim_steps_per_replan = int(replan_period / mj_model.opt.timestep)
    sim_steps_per_replan = max(sim_steps_per_replan, 1)
    step_dt = sim_steps_per_replan * mj_model.opt.timestep
    actual_frequency = 1.0 / step_dt
    print(
        f"Planning at {actual_frequency} Hz, "
        f"simulating at {1.0 / mj_model.opt.timestep} Hz"
    )

    # Create a data structure for the controller to run rollouts from.
    num_envs = controller.num_samples * controller.num_randomizations
    mjx_data = controller.task.make_data(num_envs=num_envs)
    mjx_data = mjx_data.replace(
        qpos=jnp.array(mj_data.qpos, dtype=jnp.float32),
        qvel=jnp.array(mj_data.qvel, dtype=jnp.float32),
        mocap_pos=jnp.array(mj_data.mocap_pos, dtype=jnp.float32),
        mocap_quat=jnp.array(mj_data.mocap_quat, dtype=jnp.float32),
    )

    # Initialize the controller
    policy_params = controller.init_params(initial_knots=initial_knots)
    jit_optimize = jax.jit(controller.optimize)
    jit_interp_func = jax.jit(controller.interp_func)

    # Warm-up the controller
    print("Jitting the controller...")
    st = time.time()
    policy_params, rollouts = jit_optimize(mjx_data, policy_params)
    policy_params, rollouts = jit_optimize(mjx_data, policy_params)

    tq = jnp.arange(0, sim_steps_per_replan) * mj_model.opt.timestep
    tk = policy_params.tk
    knots = policy_params.mean[None, ...]
    _ = jit_interp_func(tq, tk, knots)
    _ = jit_interp_func(tq, tk, knots)
    print(f"Time to jit: {time.time() - st:.3f} seconds")

    num_rollouts = rollouts.controls.shape[1]
    if trace_idxs is not None and max_traces is not None:
        if max_traces != len(trace_idxs):
            raise ValueError(
                f"max_traces ({max_traces}) conflicts with "
                f"len(trace_idxs) ({len(trace_idxs)}). "
                "Set only one, or ensure they match."
            )
    if trace_idxs is not None:
        trace_idxs = [i for i in trace_idxs if i < num_rollouts]
    elif max_traces is not None and max_traces < num_rollouts:
        # Spread traces uniformly across the rollout batch.
        trace_idxs = np.linspace(
            0, num_rollouts - 1, max_traces, dtype=int
        ).tolist()
    else:
        trace_idxs = list(range(num_rollouts))
    num_traces = len(trace_idxs)

    # Per-domain trace setup.
    if show_domain_traces and rollouts.trace_sites_per_domain is None:
        raise RuntimeError(
            "show_domain_traces=True but the controller did not produce "
            "per-domain trace data. Use a controller with num_randomizations "
            ">= 1 routed through rollout_with_randomizations."
        )
    num_domains = (
        rollouts.trace_sites_per_domain.shape[0] if show_domain_traces else 1
    )
    if show_domain_traces:
        if domain_trace_colors is not None:
            domain_palette = np.asarray(domain_trace_colors, dtype=np.float32)
            if domain_palette.shape != (num_domains, 4):
                raise ValueError(
                    f"domain_trace_colors must have shape ({num_domains}, 4), "
                    f"got {domain_palette.shape}"
                )
        else:
            domain_palette = _default_domain_palette(
                num_domains, alpha=float(trace_color[3])
            )
    else:
        domain_palette = np.asarray(trace_color, dtype=np.float32)[None, :]

    # Ghost reference setup
    if reference is not None:
        ref_data = mujoco.MjData(mj_model)
        assert reference.shape[1] == mj_model.nq
        ref_data.qpos[:] = reference[0, :]
        mujoco.mj_forward(mj_model, ref_data)

        vopt = mujoco.MjvOption()
        vopt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True  # Transparent.
        pert = mujoco.MjvPerturb()
        catmask = mujoco.mjtCatBit.mjCAT_DYNAMIC  # only show dynamic bodies

    # Initialize video recording if enabled
    recorder = None
    if record_video:
        # Video dimensions
        width, height = 720, 480
        # Create the video recorder
        recorder = VideoRecorder(
            output_dir=os.path.join(ROOT, "recordings"),
            width=width,
            height=height,
            fps=actual_frequency,
        )
        # Ensure model visual offscreen buffer is compatible with video
        # recording
        mj_model.vis.global_.offwidth = width
        mj_model.vis.global_.offheight = height
        if not recorder.start():
            record_video = False
        renderer = mujoco.Renderer(mj_model, height=height, width=width)

    # Start the simulation
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        if fixed_camera_id is not None:
            # Set the custom camera
            viewer.cam.fixedcamid = fixed_camera_id
            viewer.cam.type = 2

        # Set up rollout traces
        if show_traces:
            num_trace_sites = len(controller.task.trace_site_ids)
            geom_idx = 0
            for _ in range(num_trace_sites):
                for d in range(num_domains):
                    rgba = domain_palette[d]
                    for _ in range(num_traces * controller.ctrl_steps):
                        mujoco.mjv_initGeom(
                            viewer.user_scn.geoms[geom_idx],
                            type=mujoco.mjtGeom.mjGEOM_LINE,
                            size=np.zeros(3),
                            pos=np.zeros(3),
                            mat=np.eye(3).flatten(),
                            rgba=np.asarray(rgba),
                        )
                        viewer.user_scn.ngeom += 1
                        geom_idx += 1

        # Add geometry for the ghost reference
        if reference is not None:
            mujoco.mjv_addGeoms(
                mj_model, ref_data, vopt, pert, catmask, viewer.user_scn
            )

        while viewer.is_running():
            start_time = time.time()

            # Set the start state for the controller
            mjx_data = mjx_data.replace(
                qpos=jnp.array(mj_data.qpos, dtype=jnp.float32),
                qvel=jnp.array(mj_data.qvel, dtype=jnp.float32),
                mocap_pos=jnp.array(mj_data.mocap_pos, dtype=jnp.float32),
                mocap_quat=jnp.array(mj_data.mocap_quat, dtype=jnp.float32),
                time=jnp.array(mj_data.time, dtype=jnp.float32),
            )

            # Do a replanning step
            plan_start = time.time()
            policy_params, rollouts = jit_optimize(mjx_data, policy_params)
            plan_time = time.time() - plan_start

            # Visualize the rollouts
            if show_traces:
                if show_domain_traces:
                    trace_data = rollouts.trace_sites_per_domain
                else:
                    trace_data = rollouts.trace_sites[None, ...]
                ii = 0
                for k in range(num_trace_sites):
                    for d in range(num_domains):
                        for i in trace_idxs:
                            for j in range(controller.ctrl_steps):
                                mujoco.mjv_connector(
                                    viewer.user_scn.geoms[ii],
                                    mujoco.mjtGeom.mjGEOM_LINE,
                                    trace_width,
                                    trace_data[d, i, j, k],
                                    trace_data[d, i, j + 1, k],
                                )
                                ii += 1

            # Update the ghost reference
            if reference is not None:
                t_ref = mj_data.time * reference_fps
                i_ref = int(t_ref)
                i_ref = min(i_ref, reference.shape[0] - 1)
                ref_data.qpos[:] = reference[i_ref]
                mujoco.mj_forward(mj_model, ref_data)
                mujoco.mjv_updateScene(
                    mj_model,
                    ref_data,
                    vopt,
                    pert,
                    viewer.cam,
                    catmask,
                    viewer.user_scn,
                )

            # query the control spline at the sim frequency
            # (we assume the sim freq is the same as the low-level ctrl freq)
            sim_dt = mj_model.opt.timestep
            t_curr = mj_data.time

            tq = jnp.arange(0, sim_steps_per_replan) * sim_dt + t_curr
            tk = policy_params.tk
            # print("Knots: ", policy_params.mean)
            knots = policy_params.mean[None, ...]
            us = np.asarray(jit_interp_func(tq, tk, knots))[0]  # (ss, nu)

            # simulate the system between spline replanning steps
            for i in range(sim_steps_per_replan):
                # print("Control: ", us[i])
                # print(
                #     "Control mapped: ",
                #     controller.task.control_mapper_mj(mj_data, us[i]),
                # )
                mj_data.ctrl[:] = np.array(
                    controller.task.control_mapper_mj(mj_data, us[i])
                )
                mujoco.mj_step(mj_model, mj_data)
                viewer.sync()

                # Capture frame if recording
                if record_video and recorder.is_recording:
                    renderer.update_scene(mj_data, viewer.cam)
                    frame = renderer.render()
                    recorder.add_frame(frame.tobytes())

            # Try to run in roughly realtime
            elapsed = time.time() - start_time
            if elapsed < step_dt:
                time.sleep(step_dt - elapsed)

            # Print some timing information
            rtr = step_dt / (time.time() - start_time)
            print(
                f"Realtime rate: {rtr:.2f}, plan time: {plan_time:.4f}s",
                end="\r",
            )

    # Preserve the last printout
    print("")

    # Close the video recorder if recording was enabled
    if record_video and recorder is not None:
        recorder.stop()
