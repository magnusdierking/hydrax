import argparse

import jax
import mujoco
from evosax.algorithms.distribution_based import (
    CMA_ES,
    GradientlessDescent,
    Open_ES,
    RandomSearch,
    SimulatedAnnealing,
    xNES,
)

from hydrax.algs import CEM, DIAL, MPPI, MTP, Evosax, PredictiveSampling
from hydrax.risk import WorstCase
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.bugtrap import BugTrap, Bugtrap

"""
Run an interactive simulation of the bug-trap navigation task.

Double click on the green target, then drag it around with [ctrl + right-click].
The starting pointmass is placed inside a U-shaped barrier that creates a
local minimum for purely local samplers.

Two task variants are available:
  --task bugtrap   Contact-physics Bugtrap (default)
  --task sdf       SDF-wall BugTrap (particle_navigation model)
"""

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Run an interactive simulation of the bug-trap task."
)
parser.add_argument(
    "--warp",
    action="store_true",
    help="Whether to use the (experimental) MjWarp backend. (default: False)",
    required=False,
)
parser.add_argument(
    "--task",
    choices=["bugtrap", "sdf"],
    default="bugtrap",
    help="Task variant: 'bugtrap' (contact physics) or 'sdf' (SDF walls). (default: bugtrap)",
)
subparsers = parser.add_subparsers(
    dest="algorithm", help="Sampling algorithm (choose one)"
)
subparsers.add_parser("ps", help="Predictive Sampling")
subparsers.add_parser("mppi", help="Model Predictive Path Integral Control")
subparsers.add_parser("cem", help="Cross-Entropy Method")
subparsers.add_parser("mtp", help="Model Tensor Planning")
subparsers.add_parser("cmaes", help="CMA-ES")
subparsers.add_parser("openes", help="OpenAI-ES")
subparsers.add_parser("sa", help="Simulated Annealing")
subparsers.add_parser("xnes", help="Exponential Natural Evolution Strategy")
subparsers.add_parser("gld", help="Gradient-Less Descent")
subparsers.add_parser("rs", help="Uniform Random Search")
subparsers.add_parser(
    "dial", help="Diffusion-Inspired Annealing for Legged MPC (DIAL)"
)
args = parser.parse_args()

# Define the task (cost and dynamics)
if args.task == "sdf":
    task = BugTrap(impl="warp" if args.warp else "jax")
else:
    task = Bugtrap(impl="warp" if args.warp else "jax")

# Set the controller based on command-line arguments
if args.algorithm == "ps" or args.algorithm is None:
    print("Running predictive sampling")
    ctrl = PredictiveSampling(
        task,
        num_samples=16,
        noise_level=0.1,
        num_randomizations=10,
        risk_strategy=WorstCase(),
        plan_horizon=0.25,
        spline_type="zero",
        num_knots=11,
    )

elif args.algorithm == "mppi":
    print("Running MPPI")
    ctrl = MPPI(
        task,
        num_samples=16,
        noise_level=0.3,
        temperature=0.01,
        plan_horizon=0.25,
        spline_type="zero",
        num_knots=11,
    )

elif args.algorithm == "cem":
    print("Running CEM")
    ctrl = CEM(
        task,
        num_samples=32,
        num_elites=8,
        sigma_start=0.3,
        sigma_min=0.05,
        explore_fraction=0.5,
        plan_horizon=0.25,
        spline_type="zero",
        num_knots=11,
    )

elif args.algorithm == "mtp":
    print("Running MTP")
    ctrl = MTP(
        task,
        num_samples=32,
        m_pts=4,
        num_elites=1,
        sigma_start=0.7,
        sigma_min=0.5,
        sigma_max=1.0,
        beta=1.0,
        alpha=0.1,
        mtp_interpolation="akima",
        plan_horizon=1.0,
        spline_type="zero",
        num_knots=11,
    )

elif args.algorithm == "cmaes":
    print("Running CMA-ES")
    ctrl = Evosax(
        task,
        CMA_ES,
        num_samples=16,
        plan_horizon=0.25,
        spline_type="zero",
        num_knots=11,
    )

elif args.algorithm == "gld":
    print("Running Gradient-Less Descent (GLD)")
    ctrl = Evosax(
        task,
        GradientlessDescent,
        num_samples=16,
        plan_horizon=0.25,
        spline_type="zero",
        num_knots=11,
    )

elif args.algorithm == "openes":
    print("Running OpenAI-ES")
    ctrl = Evosax(
        task,
        Open_ES,
        num_samples=16,
        plan_horizon=0.25,
        spline_type="zero",
        num_knots=11,
    )

elif args.algorithm == "sa":
    print("Running Simulated Annealing")
    ctrl = Evosax(
        task,
        SimulatedAnnealing,
        num_samples=16,
        plan_horizon=0.25,
        spline_type="zero",
        num_knots=11,
    )

elif args.algorithm == "xnes":
    print("Running Exponential Natural Evolution Strategy")
    ctrl = Evosax(
        task,
        xNES,
        num_samples=16,
        plan_horizon=0.25,
        spline_type="zero",
        num_knots=11,
    )

elif args.algorithm == "rs":
    print("Running uniform random search")
    sampling_fn = lambda key: jax.random.uniform(
        key, shape=(11 * 2), minval=-1.0, maxval=1.0
    )
    ctrl = Evosax(
        task,
        RandomSearch,
        sampling_fn=sampling_fn,
        num_samples=16,
        plan_horizon=0.25,
        spline_type="zero",
        num_knots=11,
    )

elif args.algorithm == "dial":
    print("Running Diffusion-Inspired Annealing for Legged MPC (DIAL)")
    ctrl = DIAL(
        task,
        num_samples=16,
        noise_level=0.4,
        beta_opt_iter=1.0,
        beta_horizon=1.0,
        temperature=0.001,
        plan_horizon=0.25,
        spline_type="zero",
        num_knots=11,
        iterations=5,
    )
else:
    parser.error("Invalid algorithm")

# Define the model used for simulation
mj_model = task.mj_model
mj_data = mujoco.MjData(mj_model)

if args.task == "sdf":
    mj_data.qpos[:2] = [-0.15, 0.0]
    mj_data.mocap_pos[0] = [0.25, 0.0, 0.01]

# Run the interactive simulation
run_interactive(
    ctrl,
    mj_model,
    mj_data,
    frequency=50,
    show_traces=False,
    max_traces=5,
)
