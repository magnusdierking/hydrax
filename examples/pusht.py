import argparse
from copy import deepcopy

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

from hydrax.algs import CEM, DIAL, MPPI, Evosax, PredictiveSampling
from hydrax.risk import WorstCase
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.pusht import PushT

"""
Run an interactive simulation of the push-T task.

Double click on the green target, then drag it around with [ctrl + right-click].
"""

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Run an interactive simulation of the push-T task."
)
parser.add_argument(
    "--warp",
    action="store_true",
    help="Whether to use the (experimental) MjWarp backend. (default: False)",
)
subparsers = parser.add_subparsers(
    dest="algorithm", help="Sampling algorithm (choose one)"
)
subparsers.add_parser("ps", help="Predictive Sampling")
subparsers.add_parser("mppi", help="Model Predictive Path Integral Control")
subparsers.add_parser("cem", help="Cross-Entropy Method")
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

# ============================================================
# Global options — shared across all algorithms
# ============================================================
NUM_SAMPLES = 64
NUM_RANDOMIZATIONS = 4
PLAN_HORIZON = 0.5
SPLINE_TYPE = "zero"
NUM_KNOTS = 6

# Shared keyword arguments passed to every controller
shared_kwargs = dict(
    num_samples=NUM_SAMPLES,
    num_randomizations=NUM_RANDOMIZATIONS,
    plan_horizon=PLAN_HORIZON,
    spline_type=SPLINE_TYPE,
    num_knots=NUM_KNOTS,
)


# Define the task (cost and dynamics)
task = PushT(impl="warp" if args.warp else "jax")

# Set the controller based on command-line arguments
if args.algorithm == "ps" or args.algorithm is None:
    print("Running predictive sampling")
    ctrl = PredictiveSampling(
        task,
        noise_level=0.4,
        **shared_kwargs,
    )

elif args.algorithm == "mppi":
    print("Running MPPI")
    ctrl = MPPI(
        task,
        noise_level=0.3,
        temperature=0.01,
        **shared_kwargs,
    )

elif args.algorithm == "cem":
    print("Running CEM")
    ctrl = CEM(
        task,
        num_elites=8,
        sigma_start=0.3,
        sigma_min=0.05,
        explore_fraction=0.5,
        **shared_kwargs,
    )

elif args.algorithm == "cmaes":
    print("Running CMA-ES")
    ctrl = Evosax(
        task,
        CMA_ES,
        **shared_kwargs,
    )

elif args.algorithm == "gld":
    print("Running Gradient-Less Descent (GLD)")
    ctrl = Evosax(
        task,
        GradientlessDescent,
        **shared_kwargs,
    )

elif args.algorithm == "openes":
    print("Running OpenAI-ES")
    ctrl = Evosax(
        task,
        Open_ES,
        **shared_kwargs,
    )

elif args.algorithm == "sa":
    print("Running Simulated Annealing")
    ctrl = Evosax(
        task,
        SimulatedAnnealing,
        **shared_kwargs,
    )

elif args.algorithm == "xnes":
    print("Running Exponential Natural Evolution Strategy")
    ctrl = Evosax(
        task,
        xNES,
        **shared_kwargs,
    )

elif args.algorithm == "rs":
    print("Running uniform random search")
    sampling_fn = lambda key: jax.random.uniform(
        key, shape=(NUM_KNOTS * task.mj_model.nu,), minval=-1.0, maxval=1.0
    )
    ctrl = Evosax(
        task,
        RandomSearch,
        sampling_fn=sampling_fn,
        **shared_kwargs,
    )

elif args.algorithm == "dial":
    print("Running Diffusion-Inspired Annealing for Legged MPC (DIAL)")
    ctrl = DIAL(
        task,
        noise_level=0.4,
        beta_opt_iter=1.0,
        beta_horizon=1.0,
        temperature=0.001,
        iterations=5,
        **shared_kwargs,
    )
else:
    parser.error("Invalid algorithm")

# Define the model used for simulation
mj_model = deepcopy(task.mj_model)
mj_model.opt.timestep = 0.001
mj_model.opt.iterations = 100
mj_model.opt.ls_iterations = 50
mj_data = mujoco.MjData(mj_model)
mj_data.qpos = [0.1, 0.1, 1.3, 0.0, 0.0]

# Run the interactive simulation
run_interactive(
    ctrl,
    mj_model,
    mj_data,
    frequency=50,
    show_traces=False,
)
