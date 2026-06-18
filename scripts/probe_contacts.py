"""Probe the per-environment contact/constraint budget for each task.

Run with:
    uv run python scripts/probe_contacts.py

For each task this runs a short CPU-side MuJoCo simulation with random controls
and reports the peak mj_data.ncon and mj_data.nefc values, which map to
nconmax/naconmax and njmax respectively. A safety factor is applied to give
recommended values.
"""

import sys
from pathlib import Path

import mujoco
import numpy as np

# Make sure hydrax is importable from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from hydrax.tasks.bugtrap import BugTrap, Bugtrap  # noqa: E402
from hydrax.tasks.cart_pole import CartPole  # noqa: E402
from hydrax.tasks.crane import Crane  # noqa: E402
from hydrax.tasks.cube import CubeRotation  # noqa: E402
from hydrax.tasks.double_cart_pole import DoubleCartPole  # noqa: E402
from hydrax.tasks.humanoid_mocap import HumanoidMocap  # noqa: E402
from hydrax.tasks.humanoid_standup import HumanoidStandup  # noqa: E402
from hydrax.tasks.particle import Particle  # noqa: E402
from hydrax.tasks.pendulum import Pendulum  # noqa: E402
from hydrax.tasks.pusht import PushT  # noqa: E402
from hydrax.tasks.pusht_fr3 import PushTFr3  # noqa: E402
from hydrax.tasks.walker import Walker  # noqa: E402


def probe(task_instance, num_steps: int = 1000, safety_factor: float = 3.0):
    """Run a random CPU simulation and return peak contact counts."""
    model = task_instance.mj_model
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    rng = np.random.default_rng(42)
    max_ncon = 0
    max_nefc = 0

    # Check if there are bounded actuators; fall back to [-1, 1] if not
    ctrl_lo = np.where(
        model.actuator_ctrllimited.astype(bool),
        model.actuator_ctrlrange[:, 0],
        -1.0,
    )
    ctrl_hi = np.where(
        model.actuator_ctrllimited.astype(bool),
        model.actuator_ctrlrange[:, 1],
        1.0,
    )

    for _ in range(num_steps):
        data.ctrl[:] = rng.uniform(ctrl_lo, ctrl_hi)
        mujoco.mj_step(model, data)
        max_ncon = max(max_ncon, data.ncon)
        max_nefc = max(max_nefc, data.nefc)

    rec_ncon = int(np.ceil(max_ncon * safety_factor))
    rec_nac = rec_ncon  # naconmax <= nconmax
    rec_nj = int(np.ceil(max_nefc * safety_factor))

    return max_ncon, max_nefc, rec_ncon, rec_nac, rec_nj


TASKS = [
    ("Pendulum",        Pendulum),
    ("CartPole",        CartPole),
    ("DoubleCartPole",  DoubleCartPole),
    ("Particle",        Particle),
    ("Crane",           Crane),
    ("Walker",          Walker),
    ("HumanoidStandup", HumanoidStandup),
    ("PushT",           PushT),
    ("CubeRotation",    CubeRotation),
    ("Bugtrap",         Bugtrap),
    ("BugTrap(SDF)",    BugTrap),
    ("PushTFr3",        PushTFr3),
]

# HumanoidMocap requires motion-capture data; skip for now
SKIP = {"HumanoidMocap"}

fmt = "{:<20} {:>8} {:>8} {:>10} {:>10} {:>10}"
print(fmt.format("Task", "ncon", "nefc", "ncon_per_env", "nac_per_env", "nj_per_env"))
print("-" * 72)

for name, TaskCls in TASKS:
    if name in SKIP:
        print(fmt.format(name, "-", "-", "SKIP", "SKIP", "SKIP"))
        continue
    try:
        task = TaskCls()
    except Exception as e:
        print(fmt.format(name, "ERR", "ERR", str(e)[:30], "", ""))
        continue

    peak_ncon, peak_nefc, rec_ncon, rec_nac, rec_nj = probe(task)

    no_contacts = rec_ncon == 0
    print(fmt.format(
        name,
        peak_ncon,
        peak_nefc,
        rec_ncon if not no_contacts else "0 (skip)",
        rec_nac if not no_contacts else "0 (skip)",
        rec_nj if not no_contacts else "0 (skip)",
    ))
