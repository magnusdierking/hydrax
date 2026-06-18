# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (uv preferred)
uv sync

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_pendulum.py

# Lint (check only)
uv run ruff check .

# Format + lint (auto-fix)
uv run ruff check --fix . && uv run ruff format .

# Run an example
uv run python examples/pendulum.py mppi
uv run python examples/pusht.py --warp cem
```

Ruff is configured at line-length=80, Google-style docstrings, with isort. The `N806` (uppercase variables in functions) and `PLR2004` (magic values) rules are disabled — matrices and math constants can use conventional uppercase names.

## Architecture

Hydrax is a **sampling-based MPC framework** that runs physics rollouts in parallel on GPU via JAX/MJX. The core loop: sample a batch of control trajectories → roll them out in parallel → compute costs → update the sampling distribution → repeat.

### Two base classes to understand first

**`Task` (`hydrax/task_base.py`)** — defines the optimal control problem:
- Wraps a MuJoCo model (`self.mj_model`) and its MJX counterpart (`self.model`)
- Subclasses must implement `running_cost(state, control)` and `terminal_cost(state)`
- Optional overrides: `domain_randomize_model(rng)` / `domain_randomize_data(rng)` for domain randomization, `control_mapper_mjx` / `control_mapper_mj` for non-trivial actuator mappings (e.g., task-space IK), `make_data()` for custom contact buffer sizing
- `trace_sites` kwarg in `__init__` specifies which MuJoCo sites to record for visualization

**`SamplingBasedController` (`hydrax/alg_base.py`)** — defines the MPC algorithm:
- Subclasses must implement `sample_knots(params) -> (knots, params)` and `update_params(params, rollouts) -> params`
- The base class handles: trajectory rollout via `jax.vmap` over samples, domain randomization via `jax.vmap` over model variants, risk aggregation, spline interpolation, and the optimize loop
- `SamplingParams` holds `(tk, mean, rng)`; algorithms extend this with their own fields (e.g., `cov`, `elites`)
- `Trajectory` holds `(controls, knots, costs, trace_sites)` — `rollouts.knots` is what `update_params` receives

### Control parameterization

Controls are **spline knots**, not raw action sequences. The base class interpolates knots → dense control trajectory before simulation. `spline_type` ∈ `{"zero", "linear", "cubic", "akima"}` and `num_knots` are constructor arguments on every algorithm. Spline utilities live in `hydrax/utils/spline.py`.

### Domain randomization

When `num_randomizations > 1`, the controller vmaps over a batch of randomized models. The risk strategy (`hydrax/risk.py`) reduces per-randomization costs to a scalar: `AverageCost` (default), `WorstCase`, `BestCase`, `CVaR`, etc.

### Simulation

- `run_interactive` (`hydrax/simulation/deterministic.py`) — synchronous MuJoCo viewer loop, used in all examples
- `run_asynchronous` (`hydrax/simulation/asynchronous.py`) — separate processes for controller and physics, better for wall-clock fidelity
- Both accept a controller, an `mj_model`, and `mj_data`; `run_interactive` accepts `show_traces`, `trace_idxs`, `max_traces` for visualization

### Adding a new task

1. Create `hydrax/tasks/my_task.py` subclassing `Task`
2. Add a MuJoCo XML model under `hydrax/models/my_task/`
3. Implement `running_cost` and `terminal_cost`; optionally override `domain_randomize_*` and `control_mapper_*`
4. Add an example in `examples/my_task.py`
5. Add a test in `tests/test_my_task.py` (see `tests/test_pendulum.py` for minimal pattern)

### Adding a new algorithm

1. Create `hydrax/algs/my_alg.py` subclassing `SamplingBasedController`
2. Define a `MyAlgParams(SamplingParams)` dataclass (flax struct) with extra fields
3. Implement `init_params`, `sample_knots`, `update_params`
4. Export from `hydrax/algs/__init__.py`

### MjWarp backend

Many examples accept `--warp` to switch from the default MJX (JAX) backend to the experimental MjWarp backend. Task `__init__` passes `impl` to `super().__init__()`.
