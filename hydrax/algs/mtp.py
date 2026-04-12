"""Model Tensor Planning (MTP).

A hybrid CEM/MPPI controller that samples trajectories from both a random
tensor graph (MTP samples) and Gaussian perturbations around the mean (MPPI
samples).  MTP graph samples use fewer control points for coarse exploration,
projected to the shared knot space via a B-spline basis matrix.  The base
class spline interpolation handles the final knot-to-trajectory mapping.
"""

from typing import Literal, Tuple

import jax
import jax.numpy as jnp
from flax.struct import dataclass

from hydrax.alg_base import SamplingBasedController, SamplingParams, Trajectory
from hydrax.risk import RiskStrategy
from hydrax.task_base import Task
from hydrax.utils.spline import clamped_knot_vector, compute_bspline_basis


@dataclass
class MTPParams(SamplingParams):
    """Policy parameters for Model Tensor Planning.

    Attributes:
        tk: The knot times of the control spline.
        mean: The mean of the control spline knot distribution.
        rng: The pseudo-random number generator key.
        cov: Diagonal standard deviation for the Gaussian branch.
        elites: The best control knot sequences from the previous iteration.
        beta: The fraction of samples allocated to MTP.
    """

    cov: jax.Array
    elites: jax.Array
    beta: jax.Array


class MTP(SamplingBasedController):
    """Model Tensor Planning — a hybrid CEM/MPPI sampling controller.

    MTP splits the sample budget into three groups:

    1. **Previous best** — the current mean is always included as a sample.
    2. **MTP (tensor-graph) samples** — ``M+1`` coarse control points are
       drawn from a random graph and projected to ``num_knots`` via a
       B-spline basis matrix.  Fewer control points = more explorative.
    3. **MPPI (Gaussian) samples** — Gaussian noise around the mean,
       operating directly in the ``num_knots`` space for local refinement.

    Elite samples from the previous iteration can optionally be injected.
    """

    def __init__(
        self,
        task: Task,
        num_samples: int,
        # MTP graph parameters
        num_layers: int = 3,
        nodes_per_layer: int = 50,
        # B-spline parameters
        degree: int = 2,
        # CEM/MPPI parameters
        num_elites: int = 5,
        sigma_start: float = 0.5,
        sigma_min: float = 0.1,
        sigma_max: float = 1.0,
        temperature: float = 0.1,
        beta: float = 0.1,
        alpha: float = 0.5,
        # General controller args
        num_randomizations: int = 1,
        risk_strategy: RiskStrategy = None,
        seed: int = 0,
        plan_horizon: float = 1.0,
        spline_type: Literal["zero", "linear", "cubic"] = "linear",
        num_knots: int = 8,
        iterations: int = 1,
        # Elites
        keep_elites: int = 1,
    ) -> None:
        """Initialize the MTP controller.

        Args:
            task: The dynamics and cost for the system we want to control.
            num_samples: Total number of control sequences to sample.
            num_layers: Number of layers in the MTP tensor graph (M).
                Graph samples have ``M + 1`` coarse control points
                (anchor + M layer picks), which are projected to
                ``num_knots`` via B-spline interpolation.
            nodes_per_layer: Number of random nodes per layer (N).
            degree: B-spline degree (≥ 2) for the graph → knots projection.
            num_elites: Number of elites used for the CEM update.
            sigma_start: Initial standard deviation.
            sigma_min: Minimum standard deviation (clamp).
            sigma_max: Maximum standard deviation (clamp).
            temperature: Softmax temperature for weighting.
            beta: Fraction of samples allocated to MTP graph sampling.
            alpha: Momentum factor for the mean/cov update (0 = no momentum).
            num_randomizations: Number of domain randomizations.
            risk_strategy: Risk aggregation strategy.
            seed: Random seed for domain randomization.
            plan_horizon: Planning horizon in seconds.
            spline_type: Spline type for base class interpolation.
            num_knots: Number of control spline knots (shared output space).
            iterations: Number of optimisation iterations per step.
            keep_elites: Number of elite knot sequences to inject.
        """
        assert degree >= 2, "B-spline degree must be at least 2."
        super().__init__(
            task,
            num_randomizations=num_randomizations,
            risk_strategy=risk_strategy,
            seed=seed,
            plan_horizon=plan_horizon,
            spline_type=spline_type,
            num_knots=num_knots,
            iterations=iterations,
        )

        self.num_samples = num_samples
        self.degree = degree
        self.num_layers = num_layers
        self.nodes_per_layer = nodes_per_layer
        self.num_elites = num_elites
        self.sigma_start = sigma_start
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.keep_elites = max(1, min(keep_elites, num_elites))

        # Sample budget uses masking on beta dynamically

        # Pre-compute B-spline basis for MTP graph → knots projection.
        # Maps M+1 coarse control points to num_knots dense knot values.
        mtp_ctrl_pts = self.num_layers + 1  # anchor + M layers
        bspline_knots = clamped_knot_vector(mtp_ctrl_pts, self.degree)
        self.bspline_basis = jnp.asarray(
            compute_bspline_basis(
                bspline_knots,
                self.degree,
                self.num_knots,
            ),
            dtype=jnp.float32,
        )

    def init_params(
        self, initial_knots: jax.Array = None, seed: int = 0
    ) -> MTPParams:
        """Initialize policy parameters."""
        _params = super().init_params(initial_knots, seed)
        cov = jnp.full_like(_params.mean, self.sigma_start)
        elites = jnp.repeat(_params.mean[None, ...], self.keep_elites, axis=0)
        beta = jnp.array(self.beta, dtype=jnp.float32)
        return MTPParams(
            tk=_params.tk,
            mean=_params.mean,
            rng=_params.rng,
            cov=cov,
            elites=elites,
            beta=beta,
        )

    def sample_knots(self, params: MTPParams) -> Tuple[jax.Array, MTPParams]:
        """Sample control spline knots.

        Returns knots of shape ``(num_samples, num_knots, nu)`` and updated
        params (with a fresh RNG key).
        """
        rng = params.rng
        K = self.num_knots
        U = self.task.nu
        R = self.num_samples
        out = jnp.empty((R, K, U), dtype=jnp.float32)

        # --- Sample 0: the current mean ---
        out = out.at[0].set(params.mean)
        idx = 1

        # --- Inject elite samples ---
        if self.keep_elites > 0 and params.elites is not None:
            out = out.at[idx : idx + self.keep_elites].set(params.elites)
            idx += self.keep_elites

        S = R - idx  # remaining stochastic slots

        if S <= 0:
            return out, params.replace(rng=rng)

        # --- MTP branch (full S, later masked) ---
        rng, rng_pts, rng_idx = jax.random.split(rng, 3)

        # Random control points per layer: (M, N, nu)
        control_points = jax.random.uniform(
            rng_pts,
            (self.num_layers, self.nodes_per_layer, U),
            minval=self.task.u_min,
            maxval=self.task.u_max,
        )

        # Random path indices through the graph: (S, M)
        layer_indices = jax.random.randint(
            rng_idx,
            (S, self.num_layers),
            0,
            self.nodes_per_layer,
        )

        # Gather the selected control points: (S, M, nu)
        def _gather_path(indices: jax.Array) -> jax.Array:
            return control_points[jnp.arange(self.num_layers), indices]

        paths = jax.vmap(_gather_path)(layer_indices)

        # Anchor with current first knot: (S, 1, nu)
        anchor = jnp.broadcast_to(params.mean[0:1], (S, 1, U))

        # Coarse control points: (S, M+1, nu)
        full_pts = jnp.concatenate([anchor, paths], axis=1)

        # Project to num_knots via B-spline basis: (S, K, nu)
        mtp_knots = jnp.einsum("bmd, hm -> bhd", full_pts, self.bspline_basis)

        # --- MPPI (Gaussian) branch (full S, later masked) ---
        rng, rng_noise = jax.random.split(rng)
        noise = jax.random.normal(rng_noise, (S, K, U))
        mppi_knots = params.mean + params.cov * noise

        # --- Masked mixing ---
        num_mtp = jnp.floor(params.beta * S).astype(jnp.int32)
        mask = (jnp.arange(S) < num_mtp)[:, None, None]
        mixed_tail = jnp.where(mask, mtp_knots, mppi_knots)

        out = out.at[idx:R].set(mixed_tail)

        return out, params.replace(rng=rng)

    def update_params(
        self, params: MTPParams, rollouts: Trajectory
    ) -> MTPParams:
        """Update parameters using CEM with softmax weighting."""
        costs = jnp.sum(rollouts.costs, axis=1)

        _, elite_indices = jax.lax.top_k(-costs, self.num_elites)
        elite_knots = rollouts.knots[elite_indices]
        weights = jax.nn.softmax(
            -costs[elite_indices] / self.temperature, axis=0
        )
        weights = jnp.nan_to_num(weights)
        weighted = weights[:, None, None] * elite_knots

        # Weighted mean + momentum
        mean = jnp.sum(weighted, axis=0)
        mean = mean + self.alpha * (params.mean - mean)

        # Update diagonal covariance
        cov = jnp.sqrt(
            jnp.sum(
                weights[:, None, None] * (elite_knots - mean) ** 2,
                axis=0,
            )
        )
        cov = cov + self.alpha * (params.cov - cov)
        cov = jnp.clip(cov, self.sigma_min, self.sigma_max)

        # Track elites for injection
        new_elites = rollouts.knots[elite_indices[: self.keep_elites]]

        return params.replace(
            mean=mean,
            cov=cov,
            elites=new_elites,
        )
