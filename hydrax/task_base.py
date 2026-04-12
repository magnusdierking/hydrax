from abc import ABC, abstractmethod
from typing import Dict, Optional, Sequence

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx


class Task(ABC):
    """An abstract task interface, defining the dynamics and cost functions.

    The task is a discrete-time optimal control problem of the form

        minᵤ ϕ(x_{T+1}) + ∑ₜ ℓ(xₜ, uₜ)
        s.t. xₜ₊₁ = f(xₜ, uₜ)

    where the dynamics f(xₜ, uₜ) are defined by a MuJoCo model, and the costs
    ℓ(xₜ, uₜ) and ϕ(x_{T+1}) are defined by the task instance itself.
    """

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        trace_sites: Sequence[str] | None = None,
        nu: Optional[int] = None,
        ctrl_limits: Optional[Dict[str, jnp.ndarray]] = None,
        impl: str = "jax",
    ) -> None:
        """Set the model and simulation parameters.

        Args:
            mj_model: The MuJoCo model to use for simulation.
            trace_sites: A list of site names to visualize with traces.
            nu: The number of actuators in the model.
            ctrl_limits: The control limits for the model.
            impl: The backend implementation for rollouts ("jax" for standard
                  MJX or "warp" for MjWarp).

        Note: many other simulator parameters, e.g., simulator time step,
              Newton iterations, etc., are set in the model itself.
        """
        assert isinstance(mj_model, mujoco.MjModel)
        self.mj_model = mj_model
        self.model = mjx.put_model(self.mj_model, impl=impl)

        # Here we set the number of actuators to use for control
        # If nu is None, we use all actuators as in original hydrax
        self.nu = nu if nu is not None else self.model.nu

        if nu is None:
            # Set actuator limits
            self.u_min = jnp.where(
                mj_model.actuator_ctrllimited,
                mj_model.actuator_ctrlrange[:, 0],
                -jnp.inf,
            )
            self.u_max = jnp.where(
                mj_model.actuator_ctrllimited,
                mj_model.actuator_ctrlrange[:, 1],
                jnp.inf,
            )
            self.act_min = None
            self.act_max = None
        else:
            if ctrl_limits is None:
                raise ValueError("'ctrl_limits' must be provided if 'nu' is")
            self.u_min = ctrl_limits["u_min"]
            self.u_max = ctrl_limits["u_max"]
            self.act_min = jnp.where(
                mj_model.actuator_ctrllimited,
                mj_model.actuator_ctrlrange[:, 0],
                -jnp.inf,
            )
            self.act_max = jnp.where(
                mj_model.actuator_ctrllimited,
                mj_model.actuator_ctrlrange[:, 1],
                jnp.inf,
            )

        # Simulation timestep
        self.dt = mj_model.opt.timestep

        # Get site IDs for points we want to trace
        trace_sites = trace_sites or []
        self.trace_site_ids = jnp.array(
            [mj_model.site(name).id for name in trace_sites]
        )

    @abstractmethod
    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ).

        Args:
            state: The current state xₜ.
            control: The control action uₜ.

        Returns:
            The scalar running cost ℓ(xₜ, uₜ)
        """
        pass

    @abstractmethod
    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T).

        Args:
            state: The final state x_T.

        Returns:
            The scalar terminal cost ϕ(x_T).
        """
        pass

    def get_trace_sites(self, state: mjx.Data) -> jax.Array:
        """Get the positions of the trace sites at the current time step.

        Args:
            state: The current state xₜ.

        Returns:
            The positions of the trace sites at the current time step.
        """
        if len(self.trace_site_ids) == 0:
            return jnp.zeros((0, 3))

        return state.site_xpos[self.trace_site_ids]

    def domain_randomize_model(self, rng: jax.Array) -> Dict[str, jax.Array]:
        """Generate randomized model parameters for domain randomization.

        Returns a dictionary of randomized model parameters, that can be used
        with `mjx.Model.tree_replace` to create a new randomized model.

        For example, we might set the `model.geom_friction` values by returning
        `{"geom_friction": new_frictions, ...}`.

        The default behavior is to return an empty dictionary, which means no
        randomization is applied.

        Args:
            rng: A random number generator key.

        Returns:
            A dictionary of randomized model parameters.
        """
        return {}

    def domain_randomize_data(
        self, data: mjx.Data, rng: jax.Array
    ) -> Dict[str, jax.Array]:
        """Generate randomized data elements for domain randomization.

        This is the place where we could randomize the initial state and other
        `data` elements. Like `domain_randomize_model`, this method should
        return a dictionary that can be used with `mjx.Data.tree_replace`.

        Args:
            data: The base data instance holding the current state.
            rng: A random number generator key.

        Returns:
            A dictionary of randomized data elements.
        """
        return {}

    def make_data(self, **kwargs) -> mjx.Data:
        """Create a new state consistent with this task.

        By default, this just creates a new `mjx.Data` instance from the model.
        Specific tasks can override this method to set parameters that must be
        adjusted per task, e.g., nconmax and naconmax.

        TODO(vincekurtz): figure out a smarter place to set naconmax and njmax.
        N.B. when performing parallel rollouts with MjWarp, naconmax and
        njmax need to be set high enough to support constraint solving across
        *all* rollouts. This means that these parameters scale with the number
        of parallel rollouts/samples, as well as the complexity of the task.

        Args:
            **kwargs: Additional keyword arguments to pass to `mjx.make_data`.

        Returns:
            A new `mjx.Data` instance for this task.
        """
        return mjx.make_data(self.mj_model, impl=self.model.impl, **kwargs)

    ##############################################
    ##             Experimental                 ##
    ##############################################

    def control_mapper_mjx(self, state: mjx.Data, u: jax.Array) -> jax.Array:
        """Map controls from sampling space to actuator space (MJX version).

        By default, this just returns the input control.
        Specific tasks can override this method to implement custom control
        mapping, e.g. to enable task space sampling and joint space control.

        NOTE: This method is used inside the JITed rollouts and needs to be
        JIT compatible
        """
        return u

    def control_mapper_mj(
        self, state: mujoco.MjData, u: jnp.ndarray
    ) -> jnp.ndarray:
        """Map controls from sampling space to actuator space (MuJoCo version).

        By default, this just returns the input control.
        Specific tasks can override this method to implement custom control
        mapping, e.g. to enable task space sampling and joint space control.

        NOTE: This method is used in the sim to sim scripts
        """
        return u
