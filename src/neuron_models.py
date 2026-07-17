"""
neuron_models.py

Core neuron model abstractions for the Neuro-AI-Foundations project.

Phase 1 contains only the abstract BaseNeuron class. Concrete neuron models
such as PassiveNeuron, LIFNeuron, and AdExNeuron will be implemented in later
phases.

Design goals
------------
1. Provide a clean object-oriented interface for neuron models.
2. Support Euler integration as the default method for efficient simulations.
3. Support RK4 integration as an optional method for single-neuron analytical
   comparisons.
4. Keep the base class general enough to support both non-spiking and spiking
   neuron models.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional, Any, List, Union

import numpy as np


ArrayLike = Union[float, int, np.ndarray]


class BaseNeuron(ABC):
    """
    Abstract base class for neuron models.

    This class defines the shared interface and numerical integration utilities
    for all neuron models in the project.

    Concrete subclasses are expected to implement at least:

    - ``derivatives(...)``:
        Computes the time derivatives of the model state variables.

    - ``get_state()``:
        Returns the current state as a dictionary.

    - ``set_state(state)``:
        Updates the internal state from a dictionary.

    - ``reset_state()``:
        Resets the neuron to its initial/default state.

    Spiking models may also override:

    - ``check_spike(...)``:
        Determines whether a spike occurred.

    - ``handle_spike(...)``:
        Applies reset, refractory behavior, adaptation increments, etc.

    Notes
    -----
    The repository standard for network simulations is forward Euler
    integration with a small time step, typically:

    ``dt <= 0.1 ms``

    RK4 is included only as an optional method for single-neuron analytical
    comparisons. For discontinuous spiking dynamics with threshold/reset events,
    Euler stepping is usually clearer and more practical.
    """

    def __init__(
        self,
        name: str = "BaseNeuron",
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize a base neuron.

        Parameters
        ----------
        name:
            Human-readable name for the neuron model.

        params:
            Optional dictionary of model parameters. Concrete subclasses should
            define their own expected parameters and defaults.
        """
        self.name = name
        self.params: Dict[str, Any] = params.copy() if params is not None else {}

        self.spike_times: List[float] = []
        self.last_spike_time: Optional[float] = None

    # ---------------------------------------------------------------------
    # Required subclass interface
    # ---------------------------------------------------------------------

    @abstractmethod
    def derivatives(
        self,
        t: float,
        state: Dict[str, ArrayLike],
        input_current: ArrayLike = 0.0,
    ) -> Dict[str, ArrayLike]:
        """
        Compute derivatives of the neuron's state variables.

        Parameters
        ----------
        t:
            Current simulation time in milliseconds.

        state:
            Dictionary containing the current model state. For a simple
            membrane model, this may be ``{"v": voltage}``. For AdEx, this
            may include ``{"v": voltage, "w": adaptation_current}``.

        input_current:
            External input current at time ``t``. Units are model-dependent,
            but later implementations will generally use picoamperes.

        Returns
        -------
        derivatives:
            Dictionary mapping each state variable name to its derivative.

        Example
        -------
        A passive neuron subclass might return:

        ``{"v": (-(v - v_rest) + r_m * input_current) / tau_m}``
        """
        raise NotImplementedError

    @abstractmethod
    def get_state(self) -> Dict[str, ArrayLike]:
        """
        Return the current internal state of the neuron.

        Returns
        -------
        state:
            Dictionary of state variables.
        """
        raise NotImplementedError

    @abstractmethod
    def set_state(self, state: Dict[str, ArrayLike]) -> None:
        """
        Set the internal state of the neuron.

        Parameters
        ----------
        state:
            Dictionary of state variables.
        """
        raise NotImplementedError

    @abstractmethod
    def reset_state(self) -> None:
        """
        Reset the neuron to its initial/default state.

        Concrete subclasses should define what reset means for their own
        state variables.
        """
        raise NotImplementedError

    # ---------------------------------------------------------------------
    # Optional spiking hooks
    # ---------------------------------------------------------------------

    def check_spike(
        self,
        t: float,
        state: Optional[Dict[str, ArrayLike]] = None,
    ) -> bool:
        """
        Check whether the neuron has emitted a spike.

        Non-spiking models can use the default implementation, which always
        returns False.

        Parameters
        ----------
        t:
            Current simulation time in milliseconds.

        state:
            Optional state dictionary. If not provided, the neuron's internal
            state may be queried by subclasses.

        Returns
        -------
        has_spiked:
            Boolean indicating whether a spike occurred.
        """
        return False

    def handle_spike(self, t: float) -> None:
        """
        Handle post-spike updates.

        Spiking subclasses may override this method to implement:

        - voltage reset,
        - refractory-period bookkeeping,
        - adaptation jumps,
        - spike-time logging,
        - synaptic event emission.

        The base implementation only records spike timing.

        Parameters
        ----------
        t:
            Spike time in milliseconds.
        """
        self.spike_times.append(t)
        self.last_spike_time = t

    # ---------------------------------------------------------------------
    # Numerical integration utilities
    # ---------------------------------------------------------------------

    def euler_step(
        self,
        t: float,
        dt: float,
        input_current: ArrayLike = 0.0,
    ) -> Dict[str, ArrayLike]:
        """
        Advance the neuron state by one Euler step.

        The Euler method updates each state variable according to:

        ``x(t + dt) = x(t) + dt * dx/dt``

        Parameters
        ----------
        t:
            Current simulation time in milliseconds.

        dt:
            Time step in milliseconds.

        input_current:
            External current applied during this step.

        Returns
        -------
        new_state:
            Updated state dictionary.

        Notes
        -----
        Euler integration is the standard method for this repository's network
        simulations. For stable and accurate spiking simulations, use small
        time steps, typically ``dt <= 0.1 ms``.
        """
        state = self.get_state()
        derivs = self.derivatives(t, state, input_current)

        new_state = {}
        for key, value in state.items():
            if key not in derivs:
                raise KeyError(
                    f"Missing derivative for state variable '{key}' "
                    f"in neuron model '{self.name}'."
                )
            new_state[key] = value + dt * derivs[key]

        self.set_state(new_state)

        if self.check_spike(t + dt, new_state):
            self.handle_spike(t + dt)

        return self.get_state()

    def rk4_step(
        self,
        t: float,
        dt: float,
        input_current: ArrayLike = 0.0,
    ) -> Dict[str, ArrayLike]:
        """
        Advance the neuron state by one fourth-order Runge-Kutta step.

        RK4 is useful for smooth single-neuron dynamical systems and analytical
        comparisons. It is not the default method for large spiking network
        simulations in this project.

        Parameters
        ----------
        t:
            Current simulation time in milliseconds.

        dt:
            Time step in milliseconds.

        input_current:
            External current applied during this step.

        Returns
        -------
        new_state:
            Updated state dictionary.

        Notes
        -----
        For models with hard threshold/reset discontinuities, RK4 should be
        used with caution because spike events are not naturally continuous.
        """
        state = self.get_state()

        k1 = self.derivatives(t, state, input_current)

        state_k2 = self._state_add_scaled(state, k1, dt / 2.0)
        k2 = self.derivatives(t + dt / 2.0, state_k2, input_current)

        state_k3 = self._state_add_scaled(state, k2, dt / 2.0)
        k3 = self.derivatives(t + dt / 2.0, state_k3, input_current)

        state_k4 = self._state_add_scaled(state, k3, dt)
        k4 = self.derivatives(t + dt, state_k4, input_current)

        new_state = {}
        for key in state:
            new_state[key] = state[key] + (dt / 6.0) * (
                k1[key] + 2.0 * k2[key] + 2.0 * k3[key] + k4[key]
            )

        self.set_state(new_state)

        if self.check_spike(t + dt, new_state):
            self.handle_spike(t + dt)

        return self.get_state()

    def step(
        self,
        t: float,
        dt: float,
        input_current: ArrayLike = 0.0,
        method: str = "euler",
    ) -> Dict[str, ArrayLike]:
        """
        Advance the neuron state by one time step.

        Parameters
        ----------
        t:
            Current simulation time in milliseconds.

        dt:
            Time step in milliseconds.

        input_current:
            External current applied during this step.

        method:
            Numerical integration method. Supported values are:

            - ``"euler"``
            - ``"rk4"``

        Returns
        -------
        state:
            Updated state dictionary.
        """
        method = method.lower()

        if method == "euler":
            return self.euler_step(t, dt, input_current=input_current)

        if method == "rk4":
            return self.rk4_step(t, dt, input_current=input_current)

        raise ValueError(
            f"Unknown integration method '{method}'. "
            "Supported methods are 'euler' and 'rk4'."
        )

    def simulate(
        self,
        t_stop: float,
        dt: float = 0.1,
        input_current: Union[ArrayLike, Callable[[float], ArrayLike]] = 0.0,
        method: str = "euler",
        reset: bool = True,
    ) -> Dict[str, Any]:
        """
        Simulate a single neuron over time.

        Parameters
        ----------
        t_stop:
            Final simulation time in milliseconds.

        dt:
            Time step in milliseconds. For standard simulations, use
            ``dt <= 0.1 ms``.

        input_current:
            Either a constant current or a callable function of time:

            ``I = input_current(t)``

        method:
            Numerical integration method. Defaults to ``"euler"``.

        reset:
            If True, reset the neuron's internal state before simulation.

        Returns
        -------
        results:
            Dictionary containing:

            - ``"time"``:
                NumPy array of time points.

            - ``"states"``:
                Dictionary mapping state variable names to NumPy arrays.

            - ``"spike_times"``:
                NumPy array of spike times.

            - ``"input_current"``:
                NumPy array of applied input current values.

        Notes
        -----
        This simulation helper is intentionally generic. More specialized
        simulation utilities may be added later for networks and synaptic input.
        """
        if dt <= 0:
            raise ValueError("dt must be positive.")

        if t_stop <= 0:
            raise ValueError("t_stop must be positive.")

        if method.lower() == "euler" and dt > 0.1:
            # This is a warning rather than an error because educational
            # notebooks may intentionally demonstrate numerical effects.
            print(
                "Warning: Euler integration with dt > 0.1 ms may be inaccurate "
                "for spiking neuron simulations."
            )

        if reset:
            self.reset_state()

        self.spike_times = []
        self.last_spike_time = None

        time = np.arange(0.0, t_stop + dt, dt)
        initial_state = self.get_state()

        state_traces: Dict[str, List[ArrayLike]] = {
            key: [] for key in initial_state
        }
        input_trace: List[ArrayLike] = []

        for t in time:
            current_state = self.get_state()

            for key, value in current_state.items():
                state_traces[key].append(np.copy(value))

            if callable(input_current):
                I_t = input_current(t)
            else:
                I_t = input_current

            input_trace.append(np.copy(I_t))

            if t < time[-1]:
                self.step(
                    t=t,
                    dt=dt,
                    input_current=I_t,
                    method=method,
                )

        results = {
            "time": time,
            "states": {
                key: np.asarray(values)
                for key, values in state_traces.items()
            },
            "spike_times": np.asarray(self.spike_times),
            "input_current": np.asarray(input_trace),
        }

        return results

    # ---------------------------------------------------------------------
    # Internal helper methods
    # ---------------------------------------------------------------------

    @staticmethod
    def _state_add_scaled(
        state: Dict[str, ArrayLike],
        derivs: Dict[str, ArrayLike],
        scale: float,
    ) -> Dict[str, ArrayLike]:
        """
        Construct a temporary state by adding a scaled derivative.

        This helper is mainly used by RK4 integration.

        Parameters
        ----------
        state:
            Current state dictionary.

        derivs:
            Derivative dictionary.

        scale:
            Multiplicative scale applied to the derivatives.

        Returns
        -------
        new_state:
            Temporary state dictionary.
        """
        new_state = {}

        for key, value in state.items():
            if key not in derivs:
                raise KeyError(
                    f"Missing derivative for state variable '{key}' "
                    "during RK4 intermediate state construction."
                )
            new_state[key] = value + scale * derivs[key]

        return new_state

    def __repr__(self) -> str:
        """
        Return a concise string representation of the neuron model.
        """
        return f"{self.__class__.__name__}(name={self.name!r}, params={self.params!r})"
