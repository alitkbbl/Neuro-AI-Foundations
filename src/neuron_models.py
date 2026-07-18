"""
neuron_models.py
=================

Object-oriented, vectorized (NumPy) implementations of single-compartment
neuron models, forming a conceptual ladder from a passive membrane to a
full Adaptive Exponential Integrate-and-Fire (AdEx) neuron.

Unit convention (consistent throughout the whole project, matching the
convention used by Brian2 / NEST for AdEx-family models so that numbers
stay in a friendly range and are mutually dimensionally consistent):

    time         : ms
    voltage      : mV
    current      : pA
    capacitance  : pF
    conductance  : nS
    resistance   : GOhm  (= 1 / conductance[nS])

Because 1 nS * 1 mV = 1 pA and 1 pF * 1 mV/ms = 1 pA, equations of the form
``C * dv/dt = -g*(v - v_rest) + I`` are numerically consistent with no unit
conversion factors required.

Classes
-------
BaseNeuron   : Abstract base class. Defines the state vector, the Euler and
               RK4 integrators, the simulate() driver loop, and a generic
               f_i_curve() helper. Subclasses only need to implement the
               ``derivatives`` method (and, for spiking models, the
               spike-detection / reset hooks).
PassiveNeuron: Linear RC membrane. No threshold, no spike, no reset.
LIFNeuron    : PassiveNeuron + hard threshold, reset, refractory period.
AdExNeuron   : Exponential spike-onset nonlinearity + adaptation current w.
"""

from abc import ABC, abstractmethod
import warnings

import numpy as np

#: Recommended maximum Euler time step (ms) for network-scale simulations.
#: RK4 (single-neuron use only) is not bound by this recommendation.
RECOMMENDED_MAX_DT = 0.1


class BaseNeuron(ABC):
    """
    Abstract base class for all single-compartment neuron models.

    A subclass represents its dynamical state as a 1D NumPy array
    ``self.state``, where ``self.state[0]`` is *always* the membrane
    potential ``v`` (mV). Subclasses that carry extra dynamical variables
    (e.g. an adaptation current ``w``) declare their names, in order, via
    the class attribute ``state_labels`` and extend nothing but
    ``derivatives`` / ``reset_state``.

    Subclasses must implement :meth:`derivatives`. Spiking subclasses
    additionally override :meth:`check_spike` and :meth:`apply_reset`.
    """

    #: Ordered names of the state vector's components. First entry must be "v".
    state_labels = ("v",)

    def __init__(self, dt: float = 0.1, v_init: float = -65.0):
        self.dt = float(dt)
        self.v_init = float(v_init)
        self.reset_state()

    # ------------------------------------------------------------------ #
    # State management
    # ------------------------------------------------------------------ #
    def reset_state(self):
        """Re-initialize the state vector to resting conditions. Subclasses
        with extra state variables (e.g. AdEx's ``w``) should override this
        to also zero those out, and should call ``super().reset_state()``
        is not required as long as they set ``self.state`` themselves."""
        self.state = np.zeros(len(self.state_labels))
        self.state[0] = self.v_init

    @property
    def v(self) -> float:
        """Current membrane potential (mV)."""
        return float(self.state[0])

    # ------------------------------------------------------------------ #
    # Dynamics — subclasses implement this
    # ------------------------------------------------------------------ #
    @abstractmethod
    def derivatives(self, state: np.ndarray, I: float) -> np.ndarray:
        """
        Return d(state)/dt, evaluated at ``state`` with instantaneous
        external input current ``I`` (pA). Must return an array with the
        same shape as ``state``.
        """
        raise NotImplementedError

    def check_spike(self, state: np.ndarray) -> bool:
        """Return True if ``state`` should be treated as a spike event.
        The base (non-spiking, passive) implementation never spikes."""
        return False

    def apply_reset(self, state: np.ndarray) -> np.ndarray:
        """Return a new state array following a spike event. No-op by
        default; spiking subclasses override this to reset v (and bump
        any adaptation variables)."""
        return state

    # ------------------------------------------------------------------ #
    # Integrators
    # ------------------------------------------------------------------ #
    def _euler_step(self, state: np.ndarray, I: float) -> np.ndarray:
        """Forward Euler update — the standard integrator for this project,
        used for every network simulation (dt <= 0.1 ms)."""
        d = self.derivatives(state, I)
        return state + self.dt * d

    def _rk4_step(self, state: np.ndarray, I: float) -> np.ndarray:
        """Classical 4th-order Runge-Kutta update.

        Provided *only* as an analytical cross-check for single-neuron
        trajectories (e.g. "how much does Euler's local error matter at
        dt=0.1ms vs the gold-standard RK4?", explored in Notebook 01).
        RK4 is never used for network simulations in this project: each
        step requires four derivative evaluations, which is prohibitively
        expensive at the scale of hundreds-to-thousands of neurons, and
        the vectorized Euler update is what makes those simulations
        tractable. The external current I is held constant across the
        four RK4 stages (piecewise-constant-current assumption, standard
        for these models).
        """
        dt = self.dt
        k1 = self.derivatives(state, I)
        k2 = self.derivatives(state + 0.5 * dt * k1, I)
        k3 = self.derivatives(state + 0.5 * dt * k2, I)
        k4 = self.derivatives(state + dt * k3, I)
        return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def step(self, I: float, method: str = "euler") -> bool:
        """
        Advance the neuron by one ``dt`` given the instantaneous input
        current ``I`` (pA). Returns True if a spike was registered on
        this step (and applies the reset internally).
        """
        if method == "euler":
            new_state = self._euler_step(self.state, I)
        elif method == "rk4":
            new_state = self._rk4_step(self.state, I)
        else:
            raise ValueError(f"Unknown integration method '{method}'. Use 'euler' or 'rk4'.")

        self.state = new_state
        spiked = self.check_spike(self.state)
        if spiked:
            self.state = self.apply_reset(self.state)
        return spiked

    # ------------------------------------------------------------------ #
    # Simulation driver
    # ------------------------------------------------------------------ #
    def simulate(self, I_ext, T: float = None, method: str = "euler") -> dict:
        """
        Simulate the neuron and return its full trajectory.

        Parameters
        ----------
        I_ext : float or 1D array-like
            Either a constant current (pA) — in which case ``T`` (ms) is
            required — or a pre-computed time-varying current array, one
            value per time step (length implicitly sets the duration).
        T : float, optional
            Total simulated duration (ms). Required only if ``I_ext`` is scalar.
        method : {"euler", "rk4"}
            Integration method. See :meth:`_rk4_step` docstring for when
            RK4 is (and is not) appropriate.

        Returns
        -------
        dict with keys:
            "t"            : time vector (ms), shape (n_steps,)
            "I"            : the (possibly broadcast) input current, shape (n_steps,)
            "spike_times"  : 1D array of spike times (ms)
            plus one array per entry in ``self.state_labels`` (e.g. "v", "w"),
            each of shape (n_steps,).

        Note on plotting spikes
        ------------------------
        For spiking models, the raw sub-threshold trace never actually
        reaches the "peak" (it is reset the instant threshold is crossed).
        Purely for visualization, the recorded ``v`` sample at the step on
        which a spike was detected is overwritten with ``self.v_peak`` if
        the neuron defines one, else 0.0 mV — giving the traditional
        upward "spike" stroke seen in textbook figures. This does not
        affect the underlying dynamics or any recorded spike time.
        """
        if np.isscalar(I_ext):
            if T is None:
                raise ValueError("T (ms) must be given when I_ext is a scalar constant current.")
            n_steps = int(round(T / self.dt))
            I_array = np.full(n_steps, float(I_ext))
        else:
            I_array = np.asarray(I_ext, dtype=float)
            n_steps = len(I_array)

        t = np.arange(n_steps) * self.dt
        n_vars = len(self.state_labels)
        state_traces = np.zeros((n_steps, n_vars))
        spike_times = []
        spike_marker = getattr(self, "v_peak", 0.0)

        for i in range(n_steps):
            state_traces[i] = self.state
            spiked = self.step(I_array[i], method=method)
            if spiked:
                spike_times.append(t[i] + self.dt)
                state_traces[i, 0] = spike_marker

        result = {"t": t, "I": I_array, "spike_times": np.array(spike_times)}
        for idx, label in enumerate(self.state_labels):
            result[label] = state_traces[:, idx]
        return result

    # ------------------------------------------------------------------ #
    # Analysis helpers
    # ------------------------------------------------------------------ #
    def f_i_curve(self, I_values, T: float = 1000.0, transient: float = 200.0,
                  method: str = "euler") -> np.ndarray:
        """
        Compute the steady-state firing-rate (Hz) response to a range of
        constant input currents — the classic F-I (frequency-current) curve.

        The neuron's state is reset before each current level so trials are
        independent. Spikes occurring before ``transient`` ms are discarded
        to let any initial transient (e.g. adaptation build-up) settle.

        Parameters
        ----------
        I_values : 1D array-like of constant currents (pA)
        T : total duration per trial (ms)
        transient : initial period excluded from the rate estimate (ms)

        Returns
        -------
        1D array of firing rates (Hz), same length as I_values.
        """
        rates = np.zeros(len(I_values))
        for k, I in enumerate(I_values):
            self.reset_state()
            res = self.simulate(I, T=T, method=method)
            st = res["spike_times"]
            st = st[st >= transient]
            rates[k] = len(st) / ((T - transient) / 1000.0)
        return rates

    def __repr__(self):
        params = ", ".join(f"{k}={v}" for k, v in self.__dict__.items()
                            if not k.startswith("_") and k not in ("state",))
        return f"{self.__class__.__name__}({params})"


class PassiveNeuron(BaseNeuron):
    """
    Linear passive (RC) membrane — the baseline model with **no spiking
    mechanism at all**:

        C_m * dv/dt = -g_L * (v - v_rest) + I(t)

    Exists to make a pedagogical point: without a threshold, "firing rate"
    is undefined and the membrane simply acts as a low-pass filter on its
    input current, exponentially relaxing toward v_rest + I/g_L.
    """

    state_labels = ("v",)

    def __init__(self, C_m: float = 200.0, g_L: float = 10.0, v_rest: float = -65.0,
                 dt: float = 0.1, v_init: float = None):
        self.C_m = float(C_m)
        self.g_L = float(g_L)
        self.v_rest = float(v_rest)
        super().__init__(dt=dt, v_init=v_rest if v_init is None else v_init)

    @property
    def tau_m(self) -> float:
        """Membrane time constant (ms)."""
        return self.C_m / self.g_L

    @property
    def R(self) -> float:
        """Membrane input resistance (GOhm)."""
        return 1.0 / self.g_L

    def derivatives(self, state: np.ndarray, I: float) -> np.ndarray:
        v = state[0]
        dv = (-self.g_L * (v - self.v_rest) + I) / self.C_m
        return np.array([dv])


class LIFNeuron(PassiveNeuron):
    """
    Leaky Integrate-and-Fire neuron: identical sub-threshold dynamics to
    :class:`PassiveNeuron`, plus a hard threshold-and-reset spiking rule:

        v(t) >= v_th  =>  spike registered, v <- v_reset, then clamped for t_ref ms

    The workhorse model of large-scale spiking network simulation — cheap,
    and already sufficient to produce a saturating, well-behaved F-I curve.
    """

    def __init__(self, C_m: float = 200.0, g_L: float = 10.0, v_rest: float = -65.0,
                 v_th: float = -50.0, v_reset: float = -65.0, t_ref: float = 2.0,
                 dt: float = 0.1, v_init: float = None):
        self.v_th = float(v_th)
        self.v_reset = float(v_reset)
        self.t_ref = float(t_ref)
        self._ref_remaining = 0.0
        super().__init__(C_m=C_m, g_L=g_L, v_rest=v_rest, dt=dt, v_init=v_init)

    def reset_state(self):
        super().reset_state()
        self._ref_remaining = 0.0

    def check_spike(self, state: np.ndarray) -> bool:
        return state[0] >= self.v_th

    def apply_reset(self, state: np.ndarray) -> np.ndarray:
        state = state.copy()
        state[0] = self.v_reset
        self._ref_remaining = self.t_ref
        return state

    def step(self, I: float, method: str = "euler") -> bool:
        """Refractory-period gate wraps the base integrator: while
        refractory, v is clamped at v_reset and no dynamics are evaluated."""
        if self._ref_remaining > 0:
            self._ref_remaining = max(0.0, self._ref_remaining - self.dt)
            self.state[0] = self.v_reset
            return False
        return super().step(I, method=method)


class AdExNeuron(BaseNeuron):
    """
    Adaptive Exponential Integrate-and-Fire neuron (Brette & Gerstner, 2005).

    Two coupled state variables, v (membrane potential) and w (adaptation
    current):

        C_m * dv/dt = -g_L*(v - E_L) + g_L*delta_T*exp((v - v_T)/delta_T) - w + I(t)
        tau_w * dw/dt = a*(v - E_L) - w + b*tau_w * sum_f delta(t - t^f)

    The exponential term is a soft, biophysically-motivated replacement for
    LIF's hard threshold (it approximates fast sodium activation near spike
    onset). ``v_peak`` is a numerical cutoff standing in for the true
    (divergent) spike upstroke: when v reaches v_peak we register a spike
    and apply the reset

        v <- v_reset
        w <- w + b

    The ``b * tau_w * delta(t - t^f)`` term in the w-equation is implemented
    *exactly* as its integral prescribes: an instantaneous jump of size
    ``b`` in w at each spike time (not a smoothed/filtered approximation).

    Depending on (a, b, tau_w, v_reset), AdEx reproduces essentially every
    qualitative cortical firing pattern: tonic spiking, spike-frequency
    adaptation, initial bursting, regular bursting, and irregular firing.
    """

    state_labels = ("v", "w")

    def __init__(self, C_m: float = 200.0, g_L: float = 10.0, E_L: float = -65.0,
                 v_T: float = -50.0, delta_T: float = 2.0,
                 a: float = 2.0, tau_w: float = 100.0, b: float = 60.0,
                 v_reset: float = -58.0, v_peak: float = 0.0, t_ref: float = 0.0,
                 dt: float = 0.1, v_init: float = None):
        self.C_m = float(C_m)
        self.g_L = float(g_L)
        self.E_L = float(E_L)
        self.v_T = float(v_T)
        self.delta_T = float(delta_T)
        self.a = float(a)
        self.tau_w = float(tau_w)
        self.b = float(b)
        self.v_reset = float(v_reset)
        self.v_peak = float(v_peak)
        self.t_ref = float(t_ref)
        self._ref_remaining = 0.0
        super().__init__(dt=dt, v_init=E_L if v_init is None else v_init)

    # alias, matches the equation's notation used throughout the literature
    @property
    def v_rest(self) -> float:
        return self.E_L

    def reset_state(self):
        self.state = np.array([self.v_init, 0.0])
        self._ref_remaining = 0.0

    def derivatives(self, state: np.ndarray, I: float) -> np.ndarray:
        v, w = state
        # Clip the exponential's argument: purely a numerical safety net so
        # that a v transiently above v_peak (last Euler step before reset)
        # never overflows float64 — it does not alter the model near/below
        # threshold, where delta_T is small relative to the clip range.
        exp_arg = np.clip((v - self.v_T) / self.delta_T, -50.0, 50.0)
        i_exp = self.g_L * self.delta_T * np.exp(exp_arg)
        dv = (-self.g_L * (v - self.E_L) + i_exp - w + I) / self.C_m
        dw = (self.a * (v - self.E_L) - w) / self.tau_w
        return np.array([dv, dw])

    def check_spike(self, state: np.ndarray) -> bool:
        return state[0] >= self.v_peak

    def apply_reset(self, state: np.ndarray) -> np.ndarray:
        state = state.copy()
        state[0] = self.v_reset
        state[1] = state[1] + self.b  # instantaneous adaptation jump
        self._ref_remaining = self.t_ref
        return state

    def step(self, I: float, method: str = "euler") -> bool:
        if self._ref_remaining > 0:
            self._ref_remaining = max(0.0, self._ref_remaining - self.dt)
            self.state[0] = self.v_reset
            return False
        return super().step(I, method=method)

    def nullclines(self, v_range: np.ndarray, I: float = 0.0):
        """
        Evaluate both nullclines of the (v, w) phase plane over ``v_range``,
        for a given constant input current ``I`` — used by Notebook 04's
        interactive phase-plane explorer.

        v-nullcline (dv/dt = 0):
            w = -g_L*(v - E_L) + g_L*delta_T*exp((v - v_T)/delta_T) + I
        w-nullcline (dw/dt = 0):
            w = a * (v - E_L)

        Returns
        -------
        (w_on_v_nullcline, w_on_w_nullcline) : both 1D arrays, same shape as v_range.
        """
        v_range = np.asarray(v_range, dtype=float)
        exp_arg = np.clip((v_range - self.v_T) / self.delta_T, -50.0, 50.0)
        w_v_null = -self.g_L * (v_range - self.E_L) + self.g_L * self.delta_T * np.exp(exp_arg) + I
        w_w_null = self.a * (v_range - self.E_L)
        return w_v_null, w_w_null


def _dt_sanity_check(dt: float, context: str = "network simulation"):
    """Shared helper (used by network_builder) to warn when dt exceeds the
    project's recommended Euler step size for network-scale simulations."""
    if dt > RECOMMENDED_MAX_DT:
        warnings.warn(
            f"dt={dt}ms exceeds the recommended maximum of {RECOMMENDED_MAX_DT}ms "
            f"for a {context}. Euler integration can become inaccurate or unstable "
            f"at larger steps for spiking models; consider dt <= {RECOMMENDED_MAX_DT}ms.",
            stacklevel=3,
        )
