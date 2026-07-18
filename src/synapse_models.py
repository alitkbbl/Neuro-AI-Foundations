"""
synapse_models.py
==================

Synaptic dynamics and background-input generation, vectorized with NumPy so
the same classes scale from "one synapse onto one neuron" (Notebooks 01-03)
to "hundreds of thousands of synapses in a recurrent network" (Notebook 05)
without changing a line of code — only the array sizes change.

All kernels are updated with forward Euler at the caller's ``dt``, matching
the project-wide integration convention.

Classes
-------
ExponentialSynapse  : Single-exponential decay conductance/current kernel.
AlphaSynapse         : Alpha-function (rise-and-decay) kernel via two coupled
                        linear ODEs, recursively updated.
PoissonSpikeGenerator: Vectorized homogeneous Poisson spike-train source,
                        used to drive independent background noise onto
                        many neurons at once.
"""

import numpy as np


class ExponentialSynapse:
    """
    Single-exponential synaptic kernel. Each incoming spike instantaneously
    increments the synaptic variable by ``weight``; between spikes it decays
    with time constant ``tau``:

        tau * dg/dt = -g          g <- g + weight   on each incoming spike

    Vectorized over ``n_synapses`` independent channels (e.g. one entry per
    presynaptic connection onto a neuron, or one entry per postsynaptic
    neuron in a population-level aggregate current).
    """

    def __init__(self, tau: float = 5.0, weight: float = 1.0, dt: float = 0.1,
                 n_synapses: int = 1):
        self.tau = float(tau)
        self.weight = float(weight)
        self.dt = float(dt)
        self.n_synapses = n_synapses
        self.g = np.zeros(n_synapses)
        # exact solution of the decay-only step, precomputed once
        self._decay = np.exp(-self.dt / self.tau)

    def step(self, spike_input) -> np.ndarray:
        """
        Advance by one dt.

        Parameters
        ----------
        spike_input : bool/0-1 scalar or array of shape (n_synapses,)
            Whether each channel received a presynaptic spike this step.

        Returns
        -------
        Updated synaptic variable g, shape (n_synapses,).
        """
        self.g = self.g * self._decay + self.weight * np.asarray(spike_input, dtype=float)
        return self.g

    def reset(self):
        self.g[:] = 0.0


class AlphaSynapse:
    """
    Alpha-function synaptic kernel: rises and decays with the *same* time
    constant tau, peaking at exactly t = tau after an isolated input spike
    (Rall's alpha function). Implemented via the standard trick of two
    coupled linear first-order ODEs so it can be updated with a simple
    Euler step at every time point:

        dx/dt = -x / tau                         x <- x + weight on each input spike
        dg/dt = (x - g) / tau

    ``g`` is the (smoother, more biophysically realistic) synaptic
    conductance/current trace actually injected into the postsynaptic
    neuron.
    """

    def __init__(self, tau: float = 5.0, weight: float = 1.0, dt: float = 0.1,
                 n_synapses: int = 1):
        self.tau = float(tau)
        self.weight = float(weight)
        self.dt = float(dt)
        self.n_synapses = n_synapses
        self.x = np.zeros(n_synapses)
        self.g = np.zeros(n_synapses)

    def step(self, spike_input) -> np.ndarray:
        spike_input = np.asarray(spike_input, dtype=float)
        dx = -self.x / self.tau
        self.x = self.x + self.dt * dx + self.weight * spike_input
        dg = (self.x - self.g) / self.tau
        self.g = self.g + self.dt * dg
        return self.g

    def reset(self):
        self.x[:] = 0.0
        self.g[:] = 0.0


def spikes_to_current(spike_train, tau: float = 5.0, weight: float = 100.0,
                       dt: float = 0.1, kind: str = "exponential") -> np.ndarray:
    """
    Convenience wrapper for single-neuron notebooks: filter a 1D binary
    presynaptic spike train through a synaptic kernel to produce a
    continuous post-synaptic current trace, suitable for injecting into
    ``BaseNeuron.simulate(I_ext=...)`` in place of a step/constant current.

    Parameters
    ----------
    spike_train : 1D array-like of 0/1 (or bool), one entry per time step.
    tau : synaptic time constant (ms)
    weight : current jump per spike (pA)
    dt : simulation time step (ms), must match the step used elsewhere
    kind : "exponential" or "alpha"

    Returns
    -------
    1D array, same length as spike_train, of post-synaptic current (pA).
    """
    spike_train = np.asarray(spike_train, dtype=float)
    n = len(spike_train)
    if kind == "exponential":
        syn = ExponentialSynapse(tau=tau, weight=weight, dt=dt, n_synapses=1)
    elif kind == "alpha":
        syn = AlphaSynapse(tau=tau, weight=weight, dt=dt, n_synapses=1)
    else:
        raise ValueError("kind must be 'exponential' or 'alpha'")

    out = np.zeros(n)
    for i in range(n):
        out[i] = syn.step(spike_train[i])[0]
    return out


class PoissonSpikeGenerator:
    """
    Vectorized homogeneous Poisson spike-train source.

    At each time step, each of ``n_sources`` independent channels emits a
    spike with probability ``rate * dt / 1000`` (rate in Hz, dt in ms) — the
    standard Bernoulli-per-bin approximation to a Poisson process. The
    approximation is excellent whenever ``rate * dt << 1000``, comfortably
    true here (dt <= 0.1ms) for any biologically plausible firing rate.

    Used throughout Notebook 05 to inject independent background ("noise")
    drive onto every neuron in the balanced network, without which a purely
    recurrent, otherwise-quiescent network would never spike at all.
    """

    def __init__(self, rate, dt: float = 0.1, n_sources: int = 1, rng=None):
        self.dt = float(dt)
        self.n_sources = n_sources
        self.rng = np.random.default_rng() if rng is None else rng
        self.set_rate(rate)

    def set_rate(self, rate):
        """rate: scalar (Hz, applied to every source) or array of length n_sources (Hz)."""
        self.rate = np.broadcast_to(np.asarray(rate, dtype=float), (self.n_sources,)).copy()
        self.p_spike = self.rate * self.dt / 1000.0

    def step(self) -> np.ndarray:
        """Draw one time step. Returns a boolean array, shape (n_sources,)."""
        return self.rng.random(self.n_sources) < self.p_spike

    def generate(self, T: float) -> np.ndarray:
        """
        Pre-generate a full spike matrix for a duration T (ms) in one
        vectorized call — convenient when the whole raster is needed
        up-front (e.g. for plotting or for feeding a static input array
        into ``BaseNeuron.simulate``).

        Returns
        -------
        Boolean array of shape (n_steps, n_sources).
        """
        n_steps = int(round(T / self.dt))
        return self.rng.random((n_steps, self.n_sources)) < self.p_spike
