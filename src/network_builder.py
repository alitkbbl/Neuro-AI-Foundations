"""
network_builder.py
===================

Construction and simulation of a recurrently-connected spiking network,
built for scale: everything here operates on whole-population NumPy arrays
and a sparse (SciPy) connectivity matrix, never on a Python list of N
individual neuron objects. This is the difference between a 1000-neuron,
10,000-timestep simulation finishing in seconds versus minutes.

Classes
-------
NeuronPopulation   : Vectorized array of identical LIF neurons.
SparseConnectivity : Random sparse recurrent weight matrix enforcing Dale's
                      law (excitatory neurons only project positive weights,
                      inhibitory neurons only negative).
BalancedNetwork     : Combines a population + connectivity + independent
                       Poisson background drive into a runnable network,
                       reproducing the cortex-like Asynchronous-Irregular
                       (AI) firing regime (Brunel, 2000) when inhibition is
                       scaled sufficiently above excitation.

Module-level helpers
---------------------
compute_population_rate : bin a network's spike times into a firing-rate trace.
isi_cv                  : inter-spike-interval coefficient of variation
                           (a standard measure of spike-train irregularity).
"""

import numpy as np
from scipy import sparse

from .neuron_models import RECOMMENDED_MAX_DT, _dt_sanity_check
from .synapse_models import ExponentialSynapse, PoissonSpikeGenerator


class NeuronPopulation:
    """
    A vectorized population of ``n_neurons`` identical LIF neurons, all
    advanced by one Euler step in a single set of array operations. State
    (v, refractory countdown) lives in NumPy arrays of shape (n_neurons,).

    Membrane dynamics are identical to :class:`src.neuron_models.LIFNeuron`;
    see that class's docstring for the governing equation.
    """

    def __init__(self, n_neurons: int, C_m: float = 200.0, g_L: float = 10.0,
                 v_rest: float = -65.0, v_th: float = -50.0, v_reset: float = -65.0,
                 t_ref: float = 2.0, dt: float = 0.1, rng=None):
        self.n = n_neurons
        self.C_m = C_m
        self.g_L = g_L
        self.v_rest = v_rest
        self.v_th = v_th
        self.v_reset = v_reset
        self.t_ref = t_ref
        self.dt = dt
        self.rng = np.random.default_rng() if rng is None else rng
        self.reset_state()

    def reset_state(self):
        """Re-initialize membrane potentials with small random jitter around
        rest (avoids the pathological perfect synchrony that identical
        initial conditions would otherwise seed) and clear refractory timers."""
        self.v = self.v_rest + self.rng.uniform(-5.0, 5.0, self.n)
        self.ref_remaining = np.zeros(self.n)

    def step(self, I: np.ndarray) -> np.ndarray:
        """
        Advance every neuron by one dt given a per-neuron input current
        array ``I`` (pA), shape (n_neurons,).

        Returns
        -------
        Boolean array, shape (n_neurons,): which neurons spiked this step.
        """
        refractory = self.ref_remaining > 0
        active = ~refractory

        dv = (-self.g_L * (self.v - self.v_rest) + I) / self.C_m
        self.v[active] = self.v[active] + self.dt * dv[active]

        self.ref_remaining[refractory] -= self.dt
        np.clip(self.ref_remaining, 0.0, None, out=self.ref_remaining)

        spiked = (self.v >= self.v_th) & active
        self.v[spiked] = self.v_reset
        self.ref_remaining[spiked] = self.t_ref

        return spiked


class SparseConnectivity:
    """
    Random sparse recurrent connectivity matrix enforcing **Dale's law**:
    each presynaptic neuron's outgoing weights are either all-excitatory
    (positive) or all-inhibitory (negative), never mixed — matching the
    biological constraint that a real neuron releases one neurotransmitter
    type. The first ``n_exc`` neuron indices are excitatory, the remainder
    inhibitory.

    Stored as a SciPy CSR sparse matrix ``W`` of shape (n_post, n_pre): the
    recurrent input current to every neuron in one step is then the single
    sparse matrix-vector product ``W @ s``, where ``s`` is each presynaptic
    neuron's (decaying) synaptic trace.

    Inhibitory weights are scaled by a factor ``g`` relative to excitatory
    ones (``w_inh = -g * w_exc``); the "balanced network" regime (Brunel,
    2000) requires ``g`` large enough (roughly g >~ 4-5, given typical 4:1
    E:I population ratios and connection probabilities) that recurrent
    inhibition can track and cancel the mean recurrent excitation, leaving
    the net current fluctuation-dominated rather than mean-dominated —
    which is exactly what produces irregular, asynchronous spiking instead
    of synchronized runaway firing.
    """

    def __init__(self, n_exc: int, n_inh: int, p_connect: float = 0.1,
                 w_exc: float = 0.5, g: float = 5.0, rng=None,
                 self_connections: bool = False):
        self.n_exc = n_exc
        self.n_inh = n_inh
        self.n = n_exc + n_inh
        self.p_connect = p_connect
        self.w_exc = w_exc
        self.g = g
        self.w_inh = -g * w_exc
        self.rng = np.random.default_rng() if rng is None else rng
        self.W = self._build(self_connections)

    def _build(self, self_connections: bool) -> sparse.csr_matrix:
        n, n_exc = self.n, self.n_exc
        mask = self.rng.random((n, n)) < self.p_connect
        if not self_connections:
            np.fill_diagonal(mask, False)

        W = np.zeros((n, n))
        W[:, :n_exc] = mask[:, :n_exc] * self.w_exc      # excitatory presynaptic columns
        W[:, n_exc:] = mask[:, n_exc:] * self.w_inh       # inhibitory presynaptic columns
        return sparse.csr_matrix(W)

    @property
    def density(self) -> float:
        """Fraction of the n x n matrix that is actually connected."""
        return self.W.nnz / (self.n * self.n)


class BalancedNetwork:
    """
    A recurrently-connected network of LIF neurons — 80% excitatory / 20%
    inhibitory by default — with sparse random connectivity (Dale's law
    enforced) and independent homogeneous Poisson background input driving
    every cell. This is the classic Brunel-style balanced network, used
    here to demonstrate the cortex-like **Asynchronous-Irregular (AI)**
    firing regime: individually irregular spike trains (CV of ISI near 1),
    weak pairwise correlation, and a stable, low mean population rate —
    emerging precisely because strong recurrent excitation and even
    stronger recurrent inhibition are fighting each other into balance,
    rather than because input is weak.

    Parameters
    ----------
    n_neurons : total network size (80/20 E/I split by default)
    frac_exc : fraction of neurons that are excitatory
    p_connect : connection probability between any ordered pair of neurons
    w_exc : baseline excitatory synaptic weight (pA-equivalent per unit trace)
    g : inhibitory-to-excitatory weight ratio (w_inh = -g * w_exc)
    syn_tau_exc / syn_tau_inh : recurrent synaptic decay time constants (ms)
    bg_rate : firing rate (Hz) of each neuron's independent Poisson background input
    bg_weight : synaptic weight of the background input (pA-equivalent)
    bg_tau : decay time constant of the background synapse (ms)
    neuron_params : optional dict overriding NeuronPopulation defaults (C_m, g_L, v_th, ...)
    dt : Euler time step (ms) — should stay <= 0.1ms (project standard for network sims)
    seed : RNG seed, for reproducibility

    Note on default weights
    ------------------------
    ``w_exc``, ``g``, ``bg_rate``, and ``bg_weight`` default to values that
    were empirically calibrated (see Notebook 05) so the network sits in a
    stable, low-rate (~3-5 Hz), irregularly-firing (ISI CV ~0.5-0.6) regime
    with the LIF parameters' default rheobase (~150 pA). If you change
    ``C_m``, ``g_L``, ``v_th``, or ``v_rest`` via ``neuron_params``, these
    weights will very likely need re-tuning — the network's rheobase, and
    therefore everything about how much synaptic drive is "enough", moves
    with them.
    """

    def __init__(self, n_neurons: int = 1000, frac_exc: float = 0.8,
                 p_connect: float = 0.1, w_exc: float = 2.0, g: float = 5.0,
                 syn_tau_exc: float = 5.0, syn_tau_inh: float = 10.0,
                 bg_rate: float = 1000.0, bg_weight: float = 25.0, bg_tau: float = 5.0,
                 neuron_params: dict = None, dt: float = 0.1, seed: int = None):
        _dt_sanity_check(dt, context="BalancedNetwork simulation")

        self.dt = dt
        self.n_neurons = n_neurons
        self.n_exc = int(round(n_neurons * frac_exc))
        self.n_inh = n_neurons - self.n_exc
        self.rng = np.random.default_rng(seed)

        params = dict(neuron_params) if neuron_params else {}
        self.pop = NeuronPopulation(n_neurons, dt=dt, rng=self.rng, **params)
        self.conn = SparseConnectivity(self.n_exc, self.n_inh, p_connect=p_connect,
                                        w_exc=w_exc, g=g, rng=self.rng)

        # Independent Poisson background input, one source per neuron, filtered
        # through its own exponential synapse.
        self.bg_gen = PoissonSpikeGenerator(bg_rate, dt=dt, n_sources=n_neurons, rng=self.rng)
        self.bg_synapse = ExponentialSynapse(tau=bg_tau, weight=bg_weight, dt=dt,
                                              n_synapses=n_neurons)

        # Recurrent synaptic trace: one scalar per presynaptic neuron, decaying
        # with tau_exc for excitatory sources and tau_inh for inhibitory ones.
        decay = np.empty(n_neurons)
        decay[:self.n_exc] = np.exp(-dt / syn_tau_exc)
        decay[self.n_exc:] = np.exp(-dt / syn_tau_inh)
        self._rec_decay = decay
        self._s_rec = np.zeros(n_neurons)

    def reset(self):
        """Reset neuron states and all synaptic traces (connectivity is kept)."""
        self.pop.reset_state()
        self.bg_synapse.reset()
        self._s_rec[:] = 0.0

    def simulate(self, T: float = 1000.0, I_ext=None, record_v_indices=None,
                 reset: bool = True) -> dict:
        """
        Run the network for ``T`` ms.

        Parameters
        ----------
        I_ext : optional extra input current, scalar (pA, applied to every
            neuron at every step) or array of shape (n_steps, n_neurons).
        record_v_indices : iterable of neuron indices whose full membrane
            potential trace should be recorded (default: one excitatory and
            one inhibitory sample neuron). Recording all 1000 neurons'
            traces at 0.1ms resolution is unnecessary for the raster/rate
            analyses this network is meant for, so only a handful are kept.
        reset : if True (default), reset neuron/synapse state before running
            so repeated calls are independent trials.

        Returns
        -------
        dict with keys:
            "t"              : time vector (ms)
            "dt"              : the time step used (ms)
            "spike_times"     : 1D array of all spike times (ms), network-wide
            "spike_neurons"   : 1D array of the corresponding neuron indices
            "pop_rate_inst"   : instantaneous population rate per step (Hz)
            "v_samples"       : {neuron_index: v_trace} for recorded neurons
            "n_exc", "n_inh"  : population sizes, for downstream raster coloring
        """
        if reset:
            self.reset()

        if record_v_indices is None:
            record_v_indices = [0, self.n_exc]  # one E, one I sample neuron

        dt = self.dt
        n_steps = int(round(T / dt))
        t = np.arange(n_steps) * dt

        v_samples = {idx: np.zeros(n_steps) for idx in record_v_indices}
        pop_spike_counts = np.zeros(n_steps)
        spike_times_chunks = []
        spike_neurons_chunks = []

        for i in range(n_steps):
            for idx in record_v_indices:
                v_samples[idx][i] = self.pop.v[idx]

            I_rec = self.conn.W.dot(self._s_rec)
            bg_spikes = self.bg_gen.step()
            I_bg = self.bg_synapse.step(bg_spikes)

            I_total = I_rec + I_bg
            if I_ext is not None:
                I_total = I_total + (I_ext if np.isscalar(I_ext) else I_ext[i])

            spiked = self.pop.step(I_total)
            self._s_rec = self._s_rec * self._rec_decay + spiked

            n_spiked = spiked.sum()
            pop_spike_counts[i] = n_spiked
            if n_spiked:
                idxs = np.nonzero(spiked)[0]
                spike_times_chunks.append(np.full(n_spiked, t[i]))
                spike_neurons_chunks.append(idxs)

        spike_times = np.concatenate(spike_times_chunks) if spike_times_chunks else np.array([])
        spike_neurons = np.concatenate(spike_neurons_chunks) if spike_neurons_chunks else np.array([], dtype=int)
        pop_rate_inst = pop_spike_counts / self.n_neurons / (dt / 1000.0)

        return {
            "t": t, "dt": dt,
            "spike_times": spike_times, "spike_neurons": spike_neurons,
            "pop_rate_inst": pop_rate_inst,
            "v_samples": v_samples,
            "n_exc": self.n_exc, "n_inh": self.n_inh,
        }


# ---------------------------------------------------------------------- #
# Module-level analysis helpers
# ---------------------------------------------------------------------- #
def compute_population_rate(spike_times: np.ndarray, n_neurons: int, T: float,
                             bin_size: float = 1.0):
    """
    Bin network-wide spike times into a population firing-rate trace.

    Parameters
    ----------
    spike_times : 1D array of spike times (ms), pooled across all neurons.
    n_neurons : total number of neurons in the population (for normalization).
    T : total simulated duration (ms).
    bin_size : bin width (ms).

    Returns
    -------
    (bin_centers, rate) : both 1D arrays. rate is in Hz (spikes / s / neuron).
    """
    bins = np.arange(0, T + bin_size, bin_size)
    counts, _ = np.histogram(spike_times, bins=bins)
    rate = counts / n_neurons / (bin_size / 1000.0)
    bin_centers = bins[:-1] + bin_size / 2.0
    return bin_centers, rate


def isi_cv(spike_times: np.ndarray) -> float:
    """
    Coefficient of variation (std / mean) of a single neuron's inter-spike
    intervals — the standard measure of spike-train irregularity.

    CV ~= 1   : irregular, Poisson-like firing (hallmark of the AI regime)
    CV << 1   : regular, clock-like firing
    CV >> 1   : bursty firing

    Returns NaN if fewer than 3 spikes are given (need >= 2 ISIs).
    """
    spike_times = np.sort(np.asarray(spike_times))
    if len(spike_times) < 3:
        return float("nan")
    isis = np.diff(spike_times)
    return float(np.std(isis) / np.mean(isis))
