"""
Unit tests for src/network_builder.py.
"""

import numpy as np
import pytest

from src.neuron_models import LIFNeuron
from src.network_builder import (
    NeuronPopulation, SparseConnectivity, BalancedNetwork,
    compute_population_rate, isi_cv,
)


class TestNeuronPopulation:

    def test_single_neuron_population_matches_lif_neuron(self):
        """A NeuronPopulation of size 1 should reproduce LIFNeuron's
        trajectory exactly for the same parameters and input — this is the
        key consistency check that the vectorized implementation didn't
        silently change the dynamics."""
        dt = 0.1
        params = dict(C_m=200.0, g_L=10.0, v_rest=-65.0, v_th=-50.0,
                       v_reset=-65.0, t_ref=2.0, dt=dt)
        n_steps = 3000
        rng = np.random.default_rng(0)
        I = rng.uniform(0, 400, n_steps)

        single = LIFNeuron(**params)
        v_single = np.zeros(n_steps)
        for i in range(n_steps):
            v_single[i] = single.v
            single.step(I[i])

        pop = NeuronPopulation(n_neurons=1, **params, rng=np.random.default_rng(42))
        pop.v[:] = params["v_rest"]  # match LIFNeuron's deterministic init (no jitter)
        v_pop = np.zeros(n_steps)
        for i in range(n_steps):
            v_pop[i] = pop.v[0]
            pop.step(np.array([I[i]]))

        np.testing.assert_allclose(v_pop, v_single, atol=1e-8)

    def test_reset_state_reinitializes_within_jitter_range(self):
        pop = NeuronPopulation(n_neurons=100, v_rest=-65.0, rng=np.random.default_rng(0))
        assert np.all(pop.v >= -70.0) and np.all(pop.v <= -60.0)
        assert np.all(pop.ref_remaining == 0.0)

    def test_refractory_neurons_do_not_update(self):
        pop = NeuronPopulation(n_neurons=1, t_ref=5.0, rng=np.random.default_rng(0))
        pop.v[0] = pop.v_th + 5.0  # clearly above threshold -> guaranteed spike this step
        pop.step(np.array([0.0]))
        assert pop.v[0] == pop.v_reset
        v_before = pop.v[0]
        pop.step(np.array([1e6]))  # huge current should have zero effect while refractory
        assert pop.v[0] == v_before


class TestSparseConnectivity:

    def test_dales_law_enforced(self):
        conn = SparseConnectivity(n_exc=80, n_inh=20, p_connect=0.2,
                                   w_exc=1.0, g=5.0, rng=np.random.default_rng(0))
        W = conn.W.toarray()
        exc_cols = W[:, :80]
        inh_cols = W[:, 80:]
        assert np.all(exc_cols >= 0.0)
        assert np.all(inh_cols <= 0.0)

    def test_no_self_connections_by_default(self):
        conn = SparseConnectivity(n_exc=50, n_inh=50, p_connect=0.5,
                                   rng=np.random.default_rng(0))
        assert np.all(conn.W.diagonal() == 0.0)

    def test_density_close_to_requested_probability(self):
        p = 0.15
        conn = SparseConnectivity(n_exc=400, n_inh=100, p_connect=p,
                                   rng=np.random.default_rng(0))
        assert conn.density == pytest.approx(p, abs=0.02)

    def test_inhibitory_weight_scaling(self):
        conn = SparseConnectivity(n_exc=10, n_inh=10, w_exc=2.0, g=4.0,
                                   p_connect=1.0, rng=np.random.default_rng(0))
        W = conn.W.toarray()
        assert conn.w_inh == pytest.approx(-8.0)
        # any actual (nonzero) inhibitory weight should equal w_inh exactly
        inh_weights = W[:, 10:]
        nonzero = inh_weights[inh_weights != 0]
        assert np.all(nonzero == pytest.approx(-8.0))


class TestBalancedNetwork:

    def test_output_dict_shape_and_keys(self):
        net = BalancedNetwork(n_neurons=200, seed=0)
        out = net.simulate(T=100.0)
        expected_keys = {"t", "dt", "spike_times", "spike_neurons",
                          "pop_rate_inst", "v_samples", "n_exc", "n_inh"}
        assert expected_keys.issubset(out.keys())
        assert len(out["t"]) == 1000  # 100ms / 0.1ms
        assert out["n_exc"] + out["n_inh"] == 200

    def test_network_produces_spikes_with_default_params(self):
        net = BalancedNetwork(n_neurons=500, seed=1)
        out = net.simulate(T=500.0)
        assert len(out["spike_times"]) > 0

    def test_network_is_stable_not_silent_not_saturated(self):
        """A healthy balanced network should neither die out completely nor
        have every neuron firing every few ms (runaway synchrony)."""
        net = BalancedNetwork(n_neurons=500, seed=2)
        out = net.simulate(T=1000.0)
        mean_rate_hz = len(out["spike_times"]) / net.n_neurons / 1.0
        assert 0.1 < mean_rate_hz < 100.0

    def test_spike_neuron_indices_within_bounds(self):
        net = BalancedNetwork(n_neurons=300, seed=3)
        out = net.simulate(T=300.0)
        if len(out["spike_neurons"]) > 0:
            assert out["spike_neurons"].min() >= 0
            assert out["spike_neurons"].max() < 300

    def test_reproducible_with_seed(self):
        out1 = BalancedNetwork(n_neurons=200, seed=7).simulate(T=200.0)
        out2 = BalancedNetwork(n_neurons=200, seed=7).simulate(T=200.0)
        np.testing.assert_array_equal(out1["spike_times"], out2["spike_times"])
        np.testing.assert_array_equal(out1["spike_neurons"], out2["spike_neurons"])

    def test_dt_sanity_warning_for_large_dt(self):
        with pytest.warns(UserWarning):
            BalancedNetwork(n_neurons=50, dt=0.5)


class TestAnalysisHelpers:

    def test_compute_population_rate_basic(self):
        # 10 spikes uniformly distributed over 1000ms across 100 neurons
        spike_times = np.linspace(0, 999, 10)
        centers, rate = compute_population_rate(spike_times, n_neurons=100,
                                                  T=1000.0, bin_size=1000.0)
        assert len(rate) == 1
        # 10 spikes / 100 neurons / 1s = 0.1 Hz
        assert rate[0] == pytest.approx(0.1, abs=1e-6)

    def test_isi_cv_regular_train_near_zero(self):
        regular = np.arange(0, 1000, 10.0)  # perfectly periodic, ISI=10ms always
        cv = isi_cv(regular)
        assert cv == pytest.approx(0.0, abs=1e-9)

    def test_isi_cv_nan_for_too_few_spikes(self):
        assert np.isnan(isi_cv(np.array([1.0, 2.0])))

    def test_isi_cv_high_for_irregular_train(self):
        rng = np.random.default_rng(0)
        isis = rng.exponential(scale=20.0, size=200)  # Poisson-like ISIs -> CV ~= 1
        spikes = np.cumsum(isis)
        cv = isi_cv(spikes)
        assert 0.7 < cv < 1.3
