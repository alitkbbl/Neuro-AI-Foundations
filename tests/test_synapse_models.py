"""
Unit tests for src/synapse_models.py.
"""

import numpy as np
import pytest

from src.synapse_models import (
    ExponentialSynapse, AlphaSynapse, PoissonSpikeGenerator, spikes_to_current
)


class TestExponentialSynapse:

    def test_single_impulse_decays_exponentially(self):
        tau, weight, dt = 10.0, 1.0, 0.1
        syn = ExponentialSynapse(tau=tau, weight=weight, dt=dt, n_synapses=1)
        syn.step(1.0)  # single spike at t=0
        trace = [syn.step(0.0)[0] for _ in range(200)]
        trace = np.array(trace)
        t = np.arange(1, 201) * dt
        expected = weight * np.exp(-t / tau)
        np.testing.assert_allclose(trace, expected, rtol=1e-6)

    def test_reset_zeros_state(self):
        syn = ExponentialSynapse(n_synapses=3)
        syn.step(np.array([1.0, 1.0, 1.0]))
        syn.reset()
        assert np.all(syn.g == 0.0)

    def test_vectorized_over_multiple_synapses(self):
        syn = ExponentialSynapse(tau=5.0, weight=2.0, dt=0.1, n_synapses=4)
        out = syn.step(np.array([1, 0, 1, 0]))
        np.testing.assert_allclose(out, [2.0, 0.0, 2.0, 0.0])


class TestAlphaSynapse:

    def test_starts_and_returns_to_zero(self):
        syn = AlphaSynapse(tau=5.0, weight=1.0, dt=0.1, n_synapses=1)
        assert syn.g[0] == 0.0
        syn.step(1.0)
        trace = [syn.step(0.0)[0] for _ in range(2000)]  # long relaxation
        assert trace[-1] < 1e-3

    def test_peak_occurs_near_tau(self):
        tau, dt = 8.0, 0.1
        syn = AlphaSynapse(tau=tau, weight=1.0, dt=dt, n_synapses=1)
        syn.step(1.0)
        trace = []
        for _ in range(500):
            trace.append(syn.step(0.0)[0])
        trace = np.array(trace)
        peak_time = (np.argmax(trace) + 1) * dt
        # alpha function peaks at t = tau after the impulse; allow discretization slack
        assert peak_time == pytest.approx(tau, abs=1.0)


class TestPoissonSpikeGenerator:

    def test_empirical_rate_matches_requested_rate(self):
        rate_hz = 50.0
        dt = 0.1
        gen = PoissonSpikeGenerator(rate=rate_hz, dt=dt, n_sources=200,
                                     rng=np.random.default_rng(0))
        spikes = gen.generate(T=5000.0)  # (n_steps, 200)
        empirical_rate = spikes.sum() / spikes.shape[1] / (5000.0 / 1000.0)
        assert empirical_rate == pytest.approx(rate_hz, rel=0.1)

    def test_per_source_rate_array(self):
        rates = np.array([10.0, 100.0])
        gen = PoissonSpikeGenerator(rate=rates, dt=0.1, n_sources=2,
                                     rng=np.random.default_rng(1))
        spikes = gen.generate(T=10000.0)
        emp = spikes.sum(axis=0) / (10000.0 / 1000.0)
        assert emp[1] > emp[0]  # the 100Hz source should fire much more than the 10Hz one

    def test_zero_rate_never_spikes(self):
        gen = PoissonSpikeGenerator(rate=0.0, dt=0.1, n_sources=5)
        spikes = gen.generate(T=1000.0)
        assert not spikes.any()


class TestSpikesToCurrent:

    def test_output_shape_matches_input(self):
        train = np.zeros(100)
        train[[10, 50]] = 1
        out = spikes_to_current(train, tau=5.0, weight=100.0, dt=0.1, kind="exponential")
        assert out.shape == train.shape

    def test_invalid_kind_raises(self):
        with pytest.raises(ValueError):
            spikes_to_current(np.zeros(10), kind="not_a_kind")
