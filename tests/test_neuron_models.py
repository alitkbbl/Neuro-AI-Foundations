"""
Unit tests for src/neuron_models.py.

These tests check actual dynamical properties of each model (analytical
steady states, monotonic F-I curves, correct spike-triggered adaptation),
not merely that the code runs without raising.
"""

import numpy as np
import pytest

from src.neuron_models import PassiveNeuron, LIFNeuron, AdExNeuron


# --------------------------------------------------------------------- #
# PassiveNeuron
# --------------------------------------------------------------------- #
class TestPassiveNeuron:

    def test_converges_to_analytical_steady_state(self):
        neuron = PassiveNeuron(C_m=200.0, g_L=10.0, v_rest=-65.0, dt=0.1)
        I = 120.0
        res = neuron.simulate(I_ext=I, T=10 * neuron.tau_m)  # >= 10 time constants
        expected_ss = neuron.v_rest + I / neuron.g_L
        assert res["v"][-1] == pytest.approx(expected_ss, abs=0.1)

    def test_never_spikes_even_at_huge_current(self):
        neuron = PassiveNeuron()
        res = neuron.simulate(I_ext=1e5, T=100.0)
        assert len(res["spike_times"]) == 0

    def test_relaxes_back_to_rest_with_zero_current(self):
        neuron = PassiveNeuron(v_rest=-65.0)
        neuron.state[0] = -20.0  # displace away from rest
        res = neuron.simulate(I_ext=0.0, T=200.0)
        assert res["v"][-1] == pytest.approx(-65.0, abs=0.1)

    def test_tau_m_and_R_properties(self):
        neuron = PassiveNeuron(C_m=200.0, g_L=10.0)
        assert neuron.tau_m == pytest.approx(20.0)
        assert neuron.R == pytest.approx(0.1)


# --------------------------------------------------------------------- #
# LIFNeuron
# --------------------------------------------------------------------- #
class TestLIFNeuron:

    def test_subthreshold_current_never_spikes(self):
        neuron = LIFNeuron(v_rest=-65.0, v_th=-50.0, g_L=10.0)
        rheobase = neuron.g_L * (neuron.v_th - neuron.v_rest)  # 150 pA
        res = neuron.simulate(I_ext=0.5 * rheobase, T=1000.0)
        assert len(res["spike_times"]) == 0

    def test_suprathreshold_current_spikes(self):
        neuron = LIFNeuron(v_rest=-65.0, v_th=-50.0, g_L=10.0)
        rheobase = neuron.g_L * (neuron.v_th - neuron.v_rest)
        res = neuron.simulate(I_ext=2.0 * rheobase, T=500.0)
        assert len(res["spike_times"]) > 0

    def test_reset_after_spike(self):
        neuron = LIFNeuron(v_reset=-65.0, t_ref=2.0)
        neuron.simulate(I_ext=400.0, T=50.0)
        # after any spike, voltage should never exceed v_th again until dynamics evolve
        assert neuron.v <= neuron.v_th + 1e-9

    def test_refractory_period_bounds_max_rate(self):
        t_ref = 5.0  # ms -> max possible rate = 1000/5 = 200 Hz
        neuron = LIFNeuron(t_ref=t_ref)
        res = neuron.simulate(I_ext=1e4, T=1000.0)  # enormous current, would spike every step without t_ref
        rate = len(res["spike_times"]) / 1.0  # Hz over 1000ms = 1s
        assert rate <= 1000.0 / t_ref + 1.0  # small tolerance

    def test_fi_curve_is_monotonic_non_decreasing(self):
        neuron = LIFNeuron()
        I_values = np.linspace(0, 500, 10)
        rates = neuron.f_i_curve(I_values, T=800.0, transient=200.0)
        diffs = np.diff(rates)
        assert np.all(diffs >= -1e-9)  # non-decreasing

    def test_fi_curve_zero_below_rheobase(self):
        neuron = LIFNeuron(v_rest=-65.0, v_th=-50.0, g_L=10.0)
        rheobase = neuron.g_L * (neuron.v_th - neuron.v_rest)
        rates = neuron.f_i_curve([0.3 * rheobase, 0.6 * rheobase], T=500.0, transient=100.0)
        assert np.all(rates == 0.0)

    def test_euler_and_rk4_agree_qualitatively(self):
        # Both integrators should agree on "spikes vs no spikes" for a clearly
        # suprathreshold constant current, even though exact spike times differ.
        n_euler = LIFNeuron(dt=0.1)
        n_rk4 = LIFNeuron(dt=0.1)
        res_e = n_euler.simulate(I_ext=400.0, T=200.0, method="euler")
        res_r = n_rk4.simulate(I_ext=400.0, T=200.0, method="rk4")
        assert len(res_e["spike_times"]) > 0
        assert len(res_r["spike_times"]) > 0

    def test_invalid_integration_method_raises(self):
        neuron = LIFNeuron()
        with pytest.raises(ValueError):
            neuron.step(I=100.0, method="not_a_method")

    def test_scalar_current_requires_T(self):
        neuron = LIFNeuron()
        with pytest.raises(ValueError):
            neuron.simulate(I_ext=100.0)  # no T given


# --------------------------------------------------------------------- #
# AdExNeuron
# --------------------------------------------------------------------- #
class TestAdExNeuron:

    def test_spike_frequency_adaptation_increases_isis(self):
        # Strong adaptation parameters (a, b) with an above-rheobase drive
        # should produce a train whose ISIs grow, on average, over the train.
        neuron = AdExNeuron(a=4.0, b=80.0, tau_w=100.0)
        res = neuron.simulate(I_ext=400.0, T=1000.0)
        spikes = res["spike_times"]
        assert len(spikes) >= 4  # need a handful of spikes to see the trend
        isis = np.diff(spikes)
        # first ISI should be shorter than the last (adaptation slows firing down)
        assert isis[0] < isis[-1]

    def test_zero_adaptation_reduces_toward_lif_like_regularity(self):
        # With a=0, b=0, AdEx has no adaptation current at all; ISIs should
        # be much more uniform (near-constant) than with strong adaptation.
        neuron_adapt = AdExNeuron(a=4.0, b=80.0, tau_w=100.0)
        neuron_plain = AdExNeuron(a=0.0, b=0.0, tau_w=100.0)
        isis_adapt = np.diff(neuron_adapt.simulate(I_ext=400.0, T=1000.0)["spike_times"])
        isis_plain = np.diff(neuron_plain.simulate(I_ext=400.0, T=1000.0)["spike_times"])
        assert len(isis_adapt) > 2 and len(isis_plain) > 2
        cv_adapt = np.std(isis_adapt) / np.mean(isis_adapt)
        cv_plain = np.std(isis_plain) / np.mean(isis_plain)
        assert cv_plain < cv_adapt

    def test_w_jumps_by_b_on_spike(self):
        neuron = AdExNeuron(a=0.0, b=50.0, tau_w=1e6)  # freeze w decay (huge tau_w)
        res = neuron.simulate(I_ext=500.0, T=50.0)
        n_spikes = len(res["spike_times"])
        assert n_spikes >= 1
        # with negligible decay, w after N spikes should be close to N * b
        assert res["w"][-1] == pytest.approx(n_spikes * neuron.b, rel=0.15)

    def test_nullclines_match_analytical_formula(self):
        neuron = AdExNeuron(g_L=10.0, E_L=-65.0, v_T=-50.0, delta_T=2.0, a=2.0)
        v_range = np.array([-70.0, -60.0, -50.0])
        w_v, w_w = neuron.nullclines(v_range, I=0.0)
        expected_w_w = neuron.a * (v_range - neuron.E_L)
        np.testing.assert_allclose(w_w, expected_w_w)
        expected_w_v = (-neuron.g_L * (v_range - neuron.E_L)
                         + neuron.g_L * neuron.delta_T * np.exp((v_range - neuron.v_T) / neuron.delta_T))
        np.testing.assert_allclose(w_v, expected_w_v, rtol=1e-6)

    def test_v_rest_alias(self):
        neuron = AdExNeuron(E_L=-63.0)
        assert neuron.v_rest == neuron.E_L
