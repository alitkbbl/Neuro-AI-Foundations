# 🧠 Neuro-AI-Foundations

**From a single leaky membrane to a thousand-neuron balanced network — implemented, derived, and visualized from first principles.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/vectorized-NumPy-orange.svg)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-educational%20%2B%20research--grade-brightgreen.svg)]()

---

## Why this repository exists

Most computational neuroscience courses teach the Leaky Integrate-and-Fire (LIF) neuron as if it were the whole story, and most machine learning courses treat "neurons" as stateless dot products. **Neither is a spiking neuron.**

`Neuro-AI-Foundations` closes that gap by building a strict conceptual ladder, one differential equation at a time:

```
Passive membrane  →  LIF  →  Nonlinear (Exponential) I&F  →  AdEx  →  Balanced spiking network
   (no spikes)      (hard      (soft, biophysical         (+ adaptation,   (1000 neurons,
                   threshold)   spike onset)                bursting)      Dale's law, AI regime)
```

Every model is derived from its governing equation, implemented as clean, tested, vectorized NumPy code, and paired with a notebook that lets you **see** the consequence of each added term — what a hard threshold buys you, why exponential spike onset matters, what adaptation currents do to an F-I curve, and how 1000 sparsely-connected leaky neurons self-organize into the irregular, asynchronous firing regime seen in cortex.

This project sits deliberately at the intersection of **computational neuroscience** and **AI engineering practice**: the science is textbook-accurate (Gerstner, Brette & Gerstner, Destexhe), and the code is written the way you'd write production ML infrastructure — abstract base classes, vectorized state updates, unit-tested dynamics, and reproducible notebooks.

---

## Scientific scope

| Concept | Where |
|---|---|
| RC membrane charging, no spiking | `PassiveNeuron` |
| Hard-threshold spiking, refractory period, F-I curve | `LIFNeuron` |
| Exponential spike-onset nonlinearity, rheobase, spike latency | `notebooks/02` |
| Spike-frequency adaptation, tonic vs. bursting vs. initial-burst regimes | `AdExNeuron` |
| Nullclines, fixed points, phase-plane geometry of excitability | `notebooks/04` |
| Synaptic conductance/current kernels (exponential, alpha) | `synapse_models.py` |
| Dale's law, sparse random connectivity, Poisson background drive | `network_builder.py` |
| Asynchronous-Irregular (AI) cortical-like dynamics | `notebooks/05` |

---

## Repository structure

```
Neuro-AI-Foundations/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── neuron_models.py      # BaseNeuron (ABC), PassiveNeuron, LIFNeuron, AdExNeuron
│   ├── synapse_models.py     # ExponentialSynapse, AlphaSynapse, PoissonSpikeGenerator
│   └── network_builder.py    # NeuronPopulation, SparseConnectivity, BalancedNetwork
├── notebooks/
│   ├── 01_Passive_and_LIF.ipynb
│   ├── 02_Non_Linear_Integrate_and_Fire.ipynb
│   ├── 03_The_Need_for_Adaptation_AdEx.ipynb
│   ├── 04_Phase_Plane_Analysis.ipynb
│   └── 05_Balanced_Recurrent_Network.ipynb
└── tests/
    ├── test_neuron_models.py
    ├── test_synapse_models.py
    └── test_network_builder.py
```

---

## Installation

```bash
git clone https://github.com/<your-username>/Neuro-AI-Foundations.git
cd Neuro-AI-Foundations
python3 -m venv .venv && source .venv/bin/activate     # optional but recommended
pip install -r requirements.txt

# Enable interactive widgets + register the kernel
python -m ipykernel install --user --name neuroai
jupyter notebook notebooks/01_Passive_and_LIF.ipynb
```

Run the test suite:

```bash
pytest tests/ -v
```

---

## The models, briefly

### 1. Passive (RC) neuron — the baseline

$$\tau_m \frac{dv}{dt} = -(v - v_{rest}) + R\,I(t)$$

A pure leaky integrator. No threshold, no spike, no reset. It exists to demonstrate *why* a threshold is needed at all — without one, "firing rate" has no meaning, only a graded, low-pass-filtered voltage.

### 2. Leaky Integrate-and-Fire (LIF)

Same subthreshold equation as the passive neuron, plus a hard rule:

$$v(t) \geq v_{th} \implies v \leftarrow v_{reset}, \quad \text{spike recorded}, \quad \text{refractory for } t_{ref}$$

This is the workhorse of large-scale spiking network simulation because it is cheap, and it already produces a well-behaved, saturating F-I curve.

### 3. AdEx — Adaptive Exponential Integrate-and-Fire

AdEx replaces the hard threshold with an **exponential nonlinearity** (biophysically closer to Hodgkin–Huxley sodium activation) and couples it to a slow **adaptation current** `w`:

$$C_m \frac{dv}{dt} = -g_L (v - E_L) + g_L \Delta_T \exp\!\left(\frac{v - v_T}{\Delta_T}\right) - w + I(t)$$

$$\tau_w \frac{dw}{dt} = a(v - v_{rest}) - w + b\,\tau_w \sum_f \delta(t - t^f)$$

On each spike ($v \to v_{peak}$): $v \leftarrow v_{reset}$ and $w \leftarrow w + b$ (the delta-function sum above is implemented exactly this way — an instantaneous jump in `w` at each spike time, not a smoothed approximation). Depending on just four parameters ($a$, $b$, $\tau_w$, $v_{reset}$), AdEx reproduces essentially every qualitative firing pattern seen in cortex: tonic spiking, spike-frequency adaptation, initial bursting, regular bursting, and irregular firing. Notebook 04 lets you feel this out interactively.

### 4. Synapses

Post-synaptic currents/conductances are generated by kernel filters applied to incoming spike trains — `ExponentialSynapse` (single decay time constant, computed with an efficient recursive update) and `AlphaSynapse` (rise-and-decay). `PoissonSpikeGenerator` produces homogeneous Poisson background spike trains used to drive the balanced network.

### 5. Balanced recurrent network

1000 LIF neurons, 80% excitatory / 20% inhibitory (**Dale's law enforced** — a neuron's outgoing weights are all excitatory or all inhibitory, never mixed), sparse random connectivity, and inhibitory synaptic weights scaled well above excitatory ones. Driven by independent Poisson background input to every cell. The result — reproduced in Notebook 05 with raster plots and population-rate traces — is the **Asynchronous-Irregular (AI)** regime characteristic of cortical microcircuits: low pairwise correlation, high single-neuron CV of inter-spike intervals, and a stable, low mean population rate, despite (and because of) strong recurrent excitation and inhibition fighting each other into balance.

---

## Design principles

- **Object-oriented core, vectorized execution.** `BaseNeuron` defines a common interface (`step`, `simulate`, `f_i_curve`) so every model is a drop-in replacement for any other in a notebook or a network. The 1000-neuron network is *not* 1000 Python objects in a loop — state is held in NumPy arrays and updated with vectorized operations, because that is what makes a 1000-neuron, 10,000-timestep simulation run in seconds rather than minutes.
- **Euler by default, RK4 for validation.** Every network simulation uses forward Euler at $dt \leq 0.1\,\text{ms}$ — the standard choice in the field for spiking network models, since spike-timing precision at that resolution is more than sufficient and the constant-time-step vectorized update is what makes networks tractable. `BaseNeuron` additionally offers a 4th-order Runge-Kutta integrator, provided purely as an *analytical cross-check* for single-neuron trajectories (it should never be used for a spiking network — see the docstring).
- **Tested dynamics, not just tested syntax.** The test suite checks actual dynamical properties: the passive neuron converges to its analytical steady state, the LIF F-I curve is monotonic and has a correct rheobase, AdEx reproduces spike-frequency adaptation (increasing ISIs), and the balanced network is stable (does not diverge or fully silence).

---

## References

- Gerstner, W., Kistler, W. M., Naud, R., & Paninski, L. (2014). *Neuronal Dynamics*. Cambridge University Press. [Free online](https://neuronaldynamics.epfl.ch/)
- Brette, R., & Gerstner, W. (2005). Adaptive Exponential Integrate-and-Fire Model as an Effective Description of Neuronal Activity. *Journal of Neurophysiology*, 94(5).
- Brunel, N. (2000). Dynamics of Sparsely Connected Networks of Excitatory and Inhibitory Spiking Neurons. *Journal of Computational Neuroscience*, 8(3).
- Destexhe, A., Rudolph, M., & Paré, D. (2003). The high-conductance state of neocortical neurons in vivo. *Nature Reviews Neuroscience*, 4(9).

## License

MIT — see `LICENSE`. Educational and research use is warmly encouraged; please cite the primary literature above if this repository informs published work.
