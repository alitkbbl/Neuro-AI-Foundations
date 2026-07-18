# Neuro-AI-Foundations


<p align="center">
      <img src="assets/banner.png" alt="Project Banner" width="100%">
</p>

---

### 📌 The Evolutionary Path

Each model is derived directly from its governing differential equation, with complexity added incrementally:

| Step | Model | Key Feature | Spike Behaviour |
|:---:|:---|:---|:---|
| 1 | Passive Membrane | RC Circuit | None |
| 2 | Leaky I&F (LIF) | Hard Threshold | Fixed threshold spike |
| 3 | Exponential I&F (EIF) | Exponential nonlinearity | Biophysical onset |
| 4 | AdEx | Adaptation current | Bursting & adaptation |
| 5 | Balanced Network | Sparse connectivity | Irregular, asynchronous firing |

### Engineering Highlights

Designed with production-grade standards to ensure accuracy and scalability:

*   **Vectorized Implementation:** Core dynamics are built with NumPy, optimized for efficient matrix calculations and high-performance simulation.
*   **Modular Architecture:** Utilizes Abstract Base Classes (ABCs) to define extensible, reusable neuron dynamics.
*   **Test-Driven Design:** Includes a comprehensive suite of unit tests to validate numerical accuracy and system stability across all models.
*   **Reproducibility:** Every model is accompanied by detailed Jupyter notebooks, documenting the mathematical derivation and visualizing the resulting dynamics.

### Project Objective
The goal of `Neuro-AI-Foundations` is to provide a clean, rigorous software foundation for spiking neural simulations. It enables researchers and engineers to focus on mathematical analysis and emergent network behaviors, rather than implementation overhead.

---

## ⚙️ Models used & key features

| Model | Class | Key features |
|---|---|---|
| **Passive (RC) membrane** | `PassiveNeuron` | Linear leaky integrator, **no spike mechanism**. Baseline used to show that a threshold — not just leaky integration — is what makes "firing rate" meaningful at all. |
| **Leaky Integrate-and-Fire** | `LIFNeuron` | Adds a hard voltage threshold, reset, and refractory period on top of the passive equation. Cheap and vectorizable — the workhorse of the 1000-neuron network. |
| **Exponential I&F (EIF)** | `AdExNeuron` (with `a=0, b=0`) | Replaces the hard threshold with a biophysically-motivated exponential nonlinearity. Produces a genuine spike upstroke, a lower/soft rheobase, and Type I excitability. |
| **AdEx (Adaptive Exp. I&F)** | `AdExNeuron` | Full model: exponential spike onset **+** a second state variable, the adaptation current `w`, which jumps at every spike and decays between them. Reproduces tonic spiking, spike-frequency adaptation, initial bursting, and regular bursting from the same two equations. |
| **Synaptic kernels** | `ExponentialSynapse`, `AlphaSynapse` | Convert discrete spike trains into continuous post-synaptic current, recursively updated at every Euler step. |
| **Background drive** | `PoissonSpikeGenerator` | Vectorized, independent homogeneous Poisson spike sources — used to drive every neuron in the network with realistic background noise. |
| **Balanced recurrent network** | `NeuronPopulation`, `SparseConnectivity`, `BalancedNetwork` | 1000 LIF neurons, 80% excitatory / 20% inhibitory, **Dale's law enforced**, sparse random connectivity, inhibition scaled well above excitation. Reproduces the cortex-like Asynchronous-Irregular (AI) firing regime. |

**Implementation features common to all models:**
- **Object-oriented core, vectorized execution.** `BaseNeuron` defines a common interface (`step`, `simulate`, `f_i_curve`) so every model is a drop-in replacement for any other. The 1000-neuron network is NumPy arrays updated in bulk, not 1000 Python objects in a loop.
- **Euler by default, RK4 for validation.** Every network simulation uses forward Euler at dt ≤ 0.1ms — the project standard. `BaseNeuron` additionally offers an optional 4th-order Runge-Kutta integrator, provided purely as a single-neuron analytical cross-check (see Notebook 01 and 02).
- **Tested dynamics, not just tested syntax.** 45 unit tests check actual dynamical properties: analytical steady states, monotonic F-I curves, Dale's-law sign constraints, spike-triggered adaptation, and network stability.

---

## 📓 Notebooks

| Notebook | What it covers |
|---|---|
| [`01_Passive_and_LIF.ipynb`](notebooks/01_Passive_and_LIF.ipynb) | Builds the passive membrane and LIF neuron, derives the F-I curve, shows how the refractory period caps it, and validates Euler against RK4 for LIF's linear dynamics. Ends by driving a LIF neuron with a realistic Poisson-filtered synaptic current instead of a step. |
| [`02_Non_Linear_Integrate_and_Fire.ipynb`](notebooks/02_Non_Linear_Integrate_and_Fire.ipynb) | Introduces the exponential (EIF) nonlinearity, compares it directly against LIF's hard threshold (voltage trajectories, rheobase, F-I curves, spike latency), and revisits Euler vs. RK4 — this time for genuinely nonlinear dynamics, where the choice of integrator actually matters. |
| [`03_The_Need_for_Adaptation_AdEx.ipynb`](notebooks/03_The_Need_for_Adaptation_AdEx.ipynb) | Switches on the AdEx adaptation current `w`, visualizes spike-frequency adaptation directly (instantaneous vs. steady-state F-I curves), and shows a gallery of four qualitatively different firing patterns from the same two equations. |
| [`04_Phase_Plane_Analysis.ipynb`](notebooks/04_Phase_Plane_Analysis.ipynb) | Explains *why* those firing patterns emerge, geometrically, via the `(v, w)` phase plane and its nullclines/fixed points. Includes a live `ipywidgets` explorer — drag `a`, `b`, `τ_w`, `v_reset`, and `I` and watch the nullclines and firing pattern shift in real time. |
| [`05_Balanced_Recurrent_Network.ipynb`](notebooks/05_Balanced_Recurrent_Network.ipynb) | Scales up to a 1000-neuron recurrent LIF network with Dale's law and Poisson background drive, and demonstrates the Asynchronous-Irregular (AI) regime with raster plots, population rate, ISI-CV statistics, and a direct empirical test of the excitation/inhibition balance mechanism. |

---

## 🛠️ Installation

**1. Clone the repository:**
```bash
git clone https://github.com/alitkbble/Neuro-AI-Foundations.git
cd Neuro-AI-Foundations
```

**2. Set up a virtual environment (Recommended):**
Using an isolated environment ensures dependency versions do not conflict with your other projects.

```bash
# Create the virtual environment
python3 -m venv .venv

# Activate on Linux / macOS:
source .venv/bin/activate

# Activate on Windows:
.venv\Scripts\activate
```

**3. Install dependencies:**
```bash
# Upgrade pip to avoid installation issues with scientific packages
python -m pip install --upgrade pip

# Install the required packages
pip install -r requirements.txt
```

**4. Register the Jupyter Kernel and Launch:**
To ensure Jupyter runs within the correct environment and interactive widgets work properly, register the local kernel before launching:

```bash
python -m ipykernel install --user --name=neuroai --display-name "Python (Neuro-AI)"
jupyter notebook notebooks/01_Passive_and_LIF.ipynb
```

Run the test suite:

```bash
pytest tests/ -v
```

---

## 📜 The models, briefly

### Passive (RC) neuron — the baseline

$$\tau_m \frac{dv}{dt} = -(v - v_{rest}) + R\,I(t)$$

A pure leaky integrator. No threshold, no spike, no reset. It exists to demonstrate *why* a threshold is needed at all — without one, "firing rate" has no meaning, only a graded, low-pass-filtered voltage.

### Leaky Integrate-and-Fire (LIF)

Same subthreshold equation as the passive neuron, plus a hard rule:

$$v(t) \geq v_{th} \implies v \leftarrow v_{reset}, \quad \text{spike recorded}, \quad \text{refractory for } t_{ref}$$

This is the workhorse of large-scale spiking network simulation because it is cheap, and it already produces a well-behaved, saturating F-I curve.

### AdEx — Adaptive Exponential Integrate-and-Fire

AdEx replaces the hard threshold with an **exponential nonlinearity** (biophysically closer to Hodgkin–Huxley sodium activation) and couples it to a slow **adaptation current** `w`:

$$C_m \frac{dv}{dt} = -g_L (v - E_L) + g_L \Delta_T \exp\!\left(\frac{v - v_T}{\Delta_T}\right) - w + I(t)$$

$$\tau_w \frac{dw}{dt} = a(v - v_{rest}) - w + b\,\tau_w \sum_f \delta(t - t^f)$$

On each spike ($v \to v_{peak}$): $v \leftarrow v_{reset}$ and $w \leftarrow w + b$ (the delta-function sum above is implemented exactly this way — an instantaneous jump in `w` at each spike time, not a smoothed approximation). Depending on just four parameters ($a$, $b$, $\tau_w$, $v_{reset}$), AdEx reproduces essentially every qualitative firing pattern seen in cortex: tonic spiking, spike-frequency adaptation, initial bursting, regular bursting, and irregular firing. Notebook 04 lets you feel this out interactively.

### Synapses

Post-synaptic currents/conductances are generated by kernel filters applied to incoming spike trains — `ExponentialSynapse` (single decay time constant, computed with an efficient recursive update) and `AlphaSynapse` (rise-and-decay). `PoissonSpikeGenerator` produces homogeneous Poisson background spike trains used to drive the balanced network.

### Balanced recurrent network

1000 LIF neurons, 80% excitatory / 20% inhibitory (**Dale's law enforced** — a neuron's outgoing weights are all excitatory or all inhibitory, never mixed), sparse random connectivity, and inhibitory synaptic weights scaled well above excitatory ones. Driven by independent Poisson background input to every cell. The result — reproduced in Notebook 05 with raster plots and population-rate traces — is the **Asynchronous-Irregular (AI)** regime characteristic of cortical microcircuits: low pairwise correlation, high single-neuron CV of inter-spike intervals, and a stable, low mean population rate, despite (and because of) strong recurrent excitation and inhibition fighting each other into balance.

---

## 📈 Main results

The primary objective of this project is to show that everything culminates in a **genuinely emergent, network-level phenomenon** — one that no single neuron in the simulation, however sophisticated, produces on its own.

Running the default 1000-neuron network in `05_Balanced_Recurrent_Network.ipynb` (800 excitatory / 200 inhibitory, 10% connection probability, inhibition scaled 5x above excitation) produces:

- A **scattered, non-synchronized raster** — no visible stripes or population-wide bursts — with a population firing rate that fluctuates around a stable mean (~3 Hz here) rather than oscillating or diverging.
- Individually **irregular** spike trains: a mean ISI coefficient of variation around 0.5, well above the near-zero CV of a clock-like or purely mean-driven neuron, though below the CV = 1 of a strictly memoryless Poisson process.
- A direct demonstration that this stability is a **balance** effect, not a weak-input effect: sweeping the inhibitory scaling factor `g` shows the population rate rising with excitatory strength when inhibition is weak, and becoming largely *insensitive* to excitatory strength once inhibition is strong enough to track and cancel it.

That's the payoff of the whole ladder: the neurons doing the work here are the simplest model in the entire project (plain LIF, no adaptation, no nonlinearity) — the richness comes entirely from Dale's law, sparse recurrent connectivity, and the excitation/inhibition balance, at scale.

![Balanced network raster and population rate](assets/balanced_network_result.png)

---

## ⚖️ License

MIT — see [LICENSE](LICENSE). Educational and research use is warmly encouraged; please cite the primary literature referenced above if this repository informs published work.
