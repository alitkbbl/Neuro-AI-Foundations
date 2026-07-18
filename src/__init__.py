"""
Neuro-AI-Foundations source package.

Exposes the core neuron, synapse, and network-building classes at the
package level so notebooks can do e.g.:

    from src.neuron_models import LIFNeuron, AdExNeuron
    from src.synapse_models import ExponentialSynapse, PoissonSpikeGenerator
    from src.network_builder import BalancedNetwork
"""

from .neuron_models import BaseNeuron, PassiveNeuron, LIFNeuron, AdExNeuron
from .synapse_models import ExponentialSynapse, AlphaSynapse, PoissonSpikeGenerator
from .network_builder import NeuronPopulation, SparseConnectivity, BalancedNetwork

__all__ = [
    "BaseNeuron", "PassiveNeuron", "LIFNeuron", "AdExNeuron",
    "ExponentialSynapse", "AlphaSynapse", "PoissonSpikeGenerator",
    "NeuronPopulation", "SparseConnectivity", "BalancedNetwork",
]

__version__ = "0.1.0"
