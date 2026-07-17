"""
Network construction and simulation utilities.

This file will be completed in later phases.
"""
import numpy as np

class VectorizedLIFNetwork:
    """
    A network of LIF neurons simulated using vectorized NumPy operations 
    for high performance. Used to study Balanced Recurrent Networks.
    """
    def __init__(self, num_neurons=1000, dt=0.1, tau_m=20.0, E_L=-70.0, v_th=-50.0, v_reset=-65.0):
        self.N = num_neurons
        self.dt = dt
        self.tau_m = tau_m
        self.E_L = E_L
        self.v_th = v_th
        self.v_reset = v_reset
        
        # State vector: initialized uniformly between reset and threshold
        self.v = np.random.uniform(v_reset, v_th, self.N)
        self.spikes = [] # Will store tuples of (time, neuron_id)
        
        # Synaptic weights matrix (N x N) - Setup as a sparse representation or dense
        self.W = np.zeros((self.N, self.N)) 

    def connect_random(self, p_connect=0.1, weight_mu=1.0):
        """Creates a random Erdos-Renyi connectivity graph."""
        mask = np.random.rand(self.N, self.N) < p_connect
        self.W[mask] = weight_mu
        np.fill_diagonal(self.W, 0.0) # No self-connections

    def step(self, t, I_external_vector):
        """
        Advances the entire network by one time step `dt`.
        I_external_vector is an array of size N representing input to each neuron.
        """
        # 1. Update voltages using vectorized Euler
        dv = (-(self.v - self.E_L) + I_external_vector) / self.tau_m
        self.v += dv * self.dt
        
        # 2. Check for spikes (Boolean mask)
        spiked_mask = self.v >= self.v_th
        spiked_indices = np.where(spiked_mask)[0]
        
        # 3. Record spikes
        for idx in spiked_indices:
            self.spikes.append((t, idx))
            
        # 4. Compute synaptic input for the next step (Matrix multiplication)
        # W[:, spiked_indices] sums the weights from spiking presynaptic neurons
        synaptic_input = np.sum(self.W[:, spiked_indices], axis=1)
        
        # 5. Reset spiked neurons
        self.v[spiked_mask] = self.v_reset
        
        return synaptic_input
