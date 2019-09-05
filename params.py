
# neuron params
C_m = 250.0  # capacitance (pF)
E_L = -70.0  # leak reversal potential (mV)
# tau_m = 15.0  # membrane time constant (ms) = C_m / g_L
V_reset = - 60.0  # reset potential
tau_ref = 2.0  # absolute refractory period
g_L = 16.7  # peak conductance (nS)
V_th = -50.0  # threshold (mV)

# synapse params
tau_E = 5.0  # syn. decay time constant exci.
tau_I = 10.0  # syn. decay time constant inhi.
V_revr_E = 0.0  # exci. reversal potential
V_revr_I = -80.0  # inhi. reversal potential

# connection params
d = 1.5  # synaptic transmission delay
gbar_E = 1.0  # exci. syn. cond.
gamma = 16  # scaling factor for inhi. syn. cond.
gbar_I = -1 * gamma * gbar_E  # inhi. syn. cond.
epsilon = 0.1  # internal recurrent rand. conn. prob., sparse connection
p_x_0 = epsilon  # background connection prob. for the input module
p_x_rest = 0.25 * epsilon  # same for other modules
p_ff = 0.75 * epsilon  # feed-forward intra connection prob.
                           
# models and default settings
neuron_model = "iaf_cond_exp"



# network params
N = 10000  # total num. of neurons in each module
N_E = int(0.8 * N)  # exci.
N_I = int(0.2 * N)  # inhi.
N_E_speci = int(epsilon * N_E)  # num. of stimulus specific exci. neurons in each module, not needed for random network
N_I_speci = int(epsilon * N_I)  # inhi.
# K_E = int(epsilon * N_E) # exci.pre synapse number for each neuron
# K_I = int(epsilon * N_I) # inhi.
module_depth = 4

# background noise
v_x = 5.0 # intensity of a Poisson process
N_X = N_E # num. of background neurons
K_x_0 = int(p_x_0 * N_X) # num. of synapses for background noise for the input module
K_x_rest = int(p_x_rest * N_X)
#
# # inter-module connection
# K_M = int(p_ff * N_E) # num. of exci. syn. between modules

# input stimuli
num_stimulus = 10  # num. of stimuli
delta = 3  # Poisson process rate factor for stimuli
t_onset = 1.0

# stimuli onset time
t_asterisk = 200.0  # how long each Poisson process lasts
input_spike_len = 800  # dim of input in spikes
