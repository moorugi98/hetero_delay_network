import sys
import numpy as np
import pylab
import nest

from params import *
import analysis


"""
select network type and stimuli existence
"""
sim_time = float(eval(sys.argv[3]))  # simulate time
mode = sys.argv[1]  # input should be either "random" or "structured"
stimuli_set = eval(sys.argv[2])  # input should be either "True" or "False"
print("topology: ", sys.argv[1])
print("stimuli on?: ", stimuli_set)
print("simulate for {} ms with {} extra onset time".format(sim_time, t_onset))



"""
initialise neurons and recording devices and connect recording devices
"""
pop_exci = []
pop_inhi = []
pop = []
spike_device = []



# init. neurons
for mod_i in range(module_depth):  # iterate over different modules
    current_mod_exci = nest.Create(model, N_E)  # excitatory neurons for each module
    pop_exci.append(current_mod_exci)
    current_mod_inhi = nest.Create(model, N_I)  # inhibitory
    pop_inhi.append(current_mod_inhi)
    current_mod_pop = current_mod_exci + current_mod_inhi
    pop.append(current_mod_pop)  # whole population for each module
    nest.SetStatus(current_mod_pop, "V_m", np.random.uniform(E_L, V_th, len(current_mod_pop))) # random init.membr.pot.
for mod_i in range(module_depth):  # don't use the same for loop to avoid id getting mixed
    # spike detector for each module
    spike_device.append(nest.Create("spike_detector", params={"withgid": True, "withtime": True, "start": t_onset}))
    nest.Connect(pop[mod_i], spike_device[mod_i], conn_spec={"rule": "all_to_all"})  # connect spike det with neurons



"""
create background noise input Poisson generator and connect them to neurons
"""
poisson_gen = nest.Create("poisson_generator", params={"rate": v_x * K_x_0}) # K_x different generator summed-up
nest.Connect(poisson_gen, pop[0], conn_spec={"rule": "all_to_all"})
if module_depth > 1: # for other modules with reduced num. synapses
    poisson_gen_rest = nest.Create("poisson_generator", params={"rate":v_x * K_x_rest})
    for mod_i in range(1, module_depth):
        nest.Connect(poisson_gen_rest, pop[mod_i], conn_spec={"rule": "all_to_all"})



"""
intra-module recurrent connections
"""
for mod_i in range(module_depth):
    nest.Connect(pop_exci[mod_i], pop[mod_i],
                 conn_spec={"rule": "pairwise_bernoulli", "p": epsilon}, syn_spec={"model": "syn_exci"})
    nest.Connect(pop_inhi[mod_i], pop[mod_i],
                 conn_spec={"rule": "pairwise_bernoulli", "p": epsilon}, syn_spec={"model": "syn_inhi"})



"""
inter-module feed-forward connections
"""
for mod_i in range(module_depth-1):
    nest.Connect(pop_exci[mod_i], pop[mod_i+1],
                 conn_spec={"rule":"pairwise_bernoulli", "p":p_ff}, syn_spec={"model": "syn_exci"})



"""
if stimuli is on, create input stimuli and corresponding poisson gen. and connect them to the input module
"""
if stimuli_set:
    # voltmeter for the whole network, record at every stimuli offset time
    voltage_device = nest.Create("voltmeter", params={"withgid": True, "withtime": True,
                                                      "interval": t_asterisk, "start": t_onset})
    # connect voltmeter to neurons
    [nest.Connect(voltage_device, pop_layer, conn_spec={"rule": "all_to_all"}) for pop_layer in pop]

    # init. stimuli pattern
    sym_seq_len = int(sim_time//t_asterisk)  # how many presentation during the simulation occurs
    sym_seq = np.zeros((num_stimulus, sym_seq_len), dtype=bool)
    # for each presentation time select just one stimulus to be activated
    for time_index in range(sym_seq_len):
        sym_seq[:, time_index][np.random.randint(low=0, high=num_stimulus)] = True

    gen_stim = [] # list of poisson generators functioning as stimuli
    for stim_sym in sym_seq: # for each symbolic repr. of stimuli
        stim_time = np.arange(t_onset, t_onset + t_asterisk*sym_seq_len, t_asterisk)  # time points to turn on/off
        stim_rate = np.zeros(stim_time.shape[0])
        stim_rate[stim_sym] = delta * v_x * input_spike_len  # assign right rate if it's on, 3*5*800 (linear sum-up)
        # create generator for each stimulus
        inhogen = nest.Create("inhomogeneous_poisson_generator")  # HERE IS THE ERROR (cannot set params by init.)
        nest.SetStatus(inhogen, {"rate_times": list(stim_time), "rate_values": list(stim_rate)})
        gen_stim.append(inhogen)



"""
switch for different network configurations
"""
if mode == "random":
    if stimuli_set:
        # connect stimuli generators to random sub-populations
        for stim_i in range(num_stimulus):
            nest.Connect(gen_stim[stim_i], tuple(np.random.choice(pop_exci[0], N_E_speci, replace=False)),
                         conn_spec={"rule": "all_to_all"})
            nest.Connect(gen_stim[stim_i], tuple(np.random.choice(pop_inhi[0], N_I_speci, replace=False)),
                         conn_spec={"rule": "all_to_all"})


    pass  # nothing more to do


elif mode == "structured":

    """
    select stimuli specific populations for each stimulus and each module
    """
    specific_exci = [[tuple(np.random.choice(pop_exci[mod_i], N_E_speci, replace=False))
    for stim_i in range(num_stimulus)] for mod_i in range(module_depth)]
    specific_inhi = [[tuple(np.random.choice(pop_inhi[mod_i], N_I_speci, replace=False))
    for stim_i in range(num_stimulus)] for mod_i in range(module_depth)]



    """
    stimuli specific feed-forward inter-module connection
    """
    if module_depth > 1:
        [[nest.Connect(specific_exci[mod_i][stim_i], specific_exci[mod_i+1][stim_i] + specific_inhi[mod_i+1][stim_i],
        conn_spec={"rule": "pairwise_bernoulli", "p": p_ff}) for stim_i in range(num_stimulus)]
        for mod_i in range(module_depth - 1)]  # exci. neurons connect to every stim.speci. neurons in the next module



    """
    random feed-forward inter-module connection
    """
    [nest.Connect(tuple(set(pop_exci[mod_i]) - set(specific_exci[mod_i][:])), pop[mod_i + 1],
    conn_spec={"rule": "pairwise_bernoulli", "p": p_ff})  # select only non-stimuli-speci. neurons as source
    for mod_i in range(module_depth - 1)]



    if stimuli_set:
        # connect generators to neurons
        for stim_i in range(num_stimulus):
            nest.Connect(gen_stim[stim_i], specific_exci[0][stim_i], conn_spec={"rule": "all_to_all"})
            nest.Connect(gen_stim[stim_i], specific_inhi[0][stim_i], conn_spec={"rule": "all_to_all"})


else:
    print("you fuckin' moron")
    exit()



"""
simulate
"""
nest.Simulate(sim_time+t_onset)



"""
collect data from recording devices
"""
if stimuli_set:
    data_volt = nest.GetStatus(voltage_device, "events")[0]  # data from voltmeter
    volt_times, volt_values = data_volt["times"], data_volt["V_m"]
    reshape_arr = lambda flat: np.reshape(flat, (-1, module_depth, N))  # reshape the data in array format
    volt_times = reshape_arr(volt_times)
    volt_values = reshape_arr(volt_values)

spike_times = []
spike_senders = []
for mod_i in range(module_depth):
    data_spike = nest.GetStatus(spike_device[mod_i], "events")[0]
    spike_times.append(data_spike["times"])  # list of spike times with each component repr. layer
    spike_senders.append(data_spike["senders"])

# compute useful measures
measures = np.zeros((3,module_depth))
for mod_i in range(module_depth):
    measures[:,mod_i] = [analysis.pairwise_corr(spike_times=spike_times[mod_i], spike_senders=spike_senders[mod_i],
                                           record_time=sim_time),
                         analysis.avg_firing_rate(spike_senders=spike_senders[mod_i], record_time=sim_time, N=N),
                         analysis.fano_factor(spike_times[mod_i])]

    print("corr: ", analysis.pairwise_corr(spike_times=spike_times[mod_i], spike_senders=spike_senders[mod_i],
                                           record_time=sim_time))
    print("avg.firing rate: ", analysis.avg_firing_rate(spike_senders=spike_senders[mod_i], record_time=sim_time, N=N))
    print("Fano factor: ", analysis.fano_factor(spike_times[mod_i]))

mod_i = 2
analysis.plot_raster(filename="raster_{}_stim={}.pdf".format(mode,stimuli_set),
                     spike_times=spike_times[mod_i], spike_senders=spike_senders[mod_i], layer=mod_i, num_to_plot=100)
titles = ["synchrony", "firing rate", "variability"]
ylabels  = ["Pearson CC", "spikes/sec", "Fano factor"]
for measure_i in range(3):
    analysis.plot_result(filename="{}_{}_stim={}.pdf".format(titles[measure_i],mode,stimuli_set),
                         arr_to_plot=measures[measure_i], title=titles[measure_i], ylabel=ylabels[measure_i])
# analysis.plot_V_m(times=volt_times[:,mod_i,0], voltages=volt_values[:,mod_i,:])
# np.save("train_res_{}.npy".format(mode),
#         analysis.train(volt_values=volt_values[1:], target_output=sym_seq))  # omit the first recording step
