import sys
import os
import numpy as np
import nest

from params import *
from helpers import reshape_arr



"""
select network type and stimuli onset
"""
network_mode = sys.argv[1]  # input should be either "noise", "random" or "topo"
sim_time = float(eval(sys.argv[2]))  # simulate time
if int(sim_time) % int(t_asterisk) != 0:
    print("time of simulation should be integer multiples of each stimulus presentation time.")
    exit()
runindex = eval(sys.argv[3])  # n-th trial

print("network network_mode: ", sys.argv[1])
print("simulate for {} ms with {} ms extra onset time".format(sim_time, t_onset))



"""
set the network and simulation environment
"""
nest.SetKernelStatus(params={"total_num_virtual_procs": num_process, "local_num_threads": num_threads,
                             "overwrite_files": True, "print_time": timeprint})
nest.SetDefaults("iaf_cond_exp", {"E_L": E_L, "C_m": C_m, "t_ref": tau_ref, "V_th": V_th, "V_reset": V_reset,
                                  "E_ex": V_revr_E, "E_in": V_revr_I, "g_L": g_L, "tau_syn_ex": tau_E,
                                  "tau_syn_in": tau_I})
nest.CopyModel("static_synapse", "syn_exci", params={"delay": d, "weight": gbar_E})
nest.CopyModel("static_synapse", "syn_inhi", params={"delay": d, "weight": gbar_I})



"""
setting delays using nest internal functions
"""
delay_mode = []
delay_param = []
delay_dict = []

delay_mode.append(sys.argv[4])
delay_mode.append(sys.argv[5])
delay_param.append(eval(sys.argv[6]))
delay_param.append(eval(sys.argv[7]))

for _ in range(2):
    # intra-module recurrent connections delay
    print("delays:")
    if delay_mode[_] == "null":  # use a integer delay
        if delay_param[_] == "null":
            print("no specification")
            delay_dict.append(d)
        else:
            d_prime = delay_param[_]
            print("set delay to ", d_prime)
            delay_dict.append(d_prime)

    elif delay_mode[_] == "unimodal":  # use a Gaussian dist. of delay
        sigma = delay_param[_]
        print("unimodal dist. with sigma={}".format(sigma))
        delay_dict.append({"distribution": "normal_clipped", "low": 0.1, "mu": d, "sigma": sigma})

    else:
        print("fucked up")
        exit()

print("delay dict: ", delay_dict)



"""
initialise neurons and recording devices and connect recording devices
"""
pop_exci = []
pop_inhi = []
pop = []
spike_device = []

for mod_i in range(module_depth):  # iterate over different modules
    current_mod_exci = nest.Create(neuron_model, N_E)  # excitatory neurons for each module
    pop_exci.append(current_mod_exci)
    current_mod_inhi = nest.Create(neuron_model, N_I)  # inhibitory
    pop_inhi.append(current_mod_inhi)
    current_mod_pop = current_mod_exci + current_mod_inhi
    pop.append(current_mod_pop)  # whole population for each module
    nest.SetStatus(current_mod_pop, "V_m",
                   np.random.uniform(E_L, V_th, len(current_mod_pop)))  # random init. membrane potential
for mod_i in range(module_depth):  # don't use the same for loops to avoid gid getting mixed
    # spike detector for each module. Record only for 10 seconds
    spike_device.append(
        nest.Create("spike_detector", params={"withgid": True, "withtime": True, "start": t_onset,
                                              "stop": endspiketime + t_onset, "binary": True, "to_memory": True,
                                              "to_file": False}))
    nest.Connect(pop[mod_i], spike_device[mod_i], conn_spec={"rule": "all_to_all"})  # connect det. with neurons



"""
create background noise input Poisson generator and connect them to neurons
"""
poisson_gen = nest.Create("poisson_generator", params={"rate": v_x * K_x_0})  # K_x different generator summed-up
nest.Connect(poisson_gen, pop[0], conn_spec={"rule": "all_to_all"})
poisson_gen_rest = nest.Create("poisson_generator", params={"rate": v_x * K_x_rest})
for mod_i in range(1, module_depth):
    nest.Connect(poisson_gen_rest, pop[mod_i], conn_spec={"rule": "all_to_all"})



"""
intra-module recurrent connections. Every possible connection is connected with probability @epsilon(=0.1)
"""
for mod_i in range(module_depth):
    nest.Connect(pop_exci[mod_i], pop[mod_i],
                 conn_spec={"rule": "pairwise_bernoulli", "p": epsilon}, syn_spec={"model": "syn_exci",
                                                                                   "delay": delay_dict[0]})

    nest.Connect(pop_inhi[mod_i], pop[mod_i],
                 conn_spec={"rule": "pairwise_bernoulli", "p": epsilon}, syn_spec={"model": "syn_inhi",
                                                                                   "delay": delay_dict[0]})



"""
if stimuli is on, create input stimuli and corresponding poisson gen. and connect them to the input module
"""
if (network_mode == "random") or (network_mode == "topo"):
    # voltmeter for the whole network, record at every stimuli offset time
    voltage_device = nest.Create("voltmeter", params={"withtime": False, "withgid": True,
                                                      "interval": t_asterisk, "start": t_onset, "to_file": False,
                                                      "to_memory": True})
    # connect voltmeter to exci. neurons
    [nest.Connect(voltage_device, epop_module, conn_spec={"rule": "all_to_all"}) for epop_module in pop_exci]

    # init. stimuli pattern
    sym_seq_len = int(sim_time//t_asterisk)  # how many presentation during the simulation occurs
    sym_seq = np.zeros((num_stimulus, sym_seq_len), dtype=bool)
    # for each presentation time select just one stimulus to be activated
    for time_index in range(sym_seq_len):
        sym_seq[:, time_index][np.random.randint(low=0, high=num_stimulus)] = True

    gen_stim = []  # list of poisson generators functioning as stimuli
    for stim_sym in sym_seq:  # for each symbolic repr. of stimuli
        stim_time = np.arange(t_onset, t_onset + t_asterisk*sym_seq_len, t_asterisk)  # time points to turn on/off
        stim_rate = np.zeros(stim_time.shape[0])
        stim_rate[stim_sym] = delta * v_x * input_spike_len  # assign right rate if it's on, 3*5*800 (linear sum-up)
        # create generator for each stimulus
        inho_gen = nest.Create("inhomogeneous_poisson_generator")
        nest.SetStatus(inho_gen, {"rate_times": list(stim_time), "rate_values": list(stim_rate)})
        gen_stim.append(inho_gen)



"""
switch for different network configurations
"""
####
if (network_mode == "noise") or (network_mode == "random"):
####
    # inter-module feed-forward connections. Every exci. neuron projects to the next module with prob. @p_ff(=0.075)
    for mod_i in range(module_depth - 1):
        nest.Connect(pop_exci[mod_i], pop[mod_i + 1],
                     conn_spec={"rule": "pairwise_bernoulli", "p": p_ff}, syn_spec={"model": "syn_exci",
                                                                                    "delay": delay_dict[1]})

    # connect stimuli generators to random sub-populations
    if network_mode == "random":
        for stim_i in range(num_stimulus):
            nest.Connect(gen_stim[stim_i], tuple(np.random.choice(pop_exci[0], N_E_speci, replace=False)),
                         conn_spec={"rule": "all_to_all"})
            nest.Connect(gen_stim[stim_i], tuple(np.random.choice(pop_inhi[0], N_I_speci, replace=False)),
                         conn_spec={"rule": "all_to_all"})


####
elif network_mode == "topo":
####
    # printconnec = lambda source, target: print(len(nest.GetStatus(
    # nest.GetConnections(source=source, target=target), "target")))

    # select stimuli specific neurons and save them in lists of lists. specific_exci[module][stimuli]
    specific_exci = [[tuple(np.random.choice(pop_exci[mod_i], N_E_speci, replace=False))
                     for stim_i in range(num_stimulus)] for mod_i in range(module_depth)]
    specific_inhi = [[tuple(np.random.choice(pop_inhi[mod_i], N_I_speci, replace=False))
                     for stim_i in range(num_stimulus)] for mod_i in range(module_depth)]


    # stimuli specific feed-forward inter-module connection. Stimulus specific neurons connect to the next module
    # specific neurons with probability @p_ff(=0.075)
    [[nest.Connect(specific_exci[mod_i][stim_i], specific_exci[mod_i+1][stim_i] + specific_inhi[mod_i+1][stim_i],
                   conn_spec={"rule": "pairwise_bernoulli", "p": p_ff}, syn_spec={"model": "syn_exci",
                                                                                  "delay": delay_dict[1]})
      for stim_i in range(num_stimulus)] for mod_i in range(module_depth - 1)]



    # random feed-forward inter-module connection. Only stimuli non-specific exci. neurons projects randomly.
    for mod_i in range(module_depth-1):
        all_specific_exci = []
        for subpop in specific_exci[mod_i]:
            all_specific_exci.extend(list(subpop))  # now a list with gids of all neurons that are stimuli-specific

        nonspeci_exci = tuple(set(pop_exci[mod_i]).difference(set(all_specific_exci)))  # all neurons that aren't speci.
        nest.Connect(nonspeci_exci, pop[mod_i + 1],  # only non-specific ones project randomly to the next layer
                          conn_spec={"rule": "pairwise_bernoulli", "p": p_ff}, syn_spec={"model": "syn_exci",
                                                                                         "delay": delay_dict[1]})

    # connect gen. to the input module
    for stim_i in range(num_stimulus):
        nest.Connect(gen_stim[stim_i], specific_exci[0][stim_i], conn_spec={"rule": "all_to_all"})
        nest.Connect(gen_stim[stim_i], specific_inhi[0][stim_i], conn_spec={"rule": "all_to_all"})


####
else:
####
    print("you typed in the wrong network name")
    exit()








"""
simulate with extra time for stimuli to be set
"""
nest.Simulate(sim_time + t_onset)


spike_times = []
spike_senders = []
for mod_i in range(module_depth):
    data_spike = nest.GetStatus(spike_device[mod_i], "events")[0]
    spike_times.append(data_spike["times"])  # list of spike times with each component repr. layer
    spike_senders.append(data_spike["senders"])
    np.save(PATH + "spiketimes_run={}_{}_intra={}{}_inter={}{}.npy".
            format(runindex, network_mode, delay_mode[0], delay_param[0], delay_mode[1], delay_param[1]),
               np.array(spike_times))
    np.save(PATH + "spikesenders_run={}_{}_intra={}{}_inter={}{}.npy".
            format(runindex, network_mode, delay_mode[0], delay_param[0], delay_mode[1], delay_param[1]),
               np.array(spike_senders))




if (network_mode=="random") or (network_mode=="topo"):
    np.savetxt(PATH + "stimuli_run={}_{}_intra={}{}_inter={}{}".
               format(runindex, network_mode, delay_mode[0], delay_param[0], delay_mode[1], delay_param[1]), sym_seq)

    data_volt = nest.GetStatus(voltage_device, "events")[0]  # data from voltmeter
    volt_values = data_volt["V_m"]
    volt_values = reshape_arr(volt_values)
    np.save(PATH + "volt_run={}_{}_intra={}{}_inter={}{}.npy".
            format(runindex, network_mode, delay_mode[0], delay_param[0], delay_mode[1], delay_param[1]),
            volt_values)
