import sys
import numpy as np
import pandas as pd
import pylab
import nest

from params import *
import analysis



"""
select network type and stimuli onset
"""
runnum = eval(sys.argv[4])  # how often should the simulation be repeated?
sim_time = float(eval(sys.argv[3]))  # simulate time
if int(sim_time) % int(t_asterisk) != 0:
    print(sim_time, t_asterisk)
    print("time of simulation should be integer multiples of each stimulus presentation time.")
    exit()
mode = sys.argv[1]  # input should be either "random" or "structured"
stimuli_set = eval(sys.argv[2])  # input should be either "True" or "False"
print("topology: ", sys.argv[1])
print("stimuli on?: ", stimuli_set)
print("simulate for {} ms with {} extra onset time".format(sim_time, t_onset))



"""
initialise data structures to save the data
"""
# where important metrics will get saved
measures = pd.DataFrame(index=pd.MultiIndex.from_product([np.arange(runnum), np.arange(module_depth)]),
                        columns=["synchrony", "irregularity", "firing rate", "variability"])
measures.reset_index(level=1, inplace=True)
measures.rename(columns={"level_1": "module index"}, inplace=True)  # row is the index of run, everything else is column
train_res = []  # save the training accuracy score for each run



for runindex in range(runnum):
    """
    reset the network
    """
    nest.ResetKernel()
    nest.SetDefaults("iaf_cond_exp", {"E_L": E_L, "C_m": C_m, "t_ref": tau_ref, "V_th": V_th, "V_reset": V_reset,
                                      "E_ex": V_revr_E, "E_in": V_revr_I, "g_L": g_L, "tau_syn_ex": tau_E,
                                      "tau_syn_in": tau_I})
    nest.CopyModel("static_synapse", "syn_exci", params={"delay": d, "weight": gbar_E})
    nest.CopyModel("static_synapse", "syn_inhi", params={"delay": d, "weight": gbar_I})



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
                       np.random.uniform(E_L, V_th, len(current_mod_pop))) # random init.membrane potential
    for mod_i in range(module_depth):  # don't use the same for loops to avoid gid getting mixed
        # spike detector for each module
        spike_device.append(nest.Create("spike_detector", params={"withgid": True, "withtime": True, "start": t_onset}))
        nest.Connect(pop[mod_i], spike_device[mod_i], conn_spec={"rule": "all_to_all"})  # connect det. with neurons



    """
    create background noise input Poisson generator and connect them to neurons
    """
    poisson_gen = nest.Create("poisson_generator", params={"rate": v_x * K_x_0})  # K_x different generator summed-up
    nest.Connect(poisson_gen, pop[0], conn_spec={"rule": "all_to_all"})
    if module_depth > 1:  # for other modules with reduced num. synapses
        poisson_gen_rest = nest.Create("poisson_generator", params={"rate": v_x * K_x_rest})
        for mod_i in range(1, module_depth):
            nest.Connect(poisson_gen_rest, pop[mod_i], conn_spec={"rule": "all_to_all"})



    """
    intra-module recurrent connections. Every possible connection is connected with probability @epsilon(=0.1)
    """
    for mod_i in range(module_depth):
        nest.Connect(pop_exci[mod_i], pop[mod_i],
                     conn_spec={"rule": "pairwise_bernoulli", "p": epsilon}, syn_spec={"model": "syn_exci"})
        nest.Connect(pop_inhi[mod_i], pop[mod_i],
                     conn_spec={"rule": "pairwise_bernoulli", "p": epsilon}, syn_spec={"model": "syn_inhi"})



    """
    inter-module feed-forward connections. Every exci. neuron projects to the next module with prob. @p_ff(=0.075)
    """
    for mod_i in range(module_depth-1):
        nest.Connect(pop_exci[mod_i], pop[mod_i+1],
                     conn_spec={"rule": "pairwise_bernoulli", "p": p_ff}, syn_spec={"model": "syn_exci"})



    """
    if stimuli is on, create input stimuli and corresponding poisson gen. and connect them to the input module
    """
    if stimuli_set:
        # voltmeter for the whole network, record at every stimuli offset time
        voltage_device = nest.Create("voltmeter", params={"withgid": True, "withtime": True,
                                                          "interval": t_asterisk, "start": t_onset})
        # connect voltmeter to exci. neurons
        [nest.Connect(voltage_device, epop_module, conn_spec={"rule": "all_to_all"}) for epop_module in pop_exci]

        # init. stimuli pattern
        sym_seq_len = int(sim_time//t_asterisk)  # how many presentation during the simulation occurs
        sym_seq = np.zeros((num_stimulus, sym_seq_len), dtype=bool)
        # for each presentation time select just one stimulus to be activated
        for time_index in range(sym_seq_len):
            sym_seq[:, time_index][np.random.randint(low=0, high=num_stimulus)] = True

        gen_stim = [] # list of poisson generators functioning as stimuli
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
    ######
    if mode == "random":
    ######
        if stimuli_set:
            # connect stimuli generators to random sub-populations
            for stim_i in range(num_stimulus):
                nest.Connect(gen_stim[stim_i], tuple(np.random.choice(pop_exci[0], N_E_speci, replace=False)),
                             conn_spec={"rule": "all_to_all"})
                nest.Connect(gen_stim[stim_i], tuple(np.random.choice(pop_inhi[0], N_I_speci, replace=False)),
                             conn_spec={"rule": "all_to_all"})
        pass  # nothing more to do


    ######
    elif mode == "structured":
    ######
        """
        select stimuli specific populations for each stimulus and each module
        """
        specific_exci = [[tuple(np.random.choice(pop_exci[mod_i], N_E_speci, replace=False))
                         for stim_i in range(num_stimulus)] for mod_i in range(module_depth)]
        specific_inhi = [[tuple(np.random.choice(pop_inhi[mod_i], N_I_speci, replace=False))
                         for stim_i in range(num_stimulus)] for mod_i in range(module_depth)]



        """
        stimuli specific feed-forward inter-module connection. Stimulus specific neurons connect to the next module specific
        neurons with probability @p_ff(=0.075)
        """
        if module_depth > 1:
            [[nest.Connect(specific_exci[mod_i][stim_i], specific_exci[mod_i+1][stim_i] + specific_inhi[mod_i+1][stim_i],
                           conn_spec={"rule": "pairwise_bernoulli", "p": p_ff}, syn_spec={"model": "syn_exci"})
              for stim_i in range(num_stimulus)] for mod_i in range(module_depth - 1)]



        """
        random feed-forward inter-module connection. Only stimuli non-specific exci. neurons projects randomly. 
        """
        [nest.Connect(tuple(set(pop_exci[mod_i]) - set(specific_exci[mod_i][:])), pop[mod_i + 1],
        conn_spec={"rule": "pairwise_bernoulli", "p": p_ff}, syn_spec={"model": "syn_exci"})
        for mod_i in range(module_depth - 1)]



        if stimuli_set:
            # connect generators to neurons
            for stim_i in range(num_stimulus):
                nest.Connect(gen_stim[stim_i], specific_exci[0][stim_i], conn_spec={"rule": "all_to_all"})
                nest.Connect(gen_stim[stim_i], specific_inhi[0][stim_i], conn_spec={"rule": "all_to_all"})


    #######
    else:
    #######
        print("you fuckin' moron")
        exit()



    import time
    print("bis zum Simulationsanfang: ", time.process_time())
    """
    simulate with extra time for stimuli to be set
    """
    nest.Simulate(sim_time + t_onset)
    print("nach der Simulation: ", time.process_time())









    """
    collect data from recording devices
    """
    # membrane potential data is only needed if the stimuli were presented
    if stimuli_set:
        data_volt = nest.GetStatus(voltage_device, "events")[0]  # data from voltmeter
        volt_times, volt_values = data_volt["times"], data_volt["V_m"]
        reshape_arr = lambda flat: np.reshape(flat, (-1, module_depth, N_E))  # reshape the data in array format,
        volt_times = reshape_arr(volt_times)
        volt_values = reshape_arr(volt_values)

    # spike detector data
    spike_times = []
    spike_senders = []
    for mod_i in range(module_depth):
        data_spike = nest.GetStatus(spike_device[mod_i], "events")[0]
        spike_times.append(data_spike["times"])  # list of spike times with each component repr. layer
        spike_senders.append(data_spike["senders"])

    # compute useful measures
    for mod_i in range(module_depth):
        measures.iloc[runindex*module_depth + mod_i, 1:] = [analysis.pairwise_corr(spike_times=spike_times[mod_i],
                                                                                   spike_senders=spike_senders[mod_i],
                                                                                   record_time=sim_time),
                             analysis.revised_local_variation(spike_times=spike_times[mod_i],
                                                              spike_senders=spike_senders[mod_i]),
                             analysis.avg_firing_rate(spike_senders=spike_senders[mod_i],
                                                      record_time=sim_time, N=N),
                             analysis.fano_factor_normed(spike_times=spike_times[mod_i],
                                                         record_time=sim_time)]

    # save the membrane potential and the training result
    if stimuli_set:
        # save the voltage of each run
        np.save("volt_run={}_{}_stim={}.npy".format(mode, runindex, stimuli_set), volt_values)
        train_res.append(analysis.train(volt_values=volt_values, target_output=sym_seq))


# save data
measures = measures.astype(float)  # make sure that all the values saved are numeric

godf = lambda nparray: pd.DataFrame(nparray, dtype="float").stack().reset_index(level=-1, inplace=False)
print("1 accuracy: ", train_res[0])
accuracy, MSE = godf(train_res[0]), godf(train_res[1])
accuracy.rename(columns={"level_1": "module index", 0: "accuracy"}, inplace=True)
MSE.rename(columns={"level_1": "module index", 0: "MSE"}, inplace=True)

add_column = lambda df:df.insert(len(df.columns), "network_type", mode + "_" + str(stimuli_set))
[add_column(data) for data in [measures, accuracy, MSE]]

print("2 accuracy: ", accuracy)
measures.to_pickle("measures_{}_stim={}.p".format(mode, stimuli_set))
accuracy.to_pickle("accuracy_{}_stim={}.p".format(mode, stimuli_set))
MSE.to_pickle("MSE_{}_stim={}.p".format(mode, stimuli_set))


"""
raster plot is done in this script because I don't want to save spikes
"""
mod_i = 2  # which layer to plot (M2)
analysis.plot_raster(filename="raster_{}_stim={}.pdf".format(mode, stimuli_set),
                     spike_times=spike_times[mod_i], spike_senders=spike_senders[mod_i], layer=mod_i, num_to_plot=500,
                     plot_time=[t_onset, sim_time+t_onset])

print("Fertig: ", time.process_time())