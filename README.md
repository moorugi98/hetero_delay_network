# hetero_delay_network

Simulate a biologically plausible feed-forward modular network of Brunel networks (Brunel, 2000) using [NEST-simulator](https://www.nest-simulator.org/). 
By default, the membrane potential and the spikes are saved as data in a given ```PATH```. Measure basic 
properties of networks such as pairwise synchrony, revised local variation, firing rate and Fano facor. Train a 
ridge classifier using a state matrix of the network to test the classification ability of the network. Delay 
distributions can be altered depending on user's need. 

- params.py: define all the necessary parameters for the simulation.
- main.py: script to simulate a network save data to file.
- helpers.py: script which entails all necessary functions for data analysis.
- measures.py: script to compute useful properties of the network. The data is saved as 
```DataFrame```.
- train.py: train a ridge classifier using a state matrix of the network.
- plotter.py: script to plot the result.
