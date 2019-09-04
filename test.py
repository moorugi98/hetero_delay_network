import numpy as np
import nest

stim_time = np.array([10.0, 210.0])
stim_rate = np.array([12000.0, 0.0])

# # 1. two different instances, use set status
# in1 = nest.Create("inhomogeneous_poisson_generator",
#                                 params={"rate_times": list(stim_time), "rate_values": list(stim_rate)})
# in2 = nest.Create("inhomogeneous_poisson_generator")
# nest.SetStatus(in2, {"rate_times": list(stim_time), "rate_values": list(stim_rate)})

# # 2. same instances, use set status
# in1 = nest.Create("inhomogeneous_poisson_generator",
#                                 params={"rate_times": list(stim_time), "rate_values": list(stim_rate)})
# in1 = nest.Create("inhomogeneous_poisson_generator")
# nest.SetStatus(in1, {"rate_times": list(stim_time), "rate_values": list(stim_rate)})

# 3. same instances, use set status for both
in1 = nest.Create("inhomogeneous_poisson_generator")
nest.SetStatus(in1, {"rate_times": list(stim_time), "rate_values": list(stim_rate)})
stim_time = np.array([3.0, 210.0])
stim_rate = np.array([1.0, 0.0])
in1 = nest.Create("inhomogeneous_poisson_generator")
nest.SetStatus(in1, {"rate_times": list(stim_time), "rate_values": list(stim_rate)})

# 4. try to append to list
list_gen = []
for _ in range(5):
    in1 = nest.Create("inhomogeneous_poisson_generator")
    nest.SetStatus(in1, {"rate_times": list(stim_time), "rate_values": list(stim_rate)})
    list_gen.append(in1)
print(list_gen)




# for _ in range(4):
#     print(_)
#     nest.Create("poisson_generator", params={"rate":[12.0,5.0], "start": [32.0,233.0]})