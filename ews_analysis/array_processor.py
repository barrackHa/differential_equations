#%% 
import numpy as np
import matplotlib.pyplot as plt

#%% 

n = sim_time_arr_len = 31
sim_time = np.linspace(0, 30, sim_time_arr_len)
win_size=9
offset=3

#%%
block_idxs = np.arange(sim_time.shape[0]-win_size, step=offset)
print("-"*n)
for i in block_idxs:
    r = n - i - win_size
    print('x'*i + '-'*win_size + 'x'*r)



# %%
fig, ax = plt.subplots(2, 1, figsize=(10, 10))
type(ax)
# %%
