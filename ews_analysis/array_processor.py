#%% 
import numpy as np
import matplotlib.pyplot as plt

#%% 

n = sim_time_arr_len = 70
# sim_time = np.linspace(0, 30, sim_time_arr_len)
sim_time = np.arange(0, sim_time_arr_len)

win_size=11
offset=3

#%%
block_idxs = np.arange(sim_time.shape[0]-win_size, step=offset)
print("-"*n)
for i in block_idxs:
    r = n - i - win_size
    print('x'*i + '-'*win_size + 'x'*r)



# %%

arr = np.arange(10)
sw_view = np.lib.stride_tricks.sliding_window_view(
    arr, 3)
# %%
s = np.lib.stride_tricks.as_strided(arr, (arr.size-4, 3), arr.strides*4)
# %%
