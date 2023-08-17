#%%
import numpy as np
import matplotlib.pyplot as plt
from ews_helper import get_ews, get_acorr_decay_time

#%%
n = 11
arr = np.zeros(n)
arr[3:8] = 1

#%% 
tmp_a = arr - np.mean(arr)
ar1 = np.correlate(tmp_a, tmp_a, mode='full') / (np.cov(arr)* (n-1))
decay_t = get_acorr_decay_time(arr)

# %%
fig, ax = plt.subplots(1)
ax.plot(np.ones_like(ar1) *(1/np.e))
ax.plot(ar1)
ax.axvline(n-1+decay_t, color='red')

# %%
plt.show()
# %%
