#%% 
import pytest
import sys 
from pathlib import Path
import numpy as np

from matplotlib import pyplot as plt

p = Path(__file__)
sys.path.append(str(p.parents[2]))

from cdp_ews.models import simulate_ts, simulate_super_pitchfork

#%%
def test_simulate_ts():
    model = lambda t, x: 0
    time = np.arange(0, 10, 1)
    y0 = [1]
    results,derivatives = simulate_ts(
        ode_model=model, time=time, y0=y0, noise=0, args=None
    )
    assert results.size == time.size
    assert np.array_equal(results, np.ones_like(results))
    assert np.array_equal(derivatives, np.zeros_like(derivatives))

    model = lambda t, x: 1
    y0 = [1]
    results,derivatives = simulate_ts(
        ode_model=model, time=time, y0=y0, noise=0, args=None
    )
    assert np.array_equal(derivatives[:-1], np.ones_like(derivatives[:-1]))
    assert np.array_equal(results.T.flatten(), time + 1)

def test_simulate_ts_with_noise():

    model = lambda t, x: 1
    time = np.arange(0, 10, 1)
    y0 = [1]
    noise = 1
    results,derivatives = simulate_ts(
        ode_model=model, time=time, y0=y0, noise=noise, args=None
    )

    assert not np.array_equal(results.T.flatten(), time + 1)
    assert np.array_equal(derivatives[:-1], np.ones_like(results[:-1]))
    
#%%

def test_simulate_super_pitchfork():
    time = np.arange(0, 4, 1)
    y0 = [1, -5]
    epsilon = 10
    noise = [1,0] # only noise x
    
    results, derivatives = simulate_super_pitchfork(
        time=time, y0=y0, epsilon=epsilon, a=1, noise=noise
    )

    print()
    plt.plot(time, results[:,0], label=f'{time.shape}')
    plt.plot(time, results[:,1], label=f'{time.shape}')
    plt.legend()
    plt.show()





# %%
if __name__ == '__main__':
    
    pytest.main([str(p)])
# %%
