#%% 
import pytest
import sys 
from pathlib import Path
import numpy as np

p = Path(__file__)
sys.path.append(str(p.parents[2]))

from cdp_ews.helpers import itoEulerMaruyama

#%%
def test_itoEulerMaruyama():
    # model, y0, time, noise, args=None, save_derivative=False
    time = np.arange(0, 10, 0.1)

    def trivial_model(x, t, args=None):
        return 0
    
    res, der = itoEulerMaruyama(
        trivial_model, y0=[0], time=time, noise=0, args=None, save_derivative=True
    )
    
    assert np.array_equal(res, np.zeros_like(res))
    assert np.array_equal(der, np.zeros_like(der))

    def trivial_model(x, t, args=None):
        return 1
    
    res, der = itoEulerMaruyama(
        trivial_model, y0=[0], time=time, noise=0, args=None, save_derivative=True
    )
    assert np.array_equal(der[:-1], np.ones_like(der[:-1]))
    assert np.array_equal(res.T.flatten(), (1*time + 0))



# %%
if __name__ == '__main__':
    
    pytest.main([str(p)])
# %%
