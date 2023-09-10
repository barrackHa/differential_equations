import numpy as np
import pandas as pd

from cdp_ews.helpers import itoEulerMaruyama

# Pitchfork bifurcation model

def super_pitchfork(t, S, epsilon, a=1):
    """
        x_dot = rx - ax^3
        r_dot = epsilon
    """
    x, r = S
    try:
        shaper = np.ones_like(r)
    except:
        try:
            shaper = np.ones(len(r))
        except:
            shaper = 1

    return [
        r*x - a*x**3,
        epsilon * shaper
    ]

def simulate_super_pitchfork(time=None, **config):
    try:
        epsilon = config['epsilon']
    except:
        epsilon = 0.1
    try:
        y0 = config['y0']
    except:
        
    results,derivatives = itoEulerMaruyama(
    model=super_pitchfork,
    y0=[x0, r0],
    time=time,
    noise=[sigma,0],args=(epsilon,),save_derivative=True
)