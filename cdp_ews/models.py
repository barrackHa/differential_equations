import numpy as np
import pandas as pd

from cdp_ews.helpers import itoEulerMaruyama

def simulate_ts(ode_model, time, y0, noise, args):
    
    results,derivatives = itoEulerMaruyama(
        model=ode_model,
        y0=y0,
        time=time,
        noise=noise, args=args, save_derivative=True
    )
    return results,derivatives

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

def simulate_super_pitchfork(time, y0, epsilon, a, noise):
        
    results, derivatives = simulate_ts(
        ode_model=super_pitchfork,
        time=time,
        y0=y0,
        noise=noise, args=(epsilon, a)
    )

    return results,derivatives