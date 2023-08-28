import numpy as np
from matplotlib import pyplot as plt

# Pitchfork bifurcation model

def dSdt(t, S, epsilon, a=1):
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
        r*x - x**3,
        epsilon * shaper
    ]

def plot_r_of_t(ax, time, r_of_t, epsilon, a=1):
    ax.plot(time, r_of_t, label='r(t)')
    ax.set_xlabel('Time')
    ax.set_ylabel('r(t)')

    ax.grid()
    # make ax grey
    ax.set_facecolor(plt.cm.gray(.85))

    t_star = time[np.where(r_of_t >= 0)[0][0]]

    ax.axvline(t_star, c='r', label='t*', ls='--')

    ax.set_ylim(-1*epsilon, epsilon)
    ax.set_xlim(t_star - 3, t_star + 3)
    ax.legend()

    ttl = r"""dr/dt=$\epsilon$, r(0)=r_0 => r(t)=r_0+$\epsilon$t)"""
    ttl += f'\n t*={t_star}, epsilon={epsilon}, r_0={r_of_t[0]}, a={a}'
    ax.set_title(ttl, fontsize=10)
    
    return ax

