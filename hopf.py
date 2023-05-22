import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import odeint, solve_ivp

def dSdt(t, S, epsilon, omega, b):
    """S = [dr/dt, dtheta/dt, dmu/dt]"""
    r, theta, mu = S
    try:
        shaper = np.ones_like(mu)
    except:
        try:
            shaper = np.ones(len(mu))
        except:
            shaper = 1

    return [
        (mu*r) - (r**3),
        omega + (b*(r**2)),
        epsilon * shaper
    ]

if __name__ == '__main__':
    r_0 = 1
    omega = 1
    b = 1
    theta_0 = 0
    mu_0 = 0
    epsilon = 1
    a = 25
    tt = np.linspace(0, 4, 1000)
    xx = np.linspace(-30, 30, 100000)

    sol = solve_ivp(
        dSdt, t_span=[min(tt), max(tt)], 
        y0=[r_0, theta_0, mu_0], t_eval=tt, args=(epsilon, omega, b)
    )

    XX = sol.y[0, : ] * np.cos(sol.y[1, : ])
    YY = sol.y[0, : ] * np.sin(sol.y[1, : ])

    fig,axs = plt.subplots(1,1,figsize=(8,8))
    axs.set_facecolor(plt.cm.gray(.95))
    # axs.plot(sol.t, sol.y[0, : ], label='r')
    axs.plot(XX, YY)

    plt.show()
