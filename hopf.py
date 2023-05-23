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
    tt = np.linspace(0, 10, 1000)
    xx = np.linspace(-500, 500, 100000)

    sol = solve_ivp(
        dSdt, t_span=[min(tt), max(tt)], 
        y0=[r_0, theta_0, mu_0], t_eval=tt, args=(epsilon, omega, b)
    )

    XX = sol.y[0, : ] * np.cos(sol.y[1, : ])
    YY = sol.y[0, : ] * np.sin(sol.y[1, : ])

    fig,axs = plt.subplots(1,1,figsize=(8,8))
    axs.set_facecolor(plt.cm.gray(.95))
    axs.plot(XX, YY)
    axs.scatter(XX[0], YY[0], c='r', s=50, label='start')
    axs.scatter(XX[-1], YY[-1], c='g', s=50, label='end')

    # Y, X = np.mgrid[-5:5:1000j, -5:5:1000j]
    # U, V, W = dSdt(sol.t,[X,Y, np.ones_like(Y)], epsilon, omega, b)
    # stream = axs.streamplot(X, Y, U, V, density = .5, color="crimson")
    RR = np.sqrt(XX**2 + YY**2)
    TT = np.arctan2(YY, XX)
    U, V, W = dSdt(sol.t,[RR,TT, 10*np.ones_like(XX)], epsilon, omega, b)
    UU = U * np.cos(TT) - V * np.sin(TT)
    VV = U * np.sin(TT) + V * np.cos(TT)
    axs.quiver(XX, YY, UU, VV, color="purple", alpha=1)

    plt.legend()
    plt.show()
