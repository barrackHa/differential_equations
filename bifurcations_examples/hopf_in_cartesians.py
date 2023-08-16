import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import odeint, solve_ivp

def dSdt(t, S, epsilon, omega, b):
    """S = [dx/dt, dy/dt, dmu/dt]"""
    XX, YY, mu = S
    # Switch to polar coordinates
    r = np.sqrt(XX**2 + YY**2)
    theta = np.arctan2(YY, XX)
    try:
        shaper = np.ones_like(mu)
    except:
        try:
            shaper = np.ones(len(mu))
        except:
            shaper = 1
    # Solve for the derivatives in polar coordinates
    r_dot = ((mu*r) - (r**3))
    theta_dot = (omega + (b*(r**2)))
    # return to cartesian coordinates
    X_dot = r_dot * np.cos(theta)
    Y_dot = r_dot * np.sin(theta)

    return [
        X_dot,
        Y_dot,
        epsilon * shaper
    ]

if __name__ == '__main__':
    # r_0 = 1
    omega = 0
    b = 1
    # theta_0 = 0
    mu_0 = 1
    epsilon = 1
    tt = np.linspace(0,200, 1000)
    # xx = np.linspace(-500, 500, 100000)
    x0,y0 = 1, 1

    # print(dSdt(t=0, S=[x0, y0, mu_0], epsilon=epsilon, omega=omega, b=b))
    # exit()
    sol = solve_ivp(
        dSdt, t_span=[min(tt), max(tt)], 
        y0=[x0, y0, mu_0], t_eval=tt, args=(epsilon, omega, b)
    )

    XX = sol.y[0, : ] * np.cos(sol.y[1, : ])
    YY = sol.y[0, : ] * np.sin(sol.y[1, : ])

    fig,axs = plt.subplots(1,1,figsize=(8,8))
    axs.set_facecolor(plt.cm.gray(.95))
    axs.plot(XX, YY)
    axs.scatter(XX[0], YY[0], c='g', s=50, label='start')
    axs.scatter(XX[-1], YY[-1], c='r', s=50, label='end')

    RR = np.sqrt(XX**2 + YY**2)
    TT = np.arctan2(YY, XX)
    U, V, W = dSdt(sol.t,[RR,TT, 5*np.ones_like(XX)], epsilon, omega, b)
    UU = U * np.cos(V)
    VV = U * np.sin(V)
    axs.quiver(XX, YY, UU, VV, color="purple", alpha=1)

    # Y, X = np.mgrid[0:2:1000j, 0:2:1000j]
    # RR = np.sqrt(X**2 + Y**2)
    # TT = np.arctan2(Y, X)
    # U, V, W = dSdt(sol.t,[X,Y, 5*np.ones_like(Y)], epsilon, omega, b)
    # UU = U * np.cos(V)
    # VV = U * np.sin(V)
    # stream = axs.streamplot(X, Y, U, V, density =.5, color="crimson")

    plt.legend()
    plt.show()
