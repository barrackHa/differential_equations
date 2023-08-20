import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import odeint, solve_ivp
from scipy import optimize
from mpl_toolkits.mplot3d import Axes3D
from ews_helper import get_ews, itoEulerMaruyama
from scipy.stats import linregress
from pathlib import Path

def plot(tt, xx, sol, epsilon, a, show_plot=True, stream_dim=5):
    # plt.plot(sol.y[0, :], sol.y[1, :])
    fig, axs = plt.subplots(3)
    # Plot solutions to ode's
    sol_plot = axs[0].plot(sol.t, sol.y[0, :], color='b', label='x(t)')
    axs[0].plot(sol.t, sol.y[1, :], color='orange', label='y(t)')
    axs[0].set_xlabel('time')
    axs[0].set_ylabel(r'x&y', rotation=0)
    axs[0].set_xlim([0, 3])
    axs[0].set_ylim(
        [
            min(sol.y[0, :]), 
            max(sol.y[0, :])+5
    ])
    axs[0].grid()
    
    box = axs[0].get_position()
    axs[0].set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    axs[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # axs[0].legend()


    # Plot phase space of derivatives
    X_dot, Y_dot = dSdt(sol.t,[xx,xx], epsilon, a)

    phase_space = axs[1].plot(xx, X_dot, color='b', label='x')
    axs[1].plot(xx, Y_dot, color='orange', label='y')
    scat = axs[1].scatter([], [], c='black')

    axs[1].grid()
    axs[1].set_xlabel('x')
    axs[1].set_ylabel(r'$\dot{x}$', rotation=0)
    axs[1].set_ylim([-2,2500])
    axs[1].set_xlim([min(xx),max(xx)])
    
    box = axs[1].get_position()
    axs[1].set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # axs[1].legend()

    # Streamlines of X - Y space 
    w = stream_dim
    Y, X = np.mgrid[-w:w:1000j, -w:w:1000j]
    U, V = dSdt(sol.t,[X,Y], epsilon, a)
    stream = axs[2].streamplot(X, Y, U, V, density = 1)
    axs[2].set_ylim([-1*w, w])
    axs[2].set_xlim([-1*w, w])

    if show_plot:
        plt.show()

    return fig, axs

def dSdt(t, S, epsilon, a):
    x, y = S
    try:
        shaper = np.ones_like(y)
    except:
        try:
            shaper = np.ones(len(y))
        except:
            shaper = 1

    return [
        y + a*x**2 - x**3,
        epsilon * shaper
    ]

if __name__ == '__main__':

    r = 1
    x0 = 0
    y0 = 0
    epsilon = 1
    gamma = 0.0001 # noise intensity
    a = 25
    tt = np.linspace(0, 4, 1000)
    xx = np.linspace(-30, 30, 100000)

    sol = solve_ivp(
        dSdt, t_span=[min(tt), max(tt)], 
        y0=[x0, y0], t_eval=tt, args=(epsilon, a)
    )

    # Euler Maruyama
    results,derivatives = itoEulerMaruyama(
        model=dSdt,
        y0=[x0, y0],
        time=tt,
        noise=[gamma,0],args=(epsilon, a),save_derivative=True
    )

    slope,intercept,_,_,_ = linregress(tt, results[:,0])
    # mu=0=intercept+slope*t iff t=-intercept/slope
    t_star = tt[np.where(results[:,0] >= 1)[0][0]]


    fig, axs = plt.subplots(3,1)
    for ax in axs:
        ax.set_facecolor(plt.cm.gray(.95))
    # for win_size in [21,41,101]:
    win_size = 21 
    for i in range (1, 2):
        block_idxs, ar1s, decays, vars = get_ews(
            sol.t, sol.y[0, : ], win_size=win_size, offset=i
        )
        block_idxs, noisy_ar1s, noisy_decays, noisy_vars = get_ews(
            sol.t, results[:,0], win_size=win_size, offset=i
        )
        
        # Plot AR1s
        axs[0].plot(sol.t[block_idxs[:len(ar1s)]], ar1s, label=f'No Noise')
        axs[0].plot(sol.t[block_idxs[:len(ar1s)]], noisy_ar1s, label=f'With Noise')
        axs[0].axvline(
            t_star, color='k', linestyle='--', 
            alpha=0.5, label='x=0'
        )

        # Plot Decays
        axs[1].plot(sol.t[block_idxs[:len(decays)]], decays, label=f'No Noise')
        axs[1].plot(sol.t[block_idxs[:len(decays)]], noisy_decays, label=f'With Noise')
        axs[1].axvline(
            t_star, color='k', linestyle='--', 
            alpha=0.5, label='x=0'
        )

        # Plot Variance
        axs[2].plot(sol.t[block_idxs[:len(vars)]], vars, label=f'No Noise')
        axs[2].plot(sol.t[block_idxs[:len(vars)]], noisy_vars, label=f'With Noise')
        axs[2].axvline(
            t_star, color='k', linestyle='--', 
            alpha=0.5, label='x=0'
        )

        for ax in axs:
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

            # Put a legend to the right of the current axis
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            # ax.legend()
    fig.savefig(Path(__file__).parents[0]/f'tmp_figs/sdnode_ews.png')
        
    fig, axs = plot(tt, xx, sol, epsilon, a, show_plot=False)
    axs[0].plot(
        tt, results[:,0], color='crimson', label='Maroyama_x(t)', ls='-.'
    )

    box = axs[0].get_position()
    axs[0].set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    axs[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # axs[0].legend()
    # fig.savefig(Path(__file__).parents[0]/f'tmp_figs/sdnode_data.png')
    plt.show()
