import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import odeint, solve_ivp
from scipy import optimize
from mpl_toolkits.mplot3d import Axes3D
from ews_helper import get_ews

def plot(tt, xx, sol, epsilon, a, show_plot=True):
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
    axs[0].legend()


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
    axs[1].legend()

    # Streamlines of X - Y space 
    w = 5
    Y, X = np.mgrid[-w:w:1000j, -w:w:1000j]
    U, V = dSdt(sol.t,[X,Y], epsilon, a)
    # print(X.shape, Y.shape, U.shape, V.shape)
    stream = axs[2].streamplot(X, Y, U, V, density = 1)
    axs[2].set_ylim([-5, 5])
    axs[2].set_xlim([-1*w, w])

    # left_atractor_line = axs[2].axvline(-1*r, c='black', ls='--')
    # right_atractor_line = axs[2].axvline(r, c='black', ls='--')
    # axs[2].axhline(0, c='black')
    # left_atractor_scat = axs[2].scatter((-1 * r),0, facecolors='none', edgecolors='black')
    # right_atractor_scat = axs[2].scatter(r,0, c='black')
    # axs[2].set_aspect('equal', adjustable='box')

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

def itoEulerMaruyama(model, y0, time, noise, args=None, save_derivative=False):
    ret_val = np.zeros((len(time), len(y0)))
    noise = np.array(noise)
    y0 = np.array(y0)
    ret_val[0, :] = y0
    dt = time[1] - time[0]
    derivatives = np.zeros((len(time), len(y0)))
    for i in range(1, len(time)):
        derivatives[i-1, :] = np.array(
            model(time[i], ret_val[i - 1, :], *args) 
                if args else model(ret_val[i - 1, :], time[i])
        ) 
        ret_val[i, :] = ret_val[i - 1, :] + \
                        derivatives[i-1, :] * dt + \
                        noise*np.random.normal(0,np.sqrt(dt),ret_val.shape[1])
        
    return ret_val if not save_derivative else (ret_val,derivatives)


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

    fig, axs = plt.subplots(1)
    axs.set_facecolor(plt.cm.gray(.95))
    # for win_size in [21,41,101]:
    win_size = 21 
    for i in range (1, 2):
        ar1s = []
        noisy_ar1s = []
        offset = i
        block_idxs = np.arange(sol.t.shape[0]-win_size, step=offset)
        x_sol = sol.y[0, : ]
        
        for i in block_idxs:
            try:
                lag0 = x_sol[i: i+win_size]
                lag1 = x_sol[i+offset: i+offset+win_size]
                n_lag0 = results[i: i+win_size, 0]
                n_lag1 = results[i+offset: i+offset+win_size, 0]
                ar1s.append(np.corrcoef(lag0, lag1)[0, 1])
                noisy_ar1s.append(np.corrcoef(n_lag0, n_lag1)[0, 1])
            except:
                print(f'error in {i}')

        
        # axs = fig.add_subplot(111, label=f'{i}', frame_on=False)        
        print(block_idxs[:len(ar1s)].shape, len(ar1s))
        axs.plot(sol.t[block_idxs[:len(ar1s)]], ar1s, label=f'Normal Form')
        axs.plot(sol.t[block_idxs[:len(ar1s)]], noisy_ar1s, label=f'With Noise')
        axs.legend()
        
    fig, axs = plot(tt, xx, sol, epsilon, a, show_plot=False)
    axs[0].plot(
        tt, results[:,0], color='crimson', label='Maroyama_x(t)', ls='-.'
    )
    # axs[0].plot(
    #     tt, results[:,1], color='green', label='Maroyama_y(t)', ls=':'
    # )
    axs[0].legend()
    plt.show()
