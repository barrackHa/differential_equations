import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import odeint, solve_ivp
from scipy import optimize
from mpl_toolkits.mplot3d import Axes3D

def plot(tt, xx, sol, epsilon, a, show_plot=True):
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
            max(sol.y[0, :])+2
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

if __name__ == '__main__':

    r = 1
    x0 = 0
    y0 = 0
    epsilon = 1
    a = 25
    tt = np.linspace(0, 4, 1000)
    xx = np.linspace(-30, 30, 100000)

    sol = solve_ivp(
        dSdt, t_span=[min(tt), max(tt)], 
        y0=[x0, y0], t_eval=tt, args=(epsilon, a)
    )

    fig, axs = plt.subplots(1)
    win_size = 21 
    for i in range (1, 20):
        ar1s = []
        offset = i
        block_idxs = np.arange(sol.t.shape[0]-win_size, step=offset)
        x_sol = sol.y[0, : ]
        
        for i in block_idxs:
            try:
                lag0 = x_sol[i: i+win_size]
                lag1 = x_sol[i+offset: i+offset+win_size]
                ar1s.append(np.corrcoef(lag0, lag1)[0, 1])
            except:
                print(f'error in {i}')

        axs = fig.add_subplot(111, label=f'{i}', frame_on=False)        
        print(block_idxs[:len(ar1s)].shape, len(ar1s))
        axs.plot(sol.t[block_idxs[:len(ar1s)]], ar1s)

        
    plot(tt, xx, sol, epsilon, a, show_plot=False)
    plt.show()
    exit()

    tot_frames = 40
    def update(frame):

        delta = frame * (2.0 / (tot_frames))
        new_r = (r - delta)**3

        updated_sol = solve_ivp(
            dxdt, t_span=[min(tt), max(tt)], 
            y0=[x0], t_eval=tt, args=(new_r,)
        )

        sol_plot[0].set_xdata(updated_sol.t)
        sol_plot[0].set_ydata(updated_sol.y[0, :])

        phase_space[0].set_ydata(dxdt(updated_sol.t,xx,new_r))

        U = dxdt(tt, X, new_r)
        V = dydt(tt, Y, new_r)
        axs[2].collections = [] # clear lines streamplot
        axs[2].patches = [] # clear arrowheads streamplot
        axs[2].streamplot(X, Y, U, V, color='blue', arrowsize=2, density=1)
        axs[2].axhline(0, c='black')

        if 0 <= new_r:
            sqrt_r = np.sqrt(new_r)
            scat.set_offsets([[sqrt_r,0], [(-1 * sqrt_r),0]])

            left_atractor_line.set_linestyle('--')
            right_atractor_line.set_linestyle('--')
            left_atractor_line.set_xdata(-1*sqrt_r)
            right_atractor_line.set_xdata(sqrt_r)
            
            axs[2].scatter((-1 * sqrt_r),0, facecolors='none', edgecolors='black')
            axs[2].scatter(sqrt_r,0, c='black')

        else:
            scat.set_offsets([None,None])
            left_atractor_line.set_linestyle('none')
            right_atractor_line.set_linestyle('none')
        
        
        print(f'frame: {frame}, new_r: {new_r}')
        return (
            sol_plot, phase_space, scat, 
            left_atractor_line, right_atractor_line
        )

    # TODO: use init_func for animation
    ani = animation.FuncAnimation(fig=fig, func=update, frames=tot_frames, interval=50)
    ani.save('./animation.gif', writer='imagemagick', fps=60)
    # plt.show()