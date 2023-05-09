import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import odeint, solve_ivp
from scipy import optimize
from mpl_toolkits.mplot3d import Axes3D

r = 1
x0 = 0
y0 = 2
tt = np.linspace(0, 70, 1000)
xx = np.linspace(-5, 5, 100)

def dxdt(t, x, r):
    return r-x**2

def dydt(t, y, r):
    return -4*y

sol = solve_ivp(
    dxdt, t_span=[min(tt), max(tt)], 
    y0=[x0], t_eval=tt, args=(r,)
)

sol_y = solve_ivp(
    dydt, t_span=[min(tt), max(tt)], 
    y0=[y0], t_eval=tt, args=(r,)
)

fig, axs = plt.subplots(3)
# Plot solutions to ode's
sol_plot = axs[0].plot(sol.t, sol.y[0, :], color='b', label='x(t)')
axs[0].plot(sol_y.t, sol_y.y[0, :], color='orange', label='y(t)')
axs[0].set_xlabel('time')
axs[0].set_ylabel(r'x', rotation=0)
axs[0].set_xlim([0, 2])
axs[0].set_ylim([-2, 2])
axs[0].grid()
axs[0].legend()

# Plot phase space of derivatives
phase_space = axs[1].plot(xx, dxdt(sol.t,xx,r), color='b', label='x')
axs[1].plot(xx, dydt(sol_y.t,xx,r), color='orange', label='y')
scat = axs[1].scatter([], [], c='black')

axs[1].grid()
axs[1].set_xlabel('x')
axs[1].set_ylabel(r'$\dot{x}$', rotation=0)
axs[1].set_ylim([-1.1,1.1])
axs[1].legend()

# Streamlines of X - Y space 
w = 5
Y, X = np.mgrid[-w:w:1000j, -w:w:1000j]
U = dxdt(tt, X, r)
V = dydt(tt, Y, r)
stream = axs[2].streamplot(X, Y, U, V, density = 1)
axs[2].set_ylim([-1, 1])
axs[2].set_xlim([-1*w, w])
left_atractor_line = axs[2].axvline(-1*r, c='black', ls='--')
right_atractor_line = axs[2].axvline(r, c='black', ls='--')
axs[2].axhline(0, c='black')
left_atractor_scat = axs[2].scatter((-1 * r),0, facecolors='none', edgecolors='black')
right_atractor_scat = axs[2].scatter(r,0, c='black')
axs[2].set_aspect('equal', adjustable='box')

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