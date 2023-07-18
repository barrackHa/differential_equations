import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define the time span
t_start = 0
t_end = 2.5
dt = 0.01
t = np.arange(t_start, t_end, dt)

# Set the value of 'a'
a = 3

# Initial condition
sqrt2 = np.sqrt(2)
# x0 = np.array([sqrt2, sqrt2])  # Updated initial condition
x0 = np.array([1, 1])

# Function for the derivative of x
def dx_dt(x, a):
    M = np.array([[0, -a], [a, 0]])
    return M.dot(x)

# Create a figure and axis for the animation
fig, ax = plt.subplots()
fig.suptitle(
    r"""Phase Space Animation - $\dot{x}$=$M_{skew}x$"""
)
lim = max(x0[0] + a, x0[1] + a)
ax.set_xlim(-1*lim, lim)
ax.set_ylim(-1*lim, lim)
ax.set_xlabel('jPC1')
ax.set_ylabel('jPC2')
ax.set_title(rf'$\lambda$={a}, $x_{0}$={x0}')
ax.set_facecolor('darkgray')  # Set plot background to gray
# Create an empty scatter plot for the points in the animation
points = ax.scatter([], [], c=[], cmap='plasma')

# Create an empty quiver plot for x_dot
quiver = ax.quiver([], [], [], [], scale=10, color='red', label=r'$\dot{x}$')

# Function to update the scatter plot and quiver plot in each animation frame
def update(frame):
    x = np.zeros((len(t), 2))
    x[0] = x0
    for i in range(1, frame+1):
        x[i] = x[i-1] + dx_dt(x[i-1], a) * dt
    points.set_offsets(x[:frame, :])
    points.set_array(np.linspace(0, 1, frame))
    
    # Calculate x_dot at the current frame
    x_dot = dx_dt(x[frame-1], a) / 2
    
    # Update the quiver plot to represent x_dot
    quiver.set_offsets([x[frame-1]])  # Updated line
    
    quiver.set_UVC([x_dot[0]], [x_dot[1]])

    # Update the line plot from x to [0, 0]
    line.set_data([x[frame-1, 0], 0], [x[frame-1, 1], 0])  # Added line
    
    return points, quiver, line

# Create an empty line plot for the line from x to [0, 0]
line, = ax.plot([], [], 'k--', label='x') 
ax.legend()  
# Create the animation
animation = FuncAnimation(fig, update, frames=len(t), interval=20, blit=True)

# Display the color bar
cbar = plt.colorbar(points)
cbar.set_label('Time [%]')

# Save the animation as a GIF
# animation.save('tmp_figs/phase_space_animation.gif', writer='pillow')

# Display the animation
plt.show()
