import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define the time span
t_start = 0
t_end = 1.3
dt = 0.01
t = np.arange(t_start, t_end, dt)

# Set the value of 'a'
a = 6

# Initial condition
sqrt2 = np.sqrt(2)
x0 = np.array([-sqrt2, sqrt2])  # Updated initial condition

# Function for the derivative of x
def dx_dt(x, a):
    M = np.array([[0, -a], [a, 0]])
    return M.dot(x)


# Create a figure and axis for the animation
fig, ax = plt.subplots()
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.set_xlabel('jPC1')
ax.set_ylabel('jPC2')
ax.set_title('Phase Space Animation')
ax.set_facecolor('darkgray')  # Set plot background to gray


# Create an empty scatter plot for the points in the animation
points = ax.scatter([], [], c=[], cmap='plasma')

# Function to update the scatter plot in each animation frame
def update(frame):
    x = np.zeros((len(t), 2))
    x[0] = x0
    for i in range(1, frame+1):
        x[i] = x[i-1] + dx_dt(x[i-1], a) * dt
    points.set_offsets(x[:frame, :])
    points.set_array(np.linspace(0, 1, frame))
    return points,

# Create the animation
animation = FuncAnimation(fig, update, frames=len(t), interval=20, blit=True)

# Display the color bar
cbar = plt.colorbar(points)
cbar.set_label('Time')

# Save the animation as a GIF
# animation.save('phase_space_animation.gif', writer='pillow')

# Display the animation
plt.show()
