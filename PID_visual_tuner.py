import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.integrate import solve_ivp

def cart_pendulum_dynamics(t, y, params):
    """
    Returns [x_dot, x_ddot, theta_dot, theta_ddot].
    State y = [x, x_dot, theta, theta_dot].
    """
    x, x_dot, theta, theta_dot = y
    M        = params['M']         # cart mass
    m        = params['m']         # pendulum bob mass
    l        = params['l']         # pendulum length
    g        = params['g']         # gravity
    Kp_x     = params['Kp_x']      # P gain for cart position
    Kd_x     = params['Kd_x']      # D gain for cart velocity
    Kp_theta = params['Kp_theta']  # P gain for pendulum angle
    Kd_theta = params['Kd_theta']  # D gain for pendulum angular velocity

    # Control law: regulate x -> 0 AND theta -> 0
    u = (-Kp_x * x
         -Kd_x * x_dot
         -Kp_theta * (theta - np.deg2rad(180))
         -Kd_theta * theta_dot)

    denom = M + m*np.sin(theta)**2

    # Equations of motion
    x_ddot = (
        u + m*np.sin(theta)*(l*theta_dot**2 + g*np.cos(theta))
    ) / denom

    theta_ddot = (
        -u*np.cos(theta)
        -m*l*(theta_dot**2)*np.sin(theta)*np.cos(theta)
        - (M + m)*g*np.sin(theta)
    ) / (l * denom)

    return [x_dot, x_ddot, theta_dot, theta_ddot]

def simulate_cart_pendulum(params, t_span, y0, steps):
    """
    Solve the ODE for the given parameters, initial conditions,
    time span, and number of time steps.
    """
    t_eval = np.linspace(t_span[0], t_span[1], steps)
    sol = solve_ivp(
        fun=lambda t, y: cart_pendulum_dynamics(t, y, params),
        t_span=t_span,
        y0=y0,
        t_eval=t_eval
    )
    return sol.t, sol.y  # (time array, solution array)

def update(val):
    params['Kp_x'] = kp_x_slider.val
    params['Kd_x'] = kd_x_slider.val
    params['Kp_theta'] = kp_theta_slider.val
    params['Kd_theta'] = kd_theta_slider.val

    # Recompute the solution with updated parameters
    t_vals, sol = simulate_cart_pendulum(params, t_span, y0, steps)

    x_vals = sol[0, :]
    theta_vals = sol[2, :]

    # Update the plot
    x_line.set_ydata(x_vals)
    theta_line.set_ydata(theta_vals)
    fig.canvas.draw_idle()

# Define the initial parameters
params = {
    'Kp_x': 1.0,
    'Kd_x': 1.0,
    'Kp_theta': 1.0,
    'Kd_theta': 1.0,
    'M': 1.0,  # Example values for other parameters
    'm': 0.1,
    'l': 1.0,
    'g': 9.81
}

# Initial conditions and simulation parameters
t_span = (0, 10)
y0 = [0, 0, np.pi, 0]  # Initial state: [x, x_dot, theta, theta_dot]
steps = 100

# Initial simulation
t_vals, sol = simulate_cart_pendulum(params, t_span, y0, steps)
x_vals = sol[0, :]
theta_vals = sol[2, :]

# --- Setup figure and axes ---
fig, ax = plt.subplots(2, 1, figsize=(10, 8))
plt.subplots_adjust(left=0.1, bottom=0.35)

# Plot for x_vals
ax[0].set_title("Cart Position (x)")
x_line, = ax[0].plot(t_vals, x_vals, lw=2)

# Plot for theta_vals
ax[1].set_title("Pendulum Angle (theta)")
theta_line, = ax[1].plot(t_vals, theta_vals, lw=2)

# Define sliders for Kp and Kd values
axcolor = 'lightgoldenrodyellow'
ax_kp_x = plt.axes([0.1, 0.2, 0.65, 0.03], facecolor=axcolor)
ax_kd_x = plt.axes([0.1, 0.15, 0.65, 0.03], facecolor=axcolor)
ax_kp_theta = plt.axes([0.1, 0.1, 0.65, 0.03], facecolor=axcolor)
ax_kd_theta = plt.axes([0.1, 0.05, 0.65, 0.03], facecolor=axcolor)

kp_x_slider = Slider(ax_kp_x, 'Kp_x', 0.1, 20.0, valinit=params['Kp_x'])
kd_x_slider = Slider(ax_kd_x, 'Kd_x', 0.1, 20.0, valinit=params['Kd_x'])
kp_theta_slider = Slider(ax_kp_theta, 'Kp_theta', 0.1, 20.0, valinit=params['Kp_theta'])
kd_theta_slider = Slider(ax_kd_theta, 'Kd_theta', 0.1, 20.0, valinit=params['Kd_theta'])

# Call update function on slider value change
kp_x_slider.on_changed(update)
kd_x_slider.on_changed(update)
kp_theta_slider.on_changed(update)
kd_theta_slider.on_changed(update)

plt.show()