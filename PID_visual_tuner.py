import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.integrate import solve_ivp

def pid_control_law(y, params, x_ref=0):
    """Compute PID control force F."""
    x, x_dot, theta, theta_dot = y

    Kp_x     = params['Kp_x']      # P gain for cart position
    Kd_x     = params['Kd_x']      # D gain for cart velocity

    Kp_theta = params['Kp_theta']  # P gain for pendulum angle
    Kd_theta = params['Kd_theta']  # D gain for pendulum angular velocity
    
    # Outer loop: Cart position control -> Target angle
    x_error = x_ref - x
    theta_ref = np.pi + (Kp_x * x_error - Kd_x * x_dot)
    
    # Inner loop: Pendulum angle control -> Force
    theta_error = theta_ref - theta
    F = Kp_theta * theta_error - Kd_theta * theta_dot
    
    return F

def cart_pendulum_dynamics(t, y, params):
    """
    Returns [x_dot, x_ddot, theta_dot, theta_ddot].
    State y = [x, x_dot, theta, theta_dot].
    """
    x, x_dot, theta, theta_dot = y
    M        = params['M']         # cart mass
    m        = params['m']         # pendulum bob mass
    l        = params['l']/2       # half of the pendulum length
    g        = params['g']         # gravity
    u_c = params.get('mu_c', 0.01 )  # Cart friction coefficient (default 0.01)
    u_p = params.get('mu_p', 0.001)  # Pendulum friction coefficient (default 0.001)

    s = np.sin(theta)
    c = np.cos(theta)
    F = pid_control_law(y, params)

    # Equations of motion
    x_ddot = (
        m * g * s * c 
        - (7/3) * (F + m * l * theta_dot**2 * s - u_c * x_dot)
        - (u_p * theta_dot * c) / l
    ) / (m * c**2 - (7/3) * M)


    theta_ddot = (
        (3 / (7 * l)) 
        * (g * s - x_ddot * c - u_p * theta_dot 
        / (m * l))
    ) 

    return [x_dot, x_ddot, theta_dot, theta_ddot]

def simulate_system(params, y0, t_span=(0, 20), steps=2000):
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
    t_vals, sol = simulate_system(params, y0)

    x_vals = sol[0, :]
    theta_vals = sol[2, :]

    # Update the plot
    x_line.set_ydata(x_vals)
    theta_line.set_ydata(theta_vals)
    fig.canvas.draw_idle()

# Define the initial parameters
params = {
    'Kp_x': 0.0,
    'Kd_x': 0.0,
    'Kp_theta': 0.0,
    'Kd_theta': 0.0,
    'M':  1.0,     # cart mass
    'm':  0.2,     # pendulum mass
    'l':  0.5,     # pendulum length
    'g':  9.81,    # gravity
}

# Initial conditions and simulation parameters
t_span = (0, 10)
y0 = [0.0, 0.0, np.deg2rad(90), 0.0]  # Initial state: [x, x_dot, theta, theta_dot]
steps = 500

# Initial simulation
t_vals, sol = simulate_system(params, y0)
x_vals = sol[0, :]
theta_vals = sol[2, :]

# --- Setup figure and axes ---
fig, ax = plt.subplots(2, 1, figsize=(10, 8))
plt.subplots_adjust(left=0.1, bottom=0.35)

# Plot for x_vals
ax[0].set_title("Cart Position (x)")
# ax[0].set_ylim([0, 4])
x_line, = ax[0].plot(t_vals, x_vals, lw=2)
des_x_line = ax[0].axhline(y=0, color='r', linestyle='--')

# Plot for theta_vals
ax[1].set_title("Pendulum Angle (theta)")
# ax[1].set_ylim([-5, 5])
theta_line, = ax[1].plot(t_vals, theta_vals, lw=2)
des_line = ax[1].axhline(y=np.pi, color='r', linestyle='--')

# Define sliders for Kp and Kd values
axcolor = 'lightgoldenrodyellow'
ax_kp_x = plt.axes([0.1, 0.2, 0.8, 0.03], facecolor=axcolor)
ax_kd_x = plt.axes([0.1, 0.15, 0.8, 0.03], facecolor=axcolor)
ax_kp_theta = plt.axes([0.1, 0.1, 0.8, 0.03], facecolor=axcolor)
ax_kd_theta = plt.axes([0.1, 0.05, 0.8, 0.03], facecolor=axcolor)

kp_x_slider = Slider(ax_kp_x, 'Kp_x', 0, 20.0, valinit=params['Kp_x'])
kd_x_slider = Slider(ax_kd_x, 'Kd_x', 0, 20.0, valinit=params['Kd_x'])
kp_theta_slider = Slider(ax_kp_theta, 'Kp_theta', 0, 20.0, valinit=params['Kp_theta'])
kd_theta_slider = Slider(ax_kd_theta, 'Kd_theta', 0, 20.0, valinit=params['Kd_theta'])

# Call update function on slider value change
kp_x_slider.on_changed(update)
kd_x_slider.on_changed(update)
kp_theta_slider.on_changed(update)
kd_theta_slider.on_changed(update)

plt.show()