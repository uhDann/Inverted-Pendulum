import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.integrate import solve_ivp

def pid_control_law(y, params, x_ref=2.2):
    """Compute PID control force F."""
    x, x_dot, theta, theta_dot = y

    Kp_x     = params['Kp_x']      # P gain for cart position
    Kd_x     = params['Kd_x']      # D gain for cart velocity

    Kp_theta = params['Kp_theta']  # P gain for pendulum angle
    Kd_theta = params['Kd_theta']  # D gain for pendulum angular velocity
    
    # Outer loop: Cart position control -> Target angle
    x_error = x_ref - x
    theta_ref = Kp_x * x_error - Kd_x * x_dot
    
    # Inner loop: Pendulum angle control -> Force
    theta_error = theta_ref - theta
    F = Kp_theta * theta_error - Kd_theta * theta_dot
    
    return F

def pid_control(y, params, dt=0.01, x_ref=1.8288):
    """
    Compute control force F
    """
    x, x_dot, theta, theta_dot = y

    Kp_x     = params['Kp_x']      # P gain for cart position
    Ki_x     = params['Ki_x']      # I gain for cart position
    Kd_x     = params['Kd_x']      # D gain for cart velocity

    Kp_theta = params['Kp_theta']  # P gain for pendulum angle
    Ki_theta = params['Ki_theta']  # I gain for pendulum angle
    Kd_theta = params['Kd_theta']  # D gain for pendulum angular velocity

    # Outer loop: Cart position control -> Target angle
    x_error = x_ref - x
    integral_x += x_error * dt
    derivative_x = (x_error - prev_x_error) / dt
    theta_ref = Kp_x * x_error + Ki_x * integral_x + Kd_x * derivative_x
    prev_x_error = x_error 

    # Inner loop: Pendulum angle control -> Force
    theta_error = theta_ref - x[2]
    integral_theta += theta_error * dt
    derivative_theta = (theta_error - prev_theta_error) / dt
    F = Kp_theta * theta_error + Ki_theta * integral_theta + Kd_theta * derivative_theta
    prev_theta_error = theta_error

    return F

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

    I = 1/3 * m * l**2             # Inertia
    d = 1                          # Friction Coeeficient

    s = np.sin(theta)
    c = np.cos(theta)
    F = pid_control_law(y, params)
    denom = (M + m) * (I + m * l**2) - m**2 * l**2 * c**2

    # Equations of motion
    x_ddot = (
        -(I + m * l**2) * d * x_dot + m * l * (I + m * l**2) * s * theta**2 + (I + m * l**2) * F - m**2 * l**2 * g * c * s
    ) / denom

    theta_ddot = (
        m * l * c * d * x_dot + m**2 * l**2 * s * c * theta_dot**2 - m * l * c * F + (M + m) * m * g * l * s
    ) / denom

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
    'Kp_x': 1.0,
    'Kd_x': 1.0,
    'Kp_theta': 1.0,
    'Kd_theta': 1.0,
    'M':  1.5,     # cart mass
    'm':  0.2,     # pendulum mass
    'l':  0.5,     # pendulum length
    'g':  9.81,    # gravity
}

# Initial conditions and simulation parameters
t_span = (0, 10)
y0 = [0.0, 0.0, np.deg2rad(182), 0.0]  # Initial state: [x, x_dot, theta, theta_dot]
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
x_line, = ax[0].plot(t_vals, x_vals, lw=2)
des_x_line = ax[0].axhline(y=0, color='r', linestyle='--')

# Plot for theta_vals
ax[1].set_title("Pendulum Angle (theta)")
theta_line, = ax[1].plot(t_vals, theta_vals, lw=2)
des_line = ax[1].axhline(y=np.pi, color='r', linestyle='--')

# Define sliders for Kp and Kd values
axcolor = 'lightgoldenrodyellow'
ax_kp_x = plt.axes([0.1, 0.2, 0.65, 0.03], facecolor=axcolor)
ax_kd_x = plt.axes([0.1, 0.15, 0.65, 0.03], facecolor=axcolor)
ax_kp_theta = plt.axes([0.1, 0.1, 0.65, 0.03], facecolor=axcolor)
ax_kd_theta = plt.axes([0.1, 0.05, 0.65, 0.03], facecolor=axcolor)

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