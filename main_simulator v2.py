import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider

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

    # 外环控制（位置控制 -> 目标角度）
    x_error = x_ref - x
    integral_x += x_error * dt
    derivative_x = (x_error - prev_x_error) / dt
    theta_ref = Kp_x * x_error + Ki_x * integral_x + Kd_x * derivative_x
    prev_x_error = x_error  # 更新误差

    # 内环控制（角度控制 -> 控制力）
    theta_error = theta_ref - x[2]
    integral_theta += theta_error * dt
    derivative_theta = (theta_error - prev_theta_error) / dt
    F = Kp_theta * theta_error + Ki_theta * integral_theta + Kd_theta * derivative_theta
    prev_theta_error = theta_error  # 更新误差

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

def animate_pendulum(t_vals, sol, params):
    """
    Create a dynamic animation of the cart-pendulum system
    using pre-computed time/value arrays.
    """
    x_vals     = sol[0, :]   # cart positions
    theta_vals = sol[2, :]   # pendulum angles

    # Unpack system dimensions
    l = params['l']

    # --- Setup figure and axes ---
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_xlim([-0.5, 2])   # Adjust as needed for your system
    ax.set_ylim([-1.0, 1.0])
    ax.set_aspect('equal')
    ax.set_title("Cart-Pendulum Animation")

    # --- Elements to draw: cart, pendulum rod, mass ---
    # Cart: We'll represent it as a simple line or small rectangle
    cart_width = 0.3
    cart_height = 0.2

    # We'll use a Line2D for the "cart" top edge, a Line2D for the rod,
    # and a small circle (matplotlib patch) for the pendulum bob.
    cart_line, = ax.plot([], [], lw=5, color='blue')         # top edge of cart
    rod_line, = ax.plot([], [], lw=2, color='black')         # pendulum rod
    bob_circle = plt.Circle((0, 0), 0.05, fc='red')          # pendulum bob
    ax.add_patch(bob_circle)

    def init():
        cart_line.set_data([], [])
        rod_line.set_data([], [])
        bob_circle.set_center((0, 0))
        return cart_line, rod_line, bob_circle

    def update(frame):
        x = x_vals[frame]
        theta = theta_vals[frame]

        # Update cart position
        cart_line.set_data([x - cart_width/2, x + cart_width/2], [0, 0])

        # Update pendulum rod position
        rod_x = [x, x + l * np.sin(theta)]
        rod_y = [0, -l * np.cos(theta)]
        rod_line.set_data(rod_x, rod_y)

        # Update pendulum bob position
        bob_circle.set_center((rod_x[1], rod_y[1]))

        return cart_line, rod_line, bob_circle

    anim = animation.FuncAnimation(
        fig,
        func=update,
        frames=len(t_vals),
        init_func=init,
        interval=1000*(t_vals[1]-t_vals[0]),  # in milliseconds
        blit=True
    )

    plt.show()

def main():
    # System parameters
    params = {
        'M':  1.0,     # cart mass
        'm':  0.2,     # pendulum mass
        'l':  0.5,     # pendulum length
        'g':  9.81,    # gravity
        'Kp_x':     9.7,
        'Kd_x':     7.0,
        'Kp_theta': 1.1,
        'Kd_theta': 2.2
    }

    # [cart pos, cart vel, pendulum angle, pendulum angular vel]
    # Initial conditions: x=0.1m, x_dot=0, theta=5°, theta_dot=0
    y0 = [0.0, 0.0, np.deg2rad(182), 0.0]

    # Solve for 5 seconds
    t_vals, sol = simulate_system(params, y0)

    # Animate
    animate_pendulum(t_vals, sol, params)

if __name__ == "__main__":
    main()