import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider

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

def simulate_system(params, y0, t_span=(0, 10), steps=2000):
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
    ax.set_xlim([-5, 5])   # Adjust as needed for your system
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
        'Kp_x':     1,
        'Kd_x':     4.01,
        'Kp_theta': 1.09,
        'Kd_theta': 4.56
    }

    # Working values kp_x = 9.7, kd_x = 7.0, kp_theta = 1.1, kd_theta = 2.2

    # [cart pos, cart vel, pendulum angle, pendulum angular vel]
    # Initial conditions: x=0.1m, x_dot=0, theta=5°, theta_dot=0
    y0 = [0.0, 0.0, np.deg2rad(90), 0.0]

    # Solve for 5 seconds
    t_vals, sol = simulate_system(params, y0)

    # Animate
    animate_pendulum(t_vals, sol, params)

if __name__ == "__main__":
    main()