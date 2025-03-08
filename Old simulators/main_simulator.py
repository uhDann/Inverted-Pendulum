import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider


def simulate_inverted_pendulum():
    params = {
        'M':  0.8,    # mass of cart
        'm':  0.2,    # mass of pendulum bob
        'l':  0.5,    # pendulum length
        'g':  9.81,   # gravitational acceleration
        
        # Gains for cart position control
        'Kp_x':     10.0,
        'Kd_x':     5.0,
        
        # Gains for pendulum angle control
        'Kp_theta': 50.0,
        'Kd_theta': 10.0
    }

    # --- Initial conditions ---
    # [cart pos, cart vel, pendulum angle, pendulum angular vel]
    # cart 10cm away from target, pendulum 5deg from upright
    x0 = [0.1, 0.0, np.deg2rad(5), 0.0]

    # --- Time span & solver setup ---
    t_span = (0, 5)
    t_eval = np.linspace(t_span[0], t_span[1], 500)

    sol = solve_ivp(
        fun=lambda t, y: cart_pendulum_dynamics(t, y, params),
        t_span=t_span,
        y0=x0,
        t_eval=t_eval
    )

    # --- Extract solution ---
    t_vals        = sol.t
    x_vals        = sol.y[0, :]
    xdot_vals     = sol.y[1, :]
    theta_vals    = sol.y[2, :]
    thetadot_vals = sol.y[3, :]

    # --- Plot results ---
    plt.figure(figsize=(10, 8))

    plt.subplot(2,2,1)
    plt.plot(t_vals, x_vals, label='x (m)')
    plt.title('Cart Position')
    plt.xlabel('Time (s)')
    plt.ylabel('x (m)')
    plt.grid(True)

    plt.subplot(2,2,2)
    plt.plot(t_vals, xdot_vals, label='x_dot (m/s)', color='tab:orange')
    plt.title('Cart Velocity')
    plt.xlabel('Time (s)')
    plt.ylabel('x_dot (m/s)')
    plt.grid(True)

    plt.subplot(2,2,3)
    plt.plot(t_vals, np.rad2deg(theta_vals), label='theta (deg)', color='tab:green')
    plt.title('Pendulum Angle')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (deg)')
    plt.grid(True)

    plt.subplot(2,2,4)
    plt.plot(t_vals, np.rad2deg(thetadot_vals), label='theta_dot (deg/s)', color='tab:red')
    plt.title('Pendulum Angular Velocity')
    plt.xlabel('Time (s)')
    plt.ylabel('Angular Velocity (deg/s)')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

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

def simulate_system(params, y0, t_span=(0, 10), steps=500):
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
    xdot_vals  = sol[1, :]
    theta_vals = sol[2, :]   # pendulum angles
    thetadot   = sol[3, :]

    # Unpack system dimensions
    l = params['l']

    # --- Setup figure and axes ---
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_xlim([-20, 20])   # Adjust as needed for your system
    ax.set_ylim([-1.0, 1.0])
    ax.set_aspect('equal')
    ax.set_title("Cart-Pendulum Animation")

    # --- Elements to draw: cart, pendulum rod, mass ---
    # Cart: We'll represent it as a simple line or small rectangle
    cart_width = 0.3
    cart_height = 0.5

    # We'll use a Line2D for the "cart" top edge, a Line2D for the rod,
    # and a small circle (matplotlib patch) for the pendulum bob.
    cart_line, = ax.plot([], [], lw=25, color='blue')         # top edge of cart
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
        'M':  1.5,     # cart mass
        'm':  0.2,     # pendulum mass
        'l':  0.5,     # pendulum length
        'g':  9.81,    # gravity

        'Kp_x':     0,
        'Kd_x':     0,
        'Kp_theta': 16.62,
        'Kd_theta': 9.4
    }

    # [cart pos, cart vel, pendulum angle, pendulum angular vel]
    # Initial conditions: x=0.1m, x_dot=0, theta=5°, theta_dot=0
    y0 = [0.0, 0.0, np.deg2rad(182), 0.0]

    # Solve for 5 seconds
    t_vals, sol = simulate_system(params, y0, t_span=(0,10), steps=500)

    # Animate
    animate_pendulum(t_vals, sol, params)

main()