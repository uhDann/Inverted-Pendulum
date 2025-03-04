import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import place_poles

def pole_placement_control(y, params, target):
    x, x_dot, theta, theta_dot = y
    M         = params['M']         # cart mass
    m         = params['m']         # pendulum bob mass
    l         = params['l']         # pendulum length
    g         = params['g']         # gravity

    I = 1/3 * m * l**2              # Inertia
    d = 1                           # Friction Coeeficient
    
    s = np.sin(theta)
    c = np.cos(theta)
    denom = (M + m) * (I + m * l**2) - m**2 * l**2 * c**2
    
    A = np.array([
        [0, 1, 0, 0],
        [0, -(I + m * l**2) * d / denom, (-m**2 * l**2 * g * (M + m) * (I + m * l**2)) / denom**2, 0],
        [0, 0, 0, 1],
        [0, m * l * d / denom, (M + m) * m * g * l / denom, 0]
    ])

    B = np.array([
        [0],
        [(I + m * l**2) / denom],
        [0],
        [-m * l / denom]
    ])

    # Check the controllability
    controllability = np.hstack([B, A @ B, A @ A @ B, A @ A @ A @ B])

    # Pole placement
    desired_poles = np.array([-2 + 0.5j, -2 - 0.5j, -4, -5])
    placed = place_poles(A, B, desired_poles)
    K = placed.gain_matrix
    F = -np.dot(K, y - target)

    return F[0]

def cart_pendulum_dynamics(t, y, params, target):
    """
    Returns [x_dot, x_ddot, theta_dot, theta_ddot].
    State y = [x, x_dot, theta, theta_dot].
    """
    x, x_dot, theta, theta_dot = y
    M         = params['M']         # cart mass
    m         = params['m']         # pendulum bob mass
    l         = params['l']         # pendulum length
    g         = params['g']         # gravity

    I = 1/3 * m * l**2              # Inertia
    d = 1                           # Friction Coeeficient
    
    s = np.sin(theta)
    c = np.cos(theta)
    denom = (M + m) * (I + m * l**2) - m**2 * l**2 * c**2
    F = pole_placement_control(y, params, target)

    # Equations of motion
    x_ddot = (
        -(I + m * l**2) * d * x_dot + m * l * (I + m * l**2) * s * theta**2 + (I + m * l**2) * F - m**2 * l**2 * g * c * s
    ) / denom

    theta_ddot = (
        m * l * c * d * x_dot + m**2 * l**2 * s * c * theta_dot**2 - m * l * c * F + (M + m) * m * g * l * s
    ) / denom

    return [x_dot, x_ddot, theta_dot, theta_ddot]


def simulate_system(params, y0, y_target, t_span=(0, 20), steps=2000):
    """
    Solve the ODE for the given parameters, initial conditions,
    time span, and number of time steps.
    """
    t_eval = np.linspace(t_span[0], t_span[1], steps)
    sol = solve_ivp(
        fun=lambda t, y: cart_pendulum_dynamics(t, y, params, y_target),
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
    ax.set_xlim([-0.5, 5])   # Adjust as needed for your system
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
        rod_y = [0, l * np.cos(theta)]
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
        'max': 15.0,   # max control force
    }

    # [cart pos, cart vel, pendulum angle, pendulum angular vel]
    # Initial conditions: x=0.1m, x_dot=0, theta=5°, theta_dot=0
    y0 = [0.0, 0.0, np.deg2rad(5), 0.0]
    # Target conditions: x=1.8288m, x_dot=0, theta=0°, theta_dot=0
    ytarget = np.array([1.8288, 0.0, 0.0, 0.0])  

    # Solve for 5 seconds
    t_vals, sol = simulate_system(params, y0, ytarget)

    # Animate
    animate_pendulum(t_vals, sol, params)

if __name__ == "__main__":
    main()