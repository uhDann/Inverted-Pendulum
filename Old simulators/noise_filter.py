import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def pid_control_law(meas_state, params, x_ref=0):
    """
    meas_state = [ x_measured, xdot_measured, theta_measured, thetadot_measured ]
    Return control force F based on PID gains.
    """
    x, x_dot, theta, theta_dot = meas_state

    Kp_x     = params['Kp_x']      # P gain for cart position
    Kd_x     = params['Kd_x']      # D gain for cart velocity
    Kp_theta = params['Kp_theta']  # P gain for pendulum angle
    Kd_theta = params['Kd_theta']  # D gain for pendulum angular velocity

    # Outer loop: cart position -> target angle
    x_error   = x_ref - x
    theta_ref = np.pi + (Kp_x * x_error - Kd_x * x_dot)

    # Inner loop: pendulum angle -> force
    theta_error = theta_ref - theta
    F = Kp_theta * theta_error - Kd_theta * theta_dot
    return F

def cart_pendulum_dynamics(t, y, F, params):
    """
    ODE for the cart-pendulum (noise-free). 
    y = [ x, x_dot, theta, theta_dot ]
    Return dy/dt as [ x_dot, x_ddot, theta_dot, theta_ddot ].
    """
    x, x_dot, theta, theta_dot = y
    
    M  = params['M']
    m  = params['m']
    l  = params['l'] / 2.0
    g  = params['g']
    mu_c = params.get('mu_c', 0.01)  # cart friction
    mu_p = params.get('mu_p', 0.001) # pendulum friction

    s = np.sin(theta)
    c = np.cos(theta)

    # EoMs (the same you had before), but now 'F' is passed in from the outside:
    x_ddot = (
        m*g*s*c 
        - (7/3) * (F + m*l*(theta_dot**2)*s - mu_c*x_dot)
        - (mu_p * theta_dot * c) / l
    ) / (m*c**2 - (7/3)*M)

    theta_ddot = (3 / (7*l)) * (g*s - x_ddot*c - (mu_p*theta_dot)/(m*l))
    
    return np.array([x_dot, x_ddot, theta_dot, theta_ddot])

def low_pass_filter(old_value, new_measurement, alpha=0.1):
    """
    Discrete-time low-pass filter: 
       filtered_{k+1} = alpha * measurement + (1 - alpha) * filtered_k
    """
    return alpha * new_measurement + (1 - alpha) * old_value

def simulate_discrete(params, y0, dt=0.001, t_final=5.0, noise_std=[0,0,0,0], alpha=0.1):
    """
    Discretely simulate from t=0 to t=t_final with time step dt.
    Return time array, all true states, and all measured states, etc.
    """
    n_steps = int((t_final/dt))
    t_vals = np.linspace(0, t_final, n_steps+1)

    # Lists to store results
    y_true_array  = np.zeros((n_steps+1, 4))
    y_meas_array  = np.zeros((n_steps+1, 4))
    y_filt_array  = np.zeros((n_steps+1, 4))
    F_array       = np.zeros(n_steps+1)

    # Initialize
    y_true = np.array(y0, dtype=float)
    # For the filter, we can initialize "filtered measurement" to the first measurement
    measurement_init = y_true + np.random.normal(0, noise_std) 
    y_filt = measurement_init.copy()

    y_true_array[0] = y_true
    y_meas_array[0] = measurement_init
    y_filt_array[0] = y_filt

    for k in range(n_steps):
        t_k   = t_vals[k]
        # 1) measure the system (noisy)
        y_meas = y_true + np.random.normal(0, noise_std)

        # 2) filter the measurement
        y_filt = low_pass_filter(y_filt, y_meas, alpha=alpha)

        # 3) get control force from filtered measurement
        F_k = pid_control_law(y_filt, params)

        # store for plotting later
        F_array[k] = F_k
        y_meas_array[k] = y_meas
        y_filt_array[k] = y_filt

        # 4) integrate the system from t_k to t_k + dt (Euler or small RK4)
        # Here is a quick (explicit) Euler as an example:
        dydt = cart_pendulum_dynamics(t_k, y_true, F_k, params)
        y_true_next = y_true + dydt*dt

        # Update
        y_true = y_true_next
        y_true_array[k+1] = y_true
    
    # last step record
    y_meas_array[-1] = y_true + np.random.normal(0, noise_std)
    y_filt_array[-1] = low_pass_filter(y_filt, y_meas_array[-1], alpha)
    F_array[-1]      = pid_control_law(y_filt_array[-1], params)

    return t_vals, y_true_array, y_meas_array, y_filt_array, F_array

def animate_pendulum(t_vals, sol, params):
    """
    Create a dynamic animation of the cart-pendulum system
    using pre-computed time/value arrays.
    """
    x_vals     = sol[:, 0]   # cart positions
    theta_vals = sol[:, 2]   # pendulum angles

    # Unpack system dimensions
    l = params['l']

    # --- Setup figure and axes ---
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_xlim([-3, 3])   # Adjust as needed for your system
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
        'M':  1.0,     
        'm':  0.2,
        'l':  0.5,
        'g':  9.81,
        'Kp_x':     1.54,
        'Kd_x':     0.84,
        'Kp_theta': 18.36,
        'Kd_theta': 0.81
    }

    # Initial state: [x, x_dot, theta, theta_dot]
    y0 = [0.0, 0.0, np.deg2rad(90), 0.0]

    # Noise standard deviations for each measurement channel
    noise_std = [0.01, 0.0, 0.01, 0.0]

    # Simulate
    dt = 0.001
    t_final = 5.0
    alpha   = 0.05  # filter "strength" in [0..1]; smaller => heavier smoothing
    t_vals, y_true, y_meas, y_filt, F_array = simulate_discrete(
        params, y0, dt, t_final, noise_std, alpha
    )

    plt.figure()
    plt.plot(t_vals, np.rad2deg(y_true[:,2]), label='True theta (deg)')
    plt.plot(t_vals, np.rad2deg(y_meas[:,2]), label='Measured theta (deg)', alpha=0.5)
    plt.plot(t_vals, np.rad2deg(y_filt[:,2]), label='Filtered theta (deg)', linestyle='--')
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Pendulum angle [deg]")
    plt.title("Pendulum Angle: True vs. Measured vs. Filtered")
    plt.show()

    # Animate
    animate_pendulum(t_vals, y_true, params)

if __name__ == '__main__':
    main()

