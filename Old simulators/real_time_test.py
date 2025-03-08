#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

###############################################################################
#                        CART-PENDULUM DYNAMICS & CONTROL                     #
###############################################################################

def cart_pendulum_dynamics(y, F, params):
    """
    Cart-pendulum equations of motion, no noise added here.

    Input:
        y = [x, x_dot, theta, theta_dot]
        F = control force
        params = dictionary of system parameters

    Returns:
        dydt = [x_dot, x_ddot, theta_dot, theta_ddot]
    """
    M  = params['M']
    m  = params['m']
    l  = params['l'] / 2
    g  = params['g']
    mu_c = params.get('mu_c', 0.01)   # cart friction
    mu_p = params.get('mu_p', 0.001)  # pendulum friction
    
    x, x_dot, theta, theta_dot = y
    s = np.sin(theta)
    c = np.cos(theta)
    
    x_ddot = (
        m*g*s*c 
        - (7/3)*(F + m*l*(theta_dot**2)*s - mu_c*x_dot)
        - (mu_p * theta_dot * c)/l
    )/(m*c**2 - (7/3)*M)

    theta_ddot = (3/(7*l))*(g*s - x_ddot*c - (mu_p*theta_dot)/(m*l))
    
    return np.array([x_dot, x_ddot, theta_dot, theta_ddot])


def pid_control_law(y_filt, params, x_ref=0.0):
    """
    PD/PID-like control for the cart-pendulum.

    Input:
        y_filt = [x, x_dot, theta, theta_dot] (filtered states)
        params = {'Kp_x', 'Kd_x', 'Kp_theta', 'Kd_theta', ...}
        x_ref  = reference cart position

    Returns:
        F = control force
    """
    x, x_dot, theta, theta_dot = y_filt
    
    Kp_x     = params['Kp_x']
    Kd_x     = params['Kd_x']
    Kp_theta = params['Kp_theta']
    Kd_theta = params['Kd_theta']

    # Outer loop: cart position -> desired pendulum angle
    # (Here we shift around pi to keep pendulum near upright or inverted.)
    x_error   = x_ref - x
    theta_ref = np.pi + (Kp_x * x_error - Kd_x * x_dot) 

    # Inner loop: pendulum angle -> control force
    theta_error = theta_ref - theta
    F = Kp_theta * theta_error - Kd_theta * theta_dot

    return F


def low_pass_filter(old_val, new_meas, alpha=0.5):
    """
    Discrete-time first-order low-pass filter:
      filtered_{k+1} = alpha * new_measurement + (1 - alpha)*filtered_k
    """
    return alpha*new_meas + (1 - alpha)*old_val


###############################################################################
#                             REAL-TIME SIMULATION                             #
###############################################################################

def simulate_and_animate(params,
                         t_final=5.0,
                         dt=0.01,
                         noise_std=[0, 0, 0, 0],
                         alpha=0.5,
                         x_ref=0.0):
    """
    Simulate the cart-pendulum in discrete steps and animate in real-time.

    Inputs:
        params    = dictionary with {M, m, l, g, Kp_x, Kd_x, Kp_theta, Kd_theta, etc.}
        t_final   = total simulation time (seconds)
        dt        = time step (seconds)
        noise_std = list of standard deviations for measurement noise [x, x_dot, theta, theta_dot]
        alpha     = low-pass filter alpha parameter
        x_ref     = reference cart position

    This function creates a live plot with:
       - Real-time cart-pendulum animation (left side)
       - Real-time plots of x(t), theta(t) (true, noisy, filtered), and control F(t) (right side)
    """
    import time

    # -------------------------------------------------------------------------
    # 1) Prepare for simulation
    # -------------------------------------------------------------------------
    n_steps = int(np.floor(t_final / dt))
    t_vals = np.linspace(0, t_final, n_steps + 1)

    # Initial conditions
    theta0 = np.deg2rad(90.0) if 'theta0' not in params else np.deg2rad(params['theta0'])
    y_true = np.array([0.0, 0.0, theta0, 0.0])  # [x, x_dot, theta, theta_dot]
    y_filt = y_true.copy()  # initial filter state

    # Storage for plotting lines over time
    time_history        = []
    x_true_history      = []
    x_meas_history      = []
    x_filt_history      = []
    theta_true_history  = []
    theta_meas_history  = []
    theta_filt_history  = []
    F_history           = []

    # -------------------------------------------------------------------------
    # 2) Set up the matplotlib figure
    # -------------------------------------------------------------------------
    plt.ion()   # turn on interactive mode
    fig = plt.figure(figsize=(12, 6))

    # --- LEFT: Cart-Pendulum Animation ---
    ax_anim = fig.add_subplot(1, 2, 1)
    ax_anim.set_xlim([-3, 3])
    ax_anim.set_ylim([-1.2, 1.2])
    ax_anim.set_aspect('equal', adjustable='box')
    ax_anim.set_title("Cart-Pendulum Animation")

    cart_width = 0.3
    (cart_line,) = ax_anim.plot([], [], lw=5, color='k')
    (rod_line,)  = ax_anim.plot([], [], lw=2, color='blue')
    bob_radius = 0.05
    bob_circle = plt.Circle((0, 0), bob_radius, fc='red')
    ax_anim.add_patch(bob_circle)

    # --- RIGHT: 3 Subplots for x(t), theta(t), and F(t) ---
    #   We stack them vertically in the right column
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(nrows=3, ncols=1, figure=fig, 
                           left=0.6, right=0.98, hspace=0.6)

    # Subplot for x(t)
    ax_x = fig.add_subplot(gs[0, 0])
    ax_x.set_title("Cart Position x(t)", pad=30)
    ax_x.set_xlabel("Time (s)")
    ax_x.set_ylabel("x (m)")
    ax_x.set_ylim([-3, 3])
    line_x_true, = ax_x.plot([], [], label='x true', color='blue')
    line_x_meas, = ax_x.plot([], [], '--', label='x noisy', color='orange')
    line_x_filt, = ax_x.plot([], [], ':', label='x filtered', color='green')
    ax_x.legend(loc='upper right')

    # Subplot for theta(t)
    ax_theta = fig.add_subplot(gs[1, 0])
    ax_theta.set_title("Pendulum Angle theta(t)", pad=30)
    ax_theta.set_xlabel("Time (s)")
    ax_theta.set_ylabel("theta (rad)")
    ax_theta.set_ylim([0, np.pi*2])
    line_theta_true, = ax_theta.plot([], [], label='theta true', color='blue')
    line_theta_meas, = ax_theta.plot([], [], '--', label='theta noisy', color='orange')
    line_theta_filt, = ax_theta.plot([], [], ':', label='theta filtered', color='green')
    ax_theta.legend(loc='upper right')

    # Subplot for control F(t)
    ax_F = fig.add_subplot(gs[2, 0])
    ax_F.set_title("Control Force F(t)",  pad=30)
    ax_F.set_xlabel("Time (s)")
    ax_F.set_ylabel("F (N)")
    ax_F.set_ylim([-10, 10])
    line_F, = ax_F.plot([], [], label='Control', color='red')
    ax_F.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

    # Function to update the animation artists (the cart & pendulum)
    def update_animation(x, theta):
        # Update cart
        cart_line.set_data([x - cart_width/2, x + cart_width/2], [0, 0])

        # Update rod
        l = params['l']
        rod_x = [x, x + l * np.sin(theta)]
        rod_y = [0, -l * np.cos(theta)]
        rod_line.set_data(rod_x, rod_y)

        # Update bob
        bob_circle.set_center((rod_x[1], rod_y[1]))

    t = 0.0
    for k in range(n_steps + 1):
        # Store data
        time_history.append(t)
        x_true_history.append(y_true[0])
        theta_true_history.append(y_true[2])

        # "Measurement" with noise
        y_meas = y_true + np.random.normal(0, noise_std)
        x_meas_history.append(y_meas[0])
        theta_meas_history.append(y_meas[2])

        # Low-pass filter
        y_filt = low_pass_filter(y_filt, y_meas, alpha=alpha)
        x_filt_history.append(y_filt[0])
        theta_filt_history.append(y_filt[2])

        # Control force
        F = pid_control_law(y_filt, params, x_ref=x_ref)
        F_history.append(F)

        # Update animation
        update_animation(y_true[0], y_true[2])

        # Update the lines in the x(t) subplot
        line_x_true.set_data(time_history, x_true_history)
        line_x_meas.set_data(time_history, x_meas_history)
        line_x_filt.set_data(time_history, x_filt_history)
        ax_x.set_xlim([0, t_final])  # keep the same time scale
        # ax_x.relim(); ax_x.autoscale_view()  # if you want dynamic y-limits

        # Update the lines in the theta(t) subplot
        line_theta_true.set_data(time_history, theta_true_history)
        line_theta_meas.set_data(time_history, theta_meas_history)
        line_theta_filt.set_data(time_history, theta_filt_history)
        ax_theta.set_xlim([0, t_final])

        # Update the line for F(t)
        line_F.set_data(time_history, F_history)
        ax_F.set_xlim([0, t_final])
        # ax_F.relim(); ax_F.autoscale_view()

        plt.draw()
        plt.pause(dt)  # allow the plot to update in "real-time"

        # 3.3) Integrate one step (simple Euler)
        if k < n_steps:
            dydt = cart_pendulum_dynamics(y_true, F, params)
            y_true = y_true + dt * dydt
            t += dt

    # -------------------------------------------------------------------------
    # 4) Post-simulation hold
    # -------------------------------------------------------------------------
    print("Simulation completed. Close the figure window to exit.")
    plt.ioff()  # turn off interactive mode
    plt.show()


###############################################################################
#                                  MAIN EXECUTION                              #
###############################################################################

if __name__ == "__main__":
    # Example usage:
    params = {
        'M': 1.0,        # cart mass
        'm': 0.2,        # pendulum mass
        'l': 0.5,        # pendulum length
        'g': 9.81,       
        'Kp_x': 1.54,    # PD gains
        'Kd_x': 0.84,
        'Kp_theta': 18.36,
        'Kd_theta': 0.81,
        # friction (optional)
        'mu_c': 0.01,
        'mu_p': 0.001,
        # initial angle in degrees (optional)
        'theta0': 90.0
    }

    # Noise standard deviations: [x, x_dot, theta, theta_dot]
    noise_std = [0.01, 0.0, 0.01, 0.0]

    # Run the real-time simulation & animation
    simulate_and_animate(
        params=params,
        t_final=5.0,
        dt=0.01,
        noise_std=noise_std,
        alpha=0.5,
        x_ref=0.0
    )
