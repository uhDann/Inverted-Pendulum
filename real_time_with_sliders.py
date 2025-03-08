import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib

# Necessary for matplotlib to work with Tkinter
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrow

# Defining the Equations of Motion and Dynamics for simulation

def cart_pendulum_dynamics(y, F, params):
    """
    Cart-pendulum equations. 
    Note: Noise is never introduced through these dynamics
    
    Input:
        y = [x, x_dot, theta, theta_dot]

    Returns:
        [x_dot, x_ddot, theta_dot, theta_ddot]
    """

    # Recalling the parameters

    M  = params['M']
    m  = params['m']
    l  = params['l'] / 2
    g  = params['g']

    # Friction coefficients, setting realistic values as defaults if none are provided
    mu_c = params.get('mu_c', 0.01)
    mu_p = params.get('mu_p', 0.001)
    
    x, x_dot, theta, theta_dot = y
    s = np.sin(theta)
    c = np.cos(theta)

    # Equations of motion
    x_ddot = (
        m*g*s*c
        - (7/3)*(F + m*l*(theta_dot**2)*s - mu_c*x_dot)
        - (mu_p * theta_dot * c)/l
    ) / (m*c**2 - (7/3)*M)

    theta_ddot = (3/(7*l))*(g*s - x_ddot*c - (mu_p*theta_dot)/(m*l))
    
    return np.array([x_dot, x_ddot, theta_dot, theta_ddot])

# CONTROLLER 1 - Defining the PID controller (For simplicity PD controller is used)

def pid_control_law(y_filt, params, x_ref=0.0):
    """
    PID control based on filtered measurement 'y_filt'
    
    Input:
        y_filt = [x, x_dot, theta, theta_dot]
        params = {'Kp_x', 'Kd_x', 'Kp_theta', 'Kd_theta'}
        x_ref = reference cart position, default=0

    Returns:
        F = control force
    """
    x, x_dot, theta, theta_dot = y_filt
    
    Kp_x     = params['Kp_x']
    Kd_x     = params['Kd_x']
    Kp_theta = params['Kp_theta']
    Kd_theta = params['Kd_theta']

    # Outer loop: cart position -> desired pendulum angle
    x_error   = x_ref - x
    theta_ref = np.pi + (Kp_x * x_error - Kd_x * x_dot)     # Offset is necessary as pendulum is inverted

    # Inner loop: pendulum angle -> control force
    theta_error = theta_ref - theta
    F = Kp_theta * theta_error - Kd_theta * theta_dot
    return F


def low_pass_filter(old_val, new_meas, alpha=0.5):
    """
    Discrete first-order LP filter:
       filt_{k+1} = alpha * new_meas + (1-alpha)*old_val

    Input:
        old_val = previous filtered value
        new_meas = new measurement
        alpha = filter coefficient
    
    Returns:
        new filtered value
    """
    return alpha*new_meas + (1 - alpha)*old_val


# SIMULATION CLASS

class CartPendulumSim:
    """

    Holds the simulation state, figure objects, and does the simulation steps.
    One step at a time will be updated after the slider values are fetched from Tk.

    Input:
        params = dictionary of parameters for the cart-pendulum
        dt = time step
        noise_std = standard deviation of Gaussian noise for [x, x_dot, theta, theta_dot]
        alpha = LP filter coefficient
        t_final = final duration for the simulation
    """
    def __init__(self, params, dt=0.01, noise_std=[0,0,0,0], alpha=0.5, t_final=20.0):
        
        # Store input settings
        self.params = params
        self.dt = dt
        self.noise_std = noise_std
        self.alpha = alpha
        self.t_final = t_final
        self.manual_force = 0.0
        self.controller_type = "PID"

        # Initial conditions
        theta0 = np.deg2rad(params.get('theta0', 90.0))
        self.y_true = np.array([0.0, 0.0, theta0, 0.0])
        self.y_filt = self.y_true.copy()

        self.t = 0.0
        self.n_steps = int(np.floor(t_final / dt))
        self.step_count = 0

        # Storage for plotting lines
        self.time_history        = []
        self.x_true_history      = []
        self.x_meas_history      = []
        self.x_filt_history      = []
        self.theta_true_history  = []
        self.theta_meas_history  = []
        self.theta_filt_history  = []
        self.F_history           = []

        # Set up the outout figure
        self.fig = plt.figure(figsize=(12,6))

        # Left side - animation
        self.ax_anim = self.fig.add_subplot(1,2,1)
        self.ax_anim.set_xlim([-3, 3])
        self.ax_anim.set_ylim([-1.2, 1.2])
        self.ax_anim.set_aspect('equal')
        self.ax_anim.set_title("Cart-Pendulum Animation")

        self.cart_width = 0.3
        (self.cart_line,) = self.ax_anim.plot([], [], lw=4, color='k')
        (self.rod_line,)  = self.ax_anim.plot([], [], lw=2, color='blue')
        bob_radius = 0.05
        self.bob_circle = plt.Circle((0,0), bob_radius, fc='red')
        self.ax_anim.add_patch(self.bob_circle)

        gs = gridspec.GridSpec(nrows=3, ncols=1, figure=self.fig, 
                               left=0.6, right=0.98, hspace=0.6)

        # Right side (1) - x(t)
        self.ax_x = self.fig.add_subplot(gs[0, 0])
        self.ax_x.set_title("Cart Position x(t)")
        self.ax_x.set_xlabel("Time (s)")
        self.ax_x.set_ylabel("x (m)")
        self.ax_x.set_ylim([-3, 3])
        (self.line_x_true,) = self.ax_x.plot([], [], label='x true', color='blue')
        (self.line_x_meas,) = self.ax_x.plot([], [], '--', label='x noisy', color='orange')
        (self.line_x_filt,) = self.ax_x.plot([], [], ':', label='x filtered', color='green')
        self.ax_x.legend(loc='upper right')

        # Right side (2) - theta(t)
        self.ax_theta = self.fig.add_subplot(gs[1, 0])
        self.ax_theta.set_title("Pendulum Angle theta(t)")
        self.ax_theta.set_xlabel("Time (s)")
        self.ax_theta.set_ylabel("theta (rad)")
        self.ax_theta.set_ylim([0, 2*np.pi])
        (self.line_theta_true,) = self.ax_theta.plot([], [], label='theta true', color='blue')
        (self.line_theta_meas,) = self.ax_theta.plot([], [], '--', label='theta noisy', color='orange')
        (self.line_theta_filt,) = self.ax_theta.plot([], [], ':', label='theta filtered', color='green')
        self.ax_theta.legend(loc='upper right')

        # Right side (3) - control F(t)
        self.ax_F = self.fig.add_subplot(gs[2, 0])
        self.ax_F.set_title("Control Force F(t)")
        self.ax_F.set_xlabel("Time (s)")
        self.ax_F.set_ylabel("F (N)")
        self.ax_F.set_ylim([-10, 10])
        (self.line_F,) = self.ax_F.plot([], [], label='Control', color='red')
        self.ax_F.legend(loc='upper right')

        plt.tight_layout()
        plt.show(block=False)  # non-blocking show

    def update_animation(self, x, theta, F):
        # Update cart
        self.cart_line.set_data([x - self.cart_width/2, x + self.cart_width/2], [0, 0])
        # Update rod
        l = self.params['l']
        rod_x = [x, x + l * np.sin(theta)]
        rod_y = [0, -l * np.cos(theta)]
        self.rod_line.set_data(rod_x, rod_y)
        self.bob_circle.set_center((rod_x[1], rod_y[1]))

        # Remove previous force arrow
        if hasattr(self, "force_arrow"):
            self.force_arrow.remove()  

        # Compute new force arrow position (Scaling factor 0.1 for visualization)
        arrow_length = 0.1 * F  # Adjust length proportional to force magnitude
        arrow_x = x
        self.force_arrow = FancyArrow(arrow_x, 0.9, arrow_length, 0, width=0.04, color='red')

        # Re-add the arrow to the figure
        self.ax_anim.add_patch(self.force_arrow)

    def do_simulation_step(self):
        """
        Perform one step of the simulation.
        """

        # Storing the data for graphing
        self.time_history.append(self.t)
        self.x_true_history.append(self.y_true[0])
        self.theta_true_history.append(self.y_true[2])

        # Adding the requested noise to "Measurement"
        y_meas = self.y_true + np.random.normal(0, self.noise_std)
        self.x_meas_history.append(y_meas[0])
        self.theta_meas_history.append(y_meas[2])

        # Filtering the signal with a low-pass filter
        self.y_filt = low_pass_filter(self.y_filt, y_meas, alpha=self.alpha)
        self.x_filt_history.append(self.y_filt[0])
        self.theta_filt_history.append(self.y_filt[2])

        # Finding the appropriate control force with the selected controller
        if self.controller_type == "PID":
            F = pid_control_law(self.y_filt, self.params, x_ref=0.0)
        else:
            F = 0.0

        # Append the history of control signals without the application of manual force
        self.F_history.append(F)

        F = F + self.manual_force

        # Updating the animation
        self.update_animation(self.y_true[0], self.y_true[2], F)

        # Updating the graphs time-series lines
        self.line_x_true.set_data(self.time_history, self.x_true_history)
        self.line_x_meas.set_data(self.time_history, self.x_meas_history)
        self.line_x_filt.set_data(self.time_history, self.x_filt_history)
        self.ax_x.set_xlim([0, self.t_final])

        self.line_theta_true.set_data(self.time_history, self.theta_true_history)
        self.line_theta_meas.set_data(self.time_history, self.theta_meas_history)
        self.line_theta_filt.set_data(self.time_history, self.theta_filt_history)
        self.ax_theta.set_xlim([0, self.t_final])

        self.line_F.set_data(self.time_history, self.F_history)
        self.ax_F.set_xlim([0, self.t_final])

        plt.draw()

        # Integrate forward (simple Euler)
        if self.step_count < self.n_steps:
            dydt = cart_pendulum_dynamics(self.y_true, F, self.params)
            self.y_true = self.y_true + self.dt * dydt
            self.t += self.dt
            self.step_count += 1

# Tkinter GUI for live control and real-time updates
class LiveCartPendulumApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Live Cart-Pendulum Control")

        # Default param dictionary
        self.params = {
            'M': 1.0,
            'm': 0.2,
            'l': 0.5,
            'g': 9.81,

            # PID gains
            'Kp_x': 1.54,
            'Kd_x': 0.84,
            'Kp_theta': 18.36,
            'Kd_theta': 0.81,

            # Initial angle in degrees
            'theta0': 90.0,

            # (Optional) Friction coefficients
            'mu_c': 0.01,   # Cart friction coefficient
            'mu_p': 0.001   # Pendulum pivot friction coefficient
        }
        self.noise_std = [0.01, 0.0, 0.01, 0.0]
        self.alpha = 0.5
        self.dt = 0.01
        self.t_final = 20.0

        # Create a simulation object once the user clicks "Start"
        self.sim = None
        self.running = False

        # Build the sliders (ttk.Scale) + numeric-value labels
        self._build_sliders()
        
        # Input force controls for disturbance
        self._build_input_force_controls()

        # Conrtoller type selection
        self._build_controller_selector()

        # Start & Stop buttons
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)

        tk.Button(btn_frame, text="Start Simulation", bg="lightgreen",
                  command=self.start_simulation).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Stop", bg="tomato",
                  command=self.stop_simulation).pack(side=tk.LEFT, padx=5)

    def _build_sliders(self):

        # Row for each slider
        frm = tk.Frame(self.root)
        frm.pack(pady=10)

        # Kp_x
        self.kp_x_var = tk.DoubleVar(value=self.params['Kp_x'])
        self.lbl_kp_x_val = tk.Label(frm, text=f"{self.kp_x_var.get():.2f}")
        tk.Label(frm, text="Kp_x").grid(row=0, column=0, sticky="w")
        scale_kp_x = ttk.Scale(frm, from_=0, to=50, orient='horizontal',
                               variable=self.kp_x_var, length=200,
                               command=lambda v: self._update_label(self.lbl_kp_x_val, v))
        scale_kp_x.grid(row=0, column=1, padx=5, sticky="we")
        self.lbl_kp_x_val.grid(row=0, column=2, padx=5)

        # Kd_x
        self.kd_x_var = tk.DoubleVar(value=self.params['Kd_x'])
        self.lbl_kd_x_val = tk.Label(frm, text=f"{self.kd_x_var.get():.2f}")
        tk.Label(frm, text="Kd_x").grid(row=1, column=0, sticky="w")
        scale_kd_x = ttk.Scale(frm, from_=0, to=50, orient='horizontal',
                               variable=self.kd_x_var, length=200,
                               command=lambda v: self._update_label(self.lbl_kd_x_val, v))
        scale_kd_x.grid(row=1, column=1, padx=5, sticky="we")
        self.lbl_kd_x_val.grid(row=1, column=2, padx=5)

        # Kp_theta
        self.kp_theta_var = tk.DoubleVar(value=self.params['Kp_theta'])
        self.lbl_kp_theta_val = tk.Label(frm, text=f"{self.kp_theta_var.get():.2f}")
        tk.Label(frm, text="Kp_theta").grid(row=2, column=0, sticky="w")
        scale_kp_theta = ttk.Scale(frm, from_=0, to=50, orient='horizontal',
                                   variable=self.kp_theta_var, length=200,
                                   command=lambda v: self._update_label(self.lbl_kp_theta_val, v))
        scale_kp_theta.grid(row=2, column=1, padx=5, sticky="we")
        self.lbl_kp_theta_val.grid(row=2, column=2, padx=5)

        # Kd_theta
        self.kd_theta_var = tk.DoubleVar(value=self.params['Kd_theta'])
        self.lbl_kd_theta_val = tk.Label(frm, text=f"{self.kd_theta_var.get():.2f}")
        tk.Label(frm, text="Kd_theta").grid(row=3, column=0, sticky="w")
        scale_kd_theta = ttk.Scale(frm, from_=0, to=50, orient='horizontal',
                                   variable=self.kd_theta_var, length=200,
                                   command=lambda v: self._update_label(self.lbl_kd_theta_val, v))
        scale_kd_theta.grid(row=3, column=1, padx=5, sticky="we")
        self.lbl_kd_theta_val.grid(row=3, column=2, padx=5)

        # alpha (LP filter)
        self.alpha_var = tk.DoubleVar(value=self.alpha)
        self.lbl_alpha_val = tk.Label(frm, text=f"{self.alpha_var.get():.2f}")
        tk.Label(frm, text="LP alpha").grid(row=4, column=0, sticky="w")
        scale_alpha = ttk.Scale(frm, from_=0, to=1, orient='horizontal',
                                variable=self.alpha_var, length=200,
                                command=lambda v: self._update_label(self.lbl_alpha_val, v))
        scale_alpha.grid(row=4, column=1, padx=5, sticky="we")
        self.lbl_alpha_val.grid(row=4, column=2, padx=5)

        # Noise x
        self.noise_x_var = tk.DoubleVar(value=self.noise_std[0])
        self.lbl_noise_x_val = tk.Label(frm, text=f"{self.noise_x_var.get():.3f}")
        tk.Label(frm, text="Noise in x").grid(row=5, column=0, sticky="w")
        scale_noise_x = ttk.Scale(frm, from_=0, to=0.1, orient='horizontal',
                                  variable=self.noise_x_var, length=200,
                                  command=lambda v: self._update_label(self.lbl_noise_x_val, v, 3))
        scale_noise_x.grid(row=5, column=1, padx=5, sticky="we")
        self.lbl_noise_x_val.grid(row=5, column=2, padx=5)

        # Noise theta
        self.noise_th_var = tk.DoubleVar(value=self.noise_std[2])
        self.lbl_noise_th_val = tk.Label(frm, text=f"{self.noise_th_var.get():.3f}")
        tk.Label(frm, text="Noise in theta").grid(row=6, column=0, sticky="w")
        scale_noise_th = ttk.Scale(frm, from_=0, to=0.1, orient='horizontal',
                                   variable=self.noise_th_var, length=200,
                                   command=lambda v: self._update_label(self.lbl_noise_th_val, v, 3))
        scale_noise_th.grid(row=6, column=1, padx=5, sticky="we")
        self.lbl_noise_th_val.grid(row=6, column=2, padx=5)

        # Theta0
        self.theta0_var = tk.DoubleVar(value=self.params['theta0'])
        self.lbl_theta0_val = tk.Label(frm, text=f"{self.theta0_var.get():.1f}")
        tk.Label(frm, text="Initial Angle (deg)").grid(row=7, column=0, sticky="w")
        scale_theta0 = ttk.Scale(frm, from_=0, to=360, orient='horizontal',
                                 variable=self.theta0_var, length=200,
                                 command=lambda v: self._update_label(self.lbl_theta0_val, v, 1))
        scale_theta0.grid(row=7, column=1, padx=5, sticky="we")
        self.lbl_theta0_val.grid(row=7, column=2, padx=5)

        # mu_c
        self.mu_c_var = tk.DoubleVar(value=self.params['mu_c'])
        self.lbl_mu_c_val = tk.Label(frm, text=f"{self.mu_c_var.get():.3f}")
        tk.Label(frm, text="Cart Friction coefficient").grid(row=8, column=0, sticky="w")
        scale_mu_c = ttk.Scale(frm, from_=0, to=1, orient='horizontal',
                                 variable=self.mu_c_var, length=200,
                                 command=lambda v: self._update_label(self.lbl_mu_c_val, v, 3))
        scale_mu_c.grid(row=8, column=1, padx=5, sticky="we")
        self.lbl_mu_c_val.grid(row=8, column=2, padx=5)

        # mu_p
        self.mu_p_var = tk.DoubleVar(value=self.params['mu_p'])
        self.lbl_mu_p_val = tk.Label(frm, text=f"{self.mu_p_var.get():.3f}")
        tk.Label(frm, text="Pivot point Friction coefficient").grid(row=9, column=0, sticky="w")
        scale_mu_p = ttk.Scale(frm, from_=0, to=1, orient='horizontal',
                                 variable=self.mu_p_var, length=200,
                                 command=lambda v: self._update_label(self.lbl_mu_p_val, v, 3))
        scale_mu_p.grid(row=9, column=1, padx=5, sticky="we")
        self.lbl_mu_p_val.grid(row=9, column=2, padx=5)

        for i in range(10):
            frm.rowconfigure(i, weight=0)
        frm.columnconfigure(1, weight=1)
    
    def _build_input_force_controls(self):
        """
        Creates an input box for applying a manual force and a button to apply it.
        """
        frm = tk.Frame(self.root)
        frm.pack(pady=10)

        tk.Label(frm, text="Manual Force (N):").pack(side=tk.LEFT, padx=5)

        self.force_input = tk.Entry(frm, width=10)
        self.force_input.pack(side=tk.LEFT, padx=5)

        self.force_button = tk.Button(frm, text="Apply Force", bg="lightblue",
                                    command=self.apply_manual_force)
        self.force_button.pack(side=tk.LEFT, padx=5)

    def apply_manual_force(self):
        """
        Reads the manual force from the input box and applies it for one step.
        """
        try:
            self.sim.manual_force = float(self.force_input.get())  # Read user input
        except ValueError:
            self.sim.manual_force = 0.0  # Default to zero if invalid input
        
    def _build_controller_selector(self):
        """
        Creates a dropdown menu to select the controller type.
        """
        frm = tk.Frame(self.root)
        frm.pack(pady=10)

        tk.Label(frm, text="Select Controller:").pack(side=tk.LEFT, padx=5)

        self.controller_var = tk.StringVar(value="PID")
        self.controller_selector = ttk.Combobox(frm, textvariable=self.controller_var, 
                                                values=["PID", "Pole Placement", "LQR"], state="readonly")
        self.controller_selector.pack(side=tk.LEFT, padx=5)

    def _update_label(self, label_widget, new_val, precision=2):
        # Called whenever a slider moves. Update the label text
        fmt = f"{{:.{precision}f}}"
        label_widget.config(text=fmt.format(float(new_val)))

    def start_simulation(self):
        # Reset the simulation
        self.running = True
        # Update self.params from the current slider values:
        self.params['Kp_x']     = self.kp_x_var.get()
        self.params['Kd_x']     = self.kd_x_var.get()
        self.params['Kp_theta'] = self.kp_theta_var.get()
        self.params['Kd_theta'] = self.kd_theta_var.get()
        self.params['theta0']   = self.theta0_var.get()
        self.params['mu_c']   = self.mu_c_var.get()
        self.params['mu_p']   = self.mu_p_var.get()

        # Noise
        self.noise_std = [self.noise_x_var.get(), 0.0,
                          self.noise_th_var.get(), 0.0]
        # Filter alpha
        self.alpha = self.alpha_var.get()

        # Create a new simulation object
        self.sim = CartPendulumSim(params=self.params,
                                   dt=self.dt,
                                   noise_std=self.noise_std,
                                   alpha=self.alpha,
                                   t_final=self.t_final)
        # Start stepping
        self.root.after(0, self.update_sim)

    def update_sim(self):
        # If still running and we haven't exceeded sim steps

        if self.running and self.sim.step_count <= self.sim.n_steps:
            # Each step, re-read the slider values so control can change on the fly:
            self.params['Kp_x']     = self.kp_x_var.get()
            self.params['Kd_x']     = self.kd_x_var.get()
            self.params['Kp_theta'] = self.kp_theta_var.get()
            self.params['Kd_theta'] = self.kd_theta_var.get()
            self.params['mu_c']   = self.mu_c_var.get()
            self.params['mu_p']   = self.mu_p_var.get()

            self.sim.controller_type = self.controller_var.get()

            self.sim.alpha = self.alpha_var.get()
            self.sim.noise_std = [self.noise_x_var.get(), 0.0,
                                  self.noise_th_var.get(), 0.0]

            # Do a simulation step with updated parameters
            self.sim.params = self.params  # pass updated gains
            self.sim.do_simulation_step()

            # Schedule the next step
            self.root.after(1, self.update_sim)  # ~1 ms delay

            self.sim.manual_force = 0.0  # Reset after applying
        else:
            print("Simulation completed or stopped.")

    def stop_simulation(self):
        self.running = False


def main():
    root = tk.Tk()
    app = LiveCartPendulumApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
