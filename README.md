# Inverted Pendulum Real-time simulation

## Overview
This project implements a **cart-pendulum system simulation** using **Python, Tkinter, and Matplotlib**. The simulation models an inverted pendulum on a cart and provides two control methods:
1. **PID Controller** (Proportional-Derivative control)
2. **Pole Placement Controller**
3. **LQR** (In progress)

The simulation includes **real-time adjustable control parameters** via a **Tkinter GUI**, allowing users to interactively modify the system's behavior and observe the results.

## Features
- **Accurate Dynamics Simulation**: Implements the physics of an inverted pendulum on a moving cart.
- **Two Control Strategies**:
  - **PID (PD) Controller**: Maintains stability using a proportional-derivative control approach.
  - **Pole Placement Controller**: Uses state feedback for precise control.
- **Real-time Parameter Adjustment**: Users can modify PID gains, friction coefficients, drag parameters, and initial conditions.
- **Interactive GUI**:
  - Sliders for tuning PID and physics parameters
  - Drop-down menu to switch between control strategies
  - Manual force input for disturbance testing
- **Live Visualization**:
  - Animated pendulum-cart motion
  - Time series plots for cart position, pendulum angle, and control force

## Installation
### Prerequisites
Ensure you have **Python 3.x** installed along with the following dependencies:

```bash
pip install numpy scipy matplotlib tk
```

### Running the Simulation
Clone the repository and execute the main script:

```bash
git clone https://github.com/uhDann/Inverted-Pendulum.git
python real_time_with_sliders.py
```

## Usage
Upon running the script, a **Tkinter-based GUI** will open, allowing you to:
- Adjust PID control parameters using sliders
- Modify system properties (friction, drag, initial angle, etc.)
- Choose between **PID** and **Pole Placement** controllers
- Apply **manual force** disturbances
- Click **Start Simulation** to begin
- Observe **live simulation results** in the animation and plots

## Files
- [real_time_with_sliders.py](https://github.com/uhDann/Inverted-Pendulum/blob/main/real_time_with_sliders.py) - Main simulator
- [symbolic_lin.py](https://github.com/uhDann/Inverted-Pendulum/blob/main/symbolic_lin.py) - Symbolic linearization for the pole-placement
- [PID_visual_tuner.py](https://github.com/uhDann/Inverted-Pendulum/blob/main/PID_visual_tuner.py) - PID visual tuner, for faster tuning and visualization

## License
This project is licensed under the **MIT License**.
