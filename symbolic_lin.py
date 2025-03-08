import sympy as sp

# Defining the symbols
x, xdot, th, thdot, F, tau_pend= sp.symbols('x xdot th thdot F tau_pend', real=True)
m, M, l, g = sp.symbols('m M l g', positive=True)
mu_c, mu_p = sp.symbols('mu_c mu_p', positive=True)

s = sp.sin(th)
c = sp.cos(th)

den = m*c**2 - (sp.Rational(7,3))*M

num = (m*g*s*c
       - (sp.Rational(7,3))*(F + m*l*(thdot**2)*s - mu_c*xdot)
       - (mu_p*thdot*c)/l)

x_ddot_expr = num / den

th_ddot_expr = (sp.Rational(3,7)/l)*(
    g*s
    - x_ddot_expr*c
    - (mu_p*thdot)/(m*l)
    - tau_pend
)

# Define the state vector derivative f = [x_dot, x_ddot, theta_dot, theta_ddot]
f1 = xdot
f2 = x_ddot_expr
f3 = thdot
f4 = th_ddot_expr

f = sp.Matrix([f1, f2, f3, f4])  # This is [dot{x}, dot{x_dot}, dot{theta}, dot{theta_dot}]

# Define the state vector X and input u
X = sp.Matrix([x, xdot, th, thdot])
u = sp.Matrix([F])

# Compute Jacobians: A = df/dX,B = df/dF
A_sym = f.jacobian(X)
B_sym = f.jacobian(u)

# Evaluate at the equilibrium: x=0, xdot=0, th=pi, thdot=0, and (optionally) F=0
equil_subs = {
    x: 0,
    xdot: 0,
    th: sp.pi,
    thdot: 0,
    F: 0,          # input = 0
    tau_pend: 0    # Typically 0 at equilibrium
}

A_lin = A_sym.subs(equil_subs)
B_lin = B_sym.subs(equil_subs)

print("A =", A_lin)
print("B =", B_lin)