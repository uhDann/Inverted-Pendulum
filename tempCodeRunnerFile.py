  x_ddot = (
        -(I + m * l**2) * d * x_dot 
        + m * l * (I + m * l**2) * s * (theta_dot**2) 
        + (I + m * l**2) * F 
        - m**2 * l**2 * g * c * s
    ) / denom

    theta_ddot = (
        m * l * c * d * x_dot 
        - m**2 * l**2 * s * c * theta_dot**2 
        - m * l * c * F 
        + (M + m) * m * g * l * s
    ) / denom