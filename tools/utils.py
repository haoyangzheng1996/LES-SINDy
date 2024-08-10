import numpy as np


def compute_derivatives(y, dt, inits, n=4):

    if n < 1:
        raise ValueError("n must be at least 1")
    if len(inits) < n:
        for i in range(n - len(inits)):
            inits.append(0)
    elif len(inits) > n:
        n = len(inits)

    # Create a dictionary to hold the derivatives
    derivatives = {f'{i + 1}th': np.zeros_like(y) for i in range(n)}

    # Apply initial conditions (assuming they are provided for the first n derivatives)
    for i in range(min(n, len(inits))):
        derivatives[f'{i + 1}th'][0] = inits[i]  # Given initial condition for i-th derivative

    for i in range(n):
        if i == 0:
            derivatives['1th'][1:-1] = (y[2:] - y[:-2]) / (2 * dt)
        elif i == 1:
            derivatives['2th'][1:-1] = (y[2:] - 2 * y[1:-1] + y[:-2]) / dt ** 2
        elif i == 2:
            derivatives['3th'][2:-2] = (y[4:] - 2 * y[3:-1] + 2 * y[1:-3] - y[:-4]) / (2 * dt ** 3)
        elif i == 3:
            derivatives['4th'][2:-2] = (y[4:] - 4 * y[3:-1] + 6 * y[2:-2] - 4 * y[1:-3] + y[:-4]) / dt ** 4
            derivatives['4th'][0] = (y[4] - 4 * y[3] + 6 * y[2] - 4 * y[1] + y[0]) / dt ** 4
        else:
            # Higher-order derivatives (central differences)
            for j in range(2, len(y) - 2):
                coeffs = [(-1) ** k * np.math.comb(i + 1, k) for k in range(i + 2)]
                terms = sum(coeffs[k] * y[j + k - i // 2] for k in range(i + 2) if 0 <= j + k - i // 2 < len(y))
                derivatives[f'{i + 1}th'][j] = terms / (dt ** (i + 1))

    return [derivatives[f'{i + 1}th'] for i in range(n)]
