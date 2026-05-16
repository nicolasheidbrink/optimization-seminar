import numpy as np


"""
Shubert function
"""
def shubert(x):
    x = np.asarray(x)
    i = np.array([1, 2, 3, 4, 5])
    
    # Create a grid for broadcasting: (n, 1) and (1, 5)
    # This computes all i-terms for all x_j simultaneously
    inner_terms = i * np.cos((i + 1) * x[:, np.newaxis] + i)
    
    # Sum across the i-dimension (axis 1) then product across the x-dimension
    s_j = np.sum(inner_terms, axis=1)
    return np.prod(s_j)

def shubert_grad(x):
    x = np.asarray(x)
    n = len(x)
    i = np.arange(1, 6)
    
    # 1. Compute S_j (sums) and dS_j (derivatives of sums)
    # Shapes: x is (n, 1), i is (1, 5) -> result is (n, 5)
    x_grid = x[:, np.newaxis]
    
    s_terms = i * np.cos((i + 1) * x_grid + i)
    ds_terms = -i * (i + 1) * np.sin((i + 1) * x_grid + i)
    
    s = np.sum(s_terms, axis=1)   # Shape (n,)
    ds = np.sum(ds_terms, axis=1) # Shape (n,)
    
    # 2. Compute "product of all others" using prefix and suffix products
    # This avoids O(n^2) operations or division by zero errors.
    prefix = np.ones(n + 1)
    suffix = np.ones(n + 1)
    
    prefix[1:] = np.cumprod(s)
    suffix[:-1] = np.cumprod(s[::-1])[::-1]
    
    # The product excluding index j is prefix[j] * suffix[j+1]
    grad_others = prefix[:-1] * suffix[1:]
    
    return ds * grad_others



"""
Altered Shubert function
"""
def altered_shubert(x):
    return shubert(x) + 0.5 * ((x[0] + 1.42513) ** 2 + (x[1] + 0.80032) ** 2)

def altered_shubert_grad(x):
    grad_shubert = shubert_grad(x)
    grad_alteration = np.array([x[0] + 1.42513, x[1] + 0.80032])
    return grad_shubert + grad_alteration



"""
Six-Hump Camel function
"""
def six_hump_camel(x):
    x1, x2 = x
    term1 = (4 - 2.1 * x1**2 + (x1**4) / 3) * x1**2
    term2 = x1 * x2
    term3 = (-4 + 4 * x2**2) * x2**2
    return term1 + term2 + term3

def six_hump_camel_grad(x):
    x1, x2 = x
    # Gradient with respect to x1
    dterm1_dx1 = (4 - 2.1 * x1**2 + (x1**4) / 3) * 2 * x1 + (4/3 * x1**3 - 4.2 * x1) * x1**2
    dterm2_dx1 = x2
    dterm3_dx1 = 0

    # Gradient with respect to x2
    dterm1_dx2 = 0
    dterm2_dx2 = x1
    dterm3_dx2 = (-4 + 4 * x2**2) * 2 * x2 + (8 * x2) * x2**2

    return np.array([dterm1_dx1 + dterm2_dx1 + dterm3_dx1, dterm1_dx2 + dterm2_dx2 + dterm3_dx2])


"""
Function 16
"""
def function_16(x, k=10, A=1):
    """Implementation of Equation (16) from image_1d2eb3.png."""
    x = np.asarray(x)
    n = len(x)
    pi = np.pi
    
    term1 = k * (np.sin(pi * x[0])**2)
    sum_part = np.sum((x[:-1] - A)**2 * (1 + k * np.sin(pi * x[1:])**2))
    term3 = (x[-1] - A)**2
    
    return (pi / n) * (term1 + sum_part + term3)

def function_16_grad(x, k=10, A=1):
    """Gradient of Equation (16) from image_1d2eb3.png."""
    x = np.asarray(x)
    n = len(x)
    pi = np.pi
    
    grad = np.zeros_like(x, dtype=float)
    s = np.sin(pi * x)
    s2 = np.sin(2 * pi * x) # derivative of sin^2(pi*x) is pi*sin(2*pi*x)

    # Initial terms
    grad[0] += k * pi * s2[0]
    grad[-1] += 2 * (x[-1] - A)

    if n > 1:
        # Derivative of (x_i - A)^2 * (1 + k*sin^2(pi*x_{i+1})) wrt x_i
        grad[:-1] += 2 * (x[:-1] - A) * (1 + k * s[1:]**2)
        # Derivative of (x_i - A)^2 * (1 + k*sin^2(pi*x_{i+1})) wrt x_{i+1}
        grad[1:] += (x[:-1] - A)**2 * k * pi * s2[1:]

    return (pi / n) * grad

def function_15(x, k=10, A=1):
    x = np.asarray(x)
    y = 1 + 0.25 * (x - 1)
    return function_16(y, k, A)

def function_15_grad(x, k=10, A=1):
    x = np.asarray(x)
    y = 1 + 0.25 * (x - 1)
    return 0.25 * function_16_grad(y, k, A)


"""
Function 17
"""
def function_17(x, k0=1.0, k1=0.1, A=1.0, l0=3.0, l1=2.0):
    """Implementation of Equation (17) from image_1cbd61.png."""
    x = np.asarray(x)
    n = len(x)
    pi = np.pi
    
    # Initial term
    term1 = k1 * (np.sin(pi * l0 * x[0])**2)
    
    # Summation term: i=1 to n-1
    sum_part = 0
    if n > 1:
        sum_part = np.sum((x[:-1] - A)**2 * (1 + k0 * np.sin(pi * l0 * x[1:])**2))
    
    # Final term (using l1 for the terminal frequency, which is not done in the paper, where l1 is defined but never used)
    term_final = (x[-1] - A)**2 * (1 + k0 * np.sin(pi * l1 * x[-1])**2)
    
    return term1 + k1 * sum_part + k1 * term_final

def function_17_grad(x, k0=1.0, k1=0.1, A=1.0, l0=3.0, l1=2.0):
    """Gradient of Equation (17) from image_1cbd61.png."""
    x = np.asarray(x)
    n = len(x)
    pi = np.pi
    grad = np.zeros_like(x, dtype=float)
    
    # Periodic terms
    s_l0 = np.sin(pi * l0 * x)
    s2_l0 = np.sin(2 * pi * l0 * x)
    s_l1 = np.sin(pi * l1 * x)
    s2_l1 = np.sin(2 * pi * l1 * x)
    
    # Coordinate 1
    grad[0] += pi * l0 * s2_l0[0]
    if n > 1:
        grad[0] += 2 * (x[0] - A) * (1 + k0 * s_l0[1]**2)
    
    # Coordinates 2 to n-1
    if n > 2:
        # Derivative wrt (x_i - A)^2
        grad[1:-1] += 2 * (x[1:-1] - A) * (1 + k0 * s_l0[2:]**2)
        # Derivative wrt sin^2(pi * l0 * x_i) from summation
        grad[1:-1] += (x[:-2] - A)**2 * k0 * pi * l0 * s2_l0[1:-1]
        
    # Coordinate n
    if n > 1:
        # From summation part
        grad[-1] += (x[-2] - A)**2 * k0 * pi * l0 * s2_l0[-1]
        
    # From final standalone term
    grad[-1] += 2 * (x[-1] - A) * (1 + k0 * s_l1[-1]**2)
    grad[-1] += (x[-1] - A)**2 * k0 * pi * l1 * s2_l1[-1]
    
    return k1 * grad


def altered_cos(x):
    x = np.asarray(x)
    return np.array([np.cos(x[0]) + 0.2 * x[0]**2])

def altered_cos_grad(x):
    x = np.asarray(x)
    return np.array([-np.sin(x[0]) + 0.4 * x[0]])