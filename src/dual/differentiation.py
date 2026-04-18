from .main import Dual

def derivative(f, x) -> float:
    """
    Compute the derivative of a scalar function at a point using dual numbers.

    Evaluates f at the dual number (x + ε), where the dual part of the result
    equals f'(x) exactly, with no numerical approximation error.

    Parameters
    ----------
    f : callable
        A scalar-valued function of one variable, written using Dual-compatible
        operations (e.g. ``x.sin()``, ``x ** 2``).
    x : int or float
        The point at which to evaluate the derivative.

    Returns
    -------
    float
        The exact derivative f'(x).

    Examples
    --------
    >>> derivative(lambda x: x.exp(), 1)
    2.718281828459045  # e^1, since d/dx e^x = e^x
    """
    return f(Dual(x, 1)).dual

def gradient(f, *args) -> list[float]:
    """
    Compute the gradient of a scalar function at a point using dual numbers.

    Performs one forward pass per input variable, each time seeding a single
    argument with dual part 1 and the rest with dual part 0. The dual part of
    each output gives the corresponding partial derivative.

    Parameters
    ----------
    f : callable
        A scalar-valued function of n variables, written using Dual-compatible
        operations.
    *args : int or float
        The point (x₁, x₂, ..., xₙ) at which to evaluate the gradient.

    Returns
    -------
    list[float]
        A vector [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ] of length n.

    Examples
    --------
    >>> gradient(lambda x, y: x**2 + x*y, 3, 4)
    [10.0, 3.0]  # ∂f/∂x = 2x+y = 10, ∂f/∂y = x = 3
    """
    return [f(*[Dual(a, 1) if i == j else Dual(a, 0)
                for j, a in enumerate(args)]).dual
            for i in range(len(args))]

def jacobian(f, *args) -> list[list[float]]:
    """
    Compute the Jacobian matrix of a vector function at a point using dual numbers.

    Performs one forward pass per input variable, each time seeding a single
    argument with dual part 1 and the rest with dual part 0. The dual parts of
    all outputs for that pass form one row of the Jacobian.

    The resulting matrix J has shape (n, m) where n is the number of input
    variables and m is the number of output variables, with J[i][j] = ∂fⱼ/∂xᵢ.

    Parameters
    ----------
    f : callable
        A vector-valued function of n variables returning a tuple or list of m
        Dual numbers, written using Dual-compatible operations.
    *args : int or float
        The point (x₁, x₂, ..., xₙ) at which to evaluate the Jacobian.

    Returns
    -------
    list[list[float]]
        An n×m matrix where entry [i][j] is ∂fⱼ/∂xᵢ.

    Examples
    --------
    >>> jacobian(lambda x, y, z: (x**2+y+z, x*y, z+x), 2, 3, 4)
    [[4, 3, 1],   # ∂/∂x: 2x=4, y=3, 1
     [1, 2, 0],   # ∂/∂y: 1, x=2, 0
     [1, 0, 1]]   # ∂/∂z: 1, 0, 1
    """
    rows = []
    for i in range(len(args)):
        outputs = f(*[Dual(a, 1) if i == j else Dual(a, 0)
                      for j, a in enumerate(args)])
        rows.append([out.dual for out in outputs])
    return rows
