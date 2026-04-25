from .core import Dual
from typing import Callable, Union

Number = Union[int, float]

def derivative(f: Callable, x: Number) -> float:
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

def gradient(f: Callable, *args: Number) -> list[float]:
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

def jacobian(f: Callable, *args: Number) -> list[list[float]]:
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

def implicit_derivative(F: Callable, *args: Number) -> list[float]:
    """Compute all implicit partial derivatives for F(x₁, x₂, ..., xₙ) = 0.

    Treats the last argument as the dependent variable and computes its
    partial derivative with respect to each of the other variables using
    the implicit differentiation formula:

        ∂xₙ/∂xᵢ = -(∂F/∂xᵢ) / (∂F/∂xₙ)

    Parameters
    ----------
    F : callable
        An implicit function of n variables, F(x₁, ..., xₙ) = 0.
        Must be written using Dual-compatible operations.
    *args : int or float
        The point (x₁, x₂, ..., xₙ) at which to evaluate the derivatives.

    Returns
    -------
    list[float]
        A list of n-1 partial derivatives [∂xₙ/∂x₁, ∂xₙ/∂x₂, ..., ∂xₙ/∂xₙ₋₁].

    Raises
    ------
    ZeroDivisionError
        If ∂F/∂xₙ = 0 at the given point.
    ValueError
        If fewer than 2 arguments are provided.

    Examples
    --------
    Unit circle F(x, y) = x² + y² - 1, dy/dx = -x/y:

    >>> implicit_derivative_n(lambda x, y: x**2 + y**2 - 1, 0.5, M.sqrt(0.75))
    [-0.5773...]   ← list with one entry since n=2

    Sphere F(x, y, z) = x² + y² + z² - 1:

    >>> F = lambda x, y, z: x**2 + y**2 + z**2 - 1
    >>> implicit_derivative_n(F, 0.5, 0.5, 1/M.sqrt(2))
    [-0.7071..., -0.7071...]   ← [∂z/∂x, ∂z/∂y]
    """
    n = len(args)
    if n < 2: raise ValueError("implicit_derivative_n requires at least 2 arguments.")

    dF_dxn = F(*[Dual(a, 1) if i == n-1 else Dual(a, 0) for i, a in enumerate(args)]).dual
    if dF_dxn == 0: raise ZeroDivisionError(f"∂F/∂x{n} = 0 at {args} thus implicit function theorem does not apply.")

    result = []
    for i in range(n - 1):
        dF_dxi = F(*[Dual(a, 1) if j == i else Dual(a, 0) for j, a in enumerate(args)]).dual
        result.append(-dF_dxi / dF_dxn)

    return result