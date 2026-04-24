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

def implicit_derivative(F: Callable, x: Number, y: Number) -> float:
    """Compute dy/dx for an implicit function F(x, y) = 0.

    Uses the implicit differentiation formula dy/dx = -(∂F/∂x) / (∂F/∂y),
    computing both partial derivatives via dual numbers in two forward passes.

    Parameters
    ----------
    F : callable
        A function of two variables representing the implicit equation F(x, y) = 0.
        Must be written using Dual-compatible operations.
    x : int or float
        The x-coordinate of the point at which to evaluate dy/dx.
    y : int or float
        The y-coordinate of the point at which to evaluate dy/dx.

    Returns
    -------
    float
        The derivative dy/dx at the point (x, y).

    Raises
    ------
    ZeroDivisionError
        If ∂F/∂y = 0 at the given point, meaning the implicit function
        theorem does not apply there.

    Examples
    --------
    Unit circle x² + y² = 1, dy/dx = -x/y:

    >>> F = lambda x, y: x**2 + y**2 - 1
    >>> implicit_derivative(F, 0.5, M.sqrt(0.75))
    -0.5773...   ← -x/y = -0.5/√0.75  ✓

    Ellipse x²/4 + y²/9 = 1, dy/dx = -9x/4y:

    >>> F = lambda x, y: x**2/4 + y**2/9 - 1
    >>> implicit_derivative(F, 1.0, 1.5*M.sqrt(3))
    -0.8660...   ✓
    """
    dF_dx = F(Dual(x, 1), Dual(y, 0)).dual
    dF_dy = F(Dual(x, 0), Dual(y, 1)).dual

    if dF_dy == 0: raise ZeroDivisionError(f"∂F/∂y = 0 at ({x}, {y}) thus implicit function theorem does not apply.")

    return -dF_dx / dF_dy