from .core import Dual, Epsilon
from .differentiation import derivative, gradient, jacobian, implicit_derivative

__all__ = [
    "Dual",
    "Epsilon",
    "derivative",
    "gradient",
    "jacobian",
    "implicit_derivative"
]

__version__ = "1.0.0"
__author__  = "Louis"