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

__version__ = "0.0.1"
__author__  = "Louis"