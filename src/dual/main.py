from typing import Iterable, Union
import math as M

Number = Union[int, float]

def _valid_input(val) -> bool:
    return isinstance(val, (int, float))

def _round_diff(x: Number) -> int:
    if x == 0.5: ValueError("round() dual part undefined for real part half integer (Dirac Delta)")
    return 0

def _floor_diff(x: Number) -> int:
    if x == int(x): ValueError("floor() dual part undefined for real part integer (Dirac Delta)")
    return 0

def _ceil_diff(x: Number) -> int:
    if x == int(x): ValueError("ceil() dual part undefined for real part integer (Dirac Delta)")
    return 0

def _sign(x: Number) -> int:
    if x == 0: raise ValueError("sign() dual part undefined when real part is 0 (Dirac Delta)")
    return (x > 0) - (x < 0)

class Dual:
    """
    A dual number of the form a + bε, where ε is the dual unit.

    Dual numbers extend the real numbers by adjoining a nilpotent unit ε
    satisfying ε² = 0. This single property makes them exact, algebraic
    differentiators — evaluating f(a + ε) yields f(a) + f'(a)ε with no
    floating point approximation, unlike finite differences.

    Parameters
    ----------
    real : int or float
        The real component a.
    dual : int or float
        The dual component b, the coefficient of ε.

    Attributes
    ----------
    real : float
        The real part a of the dual number a + bε.
    dual : float
        The dual part b of the dual number a + bε.

    Properties
    ----------
    ε² = 0
        The defining nilpotency property. All higher powers of ε vanish,
        meaning dual number arithmetic is exact and closed — no infinite
        series, no approximation.

    f(a + bε) = f(a) + b·f'(a)ε
        The fundamental identity underlying automatic differentiation.
        Any differentiable function applied to a dual number yields the
        function value in the real part and the scaled derivative in the
        dual part.

    Arithmetic Operations
    ---------------------
    Addition:
        (a + bε) + (c + dε) = (a+c) + (b+d)ε

    Subtraction:
        (a + bε) - (c + dε) = (a-c) + (b-d)ε

    Multiplication:
        (a + bε)(c + dε) = ac + (ad+bc)ε        [ε² term vanishes]

    Division:
        (a + bε) / (c + dε) = a/c + (bc-ad)/c²ε

    Power (scalar):
        (a + bε)^n = aⁿ + b·n·aⁿ⁻¹ε

    Power (dual):
        (a + bε)^(c+dε) = aᶜ + (bc·aᶜ⁻¹ + d·aᶜ·ln(a))ε

    Floor division:
        (a + bε) // (c + dε) = ⌊a/c⌋              [dual part annihilated]

    Modulo:
        (a + bε) % (c + dε) = (a%c) + (b - d·⌊a/c⌋)ε

    Examples
    --------
    Constructing dual numbers:

    >>> x = Dual(3, 1)  # 3 + 1ε
    >>> y = Dual(4, 2)  # 4 + 2ε

    Basic arithmetic:

    >>> x + y
    Dual(7, 3)

    >>> x * y
    Dual(12, 10)  # real: 3*4, dual: 3*2 + 1*4

    >>> x / y
    Dual(0.75, -0.125)

    Computing a derivative — seed dual part with 1:

    >>> f = lambda x: x ** 3
    >>> f(Dual(2, 1))
    Dual(8, 12)  # f(2)=8, f'(2)=3*2²=12

    Composing functions — chain rule is automatic:

    >>> g = lambda x: x.sin() * x ** 2
    >>> g(Dual(1, 1))
    Dual(0.8414709848, 1.6829419696)  # real: sin(1), dual: 2sin(1) + cos(1)

    Mathematical functions:

    >>> Dual(0, 1).sin()
    Dual(0.0, 1.0)   # sin(0)=0, cos(0)=1

    >>> Dual(1, 1).exp()
    Dual(2.718, 2.718)  # e^1, derivative of e^x at x=1 is also e^1

    Nilpotency:

    >>> Dual(0, 1) ** 2
    Dual(0, 0)  # ε² = 0

    >>> Dual(0, 1) ** 3
    Dual(0, 0)  # ε³ = 0

    Singularities:

    >>> Dual(0, 1).log()
    ValueError  # ln(0) undefined

    >>> Dual(math.pi/2, 1).tan()
    ZeroDivisionError  # tan undefined at π/2

    >>> Dual(1, 1).asin()
    ValueError  # √(1-x²) = 0 at boundary

    Notes
    -----
    The dual part should be interpreted as a directional derivative. Setting
    dual=1 seeds the direction along x, so the output dual part gives the
    derivative of f with respect to x. This is the foundation of forward-mode
    automatic differentiation.

    Dual numbers are not ordered beyond their real parts. Comparisons
    (``<``, ``>``, ``<=``, ``>=``) operate on real parts only, and the dual
    part carries no notion of magnitude or sign for ordering purposes.

    Dual numbers with dual=0 behave identically to real numbers under all
    operations. A Dual with both parts zero is the additive identity.

    See Also
    --------
    Epsilon : The pure dual unit ε = Dual(0, 1).
    derivative : Compute f'(x) using dual numbers.
    gradient : Compute ∇f at a point using dual numbers.
    jacobian : Compute the Jacobian matrix of a vector function.
    """
    def __init__(self, *args) -> None:
        self.real: float = 0.0
        self.dual: float = 0.0

        match len(args):
            case 0:
                return

            case 1:
                arg = args[0]
                if isinstance(arg, Dual):
                    self.real, self.dual = arg.real, arg.dual

                elif isinstance(arg, (int, float)):
                    self.real = float(arg)

                elif isinstance(arg, Iterable):
                    items = list(arg)
                    if len(items) == 2 and all(_valid_input(x) for x in items): self.real, self.dual = float(items[0]), float(items[1])
                    else: raise ValueError("Iterable argument must contain exactly two real numbers.")
                
                else:
                    raise TypeError(f"Unsupported argument type: {type(arg)!r}")

            case 2:
                a, b = args
                if not (_valid_input(a) and _valid_input(b)): raise ValueError("Both arguments must be real numbers (int or float).")
                self.real, self.dual = float(a), float(b)

            case _:
                raise ValueError("Dual() accepts 0, 1, or 2 arguments.")

    # Representation

    def __repr__(self) -> str: return f"Dual({self.real}, {self.dual})"

    def __str__(self) -> str:
        if self.real == 0:
            return f"{self.dual}ε" if self.dual != 0 else "0"
        else:
            sign = "+" if self.dual >= 0 else "-"
            return f"{self.real} {sign} {abs(self.dual)}ε"
    
    def __bool__(self) -> bool: return self.real != 0

    def __int__(self) -> int: return int(self.real)

    def __float__(self) -> float: return self.real

    # Add

    def __add__(self, other: "Dual | Number") -> "Dual":
        if isinstance(other, (int, float)):
            return Dual(self.real + other, self.dual)
        if isinstance(other, Dual):
            return Dual(self.real + other.real, self.dual + other.dual)
        return NotImplemented

    def __radd__(self, other: Number) -> "Dual": return self.__add__(other)

    # Sub

    def __sub__(self, other: "Dual | Number") -> "Dual":
        if isinstance(other, (int, float)):
            return Dual(self.real - other, self.dual)
        if isinstance(other, Dual):
            return Dual(self.real - other.real, self.dual - other.dual)
        return NotImplemented

    def __rsub__(self, other: Number) -> "Dual":
        if isinstance(other, (int, float)):
            return Dual(other - self.real, -self.dual)
        return NotImplemented
    
    # Mult

    def __mul__(self, other: "Dual | Number") -> "Dual":
        if isinstance(other, (int, float)):
            return Dual(self.real * other, self.dual * other)
        if isinstance(other, Dual):
            return Dual(
                self.real * other.real, 
                self.real * other.dual + self.dual * other.real
                )
        return NotImplemented

    def __rmul__(self, other: Number) -> "Dual": return self.__mul__(other)

    # Divisions

    def __truediv__(self, other: "Dual | Number") -> "Dual":
        if isinstance(other, (int, float)):
            return Dual(self.real / other, self.dual / other)
        if isinstance(other, Dual):
            return Dual(
                self.real / other.real, 
                (self.dual * other.real - self.real * other.dual) / (other.real**2)
                )
        return NotImplemented
    
    def __rtruediv__(self, other: Number) -> "Dual":
        if isinstance(other, (int, float)):
            return Dual(other, 0) / self
        return NotImplemented
    
    def __mod__(self, other: "Dual | Number") -> "Dual":
        if isinstance(other, (int, float)):
            return Dual(self.real % other, self.dual)
        if isinstance(other, Dual):
            return Dual(self.real % other.real, self.dual - other.dual * self.real // other.real)
        return NotImplemented
    
    def __rmod__(self, other: Number) -> float:
        if isinstance(other, (int, float)):
            return Dual(other % self.real, -(other // self.real) * self.dual)
        return NotImplemented
    
    def __floordiv__(self, other: "Dual | Number") -> "Dual":
        if isinstance(other, (int, float)):
            return Dual(self.real // other, 0)
        if isinstance(other, Dual):
            return Dual(self.real // other.real, 0)
        return NotImplemented
    
    def __rfloordiv__(self, other: Number) -> "Dual":
        if isinstance(other, (int, float)):
            return Dual(other // self.real, 0)
        return NotImplemented
    
    def __divmod__(self, other: "Dual | Number") -> tuple["Dual"]: return self // other, self % other

    # Pow

    def __pow__(self, other):
        if isinstance(other, (int, float)):
            return Dual(self.real**other, self.dual * other * self.real**(other - 1))
        if isinstance(other, Dual):
            real = self.real ** other.real
            return Dual(real, self.dual * other.real * self.real**(other.real - 1) + other.dual * real * M.log(self.real))
        return NotImplemented
    
    def __rpow__(self, other: Number) -> "Dual":
        if isinstance(other, (int, float)):
            real = other ** self.real
            return Dual(real, self.dual * real * M.log(other))
        return NotImplemented

    # Unary

    def __neg__(self) -> "Dual":
        return Dual(-self.real, -self.dual)

    def __pos__(self) -> "Dual":
        return Dual(self.real, self.dual)

    # Equality & Inequality

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Dual):
            return self.real == other.real and self.dual == other.dual
        if isinstance(other, (int, float)):
            return self.real == other and self.dual == 0.0
        return NotImplemented

    def __ne__(self, other: object) -> bool:
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    def __lt__(self, other: "Dual | Number") -> bool:
        if isinstance(other, (int, float)):
            other = Dual(other, 0.0)

        if isinstance(other, Dual):
            if self.real != other.real:
                return self.real < other.real
            return self.dual < other.dual
        return NotImplemented

    def __le__(self, other: "Dual | Number") -> bool:
        if isinstance(other, (int, float)):
            other = Dual(other, 0.0)

        if isinstance(other, Dual):
            if self.real != other.real:
                return self.real < other.real
            return self.dual <= other.dual
        return NotImplemented

    def __gt__(self, other: "Dual | Number") -> bool:
        if isinstance(other, (int, float)):
            other = Dual(other, 0.0)

        if isinstance(other, Dual):
            if self.real != other.real:
                return self.real > other.real
            return self.dual > other.dual
        return NotImplemented

    def __ge__(self, other: "Dual | Number") -> bool:
        if isinstance(other, (int, float)):
            other = Dual(other, 0.0)

        if isinstance(other, Dual):
            if self.real != other.real:
                return self.real > other.real
            return self.dual >= other.dual
        return NotImplemented

    # Hashing

    def __hash__(self) -> int:
        return hash((self.real, self.dual))
    
    def ceil(self) -> "Dual": return Dual(M.ceil(self.real), _ceil_diff(self.dual))

    def floor(self) -> "Dual": return Dual(M.floor(self.real), _floor_diff(self.dual))

    def __round__(self, ndigits = None) -> "Dual": return Dual(round(self.real, ndigits), _round_diff(self.dual))

    def __abs__(self) -> "Dual": return Dual(abs(self.real), self.dual * _sign(self.real))

    # Other Function

    def sign(self) -> "Dual": return Dual(_sign(self.real), 0)

    def exp(self) -> "Dual":
        e_a = M.exp(self.real)
        return Dual(e_a, e_a*self.dual)
    
    def log(self, base: float = M.e) -> "Dual":
        return Dual(M.log(self.real, base), self.dual/(M.log(base) * self.real))
        
    # Trigonometric Functions

    def sin(self) -> "Dual": return Dual(M.sin(self.real), self.dual * M.cos(self.real))
    
    def cos(self) -> "Dual": return Dual(M.cos(self.real), -self.dual * M.sin(self.real))
    
    def tan(self) -> "Dual": return Dual(M.tan(self.real), self.dual / M.cos(self.real)**2)

    def sec(self) -> "Dual":
        sec_a = 1 / M.cos(self.real)
        return Dual(sec_a, self.dual * sec_a * M.tan(self.real))

    def csc(self) -> "Dual":
        csc_a = 1 / M.sin(self.real)
        return Dual(csc_a, -self.dual * csc_a * (M.cos(self.real) / M.sin(self.real)))

    def cot(self) -> "Dual": return Dual(M.cos(self.real) / M.sin(self.real), -self.dual / M.sin(self.real)**2)

    def asin(self) -> "Dual": return Dual(M.asin(self.real), self.dual / M.sqrt(1 - self.real**2))

    def acos(self) -> "Dual": return Dual(M.acos(self.real), -self.dual / M.sqrt(1 - self.real**2))

    def atan(self) -> "Dual": return Dual(M.atan(self.real), self.dual / (1 + self.real**2))

    def asec(self) -> "Dual": return Dual(M.acos(1 / self.real), self.dual / (abs(self.real) * M.sqrt(self.real**2 - 1)))

    def acsc(self) -> "Dual": return Dual(M.asin(1 / self.real), -self.dual / (abs(self.real) * M.sqrt(self.real**2 - 1)))

    def acot(self) -> "Dual":  return Dual(M.pi/2 - M.atan(self.real), -self.dual / (1 + self.real**2))

    def sinh(self) -> "Dual": return Dual(M.sinh(self.real), self.dual * M.cosh(self.real))

    def cosh(self) -> "Dual": return Dual(M.cosh(self.real), self.dual * M.sinh(self.real))

    def tanh(self) -> "Dual": return Dual(M.tanh(self.real), self.dual / M.cosh(self.real)**2)

    def sech(self) -> "Dual":
        sech_a = 1 / M.cosh(self.real)
        return Dual(sech_a, -self.dual * sech_a * M.tanh(self.real))

    def csch(self) -> "Dual":
        csch_a = 1 / M.sinh(self.real)
        return Dual(csch_a, -self.dual * csch_a * (M.cosh(self.real) / M.sinh(self.real)))

    def coth(self) -> "Dual": return Dual(M.cosh(self.real) / M.sinh(self.real), -self.dual / M.sinh(self.real)**2)

    def asinh(self) -> "Dual": return Dual(M.asinh(self.real), self.dual / M.sqrt(self.real**2 + 1))

    def acosh(self) -> "Dual": return Dual(M.acosh(self.real), self.dual / M.sqrt(self.real**2 - 1))

    def atanh(self) -> "Dual": return Dual(M.atanh(self.real), self.dual / (1 - self.real**2))

    def asech(self) -> "Dual":  return Dual(M.acosh(1 / self.real), -self.dual / (self.real * M.sqrt(1 - self.real**2)))

    def acsch(self) -> "Dual": return Dual(M.asinh(1 / self.real), -self.dual / (abs(self.real) * M.sqrt(1 + self.real**2)))

    def acoth(self) -> "Dual": return Dual(M.atanh(1 / self.real), self.dual / (1 - self.real**2))

    # Misc

    def norm(self) -> float: return abs(self.real)

    def astuple(self) -> tuple: return self.real, self.dual

class Epsilon(Dual):
    """
    The pure dual unit ε, representing the dual number 0 + 1ε.

    A convenience subclass of Dual with real=0 and dual=1 fixed. Epsilon
    serves as the fundamental nilpotent unit of the dual number system,
    satisfying the defining property ε² = 0.

    Since Epsilon is a Dual with (real=0, dual=1), it inherits all Dual
    arithmetic and mathematical operations unchanged. It exists purely for
    notational clarity and convenience.
    """
    def __init__(self):
        self.real = 0
        self.dual = 1

    def __repr__(self) -> str:
        return f"Epsilon()"

    def __str__(self) -> str:
        return "ε"
    