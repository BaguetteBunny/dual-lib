# dual
> Exact, algebraic automatic differentiation via dual numbers for Python.
 
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.0.0-green.svg)](CHANGELOG.md)
 
---
 
## What Are Dual Numbers?
 
Dual numbers extend the real numbers by adjoining a nilpotent unit **ε** (epsilon), satisfying the single defining property:
 
```
ε² = 0,   ε ≠ 0
```
 
Every dual number takes the form **a + bε**, where a and b are real numbers. Arithmetic follows naturally from this rule. The key identity that makes dual numbers powerful is:
 
```
f(a + bε) = f(a) + b·f'(a)ε
```
 
for any differentiable function f. This means evaluating a function on a dual number simultaneously computes the **function value** (real part) and the **derivative** (dual part) exactly, with no approximation.
 
---
 
## Installation
 
Clone the repository and install in editable mode:
 
```bash
git clone https://github.com/BaguetteBunny/dual-numbers.git
cd dual-numbers
pip install -e .
```
 
Then in your code:
 
```python
from dual import Dual, Epsilon, derivative, gradient, jacobian
```
 
---
 
## Quick Start
 
```python
from dual import Dual, Epsilon, derivative, gradient, jacobian
 
# --- Basic construction ---
x = Dual(3, 1)       # 3 + 1ε  (seed for differentiation at x=3)
y = Dual(4, 2)       # 4 + 2ε
 
# --- Arithmetic ---
print(x + y)         # Dual(7, 3)
print(x * y)         # Dual(12, 10)
print(x ** 2)        # Dual(9, 6)   ← real: 3², dual: 2·3 = f'(3) for f=x²
 
# --- Derivative of f(x) = x³ at x = 2 ---
f = lambda x: x ** 3
result = f(Dual(2, 1))
print(result.real)   # 8.0   ← f(2)
print(result.dual)   # 12.0  ← f'(2) = 3·2² = 12
 
# --- Using the calculus helpers ---
print(derivative(lambda x: x**3, 2.0))          # 12.0
print(gradient(lambda x, y: x**2 + x*y, 3, 4)) # [10.0, 3.0]
```
 
---
 
## The Dual Class
 
### Construction
 
`Dual` accepts 0, 1, or 2 arguments:
 
```python
Dual()             # 0 + 0ε  (zero)
Dual(3)            # 3 + 0ε  (pure real)
Dual(3, 4)         # 3 + 4ε  (full dual)
Dual(other_dual)   # copy constructor
Dual([3, 4])       # from any iterable of two real numbers
```
 
### Arithmetic
 
All standard operators are supported between `Dual` and `Dual`, and between `Dual` and `int`/`float`:
 
| Expression | Real Part | Dual Part |
|---|---|---|
| `X + Y` | `a + c` | `b + d` |
| `X - Y` | `a - c` | `b - d` |
| `X * Y` | `ac` | `ad + bc` |
| `X / Y` | `a/c` | `(bc − ad) / c²` |
| `X ** Y` | `aᶜ` | `bc·aᶜ⁻¹ + d·aᶜ·ln(a)` |
| `X // Y` | `⌊a/c⌋` | `0` |
| `X % Y` | `a mod c` | `b − d·⌊a/c⌋` |
 
```python
x = Dual(3, 1)
y = Dual(2, 1)
 
x + y    # Dual(5, 2)
x * y    # Dual(6, 5)
x / y    # Dual(1.5, -0.25)
x ** 2   # Dual(9, 6)
x // y   # Dual(1, 0)
x % y    # Dual(1, -1)
```
 
Reflected operators (`__radd__`, `__rmul__`, etc.) are fully supported, so `3 + Dual(1, 2)` works as expected.
 
### Mathematical Functions
 
```python
x = Dual(1, 1)
 
x.exp()            # (e¹, e¹)        ← d/dx eˣ = eˣ
x.log()            # (0, 1)          ← d/dx ln(x) = 1/x
x.log(base=10)     # (0, 1/ln10)
x.gamma()          # (1, -γ)
x.erf()            # (~0.843, ~0.415)
abs(x)             # (|a|, b·sign(a))
```
 
### Trigonometric Functions
 
```python
x = Dual(1, 1)
 
x.sin()    # (sin(1), cos(1))
x.cos()    # (cos(1), -sin(1))
x.tan()    # (tan(1), sec²(1))
x.sec()    # (sec(1), sec(1)·tan(1))
x.csc()    # (csc(1), -csc(1)·cot(1))
x.cot()    # (cot(1), -csc²(1))
```

Their inverse, hyperbolic, and hyperbolic inverses are fully supported.
 
---
 
## The Epsilon Class
 
`Epsilon` is a convenience subclass of `Dual` representing the pure dual unit ε = 0 + 1ε:
 
```python
from dual import Epsilon
 
e = Epsilon()      # Dual(0, 1)
 
# Build dual numbers compositionally
x = 3 + 4 * Epsilon()   # Dual(3, 4)
 
# Verify the nilpotency property
Epsilon() ** 2     # Dual(0, 0)  ← ε² = 0  ✓
Epsilon() ** 3     # Dual(0, 0)  ← ε³ = 0  ✓
 
# Use as a differentiation seed
f = lambda x: x ** 3
f(2 + Epsilon()).dual    # 12.0  ← f'(2) = 3·2² = 12
```
 
## Automatic Differentiation
 
All calculus helpers live in `dual.calculus` and are exported from `dual` directly.
 
### derivative
 
Computes the exact derivative of a scalar function at a point:
 
```python
from dual import derivative
 
derivative(lambda x: x**3,   2.0)   # 12.0   ← 3x² at x=2
derivative(lambda x: x.sin(), 0.0)  # 1.0    ← cos(0)
derivative(lambda x: x.exp(), 1.0)  # 2.718  ← eˣ at x=1
```
 
### gradient
 
Computes all partial derivatives of a scalar-valued function, one forward pass per variable:
 
```python
from dual import gradient
 
# f(x, y) = x² + xy,  ∂f/∂x = 2x+y,  ∂f/∂y = x
gradient(lambda x, y: x**2 + x*y, 3, 4)
# → [10.0, 3.0]
```
 
### jacobian
 
Computes the full m×n Jacobian matrix of a vector-valued function:
 
```python
from dual import jacobian
 
# f: R³ → R³
f = lambda x, y, z: (x**2 + y + z,  x*y,  z + x)
jacobian(f, 2, 3, 4)
# → [[4, 3, 1],    ∂/∂x: 2x=4, y=3, 1
#    [1, 2, 0],    ∂/∂y: 1, x=2, 0
#    [1, 0, 1]]    ∂/∂z: 1, 0, 1
```
 
`gradient` and `derivative` are both special cases of `jacobian`:
 
```
derivative  →  n=1, m=1  →  a single number
gradient    →  m=1        →  a vector of length n
jacobian    →  full n×m   →  a matrix
```

---

## Accuracy
 
Dual number differentiation is **exact to float64 precision** (~15–16 significant digits). It has none of the approximation error intrinsic to finite differences:
 
```python
import math as M
from dual import derivative
 
x = 1.5
h = 1e-7
 
# f(x) = sin(x²),  f'(x) = 2x·cos(x²),  f'(1.5) = 3·cos(2.25) ≈ -1.8845208...
finite_diff = (M.sin((x+h)**2) - M.sin(x**2)) / h    # -1.8845212823581647  ← error from h
dual_result = derivative(lambda x: (x**2).sin(), 1.5) # -1.8845208681682175
exact       = M.cos(x**2) * 2 * x                     # -1.8845208681682175  ✓
```
 
The dual result matches the analytical exact value to all 16 digits. The finite difference accumulates error from both the finite step size `h` and floating point cancellation when subtracting two close values.

---
 
## Limitations
 
**Piecewise constant functions** — `floor`, `ceil`, `round`, and `sign` have zero derivative almost everywhere but are undefined (Dirac delta) at their discontinuity points. The library raises a `ValueError` there:
 
```python
Dual(2.0, 1).floor()   # ✅  a=2.0 is an integer → raises ValueError
Dual(2.5, 1).floor()   # ✅  a=2.5, not an integer → Dual(2.0, 0)
```
 
**Negative base with dual exponent** — `X**Y` requires `ln(a)`, so `a` must be strictly positive:
 
```python
Dual(-2, 1) ** Dual(3, 1)   # raises ValueError — ln(-2) undefined
Dual(-2, 1) ** 3             # ✅ scalar exponent is fine → Dual(-8, 12)
```
 
**Singularities** are the same as their real counterparts — `log(0)`, `tan(π/2)`, `asin(1)`, etc. all raise the expected errors.
 
**Symbolic inspection** — dual numbers are numerical, not symbolic. The derivative of `f` cannot be printed as a formula. Use [sympy](https://www.sympy.org/) if you need symbolic expressions.
 
**Bit operations** — `&`, `|`, `^`, `<<`, `>>` are not implemented. They have no meaningful derivative and operate only on integers, making dual number semantics undefined. Extract the real part explicitly if needed: `int(x.real) & int(y.real)`.
 
---
 
## License
 
MIT License — see [LICENSE](LICENSE) for details.