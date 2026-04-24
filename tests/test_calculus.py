import pytest
import math as M
from dual import derivative, gradient, jacobian

def test_derivative_polynomial():
    assert derivative(lambda x: x**3, 2.0) == pytest.approx(12.0)

def test_derivative_trig():
    assert derivative(lambda x: x.sin(), 1.0) == pytest.approx(M.cos(1.0))

def test_derivative_exp():
    assert derivative(lambda x: x.exp(), 1.0) == pytest.approx(M.e)

def test_derivative_log():
    assert derivative(lambda x: x.log(), 1.0) == pytest.approx(1.0)

def test_derivative_chain():
    # d/dx sin(x²) = 2x·cos(x²)
    assert derivative(lambda x: (x**2).sin(), 1.5) == pytest.approx(M.cos(1.5**2) * 2 * 1.5)

def test_gradient_values():
    # f(x,y) = x² + xy,  ∂f/∂x = 2x+y = 10,  ∂f/∂y = x = 3
    g = gradient(lambda x, y: x**2 + x*y, 3.0, 4.0)
    assert g[0] == pytest.approx(10.0)
    assert g[1] == pytest.approx(3.0)

def test_gradient_length():
    g = gradient(lambda x, y, z: x*y*z, 1.0, 2.0, 3.0)
    assert len(g) == 3

def test_gradient_matches_derivative():
    f = lambda x: x**3
    assert derivative(f, 2.0) == pytest.approx(gradient(lambda x: x**3, 2.0)[0])

def test_jacobian_shape():
    f = lambda x, y: [x**2 + y, x*y]
    J = jacobian(f, 3.0, 4.0)
    assert len(J) == 2       # n inputs
    assert len(J[0]) == 2    # m outputs

def test_jacobian_values():
    f = lambda x, y, z: (x**2+y+z, x*y, z+x)
    J = jacobian(f, 2.0, 3.0, 4.0)
    assert J[0][0] == pytest.approx(4.0)   # ∂f₁/∂x = 2x = 4
    assert J[0][1] == pytest.approx(3.0)   # ∂f₁/∂y = y = 3 ... wait ∂f₂/∂x = y
    assert J[1][0] == pytest.approx(1.0)   # ∂f₁/∂y = 1
    assert J[2][0] == pytest.approx(1.0)   # ∂f₁/∂z = 1

def test_jacobian_is_gradient_when_scalar():
    f  = lambda x, y: x**2 + x*y
    g  = gradient(f, 3.0, 4.0)
    J  = jacobian(lambda x, y: [f(x, y)], 3.0, 4.0)
    assert [row[0] for row in J] == pytest.approx(g)