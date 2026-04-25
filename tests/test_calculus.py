import pytest
import math as M
from dual import derivative, gradient, jacobian, implicit_derivative

def test_derivative_polynomial():
    assert derivative(lambda x: x**3, 2.0) == pytest.approx(12.0)

def test_derivative_trig():
    assert derivative(lambda x: x.sin(), 1.0) == pytest.approx(M.cos(1.0))

def test_derivative_exp():
    assert derivative(lambda x: x.exp(), 1.0) == pytest.approx(M.e)

def test_derivative_log():
    assert derivative(lambda x: x.log(), 1.0) == pytest.approx(1.0)

def test_derivative_chain():
    assert derivative(lambda x: (x**2).sin(), 1.5) == pytest.approx(M.cos(1.5**2) * 2 * 1.5)

def test_gradient_values():
    g = gradient(lambda x, y: x**2 + x*y, 3.0, 4.0)
    assert g[0] == pytest.approx(10.0)
    assert g[1] == pytest.approx(3.0)

def test_gradient_length():
    assert len(gradient(lambda x, y, z: x*y*z, 1.0, 2.0, 3.0)) == 3

def test_gradient_matches_derivative():
    f = lambda x: x**3
    assert derivative(f, 2.0) == pytest.approx(gradient(lambda x: x**3, 2.0)[0])

def test_jacobian_shape():
    f = lambda x, y: [x**2 + y, x*y]
    J = jacobian(f, 3.0, 4.0)
    assert len(J) == 2
    assert len(J[0]) == 2

def test_jacobian_values():
    f = lambda x, y, z: (x**2+y+z, x*y, z+x)
    J = jacobian(f, 2.0, 3.0, 4.0)
    assert J[0][0] == pytest.approx(4.0)
    assert J[0][1] == pytest.approx(3.0)
    assert J[1][0] == pytest.approx(1.0)
    assert J[2][0] == pytest.approx(1.0)

def test_jacobian_is_gradient_when_scalar():
    f  = lambda x, y: x**2 + x*y
    g  = gradient(f, 3.0, 4.0)
    J  = jacobian(lambda x, y: [f(x, y)], 3.0, 4.0)
    assert [row[0] for row in J] == pytest.approx(g)

def test_implicit_two_variables():
    F = lambda x, y: x**2 + y**2 - 1
    result = implicit_derivative(F, 0.5, M.sqrt(0.75))
    assert len(result) == 1
    assert result[0] == pytest.approx(-0.5 / M.sqrt(0.75))

def test_implicit_three_variables():
    F = lambda x, y, z: x**2 + y**2 + z**2 - 1
    x, y, z = 0.5, 0.5, 1/M.sqrt(2)
    result = implicit_derivative(F, x, y, z)
    assert len(result) == 2
    assert result[0] == pytest.approx(-x / z)
    assert result[1] == pytest.approx(-y / z)

def test_implicit_four_variables():
    F = lambda x, y, z, w: x**2 + y**2 + z**2 + w**2 - 1
    result = implicit_derivative(F, 0.5, 0.5, 0.5, 0.5)
    assert len(result) == 3
    assert result[0] == pytest.approx(-1.0)
    assert result[1] == pytest.approx(-1.0)
    assert result[2] == pytest.approx(-1.0)

def test_implicit_too_few_args():
    with pytest.raises(ValueError): implicit_derivative(lambda x: x**2, 1.0)

def test_implicit_zero_denominator():
    F = lambda x, y: x**2 - y**2
    with pytest.raises(ZeroDivisionError): implicit_derivative(F, 1.0, 0.0)