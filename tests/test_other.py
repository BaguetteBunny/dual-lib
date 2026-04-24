import pytest
import math as M
from dual import Dual

def test_exp_log_roundtrip():
    x = Dual(1.5, 1)
    assert x.exp().log().real == pytest.approx(x.real)
    assert x.exp().log().dual == pytest.approx(x.dual)

def test_pow_consistent_with_mul():
    x = Dual(2.0, 1)
    assert (x**2).real == pytest.approx((x*x).real)
    assert (x**2).dual == pytest.approx((x*x).dual)

def test_chain_rule_sin_x_squared():
    # d/dx sin(x²) = 2x·cos(x²)
    x = Dual(1.5, 1)
    result = (x**2).sin()
    assert result.dual == pytest.approx(2 * 1.5 * M.cos(1.5**2))

def test_product_rule():
    # d/dx (x·sin(x)) = sin(x) + x·cos(x)
    x = Dual(1.2, 1)
    result = x * x.sin()
    assert result.dual == pytest.approx(M.sin(1.2) + 1.2 * M.cos(1.2))

def test_quotient_rule():
    # d/dx (sin(x)/cos(x)) = 1/cos²(x)
    x = Dual(1.2, 1)
    result = x.sin() / x.cos()
    assert result.dual == pytest.approx(1 / M.cos(1.2)**2)

def test_eps_squared_zero():
    e = Dual(0, 1)
    assert (e**2).real == pytest.approx(0.0)
    assert (e**2).dual == pytest.approx(0.0)

def test_accuracy_vs_finite_diff():
    # dual should match analytical, finite diff should not
    x = 1.5
    h = 1e-7
    fd = (M.sin((x+h)**2) - M.sin(x**2)) / h
    exact = M.cos(x**2) * 2 * x
    from dual import derivative
    dual = derivative(lambda x: (x**2).sin(), x)
    assert abs(dual - exact) < abs(fd - exact)