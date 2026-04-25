import pytest
import math as M
from dual import Dual

def ddx_erf(a): return (2 / M.sqrt(M.pi)) * M.exp(-a**2)

def digamma(x):
    result = 0.0
    while x < 15:
        result -= 1 / x
        x += 1
    r = 1 / x
    r2 = r * r
    result += M.log(x) - 0.5*r - (1/12)*r2 + (1/120)*r2**2 - (1/252)*r2**3
    return result


def test_sign_positive():
    x = Dual(3.0, 5.0).sign()
    assert x.real == pytest.approx(1.0)
    assert x.dual == pytest.approx(0.0)

def test_sign_negative():
    x = Dual(-3.0, 5.0).sign()
    assert x.real == pytest.approx(-1.0)
    assert x.dual == pytest.approx(0.0)

def test_sign_dual_always_zero():
    assert Dual(2.0, 99.0).sign().dual == pytest.approx(0.0)
    assert Dual(-2.0, 99.0).sign().dual == pytest.approx(0.0)

def test_sign_zero_raises():
    with pytest.raises(ValueError):
        Dual(0.0, 1.0).sign()

def test_exp_real():
    assert Dual(1.2, 1.0).exp().real == pytest.approx(M.exp(1.2))

def test_exp_dual():
    assert Dual(1.2, 1.0).exp().dual == pytest.approx(M.exp(1.2))

def test_exp_dual_scaled():
    assert Dual(1.2, 3.0).exp().dual == pytest.approx(3.0 * M.exp(1.2))

def test_exp_zero():
    x = Dual(0.0, 1.0).exp()
    assert x.real == pytest.approx(1.0)
    assert x.dual == pytest.approx(1.0)

def test_exp_negative():
    x = Dual(-1.0, 1.0).exp()
    assert x.real == pytest.approx(M.exp(-1.0))
    assert x.dual == pytest.approx(M.exp(-1.0))

def test_log_natural_real():
    assert Dual(1.2, 1.0).log().real == pytest.approx(M.log(1.2))

def test_log_natural_dual():
    assert Dual(1.2, 1.0).log().dual == pytest.approx(1 / 1.2)

def test_log_natural_dual_scaled():
    assert Dual(1.2, 3.0).log().dual == pytest.approx(3.0 / 1.2)

def test_log_base10_real():
    assert Dual(1.2, 1.0).log(10).real == pytest.approx(M.log(1.2, 10))

def test_log_base10_dual():
    assert Dual(1.2, 1.0).log(10).dual == pytest.approx(1 / (1.2 * M.log(10)))

def test_log_base2_dual():
    assert Dual(1.2, 1.0).log(2).dual == pytest.approx(1 / (1.2 * M.log(2)))

def test_log_at_one():
    x = Dual(1.0, 1.0).log()
    assert x.real == pytest.approx(0.0)
    assert x.dual == pytest.approx(1.0)

def test_log_zero_raises():
    with pytest.raises((ValueError, ZeroDivisionError)):
        Dual(0.0, 1.0).log()

def test_log_negative_raises():
    with pytest.raises((ValueError, ZeroDivisionError)):
        Dual(-1.0, 1.0).log()

def test_gamma_real():
    assert Dual(5.0, 1.0).gamma().real == pytest.approx(24.0)

def test_gamma_half_real():
    assert Dual(0.5, 1.0).gamma().real == pytest.approx(M.sqrt(M.pi))

def test_gamma_dual():
    a = 3.0
    gamma_a = M.gamma(a)
    psi_a   = digamma(a)
    assert Dual(a, 1.0).gamma().dual == pytest.approx(gamma_a * psi_a)

def test_gamma_dual_scaled():
    a = 3.0
    b = 2.0
    gamma_a = M.gamma(a)
    psi_a   = digamma(a)
    assert Dual(a, b).gamma().dual == pytest.approx(b * gamma_a * psi_a)

def test_gamma_at_one():
    x = Dual(1.0, 1.0).gamma()
    assert x.real == pytest.approx(1.0)
    assert x.dual == pytest.approx(digamma(1.0))

def test_gamma_zero_raises():
    with pytest.raises((ValueError, ZeroDivisionError, OverflowError)):
        Dual(0.0, 1.0).gamma()

def test_gamma_negative_integer_raises():
    with pytest.raises((ValueError, ZeroDivisionError, OverflowError)):
        Dual(-1.0, 1.0).gamma()

def test_erf_real():
    assert Dual(1.0, 1.0).erf().real == pytest.approx(M.erf(1.0))

def test_erf_dual():
    a = 1.0
    assert Dual(a, 1.0).erf().dual == pytest.approx(ddx_erf(a))

def test_erf_dual_scaled():
    a, b = 1.0, 3.0
    assert Dual(a, b).erf().dual == pytest.approx(b * ddx_erf(a))

def test_erf_zero():
    x = Dual(0.0, 1.0).erf()
    assert x.real == pytest.approx(0.0)
    assert x.dual == pytest.approx(2 / M.sqrt(M.pi))

def test_erf_large():
    x = Dual(5.0, 1.0).erf()
    assert x.real == pytest.approx(1.0, abs=1e-10)
    assert x.dual == pytest.approx(ddx_erf(5.0))

def test_erf_negative():
    assert Dual(-1.0, 1.0).erf().real == pytest.approx(-M.erf(1.0))

def test_erfc_real():
    assert Dual(1.0, 1.0).erfc().real == pytest.approx(M.erfc(1.0))

def test_erfc_dual():
    a = 1.0
    assert Dual(a, 1.0).erfc().dual == pytest.approx(-ddx_erf(a))

def test_erfc_dual_scaled():
    a, b = 1.0, 3.0
    assert Dual(a, b).erfc().dual == pytest.approx(-b * ddx_erf(a))

def test_erfc_zero():
    x = Dual(0.0, 1.0).erfc()
    assert x.real == pytest.approx(1.0)
    assert x.dual == pytest.approx(-2 / M.sqrt(M.pi))

def test_erfc_large():
    x = Dual(5.0, 1.0).erfc()
    assert x.real == pytest.approx(0.0, abs=1e-10)

def test_erf_erfc_sum():
    a = 1.5
    assert Dual(a, 1.0).erf().real + Dual(a, 1.0).erfc().real == pytest.approx(1.0)

def test_erf_erfc_dual_opposite():
    a = 1.5
    erf_dual  = Dual(a, 1.0).erf().dual
    erfc_dual = Dual(a, 1.0).erfc().dual
    assert erf_dual + erfc_dual == pytest.approx(0.0)