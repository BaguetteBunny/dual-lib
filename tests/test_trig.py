import pytest
import math as M
from dual import Dual

def test_sin():
    x = Dual(1.2, 1).sin()
    assert x.real == pytest.approx(M.sin(1.2))
    assert x.dual == pytest.approx(M.cos(1.2))

def test_cos():
    x = Dual(1.2, 1).cos()
    assert x.real == pytest.approx(M.cos(1.2))
    assert x.dual == pytest.approx(-M.sin(1.2))

def test_tan():
    x = Dual(1.2, 1).tan()
    assert x.real == pytest.approx(M.tan(1.2))
    assert x.dual == pytest.approx(1 / M.cos(1.2)**2)

def test_sec():
    x = Dual(1.2, 1).sec()
    assert x.real == pytest.approx(1 / M.cos(1.2))
    assert x.dual == pytest.approx(M.tan(1.2) / M.cos(1.2))

def test_csc():
    x = Dual(1.2, 1).csc()
    assert x.real == pytest.approx(1 / M.sin(1.2))
    assert x.dual == pytest.approx(-M.cos(1.2) / M.sin(1.2)**2)

def test_cot():
    x = Dual(1.2, 1).cot()
    assert x.real == pytest.approx(M.cos(1.2) / M.sin(1.2))
    assert x.dual == pytest.approx(-1 / M.sin(1.2)**2)

def test_tan_singularity():
    with pytest.raises((ValueError, ZeroDivisionError)):
        Dual(M.pi / 2, 1).tan()

def test_csc_singularity():
    with pytest.raises((ValueError, ZeroDivisionError)):
        Dual(0.0, 1).csc()

def test_cot_singularity():
    with pytest.raises((ValueError, ZeroDivisionError)):
        Dual(0.0, 1).cot()

def test_asin():
    x = Dual(0.5, 1).asin()
    assert x.real == pytest.approx(M.asin(0.5))
    assert x.dual == pytest.approx(1 / M.sqrt(1 - 0.5**2))

def test_acos():
    x = Dual(0.5, 1).acos()
    assert x.real == pytest.approx(M.acos(0.5))
    assert x.dual == pytest.approx(-1 / M.sqrt(1 - 0.5**2))

def test_atan():
    x = Dual(0.5, 1).atan()
    assert x.real == pytest.approx(M.atan(0.5))
    assert x.dual == pytest.approx(1 / (1 + 0.5**2))

def test_asec():
    x = Dual(2.0, 1).asec()
    assert x.real == pytest.approx(M.acos(0.5))
    assert x.dual == pytest.approx(1 / (2.0 * M.sqrt(2.0**2 - 1)))

def test_acsc():
    x = Dual(2.0, 1).acsc()
    assert x.real == pytest.approx(M.asin(0.5))
    assert x.dual == pytest.approx(-1 / (2.0 * M.sqrt(2.0**2 - 1)))

def test_acot():
    x = Dual(0.5, 1).acot()
    assert x.real == pytest.approx(M.pi/2 - M.atan(0.5))
    assert x.dual == pytest.approx(-1 / (1 + 0.5**2))

def test_asin_domain():
    with pytest.raises((ValueError, ZeroDivisionError)):
        Dual(1.0, 1).asin()

def test_acos_domain():
    with pytest.raises((ValueError, ZeroDivisionError)):
        Dual(1.0, 1).acos()

def test_asec_domain():
    with pytest.raises((ValueError, ZeroDivisionError)):
        Dual(0.5, 1).asec()

def test_sinh():
    x = Dual(1.2, 1).sinh()
    assert x.real == pytest.approx(M.sinh(1.2))
    assert x.dual == pytest.approx(M.cosh(1.2))

def test_cosh():
    x = Dual(1.2, 1).cosh()
    assert x.real == pytest.approx(M.cosh(1.2))
    assert x.dual == pytest.approx(M.sinh(1.2))

def test_tanh():
    x = Dual(1.2, 1).tanh()
    assert x.real == pytest.approx(M.tanh(1.2))
    assert x.dual == pytest.approx(1 / M.cosh(1.2)**2)

def test_sech():
    x = Dual(1.2, 1).sech()
    assert x.real == pytest.approx(1 / M.cosh(1.2))
    assert x.dual == pytest.approx(-M.tanh(1.2) / M.cosh(1.2))

def test_csch():
    x = Dual(1.2, 1).csch()
    assert x.real == pytest.approx(1 / M.sinh(1.2))
    assert x.dual == pytest.approx(-M.cosh(1.2) / M.sinh(1.2)**2)

def test_coth():
    x = Dual(1.2, 1).coth()
    assert x.real == pytest.approx(M.cosh(1.2) / M.sinh(1.2))
    assert x.dual == pytest.approx(-1 / M.sinh(1.2)**2)

def test_csch_singularity():
    with pytest.raises((ValueError, ZeroDivisionError)):
        Dual(0.0, 1).csch()

def test_coth_singularity():
    with pytest.raises((ValueError, ZeroDivisionError)):
        Dual(0.0, 1).coth()

def test_asinh():
    x = Dual(1.2, 1).asinh()
    assert x.real == pytest.approx(M.asinh(1.2))
    assert x.dual == pytest.approx(1 / M.sqrt(1.2**2 + 1))

def test_acosh():
    x = Dual(1.5, 1).acosh()
    assert x.real == pytest.approx(M.acosh(1.5))
    assert x.dual == pytest.approx(1 / M.sqrt(1.5**2 - 1))

def test_atanh():
    x = Dual(0.5, 1).atanh()
    assert x.real == pytest.approx(M.atanh(0.5))
    assert x.dual == pytest.approx(1 / (1 - 0.5**2))

def test_asech():
    x = Dual(0.5, 1).asech()
    assert x.real == pytest.approx(M.acosh(2.0))
    assert x.dual == pytest.approx(-1 / (0.5 * M.sqrt(1 - 0.5**2)))

def test_acsch():
    x = Dual(1.2, 1).acsch()
    assert x.real == pytest.approx(M.asinh(1/1.2))
    assert x.dual == pytest.approx(-1 / (abs(1.2) * M.sqrt(1 + 1.2**2)))

def test_acoth():
    x = Dual(2.0, 1).acoth()
    assert x.real == pytest.approx(M.atanh(0.5))
    assert x.dual == pytest.approx(1 / (1 - 2.0**2))

def test_acosh_domain():
    with pytest.raises((ValueError, ZeroDivisionError)):
        Dual(0.5, 1).acosh()

def test_atanh_domain():
    with pytest.raises((ValueError, ZeroDivisionError)):
        Dual(1.0, 1).atanh()