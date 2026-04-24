import pytest
from dual import Dual

def test_floor_non_integer():
    x = Dual(2.5, 3).floor()
    assert x.real == pytest.approx(2.0)
    assert x.dual == pytest.approx(0.0)

def test_floor_integer_raises():
    with pytest.raises(ValueError):
        Dual(2.0, 1).floor()

def test_ceil_non_integer():
    x = Dual(2.5, 3).ceil()
    assert x.real == pytest.approx(3.0)
    assert x.dual == pytest.approx(0.0)

def test_ceil_integer_raises():
    with pytest.raises(ValueError):
        Dual(2.0, 1).ceil()

def test_round_non_integer():
    x = round(Dual(2.5, 3))
    assert x.real == pytest.approx(2.0)
    assert x.dual == pytest.approx(0.0)

def test_sign_positive():
    x = Dual(3, 5).sign()
    assert x.real == pytest.approx(1.0)
    assert x.dual == pytest.approx(0.0)

def test_sign_negative():
    x = Dual(-3, 5).sign()
    assert x.real == pytest.approx(-1.0)
    assert x.dual == pytest.approx(0.0)

def test_sign_zero_raises():
    with pytest.raises(ValueError):
        Dual(0, 1).sign()