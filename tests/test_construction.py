import pytest
from dual import Dual, Epsilon

def test_no_args():
    x = Dual()
    assert x.real == 0.0
    assert x.dual == 0.0

def test_single_real():
    x = Dual(3)
    assert x.real == 3.0
    assert x.dual == 0.0

def test_two_args():
    x = Dual(3, 4)
    assert x.real == 3.0
    assert x.dual == 4.0

def test_copy_constructor():
    x = Dual(3, 4)
    y = Dual(x)
    assert y.real == 3.0
    assert y.dual == 4.0

def test_from_iterable():
    x = Dual([3, 4])
    assert x.real == 3.0
    assert x.dual == 4.0

def test_from_tuple():
    x = Dual((3, 4))
    assert x.real == 3.0
    assert x.dual == 4.0

def test_too_many_args():
    with pytest.raises(ValueError):
        Dual(1, 2, 3)

def test_bad_type():
    with pytest.raises(TypeError):
        Dual("bad")

def test_bad_iterable():
    with pytest.raises(ValueError):
        Dual([1, 2, 3])

def test_epsilon_real():
    assert Epsilon().real == 0.0

def test_epsilon_dual():
    assert Epsilon().dual == 1.0

def test_epsilon_nilpotent():
    assert (Epsilon() ** 2).real == pytest.approx(0.0)
    assert (Epsilon() ** 2).dual == pytest.approx(0.0)

def test_epsilon_builds_dual():
    x = 3 + 4 * Epsilon()
    assert x.real == pytest.approx(3.0)
    assert x.dual == pytest.approx(4.0)