import pytest

numpy = pytest.importorskip("numpy")
import numpy as np
import math as M
from dual import Dual

arr = np.array([Dual(1, 1), Dual(2, 1), Dual(3, 1)], dtype=object)

def test_array_preserves_dual():
    x = np.array(Dual(3, 4))
    assert x.dtype == object
    assert x.item().real == pytest.approx(3.0)
    assert x.item().dual == pytest.approx(4.0)

def test_array_priority_mul():
    x = np.float64(2.0) * Dual(3, 1)
    assert isinstance(x, Dual)
    assert x.real == pytest.approx(6.0)
    assert x.dual == pytest.approx(2.0)

def test_array_priority_add():
    x = np.float64(2.0) + Dual(3, 1)
    assert isinstance(x, Dual)
    assert x.real == pytest.approx(5.0)

def test_ufunc_sin():
    x = np.sin(Dual(1.0, 1.0))
    assert x.real == pytest.approx(M.sin(1.0))
    assert x.dual == pytest.approx(M.cos(1.0))

def test_ufunc_arccos():
    x = np.arccos(Dual(0.5, 1.0))
    assert x.real == pytest.approx(M.acos(0.5))

def test_ufunc_tanh():
    x = np.tanh(Dual(1.0, 1.0))
    assert x.real == pytest.approx(M.tanh(1.0))

def test_ufunc_exp():
    x = np.exp(Dual(1.0, 1.0))
    assert x.real == pytest.approx(M.e)
    assert x.dual == pytest.approx(M.e)

def test_ufunc_log():
    x = np.log(Dual(1.0, 1.0))
    assert x.real == pytest.approx(0.0)
    assert x.dual == pytest.approx(1.0)

def test_ufunc_reciprocal():
    x = np.reciprocal(Dual(2.0, 1.0))
    assert x.real == pytest.approx(0.5)
    assert x.dual == pytest.approx(-0.25)

def test_ufunc_add():
    x = np.add(Dual(3, 1), Dual(4, 2))
    assert x.real == pytest.approx(7.0)
    assert x.dual == pytest.approx(3.0)

def test_ufunc_subtract():
    x = np.subtract(Dual(5, 3), Dual(2, 1))
    assert x.real == pytest.approx(3.0)
    assert x.dual == pytest.approx(2.0)

def test_ufunc_multiply():
    x = np.multiply(Dual(3, 1), Dual(4, 2))
    assert x.real == pytest.approx(12.0)
    assert x.dual == pytest.approx(10.0)

def test_ufunc_absolute():
    x = np.absolute(Dual(-3, 4))
    assert x.real == pytest.approx(3.0)
    assert x.dual == pytest.approx(-4.0)

def test_ufunc_equal():
    assert np.equal(Dual(3, 1), Dual(3, 1)) == True
    assert np.equal(Dual(3, 1), Dual(3, 2)) == False

def test_ufunc_not_equal(): assert np.not_equal(Dual(3, 1), Dual(4, 1)) == True

def test_ufunc_less(): assert np.less(Dual(2, 99), Dual(3, 0)) == True

def test_ufunc_greater_equal():
    assert np.greater_equal(Dual(4, 0), Dual(3, 99)) == True
    assert np.greater_equal(Dual(3, 0), Dual(3, 0))  == True
    
def test_isfinite_direct():
    assert M.isfinite(Dual(1.0, 1.0).real) == True
    assert M.isfinite(Dual(float('inf'), 1.0).real) == False

def test_isnan_direct():
    assert M.isnan(Dual(float('nan'), 1.0).real) == True
    assert M.isnan(Dual(1.0, 1.0).real) == False

def test_isinf_direct():
    assert M.isinf(Dual(float('inf'), 1.0).real) == True
    assert M.isinf(Dual(1.0, 1.0).real) == False

def test_ufunc_real(): assert np.real(Dual(3.0, 4.0)) == pytest.approx(3.0)

def test_np_sum():
    result = np.sum(arr)
    assert result.real == pytest.approx(6.0)
    assert result.dual == pytest.approx(3.0)

def test_np_mean():
    result = np.mean(arr)
    assert result.real == pytest.approx(2.0)
    assert result.dual == pytest.approx(1.0)

def test_np_prod():
    result = np.prod(arr)
    assert result.real == pytest.approx(6.0)   # 1*2*3

def test_vectorize_sin():
    f = np.vectorize(lambda x: x.sin(), otypes=[object])
    result = f(arr)
    assert result[0].real == pytest.approx(M.sin(1.0))
    assert result[1].real == pytest.approx(M.sin(2.0))
    assert result[2].real == pytest.approx(M.sin(3.0))

def test_vectorize_preserves_dual():
    f = np.vectorize(lambda x: x.sin(), otypes=[object])
    result = f(arr)
    assert result[0].dual == pytest.approx(M.cos(1.0))
    assert result[1].dual == pytest.approx(M.cos(2.0))
    assert result[2].dual == pytest.approx(M.cos(3.0))