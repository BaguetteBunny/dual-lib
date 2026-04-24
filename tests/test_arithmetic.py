import pytest
from dual import Dual

def test_add_dual_dual():
    x = Dual(3, 1) + Dual(4, 2)
    assert x.real == pytest.approx(7.0)
    assert x.dual == pytest.approx(3.0)

def test_add_dual_scalar():
    x = Dual(3, 1) + 2
    assert x.real == pytest.approx(5.0)
    assert x.dual == pytest.approx(1.0)

def test_radd():
    x = 2 + Dual(3, 1)
    assert x.real == pytest.approx(5.0)
    assert x.dual == pytest.approx(1.0)

def test_sub_dual_dual():
    x = Dual(5, 3) - Dual(2, 1)
    assert x.real == pytest.approx(3.0)
    assert x.dual == pytest.approx(2.0)

def test_sub_dual_scalar():
    x = Dual(5, 3) - 2
    assert x.real == pytest.approx(3.0)
    assert x.dual == pytest.approx(3.0)

def test_rsub():
    x = 5 - Dual(2, 1)
    assert x.real == pytest.approx(3.0)
    assert x.dual == pytest.approx(-1.0)

def test_mul_dual_dual():
    # (3+1ε)(4+2ε) = 12 + (3·2 + 1·4)ε = 12 + 10ε
    x = Dual(3, 1) * Dual(4, 2)
    assert x.real == pytest.approx(12.0)
    assert x.dual == pytest.approx(10.0)

def test_mul_dual_scalar():
    x = Dual(3, 2) * 4
    assert x.real == pytest.approx(12.0)
    assert x.dual == pytest.approx(8.0)

def test_rmul():
    x = 4 * Dual(3, 2)
    assert x.real == pytest.approx(12.0)
    assert x.dual == pytest.approx(8.0)

def test_truediv_dual_dual():
    x = Dual(6, 2) / Dual(3, 1)
    assert x.real == pytest.approx(2.0)
    assert x.dual == pytest.approx((2*3 - 6*1) / 3**2)

def test_truediv_dual_scalar():
    x = Dual(6, 2) / 3
    assert x.real == pytest.approx(2.0)
    assert x.dual == pytest.approx(2/3)

def test_rtruediv():
    x = 6 / Dual(3, 1)
    assert x.real == pytest.approx(2.0)
    assert x.dual == pytest.approx(-6 / 3**2)

def test_pow_scalar():
    # d/dx x³ at x=2 → 3·2² = 12
    x = Dual(2, 1) ** 3
    assert x.real == pytest.approx(8.0)
    assert x.dual == pytest.approx(12.0)

def test_pow_zero():
    x = Dual(2, 1) ** 0
    assert x.real == pytest.approx(1.0)
    assert x.dual == pytest.approx(0.0)

def test_pow_eps_zero():
    x = Dual(0, 1) ** 0
    assert x.real == pytest.approx(1.0)
    assert x.dual == pytest.approx(0.0)

def test_pow_dual():
    # X^Y = a^c + (bc·a^(c-1) + d·a^c·ln(a))ε
    x = Dual(2, 1) ** Dual(3, 1)
    import math as M
    assert x.real == pytest.approx(8.0)
    assert x.dual == pytest.approx(1*3*2**2 + 1*8*M.log(2))

def test_rpow():
    # k^X, dual = b·k^a·ln(k)
    import math as M
    x = 2 ** Dual(3, 1)
    assert x.real == pytest.approx(8.0)
    assert x.dual == pytest.approx(8 * M.log(2))

def test_floordiv_dual_dual():
    x = Dual(7, 3) // Dual(2, 1)
    assert x.real == pytest.approx(3.0)
    assert x.dual == pytest.approx(0.0)

def test_floordiv_dual_scalar():
    x = Dual(7, 3) // 2
    assert x.real == pytest.approx(3.0)
    assert x.dual == pytest.approx(0.0)

def test_rfloordiv():
    x = 7 // Dual(2, 3)
    assert x.real == pytest.approx(3.0)
    assert x.dual == pytest.approx(0.0)

def test_mod_dual_scalar():
    x = Dual(7, 3) % 2
    assert x.real == pytest.approx(1.0)
    assert x.dual == pytest.approx(3.0)  # dual survives unchanged

def test_mod_dual_dual():
    x = Dual(7, 3) % Dual(2, 1)
    assert x.real == pytest.approx(1.0)
    assert x.dual == pytest.approx(3 - 1 * (7 // 2))

def test_rmod():
    # k % X, dual = -floor(k/a) * b   NOT -(k/a)*b
    x = 7 % Dual(2, 3)
    assert x.real == pytest.approx(1.0)
    assert x.dual == pytest.approx(-(7 // 2) * 3)

def test_neg():
    x = -Dual(3, 4)
    assert x.real == pytest.approx(-3.0)
    assert x.dual == pytest.approx(-4.0)

def test_pos():
    x = +Dual(3, 4)
    assert x.real == pytest.approx(3.0)
    assert x.dual == pytest.approx(4.0)

def test_abs_positive():
    x = abs(Dual(3, 4))
    assert x.real == pytest.approx(3.0)
    assert x.dual == pytest.approx(4.0)

def test_abs_negative():
    x = abs(Dual(-3, 4))
    assert x.real == pytest.approx(3.0)
    assert x.dual == pytest.approx(-4.0)

def test_eq():
    assert Dual(3, 4) == Dual(3, 4)
    assert Dual(3, 4) != Dual(3, 5)

def test_lt():
    assert Dual(2, 99) < Dual(3, 0)

def test_bool_nonzero():
    assert bool(Dual(1, 0)) is True

def test_bool_zero():
    assert bool(Dual(0, 1)) is False

def test_float():
    assert float(Dual(3, 4)) == 3.0

def test_int():
    assert int(Dual(3.9, 4)) == 3

def test_divmod():
    q, r = divmod(Dual(7, 3), Dual(2, 1))
    assert q.real == pytest.approx(3.0)
    assert q.dual == pytest.approx(0.0)
    assert r.real == pytest.approx(1.0)