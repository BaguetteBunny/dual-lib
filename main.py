from dual import *


a = Dual(1, 5)
b = Dual(0, 1)
c = Dual(2, 3)

f = lambda x, y: (x**2+y, x*y)
print(jacobian(f, 3, 4))