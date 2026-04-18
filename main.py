from src.dual import *

a = Dual(1, 5)
b = Dual(0, 1)
c = Dual(2, 3)

jacobian_test = lambda x, y, z: (x**2+y+z, x*y, z+x)
gradient_test = lambda x, y: x**2 + x*y
derivative_test = lambda x: x.exp()

print("Jacobian:")
print(jacobian(jacobian_test, 2, 3, 4))

print("Gradient:")
print(gradient(gradient_test, 3, 4))

print("Derivative:")
print(derivative(derivative_test, 1))