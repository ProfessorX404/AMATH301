import numpy as np
from numpy.core.numeric import full
import scipy.optimize as so


def exp(x):
    return np.exp(x)


def max_x(t):
    return -x(t)


def max_dx(t):
    return -dx(t)


def x(t):
    return (10 / 3) * (exp(-(t / 24)) - exp(-(t / 2)))


def dx(t):
    return -(5 / 36) * exp(-t / 2) * (exp((11 * t) / 24) - 12)


def problem1():
    [Aai] = so.fsolve(max_dx, x0=2)
    Aaii = x(Aai)

    Abi = so.fminbound(max_x, x1=0, x2=12)
    Abii = x(Abi)
    return np.reshape([Aai, Aaii], (1, 2)), np.reshape([Abi, Abii], (2, 1))


def himmelblau(x, y):
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2


def f(xy):
    return himmelblau(xy[0], xy[1]).T


def fgrad(xy):
    x = xy[0][0]
    y = xy[1][0]
    # print("x:", x)
    # print("y:", y)
    x_grad = 4 * (x**3) - (42 * x) + (4 * x * y) + (2 * (y**2)) - 14
    y_grad = 4 * (y**3) - (26 * y) + (4 * x * y) + (2 * (x**2)) - 22
    return np.array([[x_grad], [y_grad]])


def problem2ab():
    x0 = np.array([[-3,
                   -2]]).T
    A3 = so.fmin(f, x0=x0, disp=False)
    A3 = np.reshape(A3, (2, 1))
    A4_init = fgrad(A3)
    A4 = np.linalg.norm(A4_init)
    return A3, A4


def problem2cd():
    iter = 0
    p = np.array([[-3], [-2]])  # Choose an initial guess
    # print('p: ', p)
    grad = fgrad(p)
    while((iter <= 5000) & (np.linalg.norm(grad, np.inf) > 1e-8)):
        grad = fgrad(p)  # Find which direction to go
        # print("iter:", str(iter))
        # print("norm:", str(np.linalg.norm(grad, np.inf)))
        # print("grad:", grad)
        phi = lambda t: p - t * grad  # Define the path
        f_of_phi = lambda t: f(phi(t))  # Create a function of
        # "heights along path"
        tmin = so.fminbound(f_of_phi, 0, 1)  # Find time it takes
        # to reach min height
        p = phi(tmin)  # Find the point on the path and update your guess
        iter += 1
    return np.reshape(p, (2, 1)), iter


A1, A2 = problem1()
print("A1:", A1)
print("A2:", A2)
A3, A4 = problem2ab()
print("A3:", A3)
print("A4:", A4)
A5, A6 = problem2cd()
print("A5:", A5)
print("A6:", A6)
