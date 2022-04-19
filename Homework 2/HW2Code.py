import numpy as np
import scipy.optimize as s


def bisect(f, a, b, xtol=1e-10, maxiter=1000, full_output=True):
    return s.bisect(f, a, b, xtol=xtol, maxiter=maxiter,
                    full_output=full_output)


def sqrt(x):
    return np.sqrt(x)


def problem1():
    return bisect(lambda x: ((x/(1-x))*sqrt(5/(2+x)))-.06, -1.5, .7)


def problem2a():
    return s.fsolve(lambda x: (2*(x ** 3)) - (3*(x**2)) - 9, x0=2.5)


def problem2b(exact):
    result = 2.5
    for i in range(0, 101):
        result -= (((2 * (result ** 3)) - (3 * (result ** 2)) - 9) /
                   ((6 * (result - 1)) * result))

    return abs(exact - result)


def problem2c(xtol=1e-12):
    iters = 0
    result = 2.5
    for i in range(iters, iters+101):
        if abs((2*(result ** 3)) - (3*(result**2)) - 9) < xtol:
            return iters
        else:
            result -= (((2 * (result ** 3)) - (3 * (result ** 2)) - 9) /
                       ((6 * (result - 1)) * result))
            iters += 1


def main():
    A1, P1_results = problem1()
    A2 = P1_results.iterations
    print('A1:', A1)
    print('A2:', A2)
    A3 = problem2a()
    A4 = problem2b(A3)
    A5 = problem2c()
    print('A3:', str(A3)[1:-1])
    print('A4:', str(A4)[1:-1])
    print('A5:', A5)


if __name__ == "__main__":
    main()
