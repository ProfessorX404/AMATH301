import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve

x1_exact = 0.0
x2_exact = 0.0


bisect_x1_midpoints = dict()
bisect_x2_midpoints = dict()
bisect_x1_error = dict()
bisect_x2_error = dict()
bisect_iters = 0

newtons_x1_guesses = dict()
newtons_x2_guesses = dict()
newtons_x1_error = dict()
newtons_x2_error = dict()


def exp(num):
    return np.exp(num)


def deriv(x):
    return (exp(6-(3*x*(1+exp(3*(1-x)))))*x *
            ((9*exp(3*(1-x))*x)-(3*(1+exp(3*(1-x))))
                + exp(6-(3*x*(1+exp(3*(1-x)))))))-1


def func(x):
    return (x*exp(6-(3*x*(1+exp(3*(1-x))))))-x


def calc_func(x):
    y = np.zeros(len(x))
    for i in range(len(x)):
        y[i] = func(x[i])
    return y


def bisect(f, a, b, results=dict(), iters=0, xtol=1e-12, max_iters=100000):
    # approximates a root, R, of f bounded
    # by a and b to within tolerance
    # | f(m) | < tol with m the midpoint
    # between a and b Recursive implementation

    # check if a and b bound a root
    if np.sign(f(a)) == np.sign(f(b)):
        raise Exception(
            "The scalars a and b do not bound a root")

    # get midpoint
    m = (a + b)/2
    results[iters] = m
    if np.abs(f(m)) < xtol:
        # stopping condition, report m as root
        return results
    elif np.sign(f(a)) == np.sign(f(m)):
        # case where m is an improvement on a.
        # Make recursive call with a = m
        iters += 1
        return bisect(f, m, b, results, iters)
    elif np.sign(f(b)) == np.sign(f(m)):
        # case where m is an improvement on b.
        # Make recursive call with b = m
        iters += 1
        return bisect(f, a, m, results, iters)


def newton(f, dx, x0, xtol=1e-12, max_iters=100000):
    iters = 0
    results = dict()
    results[0] = x0
    while iters <= max_iters:
        x1 = results[iters] - f(results[iters]) / dx(results[iters])
        t = abs(x1 - results[iters])
        if t < xtol:
            return results
        iters += 1
        results[iters] = x1
    return None


def problem1b():
    x = np.arange(0, 5, .01)
    fig, ax = plt.subplots(1)
    ax.plot(x, calc_func(x))
    # z = np.zeros(len(x))
    # for i in range(len(x)):
    #     z[i] = fsolve(func, x0=x[i], xtol=1e-12)
    # plt.plot(x, z)
    ax.set_ylabel("f(x)")
    ax.set_xlabel("Caterpillar Population (*1000)")
    ax.set_title("Question 1 part b.")
    plt.savefig("./Homework 2/HW2-1b.png")


def problem1c():
    x0 = .02
    z = fsolve(func, x0=x0, xtol=1e-12)
    global x1_exact
    x1_exact = z
    print("fsolve root for x0 =", x0, "is", str(z)[1:-1])


def problem1d():
    x0 = 2
    z = fsolve(func, x0=x0, xtol=1e-12)
    global x2_exact
    x2_exact = z
    print("fsolve root for x0 =", x0, "is", str(z)[1:-1])


def problem1f():
    x0_1 = [.02, .3]
    x0_2 = [1.7, 2]
    bisect_x1_midpoints = bisect(func, x0_1[0], x0_1[1])
    bisect_x2_midpoints = bisect(func, x0_2[0], x0_2[1])
    x1 = bisect_x1_midpoints[list(bisect_x1_midpoints.keys())[-1]]
    x2 = bisect_x2_midpoints[list(bisect_x2_midpoints.keys())[-1]]
    for _iter in bisect_x1_midpoints.keys():
        bisect_x1_error[_iter] = abs(x1_exact - bisect_x1_midpoints[_iter])
    for _iter in bisect_x2_midpoints.keys():
        bisect_x2_error[_iter] = abs(x2_exact - bisect_x2_midpoints[_iter])
    print("Bisect root for", x0_1[0], "< x0 <", x0_1[1], "is", x1)
    print("Bisect root for", x0_2[0], "< x0 <", x0_2[1], "is", x2)


def problem1h():
    x0_1 = .3
    x0_2 = 2
    newtons_x1_guesses = newton(func, deriv, x0_1)
    newtons_x2_guesses = newton(func, deriv, x0_2)
    x1 = newtons_x1_guesses[list(newtons_x1_guesses.keys())[-1]]
    x2 = newtons_x2_guesses[list(newtons_x2_guesses.keys())[-1]]
    for _iter in newtons_x1_guesses.keys():
        newtons_x1_error[_iter] = abs(x1_exact - newtons_x1_guesses[_iter])
    for _iter in newtons_x2_guesses.keys():
        newtons_x2_error[_iter] = abs(x2_exact - newtons_x2_guesses[_iter])
    print("Newton's root for x0 =", x0_1, "is", x1)
    print("Newton's root for x0 =", x0_2, "is", x2)


def problem1i():
    plt.clf()
    fig, [ax1, ax2] = plt.subplots(2)
    print(newtons_x1_error)
    print(bisect_x1_error)
    ax1.plot(
        newtons_x1_error.keys(),
        newtons_x1_error.values(),
        'g*', label="Newton's")
    ax1.plot(
        bisect_x1_error.keys(),
        bisect_x1_error.values(),
        'k+', label='Bisect')
    ax2.plot(
        newtons_x2_error.keys(),
        newtons_x2_error.values(),
        'g*', label="Newton's")
    ax2.plot(
        bisect_x2_error.keys(),
        bisect_x2_error.values(),
        'k+', label='Bisect')
    ax1.set_xlabel('Number of iterations')
    ax1.set_ylabel('Error')
    ax2.set_xlabel('Number of iterations')
    ax2.set_ylabel('Error')
    ax1.set_title('Error finding x1')
    ax2.set_title('Error finding x2')
    ax1.legend()
    ax2.legend()
    plt.tight_layout()
    plt.savefig('./Homework 2/HW2-1m.png')
    return ax1, ax2


def problem2(ax1, ax2):
    ax1.set_xscale('log')
    ax2.set_xscale('log')
    ax1.set_title('Error finding x1 (logarithmic)')
    ax2.set_title('Error finding x2 (logarithmic)')
    plt.savefig('./Homework 2/HW2-2.png')


def main():
    problem1b()
    problem1c()
    problem1d()
    problem1f()
    problem1h()
    ax1, ax2 = problem1i()
    problem2(ax1, ax2)


if __name__ == "__main__":
    main()


# Source for bisection method:
# pythonnumericalmethods.berkeley.edu/notebooks/chapter19.03-Bisection-Method.html
# Source for Newton's method:
# personal.math.ubc.ca/~pwalls/math-python/roots-optimization/newton/
# I did not use scipy for bisection or Newton's since it does
# not allow the client to track individual iterattions.
# I slightly altered the alternative methods to do so.
