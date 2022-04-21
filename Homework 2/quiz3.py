from scipy.optimize import newton
import numpy as np


def f(x):
    return 3*np.cos(2*x)


def df(x):
    return -6*np.sin(2*x)


def main():
    newton(f, x0=0, fprime=df)


if __name__ == "__main__":
    main()
