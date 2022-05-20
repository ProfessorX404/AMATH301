from matplotlib import cm
import numpy as np
import scipy.optimize as so
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl


def f(x, y):
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2


def problem1():
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    x = np.linspace(-5, 5, 40)
    y = np.linspace(-5, 5, 40)

    X, Y = np.meshgrid(x, y)
    p = ax.plot_surface(X, Y, f(X, Y), cmap=cm.hot)
    fig.colorbar(p)
    ax.view_init(30, 30)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x, y)")
    plt.title("Himmelblau's Function Surface Plot")
    plt.savefig("HW6Writeup_1a_surface.png")

    plt.clf()
    ax = plt.axes(projection='3d')
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    p = ax.contourf(X, Y, Z,
                    np.logspace(-1, 3, 22),
                    cmap=cm.jet, norm=mpl.colors.LogNorm(
                        vmin=Z.min(),
                        vmax=Z.max()))
    fig.colorbar(p)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x, y)")
    plt.title("Himmelblau's Function Contour Plot")
    plt.savefig("HW6Writeup_1b_contour.png")


problem1()
