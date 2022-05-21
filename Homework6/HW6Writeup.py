import time
from matplotlib import cm
import numpy as np
import scipy.optimize as so
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl


def problem2a():
    x0 = np.array([[-3,
                   -2]]).T
    A3 = [0, 0, 0]
    A3[0], A3[1] = so.fmin(f_single, x0=x0, disp=False, full_output=False)
    A3[2] = f(A3[0], A3[1])
    A3 = np.reshape(A3, (3, 1))
    return A3


def problem2d():
    iter = -1
    p = np.array([[-3], [-2]])  # Choose an initial guess
    # print('p: ', p)
    grad = fgrad(p)
    while((iter < 5000) & (np.linalg.norm(grad, np.inf) > 1e-8)):
        grad = fgrad(p)  # Find which direction to go
        # print("iter:", str(iter))
        # print("norm:", str(np.linalg.norm(grad, np.inf)))
        # print("grad:", grad)
        phi = lambda t: p - t * grad  # Define the path
        f_of_phi = lambda t: f_single(phi(t))  # Create a function of
        # "heights along path"
        tmin = so.fminbound(f_of_phi, 0, 1)  # Find time it takes
        # to reach min height
        p = phi(tmin)  # Find the point on the path and update your guess
        iter += 1
    p_ans = [p[0], p[1], f(p[0], p[1])]
    return np.reshape(p_ans, (3, 1))


def problem2d_altered(tstep, max_iter=12000):
    init_time = time.time()
    iter = -1
    p = np.array([[2], [3]])  # Choose an initial guess
    inf_tol = 1e-9
    grad = fgrad(p)
    while((iter < max_iter) & (np.linalg.norm(grad, np.inf) > inf_tol)):
        grad = fgrad(p)  # Find which direction to go
        # print("iter:", str(iter))
        # print("norm:", str(np.linalg.norm(grad, np.inf)))
        # print("grad:", grad)
        p = p - tstep * grad
        iter += 1
    total_time = time.time() - init_time
    return iter, total_time, (iter != max_iter)


def get_iters(tstep):
    return problem2d_altered(tstep)[0]


def fgrad(xy):
    x = xy[0][0]
    y = xy[1][0]
    # print("x:", x)
    # print("y:", y)
    x_grad = 4 * (x**3) - (42 * x) + (4 * x * y) + (2 * (y**2)) - 14
    y_grad = 4 * (y**3) - (26 * y) + (4 * x * y) + (2 * (x**2)) - 22
    return np.array([[x_grad], [y_grad]])


def f(x, y):
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2


def f_single(xy):
    x = xy[0]
    y = xy[1]
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
    guesses = np.array([[-4, 4], [-4, -4], [3, 2], [4, -2]])
    x_min = np.empty(len(guesses))
    y_min = np.empty(len(guesses))
    z_min = np.empty(len(guesses))
    for guess_num in range(len(guesses)):
        x_min[guess_num], y_min[guess_num] = so.fmin(
            f_single, x0=guesses[guess_num], disp=False)
        z_min[guess_num] = f(x_min[guess_num], y_min[guess_num])
    # the star located near -4, 4 can be hard to see, but it is there
    # sometimes the renderer will either miss it or put it underneath the contour plot
    ax.scatter(x_min, y_min, z_min, c='yellow', marker='*', s=15)
    fmin_ans = problem2a()
    ax.scatter(*fmin_ans, c="red", marker="o", facecolors='none',
               edgecolors='r', s=30, label="Coding 2a minimum")
    grad_desc_ans = problem2d()
    ax.scatter(
        *grad_desc_ans, c="green", marker="s", facecolors='none',
        edgecolors='g', s=30, label="Grad Descent")
    plt.legend()
    ax.view_init(30, 220)
    plt.savefig("HW6Writeup_1b_contour.png")


def problem2():
    init_time = time.time()
    t0_full = so.fmin(f_single, x0=[2, 3], xtol=1e-9,
                      maxiter=12000, disp=False, full_output=True)
    final_time = time.time() - init_time
    t0 = [t0_full[2], final_time, (t0_full[2] == 12000)]

    t1 = problem2d_altered(.01)
    t2 = problem2d_altered(.02)
    t3 = problem2d_altered(.025)
    print(
        "fmin iters: " + str(t0[0]) +
        '; fmin time: ' + str(t0[1]) +
        '; fmin converged?: ' + str(t0[2]))
    print(
        ".01  iters: " + str(t1[0]) +
        '; .01  time: ' + str(t1[1]) +
        '; .01  converged?: ' + str(t1[2]))
    print(
        ".02  iters: " + str(t2[0]) +
        '; .02  time: ' + str(t2[1]) +
        '; .02  converged?: ' + str(t2[2]))
    print(".025 iters: " + str(t3[0]), '; .025 time:' +
          str(t3[1]) + '; .025 converged?: ' + str(t3[2]))


def problem2_opt():
    return so.fmin(get_iters, x0=.02, full_output=True, disp=False)


problem1()
problem2()
print(problem2_opt())
