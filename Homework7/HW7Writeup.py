import numpy as np
import scipy.integrate as si
import matplotlib.pyplot as plt


def kitten_weight(x, mu=3.5, sigma=.73):
    return (1 / np.sqrt(2 * np.pi * (sigma**2))) *\
        np.exp(-((x - mu)**2) / (2 * sigma**2))


def problem1():
    exact = si.quad(kitten_weight, a=4, b=5)[0]

    h_list = [2**(-(i + 1)) for i in range(16)]
    step_count = [int((1 - h) / h) for h in h_list]

    left = np.ndarray((16, 1))
    right = np.ndarray((16, 1))
    midpoint = np.ndarray((16, 1))
    trapezoid = np.ndarray((16, 1))
    simpson = np.ndarray((16, 1))

    for i in range(len(left)):
        h = h_list[i]
        sc = step_count[i]
        for j in np.linspace(4 + h, 5, sc):
            left[i] += kitten_weight(j) * h

    for i in range(len(right)):
        h = h_list[i]
        sc = step_count[i]
        for j in np.linspace(4, 5 - h, sc):
            right[i] += kitten_weight(j) * h

    for i in range(len(midpoint)):
        h = h_list[i]
        sc = step_count[i]
        for j in np.linspace(4 + (h / 2), 5 - (h / 2), sc):
            midpoint[i] += kitten_weight(j) * h

    for i in range(len(trapezoid)):
        h = h_list[i]
        sc = step_count[i]
        for j in np.linspace(4, 5 - h, sc):
            trapezoid[i] += h * ((kitten_weight(j) + kitten_weight(j + h)) / 2)

    for i in range(len(simpson)):
        h = h_list[i]
        sc = step_count[i]
        for j in np.linspace(4, 5 - h, sc):
            simpson[i] += (h / 6) * (kitten_weight(j) +
                                     (4 * kitten_weight((j + (j + h)) / 2)) +
                                     kitten_weight(j + h))

    left_error = np.abs(exact - left)
    right_error = np.abs(exact - right)
    mid_error = np.abs(exact - midpoint)
    trap_error = np.abs(exact - trapezoid)
    simpson_error = np.abs(exact - simpson)

    c1 = .25
    c2 = 2.5
    o_h = [c1 * h for h in h_list]
    o_h2 = [c2 * (h**2) for h in h_list]
    plt.plot(h_list, left_error, label="Left", marker=".", linewidth=2)
    plt.plot(h_list, right_error, label="Right", marker="x", linewidth=2)
    plt.plot(h_list, mid_error, label="Midpoint", marker="*", linewidth=2)
    plt.plot(h_list, trap_error, label="Trapezoid", marker="o", linewidth=2)
    plt.plot(h_list, simpson_error, label="Simpson", marker="s", linewidth=2)
    plt.plot(h_list, o_h, label="O(h)", marker=">", linewidth=2)
    plt.plot(h_list, o_h2, label="O(h^2)", marker="<", linewidth=2)
    plt.plot(h_list,
             [1e-16 for i in range(16)],
             label="machine precision", marker='p', linewidth=2)
    plt.legend()
    plt.xlabel("step value")
    plt.ylabel("Error")
    plt.xscale('log')
    plt.yscale('log')
    plt.title(
        "Error of different numeric integrations methods\n\
vs. the step value used when computing them")
    plt.savefig("HW7_plot.png")


def sqrt_in_el_f(t, a, b):
    return np.sqrt((a**2) * (np.cos(t)**2) + (b**2) * (np.sin(t)**2))


def extra_credit():
    plt.clf()
    a = 1.5
    b = .3
    exact = si.quad(sqrt_in_el_f, a=0, b=np.pi / 2, args=(a, b))[0]

    points = [2**(i + 1) for i in range(14)]
    delta_list = [np.pi / point for point in points]
    left = np.ndarray((14, 1))
    right = np.ndarray((14, 1))
    trapezoid = np.ndarray((14, 1))

    for i in range(len(left)):
        sc = points[i]
        delta = np.pi / sc
        for j in np.linspace(0 + delta, np.pi / 2, sc):
            left[i] += sqrt_in_el_f(j, a, b) * delta
        left[i] *= 4

    for i in range(len(right)):
        sc = points[i]
        delta = np.pi / sc
        for j in np.linspace(0, (np.pi / 2) - delta, sc):
            right[i] += sqrt_in_el_f(j, a, b) * delta
        right[i] *= 4

    for i in range(len(trapezoid)):
        sc = points[i]
        delta = np.pi / sc
        for j in np.linspace(0, (np.pi / 2) - delta, sc):
            trapezoid[i] += delta * ((sqrt_in_el_f(j, a, b) +
                                      sqrt_in_el_f(j + delta, a, b)) / 2)
        trapezoid[i] *= 4

    left_error = np.abs(exact - left)
    right_error = np.abs(exact - right)
    trap_error = np.abs(exact - trapezoid)

    c1 = .25
    c2 = 2.5
    o_h = [c1 * h for h in delta_list]
    o_h2 = [c2 * (h**2) for h in delta_list]
    plt.plot(points, left_error, label="Left", marker=".", linewidth=2)
    plt.plot(points, right_error, label="Right", marker="x", linewidth=2)
    plt.plot(points, trap_error, label="Trapezoid", marker="o", linewidth=2)
    plt.plot(points, o_h, label="O(h)", marker=">", linewidth=2)
    plt.plot(points, o_h2, label="O(h^2)", marker="<", linewidth=2)
    plt.plot(points,
             [1e-16 for i in range(14)],
             label="machine precision", marker='p', linewidth=2)
    plt.legend()
    plt.xlabel("Number of points")
    plt.ylabel("Error")
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.title(
        "Error of different numeric integrations methods\n\
vs. the step count used when computing them")
    plt.savefig("hw7_extra_credit.png")


problem1()
extra_credit()
