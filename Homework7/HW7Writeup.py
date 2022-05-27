import numpy as np
import scipy.integrate as si
import matplotlib.pyplot as plt


def kitten_weight(x, mu=3.5, sigma=.73):
    return (1 / np.sqrt(2 * np.pi * (sigma**2))) *\
        np.exp(-((x - mu)**2) / (2 * sigma**2))


def problem1():
    exact = si.quad(kitten_weight, a=4, b=5)[0]
    h_list = [2**(-(i + 1)) for i in range(16)]
    left = np.ndarray((16, 1))
    for i in range(len(left)):
        h = h_list[i]
        integral = 0
        for j in np.linspace(4 + h, 5, int((1 - h) / h)):
            integral += kitten_weight(j) * h
        left[i] = integral

    right = np.ndarray((16, 1))
    for i in range(len(left)):
        h = h_list[i]
        integral = 0
        for j in np.linspace(4, 5 - h, int((1 - h) / h)):
            integral += kitten_weight(j) * h
        right[i] = integral

    midpoint = np.ndarray((16, 1))
    for i in range(len(left)):
        h = h_list[i]
        integral = 0
        for j in np.linspace(4 + (h / 2), 5 - (h / 2), int((1 - h) / h)):
            integral += kitten_weight(j) * h
        midpoint[i] = integral

    trapezoid = np.ndarray((16, 1))
    for i in range(len(left)):
        h = h_list[i]
        integral = 0
        for j in np.linspace(4, 5 - h, int((1 - h) / h)):
            integral += h * ((kitten_weight(j) + kitten_weight(j + h)) / 2)
        trapezoid[i] = integral

    simpson = np.ndarray((16, 1))
    for i in range(len(left)):
        h = h_list[i]
        integral = 0
        for j in np.linspace(4, 5 - h, int((1 - h) / h)):
            integral += (h / 6) * (kitten_weight(j) +
                                   (4 * kitten_weight((j + (j + h)) / 2)) +
                                   kitten_weight(j + h))
        simpson[i] = integral

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
    plt.plot(
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


problem1()
