import numpy as np
import scipy.integrate as si
import matplotlib.pyplot as plt


def P(x):
    return data[1][x]


def for_dif(f, x, h):
    A = 3
    B = -4
    C = 1
    D = (B + C) * h
    A = A / D
    B = B / D
    C = C / D
    return A * f(x) + B * f(x + h) + C * f(x + (2 * h))


def back_dif(f, x, h):
    A = 3
    B = -4
    C = 1
    D = (-B * h) + (-C * 2 * h)
    A = A / D
    B = B / D
    C = C / D
    return A * f(x) + B * f(x - h) + C * f(x - (2 * h))


def center_dif(f, x, h):
    A = 0
    B = 1
    C = -B
    D = (C - B) * h
    A = A / D
    B = B / D
    C = C / D
    return A * f(x) + B * f(x - h) + C * f(x + h)


def sec_order_deriv(f, x, h):
    A = -1 / h
    B = -A
    return A * f(x) + B * f(x + h)


def sec_deriv(f, x, h):
    A = 1
    B = -2
    C = -B / 2
    D = ((B + 4 * C) * h**2) / 2
    A = A / D
    B = B / D
    C = C / D
    return A * f(x) + B * f(x + (2 * h)) + C * f(x - (2 * h))


def read_plut(file_name=None):
    if file_name is None:
        file_name = "./Plutonium.csv"
    arr = np.loadtxt(file_name, delimiter=",")
    plt.show()
    return arr


def problem1():
    global data
    h = 1
    A1 = for_dif(P, 0, h)
    A2 = back_dif(P, 40, h)
    tot = 0
    for t in data[0]:
        t = int(t)
        # print(t)
        if t == 0:
            tot += (-1 / P(t)) * sec_order_deriv(P, t, h)
            # print(tot)
        elif t == np.max(data[0]):
            tot += (-1 / P(t)) * sec_order_deriv(P, t, -h)
            # print(tot)
        else:
            tot += (-1 / P(t)) * center_dif(P, t, h)
    A3 = tot / data[0].size
    A4 = np.log(2) / A3
    A5 = sec_deriv(P, 27, h)
    return A1, A2, A3, A4, A5


def kitten_weight(x, mu=3.5, sigma=.73):
    return (1 / np.sqrt(2 * np.pi * (sigma**2))) *\
        np.exp(-((x - mu)**2) / (2 * sigma**2))


def problem2():
    A6 = si.quad(kitten_weight, a=4, b=5)[0]

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

    A7 = left
    A8 = right
    A9 = midpoint
    A10 = trapezoid
    A11 = simpson
    return A6, A7, A8, A9, A10, A11


global data
data = read_plut()
A1, A2, A3, A4, A5 = problem1()
print("A1:", A1)
print("A2:", A2)
print("A3:", A3)
print("A4:", A4)
print("A5:", A5)
A6, A7, A8, A9, A10, A11 = problem2()
print("A6:", A6)
print("A7:", A7)
print("A8:", A8)
print("A9:", A9)
print("A10:", A10)
print("A11:", A11)
