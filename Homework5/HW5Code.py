import scipy.optimize as so
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import polyfit


def exp(x):
    return np.exp(x)


def sin(x):
    return np.sin(x)


def sum_squared_error(args, data):
    [a, r, b] = args
    error = 0
    for i in range(len(data[0])):
        error += (data[1][i] - ((a * exp(r * data[0][i])) + b))**2
    return error


def average_error(args, data):
    [a, r, b] = args
    error = 0
    for i in range(len(data[0])):
        error += abs(data[1][i] - ((a * exp(r * data[0][i])) + b))
    return error


def max_error(args, data):
    [a, r, b] = args
    error = 0
    for i in range(len(data[0])):
        dif = abs(data[1][i] - ((a * exp(r * data[0][i])) + b))
        if dif > error:
            error = dif
    return error


def sse_osc(args, data):
    [a, r, b, c, d, e] = args
    error = 0
    for i in range(len(data[0])):
        error += (data[1][i] - ((a * exp(r * data[0][i])
                                 ) + b + (c*sin(d*(data[0][i] - e)))))**2
    return error


def solve_poly(coefs, x):
    sol = 0
    for i in range(len(coefs)):
        sol += coefs[i] * (x**(i + 1))
    return sol


def problem1():
    data = np.loadtxt("CO2_data.csv", delimiter=",")
    # plt.plot(data, label="data")
    # plt.legend()
    x = []
    y = []
    for data in data:
        x.append(data[0])
        y.append(data[1])
    co2_data = (x, y)
    # plt.plot(*co2_data)
    # plt.legend()
    # plt.savefig("co2_test_plt.png")
    a = 30.
    r = .03
    b = 300
    c = -5
    d = 4
    e = 0
    sse, sse_val = so.fmin(
        sum_squared_error, x0=[a, r, b],
        args=(co2_data, ),
        disp=False, full_output=True)[:2]
    sse = np.reshape(sse, (1, 3))
    mae = np.reshape(so.fmin(average_error, x0=[a, r, b],
                             args=(co2_data, ),
                             disp=False), (1, 3))
    maxe = np.reshape(so.fmin(max_error, x0=[a, r, b],
                              args=(co2_data, ),
                              disp=False), (1, 3))
    osce, osce_val = so.fmin(
        sse_osc, x0=[a, r, b, c, d, e],
        args=(co2_data, ), disp=False, maxiter=2000, full_output=True)[:2]
    osce = np.reshape(osce, (1, 6))

    return sse, sse_val, mae, maxe, osce, osce_val


def problem2():
    salmon_data = np.loadtxt('salmon_data.csv', delimiter=',')
    salmon_2021 = 489523
    x = []
    y = []
    for data in salmon_data:
        x.append(data[0])
        y.append(data[1])
    p1 = polyfit(x, y, 1)
    p3 = polyfit(x, y, 3)
    p5 = polyfit(x, y, 5)
    err1 = abs(np.polyval(p1, 2021) - salmon_2021) / salmon_2021
    err2 = abs(np.polyval(p3, 2021) - salmon_2021) / salmon_2021
    err3 = abs(np.polyval(p5, 2021) - salmon_2021) / salmon_2021
    return np.reshape(p1, (1, 2)), np.reshape(p3, (1, 4)), np.reshape(
        p5, (1, 6)), np.reshape(
        [err1, err2, err3],
        (1, 3))


A1, A2, A3, A4, A5, A6 = problem1()
A7, A8, A9, A10 = problem2()
print("A1:", A1)
print("A2:", A2)
print("A3:", A3)
print("A4:", A4)
print("A5:", A5)
print("A6:", A6)
print("A7:", A7)
print("A8:", A8)
print("A9:", A9)
print("A10:", A10)
