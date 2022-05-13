import scipy.optimize as so
import numpy as np
from numpy.polynomial import Polynomial as poly


def exp(x):
    return np.exp(x)


def sin(x):
    return np.sin(x)


def expon(args, t):
    [a, r, b] = args
    return (a * exp(r * t)) + b


def exp_sin(args, t):
    [a, r, b, c, d, e] = args
    return expon([a, r, b], t) + c*sin(d*(t - e))


def sum_squared_error(args, data):
    error = 0
    for i in range(len(data[0])):
        error += (data[1][i] - expon(args, data[0][i]))**2
    return error


def average_error(args, data):
    error = 0
    for i in range(len(data[0])):
        error += abs(data[1][i] - expon(args, data[0][i]))
    return error


def max_error(args, data):
    error = 0
    for i in range(len(data[0])):
        dif = abs(data[1][i] - expon(args, data[0][i]))
        if dif > error:
            error = dif
    return error


def sse_osc(args, data):
    error = 0
    for i in range(len(data[0])):
        error += (data[1][i] - exp_sin(args, data[0][i]))**2
    return error


def problem1():
    data = np.loadtxt("CO2_data.csv", delimiter=",")
    x = [data[0] for data in data]
    y = [data[1] for data in data]

    co2_data = (x, y)
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
    maxe = np.reshape(so.fmin(max_error, x0=([a, r, b]),
                              args=(co2_data, ),
                              disp=False), (1, 3))
    osce, osce_val = so.fmin(
        sse_osc, x0=[*sse[0], c, d, e],
        args=(co2_data, ), disp=False, maxiter=2000, full_output=True)[:2]
    osce = np.reshape(osce, (1, 6))
    return sse, sse_val, mae, maxe, osce, osce_val


def problem2():
    data = np.loadtxt('salmon_data.csv', delimiter=',')
    salmon_2021 = 489523
    x = [data[0] for data in data]
    y = [data[1] for data in data]
    p1 = poly.fit(x, y, 1)
    p3 = poly.fit(x, y, 3)
    p5 = poly.fit(x, y, 5)
    err1 = abs(p1(2021) - salmon_2021) / salmon_2021
    err2 = abs(p3(2021) - salmon_2021) / salmon_2021
    err3 = abs(p5(2021) - salmon_2021) / salmon_2021
    return np.reshape(
        p1.coef, (1, 2)), np.reshape(
        p3.coef, (1, 4)), np.reshape(
        p5.coef, (1, 6)), np.reshape(
        [err1, err2, err3],
        (1, 3))


A1, A2, A3, A4, A5, A6 = problem1()
A7, A8, A9, A10 = problem2()
