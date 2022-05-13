import scipy.optimize as so
import matplotlib.pyplot as plt
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
    plt.clf()
    fig, ax = plt.subplots(1)
    ax.plot(*co2_data, '-.k', markersize=2, label="Original Data")
    ax.set_xbound(63)
    ax.plot(co2_data[0], [exp_sin(osce[0], d) for d in co2_data[0]],
            markersize=2, color='blue', label="Exponential + Sinusoidal")
    ax.plot(
        co2_data[0],
        [expon(sse[0], d) for d in co2_data[0]],
        color='red', markersize=2, label="Exponential")
    plt.legend()
    ax.set_xlabel("Years since January 1958")
    ax.set_ylabel("Atmosphere CO_2")
    ax.set_title("CO_2 Measurements at Mauna Loa")
    plt.savefig("hw5_writeup_p1a.png")

    actual_jan_22 = 416.71
    actual_2021_avg = 416.2467
    print("Exp prediction:", expon(sse[0], 2022-1958))
    print("Exp+sin prediction:", exp_sin(osce[0], 2022-1958))
    print("Exp error:", abs(actual_jan_22 - expon(sse[0], 2022-1958)))
    print("Exp+sin error:", abs(actual_jan_22 - exp_sin(osce[0], 2022-1958)))
    exp_avg = np.average([expon(sse[0], t)
                          for t in np.linspace(2021-1958, 2022-1958)])
    exp_sin_avg = np.average([exp_sin(osce[0], t)
                              for t in np.linspace(2021-1958, 2022-1958)])
    print("Exp average:", exp_avg)
    print("Exp+sin average:", exp_sin_avg)
    print("Exp error on avg:", abs(actual_2021_avg - exp_avg))
    print("Exp+sin error on avg:", abs(actual_2021_avg - exp_sin_avg))
    return sse, sse_val, mae, maxe, osce, osce_val


def problem2():
    plt.clf()
    data = np.loadtxt('salmon_data.csv', delimiter=',')
    salmon_2021 = 489523
    x = [data[0] for data in data]
    y = [data[1] for data in data]

    p1 = poly.fit(x, y, 1)
    p3 = poly.fit(x, y, 3)
    p5 = poly.fit(x, y, 5)
    print(p1.coef)
    print(p3.coef)
    print(p5.coef)
    err1 = abs(p1(2021) - salmon_2021) / salmon_2021
    err2 = abs(p3(2021) - salmon_2021) / salmon_2021
    err3 = abs(p5(2021) - salmon_2021) / salmon_2021

    fig, ax = plt.subplots(1)
    ax.plot(x, y, '-k.',  markersize=2, label="Original Data")
    ax.set_xbound(lower=1930, upper=2020)
    ax.set_ybound(lower=1e5, upper=1.5e6)
    ax.plot(x, [p1(x) for x in x], markersize=2, color='blue', label="P1")
    ax.plot(x, [p3(x) for x in x], markersize=2, color='red', label="P3")
    ax.plot(x, [p5(x) for x in x], markersize=2, color='magenta', label="P5")
    plt.legend()
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Salmon")
    ax.set_title("Number of Salmon from 1930 to 2020")
    plt.savefig("hw5_writeup_p2a.png")

    print("P1 2050 prediction:", p1(2050))
    print("P3 2050 prediction:", p3(2050))
    print("P5 2050 prediction:", p5(2050))
    return np.reshape(
        p1.coef, (1, 2)), np.reshape(
        p3.coef, (1, 4)), np.reshape(
        p5.coef, (1, 6)), np.reshape(
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
