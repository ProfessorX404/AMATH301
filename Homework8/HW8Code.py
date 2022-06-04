import numpy as np
import scipy.optimize as so


def dpdt(t, p):
    # dp/pt, where p(t) is the ratio of dextral
    # (right-handed) snails to the total population.
    return p * (1 - p) * (p - .5)


def forward_euler(odefun, tspan, y0):
    # Forward Euler method
    # Solves the differential equation y' = f(t, y) at the times
    # specified by the vector tspan and with initial condition y0.
    # - odefun is an anonymous function of the form odefun = @(t, v) ...
    # - tspan is a row or column vector
    # - y0 is a number

    dt = tspan[1] - tspan[0]
    # Calculate dt from the t values
    y = np.zeros((len(tspan), 1))
    # Setup our solution column vector
    y[0] = y0
    # Define the initial condition
    for k in range(0, len(y) - 1):
        y[k + 1] = y[k] + dt * odefun(tspan[k], y[k])
        # Forward Euler step
    return y


def backward_euler(odefun, tspan, y0):  # wrong root-finding problem?
    dt = tspan[len(tspan) - 1] - tspan[len(tspan) - 2]
    # Calculate dt from the t values
    y = np.zeros((len(tspan), 1))
    # Setup our solution column vector
    y[0] = y0
    # Define the initial condition
    for k in range(0, len(y) - 1):
        g = lambda x: x - y[k] - dt * odefun(tspan[k + 1], x)
        y[k + 1] = so.fsolve(g, y[k])
    return y


def midpoint(odefun, tspan, y0):
    dt = tspan[1] - tspan[0]
    # Calculate dt from the t values
    y = np.zeros((len(tspan), 1))
    # Setup our solution column vector
    y[0] = y0
    # Define the initial condition
    for k in range(0, len(y) - 1):
        k1 = odefun(tspan[k], y[k])
        k2 = odefun(tspan[k] + (dt / 2), y[k] + (dt / 2) * k1)
        y[k + 1] = y[k] + dt * k2
    return y


def rk4(odefun, tspan, y0):
    dt = tspan[1] - tspan[0]
    # Calculate dt from the t values
    y = np.zeros((len(tspan), 1))
    # Setup our solution column vector
    y[0] = y0
    # Define the initial condition
    for k in range(0, len(y) - 1):
        k1 = dt * odefun(tspan[k], y[k])
        k2 = dt * odefun(tspan[k] + (dt / 2), y + (dt / 2))
        k3 = dt * odefun(tspan[k] + (dt / 2), y + (dt / 2))
        k4 = dt * odefun(tspan[k] + dt, y[k] + k3)
        y[k + 1] = y[k] + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return y


def problem1():
    p0 = .75
    interval = np.arange(0, 16, 1)
    A1 = forward_euler(dpdt, interval, p0)
    A2 = backward_euler(dpdt, interval, p0)
    A3 = midpoint(dpdt, interval, p0)
    A4 = midpoint(dpdt, interval, p0)
    return A1, A2, A3, A4


def SOE_forward_euler(A, tspan, x0):
    dt = tspan[1] - tspan[0]
    # Calculate dt from the t values
    x = np.zeros((len(tspan), 2, 1))
    # x = x.astype(np.object)
    # Setup our solution column vector
    x[0] = x0
    # Define the initial condition
    for k in range(0, len(x) - 1):
        x[k + 1] = x[k] + dt * np.dot(A, x[k])
        # Forward Euler step
    return x


def dr(a, t):
    return a * R(t) + J(t)


def dj(a, t):
    return -R(t) - a * J(t)


def problem2():
    a = 1 / np.sqrt(2)
    A5 = [[a, 1],
          [-1, -a]]
    A5 = np.reshape(A5, (2, 2))
    dt = .01
    interval = np.arange(0, 20 + dt / 2, dt)
    A6 = SOE_forward_euler(A5, interval, [[2], [1]])
    return A5, A6


A1, A2, A3, A4 = problem1()
print('A1:', A1)
print('A2:', A2)
print('A3:', A3)
print('A4:', A4)
A5, A6 = problem2()
print('A5:', A5)
print('A6:', A6)
