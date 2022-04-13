import matplotlib.pyplot as plt
import math
import numpy as np


def fact(num):
    return math.factorial(num)


pi = math.pi


def cos(x, n):
    cosin = 0
    for k in range(0, n+1):
        cosin += ((((-1) ** k)*(x ** (2*k))) / fact(2*k))
    # print(n, cosin)
    return cosin


def main():
    fig, ax = plt.subplots(1)
    x = np.arange(-pi, pi, .1)
    pure_cos, = ax.plot(x, np.cos(x), 'k-', label='Actual cos', linewidth=2)
    n1, = ax.plot(x, [cos(x, 1) for x in x], 'b--', label='n=1', linewidth=2)
    n3, = ax.plot(x, [cos(x, 3) for x in x], 'r-.', label='n=3', linewidth=2)
    n14, = ax.plot(x, [cos(x, 14) for x in x], 'm:', label='n=14', linewidth=2)
    ax.set_title('cos(x) and its Taylor approximations.')
    ax.set_ylabel('Approximations')
    ax.set_xlabel('x-values')
    ax.legend(handles=[pure_cos, n1, n3, n14])
    plt.savefig('HW1_plot.png')


if __name__ == "__main__":
    main()
