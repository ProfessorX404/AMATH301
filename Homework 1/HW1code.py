import math

import numpy

pi = math.pi


def cos(num):
    return math.cos(num)


def tan(num):
    return math.tan(num)


def exp(num):
    return math.exp(num)


def problem_one():
    x = [cos(0), cos(pi / 4), cos(pi / 2), cos((3 * pi) / 4), cos(pi)]
    return x


def problem_two():
    u_start = 3
    u_stop = 4
    u_inc = 6
    u = numpy.linspace(u_start, u_stop, u_inc)
    v = [0.0, .75, 1.5, 2.25, 3.0, 3.75]
    w = []
    for u_val in u:
        w.append(u_val ** 3)
    x = []
    for i in range(len(u)):
        x.append(tan(u[i]) + exp(v[i]))

    return [w, x]


def problem_three():
    row_one = [1, 2, -1, 0]
    row_two = [3, -1, 6, 9]
    A = [row_one, row_two]
    return A


def problem_four():
    z = numpy.arange(-6, 3, .01)
    a = []
    for i in range(len(z)):
        if i % 3 == 0:
            a.append(z[i])
    return a


def problem_five():
    x = 21
    y = 21
    A = []
    for i in range(1, y + 1):
        A.append([1/(i + j - 1) for j in range(1, x + 1)])

    B = A
    row_def = 9
    row_ref = 8
    for x in range(1, x):
        B[row_def][x] = 4 * B[row_ref][x]

    last_row_ct = 9
    first_col_ct = 7
    C = A[(y - last_row_ct):][:first_col_ct]
    return [A, B, C]


def problem_six():
    k = 1
    y1 = 0
    for i in range(k, 100001):
        y1 += .1

    y2 = 0
    for i in range(k, 100000001):
        y2 += .1

    y3 = 0
    for i in range(k, 100000001):
        y3 += .25

    y4 = 0
    for i in range(k, 100000001):
        y4 += .5

    x1 = abs(10000 - y1)
    x2 = abs(y2 - 10000000)
    x3 = abs(25000000-y3)
    x4 = abs(y4 - 50000000)
    print('x1 type: ', type(x1))
    print('x2 type: ', type(x2))
    print('x3 type: ', type(x3))
    print('x4 type: ', type(x4))

    return [x1, x2, x3, x4]


def main():
    A1 = problem_one()
    # print(A1)
    print('Problem 1 complete!')
    A2, A3 = problem_two()
    # print(A2)
    # print(A3)
    print('Problem 2 complete!')
    A4 = problem_three()
    # print(A4)
    print('Problem 3 complete!')
    A5 = problem_four()
    # print(A5)
    print('Problem 4 complete!')
    A6, A7, A8 = problem_five()
    # print(A6)
    # print(A7)
    # print(A8)
    print('Problem 5 complete!')
    A9, A10, A11, A12 = problem_six()
    print(A9)
    print(A10)
    print(A11)
    print(A12)
    print('Problem 6 complete!')


if __name__ == "__main__":
    main()
