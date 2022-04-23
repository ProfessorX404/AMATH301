import numpy as np
import scipy.linalg as s

pi = np.pi


def cos(theta):
    # RADIANS
    return np.cos(theta)


def sin(theta):  # RADIANS
    return np.sin(theta)


def create_vector_rotation_matrix(theta):
    return np.array([[cos(theta), -sin(theta), 0],
                     [sin(theta), cos(theta), 0], [0, 0, 1.]])


def problem1a():
    return create_vector_rotation_matrix((2 * pi) / 5)


def problem1b(matrix, orig_vector):
    return s.solve(a=matrix, b=orig_vector)


def problem1c(matrix, orig_vector):
    P, L, U = s.lu(matrix)
    b = orig_vector
    LUx = s.solve(P, b)
    Ux = s.solve(L, LUx)
    x = s.solve(U, Ux)
    return x


def row(
        f1=0, f2=0, f3=0, f4=0, f5=0, f6=0, f7=0, f8=0, f9=0, f10=0, f11=0,
        f12=0, f13=0, f14=0, f15=0, f16=0, f17=0, f18=0, f19=0):
    template = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10,
                f11, f12, f13, f14, f15, f16, f17, f18, f19]
    row = np.array(template)
    return row


def problem2a():
    one = 1/np.sqrt(17)
    four = 4/np.sqrt(17)
    return np.array([
        row(f1=-one, f2=1, f12=one),  # Eq. 1
        row(f1=-four, f12=-four),  # Eq. 2
        row(f2=-1, f3=1, f13=-one, f14=one),  # Eq. 3
        row(f13=-four, f14=-four),  # Eq. 4
        row(f3=-1, f4=1, f15=-one, f16=one),  # Eq. 5
        row(f15=-four, f16=-four),  # Eq. 6
        row(f4=-1, f5=1, f17=-one, f18=one),  # Eq. 7
        row(f17=-four, f18=-four),  # Eq. 8
        row(f5=-1, f6=one, f19=-one),  # Eq. 9
        row(f6=-four, f19=-four),  # Eq. 10
        row(f6=-one, f7=-1),  # Eq. 11
        row(f7=1, f8=-1, f18=-one, f19=one),  # Eq. 12
        row(f18=four, f19=four),  # Eq. 13, = W8
        row(f8=1, f9=-1, f16=-one, f17=one),  # Eq. 14
        row(f16=four, f17=four),  # Eq. 15
        row(f9=1, f10=-1, f14=-one, f15=one),  # Eq. 16
        row(f14=four, f15=four),  # Eq. 17, = W10
        row(f10=1, f11=-1, f12=-one, f13=one),  # Eq. 18
        row(f12=four, f13=four),  # Eq. 19, = W11
    ])


def problem2d(A):
    b = np.zeros((19, 1))

    W8 = 10000
    W9 = 9700
    W10 = 12000
    W11 = 21500
    b[12] = W8
    b[14] = W9
    b[16] = W10
    b[18] = W11
    return s.solve(A, b), b


def problem2e(A):
    return s.lu(A)


def problem2f(x):
    return np.amax(np.abs(x))


def problem2g(A, x, b, max_weight=42000, weight_inc=5):
    forces = x
    while problem2f(forces) <= max_weight:
        forces = s.solve(A, b)
        b[16] += weight_inc
    x = np.abs(x)
    x = x[x > 42000]
    x = np.amin(x)
    return b[16], x


def main():
    global A1, A2, A3, A4, A5, A6, A7, A8, A9
    A1 = problem1a()
    print("A1 =", A1)
    orig_vector = np.array([pi, 3., 4.]).reshape(3, 1)
    A2 = problem1b(A1, orig_vector)
    print("A2 =", A2)
    A3 = problem1c(A1, orig_vector)
    print('A3 =', A3)
    # print("Problem 1 complete!")
    A4 = problem2a()
    A5 = problem2d(A4)[0]
    print('A4 =', A4)
    print('A5 =', A5)
    print('norm(A):', np.linalg.norm(A4), '(should be 1.9760)')
    print('sum(sum(A)):', sum(sum(A4)), '(should be -3.1828)')
    P, L, U = problem2e(A4)
    A6 = L
    print('A6 =', A6)
    A7 = problem2f(A5)
    print('A7 =', A7)
    A8, A9 = problem2g(A4, A5, problem2d(A4)[1])
    A9 += 1
    print('A8 =', A8)
    print('A9 =', A9)


A1 = problem1a()
orig_vector = np.array([pi, 3., 4.]).reshape(3, 1)
A2 = problem1b(A1, orig_vector)
A3 = problem1c(A1, orig_vector)
A4 = problem2a()
A5 = problem2d(A4)[0]
P, L, U = problem2e(A4)
A6 = L
A7 = problem2f(A5)
A8, A9 = problem2g(A4, A5, problem2d(A4)[1])
A9 += 1

if __name__ == "__main__":
    main()
