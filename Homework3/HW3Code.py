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
    P, L, U = s.lu(orig_vector)
    b = matrix
    print('P*b Shape: ', np.shape(P*b))
    print('L Shape: ', np.shape(P*b))
    print('L:', L)
    print('U Shape: ', np.shape(P*b))
    y = s.solve(L, P*b)
    x = s.solve(U, y)
    return x


def main():
    A1 = problem1a()
    print("A1:", A1)
    orig_vector = np.array([pi, 3., 4.]).reshape(3, 1)
    A2 = problem1b(A1, orig_vector)
    print("A2:", A2)
    A3 = problem1c(A1, orig_vector)
    print('A3:', A3)


if __name__ == "__main__":
    main()
