import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


def problem1():
    A = np.loadtxt('Homework4\particle_position.csv', delimiter=',')
    mean = np.mean(A, axis=1)
    A = A - mean[:, None]
    U, S, V = np.linalg.svd(A, full_matrices=False)
    u1 = U[:, 0]
    v1 = V[0]
    u2 = U[:, 1]
    v2 = V[1]
    print(np.shape(S))
    A_rank1 = np.outer(u1, v1) * S[0]
    A_rank2 = np.outer(u2, v2) * S[1]
    error1 = np.linalg.norm(A-A_rank1)
    error2 = np.linalg.norm(A-A_rank2)

    # plt.xkcd()
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    ax.scatter3D(A[0], A[1], A[2], label="A", c="black")
    ax.scatter3D(A_rank1[0], A_rank1[1], A_rank1[2], label="A1", c="red")
    ax2.scatter3D(A[0], A[1], A[2], label="A", c="black")
    ax2.scatter3D(A_rank2[0], A_rank2[1], A_rank2[2], label="A2", c="red")
    ax.legend(loc="best")
    ax2.legend(loc="best")
    plt.show()
    return S, error1, error2


def main():
    A1, A2, A3 = problem1()


if __name__ == "__main__":
    main()
