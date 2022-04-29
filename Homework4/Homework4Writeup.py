import cv2
import numpy as np
import matplotlib.pyplot as plt


def getRankApprox(A, r, return_s=False):
    U, S, V = np.linalg.svd(A, full_matrices=False)  # pdq
    if(return_s):
        return np.matrix(U[:, :r]) * np.diag(S[:r]) * np.matrix(V[:r, :]), S
    return np.matrix(U[:, :r]) * np.diag(S[:r]) * np.matrix(V[:r, :])


def problem1():
    A = np.loadtxt('particle_position.csv', delimiter=',')
    mean = np.mean(A, axis=1)
    A = A - mean[:, None]
    A_rank1, S = getRankApprox(A, 1, return_s=True)
    A_rank2 = getRankApprox(A, 2)
    print(S)
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    ax.scatter3D(A[0], A[1], A[2], label="A", c="black")
    ax.scatter3D(A_rank1[0], A_rank1[1], A_rank1[2], label="A1", c="red")

    ax2.scatter3D(A[0], A[1], A[2], label="A", c="black")
    ax2.scatter3D(A_rank2[0], A_rank2[1], A_rank2[2], label="A2", c="red")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("z")
    ax.legend(loc="best")
    ax2.legend(loc="best")
    ax.set_title("Rank-1 approx.")
    ax2.set_title("Rank-2 approx.")
    fig.suptitle("HW4 Writeup Problem 1a")
    plt.savefig("HW4 Writeup P1.png")


def problem2(numvalues=15, energy_val=.92):
    plt.clf()
    A = cv2.imread('olive.jpg', 0)
    S = getRankApprox(A, 1, True)[1]
    min_r_val = np.size(S)
    for r in range(1, min_r_val):
        if min_r_val > r:
            if sum(S[:r]) / sum(S) > energy_val:
                min_r_val = r
    print(min_r_val)
    fig = plt.figure()

    fig.add_subplot(2, 2, 1)
    plt.imshow(A, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.title("Original Image")

    fig.add_subplot(2, 2, 2)
    plt.imshow(getRankApprox(A, 1), cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.title("Rank-1 Approx.")

    fig.add_subplot(2, 2, 3)
    plt.imshow(getRankApprox(A, 10), cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.title("Rank-10 Approx.")

    fig.add_subplot(2, 2, 4)
    plt.imshow(getRankApprox(A, min_r_val), cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.title("Rank-" + str(min_r_val) + " Approx.")

    plt.savefig("HW4 Writeup P2.png")

    print(np.shape(A)[0], np.shape(A)[1], min_r_val)
    print(np.shape(A)[0] * np.shape(A)[1])
    print(min_r_val * (1 + np.shape(A)[0] + np.shape(A)[1]))


def problem3():
    A = np.loadtxt('Problem3_Image.csv', delimiter=',')
    A_noisy = np.loadtxt('Problem3_Image_Noisy.csv', delimiter=',')
    A_rank2, S = getRankApprox(A_noisy, 2, True)
    rank2_energy = sum(S[:2]) / sum(S)  # part a
    error2_noisy = np.linalg.norm(A_noisy-A_rank2)  # part b) i.
    error2 = np.linalg.norm(A-A_rank2)  # part b) ii.

    print(np.max(A))

    fig, ax = plt.subplots(1)
    ax.plot(S, 'ob')
    ax.set_yscale('log')
    fig.suptitle("Plot of singular values of A_noisy")

    plt.savefig("HW4 Writeup P3a")
    plt.clf()
    fig = plt.figure()

    fig.add_subplot(1, 3, 1)
    plt.imshow(A, cmap='gray')
    plt.axis('off')
    plt.title("True Image")

    fig.add_subplot(1, 3, 2)
    plt.imshow(A_noisy, cmap='gray')
    plt.axis('off')
    plt.title("Noisy Image")

    fig.add_subplot(1, 3, 3)
    plt.imshow(getRankApprox(A_noisy, 2), cmap='gray')
    plt.axis('off')
    plt.title("Rank-2 Approx.")
    plt.tight_layout()
    plt.savefig("HW4 Writeup P3b.png")

    print(rank2_energy, error2_noisy, error2)


problem1()
problem2()
problem3()
