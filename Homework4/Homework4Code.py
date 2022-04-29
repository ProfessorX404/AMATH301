import cv2
import numpy as np


def getRankApprox(A, r, return_s=False):
    U, S, V = np.linalg.svd(A, full_matrices=False)  # pdq
    if(return_s):
        return np.matrix(U[:, :r - 1]) * np.diag(S[:r - 1]) * np.matrix(V[:r - 1, :]), S
    return np.matrix(U[:, :r - 1]) * np.diag(S[:r - 1]) * np.matrix(V[:r - 1, :])


def problem1():
    A = np.loadtxt('particle_position.csv', delimiter=',')
    mean = np.mean(A, axis=1)
    A = A - mean[:, None]
    A_rank1, S = getRankApprox(A, 1, True)
    A_rank2 = getRankApprox(A, 2)
    error1 = np.linalg.norm(A-A_rank1)
    error2 = np.linalg.norm(A-A_rank2)
    return S, error1, error2


def problem2(numvalues=15, energy_val=.92):
    A = cv2.imread('olive.jpg', 0)
    approx, S = getRankApprox(A, 1, True)
    largest_values = S[:numvalues]  # part a
    rank1_energy = S[0] / sum(S)  # part b
    rank15_energy = sum(S[:numvalues]) / sum(S)  # part c
    min_r_val = np.size(S)
    for r in range(1, min_r_val):
        if min_r_val > r:
            if sum(S[:r]) / sum(S) > energy_val:
                min_r_val = r

    return np.reshape(
        largest_values, [15, 1]), rank1_energy, rank15_energy, min_r_val


def problem3():
    A = np.loadtxt('Problem3_Image.csv', delimiter=',')
    A_noisy = np.loadtxt('Problem3_Image_Noisy.csv', delimiter=',')
    A_rank2, S = getRankApprox(A_noisy, 2, True)
    rank2_energy = sum(S[:1]) / sum(S)  # part a
    error2_noisy = np.linalg.norm(A_noisy-A_rank2)  # part b) i.
    error2 = np.linalg.norm(A-A_rank2)  # part b) ii.
    return rank2_energy, error2_noisy, error2


A1, A2, A3 = problem1()
A4, A5, A6, A7 = problem2()
A8, A9, A10 = problem3()
