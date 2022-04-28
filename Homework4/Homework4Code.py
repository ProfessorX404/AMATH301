import cv2
import numpy as np


# must run from inside Homework4 directory

def problem1():
    A = np.loadtxt('particle_position.csv', delimiter=',')
    mean = np.mean(A, axis=1)
    A = A - mean[:, None]
    U, S, V = np.linalg.svd(A, full_matrices=False)
    A_rank1 = U[:, :0] @ np.diag(S[:0]) @ V[:0, :]
    A_rank2 = U[:, :1] @ np.diag(S[:1]) @ V[:1, :]
    error1 = np.linalg.norm(A-A_rank1)
    error2 = np.linalg.norm(A-A_rank2)
    return S, error1, error2


def problem2(numvalues=15, energy_val=.92):
    img = cv2.imread('olive.jpg', 0)
    U, S, V = np.linalg.svd(img, full_matrices=False)
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
    U, S, V = np.linalg.svd(A, full_matrices=False)
    A_rank2 = U[:, :1] @ np.diag(S[:1]) @ V[:1, :]
    rank2_energy = sum(S[:1]) / sum(S)  # part a
    error2_noisy = np.linalg.norm(A_noisy-A_rank2)  # part b) i.
    error2 = np.linalg.norm(A-A_rank2)  # part b) ii.
    return rank2_energy, error2_noisy, error2


A1, A2, A3 = problem1()
A4, A5, A6, A7 = problem2()
A8, A9, A10 = problem3()
