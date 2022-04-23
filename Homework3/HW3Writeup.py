import numpy as np
import scipy.linalg as s
import time


def problemA():
    N = 1500
    offDiag = np.ones(N-1)
    A = -1*np.diag(np.ones(N)) + 4*np.diag(offDiag,
                                           1) + 4*np.diag(offDiag, -1)
    return A


def problemB(A):
    start = time.time()
    r = 0
    for i in range(120):
        b = np.random.rand(1500, 1)
        x = s.solve(A, b)
        r += np.linalg.norm(np.matmul(A, x)-b)
    stop = time.time()-start
    return stop, r


def problemC(A):
    start = time.time()
    r = 0
    P, L, U = s.lu(A)
    for i in range(120):
        b = np.random.rand(1500, 1)
        x = s.solve(np.matmul(P, np.matmul(L, U)), b)
        r += np.linalg.norm(np.matmul(A, x)-b)
    stop = time.time()-start
    return stop, r


def problemD(A):
    start = time.time()
    a = np.linalg.inv(A)
    r = 0
    for i in range(120):
        b = np.random.rand(1500, 1)
        x = np.matmul(a, b)
        r += np.linalg.norm(np.matmul(A, x)-b)
    stop = time.time()-start
    return stop, r


def main():
    A = problemA()
    print(problemB(A))
    print(problemC(A))
    print(problemD(A))


if __name__ == "__main__":
    main()
