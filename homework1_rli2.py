import numpy as np

def problem1 (A, B):
    return A + B
def problem2 (A, B, C):
    return np.dot(A,B)-C
def problem3 (A, B, C):
    return A*B + C.T
def problem4 (x, y):
    return np.dot(x,y)
def problem5 (A):
    return np.zeros(A.shape)
def problem6 (A):
    return np.ones(A.shape)
def problem7 (A, x):
    return np.linalg.solve(A, x)
def problem8 (A, x):
    return np.transpose(np.linalg.solve(A.T, x.T))
def problem9 (A, alpha):
    return A + alpha*np.eye(A.shape[1])
def problem10 (A, i, j):
    return A[i,j]
def problem11 (A, i):
    return np.sum(A,axis=1)[i]
def problem12 (A, c, d):
    A[A<c] = 0
    A[A>d] = 0
    return np.mean(A[np.nonzero(A)])

def problem13(A, k):
    w, v = np.linalg.eig(A)
    return np.fliplr(v[:, -k:])

def problem14(x, k, m, s):
    mz = m * np.ones((len(x), 1))
    sI = s * np.eye(len(x))
    return np.random.multivariate_normal(np.add(x, mz).squeeze(), sI, k).transpose()

def test():
    A = np.array([[1,2],[3, 4]])
    B = np.array([[1,2,],[3, 4]])
    C = np.array([[1, 2],[3, 4]])
    x = np.array([2,3,1,0])
    y = np.array([2,3,1,0])
    z1= np.array([2,3]).reshape((2,1))
    z2= np.array([2,3]).reshape((1,2))
    alpha = np.random.rand(1,1)
    print(problem1(A,B))
    print(problem2(A,B,C))
    print(problem3(A,B,C))
    print(problem4(x,y))
    print(problem5(A))
    print(problem6(A))
    print(problem7(A,z1))
    print(problem8(A,z2))
    print(problem9(A,alpha))
    print(problem10(A,0,1))
    print(problem11(A,0))
    print(problem12(A,2,3))
    print(problem13(A,1))
    print(problem14(z1,1,2,3))
