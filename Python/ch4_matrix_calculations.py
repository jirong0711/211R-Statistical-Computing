# Slide 4 - Gaussian Elimination : Row Operations
import numpy as np

def rowswap(m, row1, row2):
  mat = np.copy(m)
  row_temp = mat[row1, :]
  m[row1, :]  = m[row2,:]
  m[row2, :] = row_temp
  return m

def rowscale(m, row, k):
  m[row,:] = k*m[row,:]
  return m

def rowtransform(m, row1, row2, k):
  m[row2,:] = m[row2,:] + k*m[row1,:]
  return m

A = np.array([2,-5,4,-3,1,-2,1,5,1,-4,6,10], dtype="float64").reshape(3,-1)
A1 = rowtransform(A, 0,1, -A[1,0]/A[0,0])
A1 = rowtransform (A1,0,2, -A1[2,0]/A1[0,0])
A1 = rowtransform (A1, 1,2, -A1[2,1]/A1[1,1])
print(A1)


# Slide 5 - Gaussian Elimination : Row Operations
import numpy as np

def rowtransform(m, row1, row2, k):
  m[row2,:] = m[row2,:] + k*m[row1,:]
  return m

A = np.array([2,-5,4,1,-2,1,1,-4,6], dtype=float).reshape(3,-1)
B = np.array([-3,5,10], dtype=float).reshape(3,-1)
U_pls = np.c_[A, B]

def GaussElim(mat):
  p = mat.shape[0]
  for i in range(p-1):
    for j in range(i+1, p):
      k = -mat[j,i]/mat[i,i]
      mat = rowtransform(mat, i, j, k)
  return mat

GaussElim(U_pls)

# Slide 6 - Gaussian Elimination : Back Substitution
import numpy as np
A = np.array([2,-5,4,1,-2,1,1,-4,6], dtype=float).reshape(3,-1)
B = np.array([-3,5,10], dtype=float).reshape(3,-1)

def rowtransform(m, row1, row2, k):
  m[row2,:] = m[row2,:] + k*m[row1,:]
  return m


def GaussElim(mat):
  p = mat.shape[0]
  for i in range(p-1):
    for j in range(i+1, p):
      k = -mat[j,i]/mat[i,i]
      mat = rowtransform(mat, i, j, k)
  return mat


def backsub(mat):
  p = mat.shape[1]-1
  x = np.zeros(p)
  for i in range(p-1,-1,-1):
    x[i] = mat[i,p]
    if i < p-1:
      for j in range(i+1, p):
        x[i] = x[i] - mat[i,j]*x[j]
    x[i] = x[i]/mat[i,i]
  return x

U_pls = GaussElim(np.c_[A, B])
print(backsub(U_pls))


# Slide 7 - LU Decomposition : Forward Substitution
import numpy as np
A = np.array([2,-5,4,1,-2,1,1,-4,6], dtype=float).reshape(3,-1)
b = np.array([-3,5,10], dtype=float).reshape(3,-1)

def rowtransform(m, row1, row2, k):
  m[row2,:] = m[row2,:] + k*m[row1,:]
  return m


def GaussElim(mat):
  p = mat.shape[0]
  for i in range(p-1):
    for j in range(i+1, p):
      k = -mat[j,i]/mat[i,i]
      mat = rowtransform(mat, i, j, k)
  return mat


def forsub(mat):
  p = mat.shape[1]-1 ; y = np.zeros(p)
  for i in range(p):
    y[i] = mat[i, p]
    if i > 0:
      for j in range(i):
        y[i] = y[i]-mat[i,j]*y[j]
    y[i] = y[i] / mat[i, i]
  return y

temp =  GaussElim(A)
L_pls =  np.transpose(temp)
print(forsub(np.c_[L_pls, b]))



# Slide 8 - Gaussian Elimination : Pivoting
import numpy as np
A = np.array([0,-5,4,1,2,1,1,-4,6], dtype=float).reshape(3,-1)
b = np.array([-3,5,10], dtype=float).reshape(3,-1)
U_pls = np.c_[A,b]

def rowswap(m, row1, row2):
  mat = np.copy(m)
  row_temp = mat[row1, :]
  m[row1, :]  = m[row2,:]
  m[row2, :] = row_temp
  return m

def rowscale(m, row, k):
  m[row,:] = k*m[row,:]
  return m

def rowtransform(m, row1, row2, k):
  m[row2,:] = m[row2,:] + k*m[row1,:]
  return m

def GaussElim2(mat):
  p = mat.shape[0]
  for i in range(p):
    if abs(mat[i,i]) == 0 and i < p-1:
      mat = rowswap(mat, i, i+1)
    mat = rowscale(mat, i, 1/mat[i, i])
    if i < p - 1:
      for k in range(i + 1, p):
        a = -mat[k, i] / mat[i, i]
        mat = rowtransform(mat, i, k, a)

  return mat

print(GaussElim2(U_pls))


# Slide 9 - Matrix inversion in R
import numpy as np
A = np.array([2,-5,4,1,-2,1,1,-4,6], dtype=float).reshape(3,-1)

def rowswap(m, row1, row2):
  mat = np.copy(m)
  row_temp = mat[row1, :]
  m[row1, :]  = m[row2,:]
  m[row2, :] = row_temp
  return m

def rowscale(m, row, k):
  m[row,:] = k*m[row,:]
  return m

def rowtransform(m, row1, row2, k):
  m[row2,:] = m[row2,:] + k*m[row1,:]
  return m

def backsub(mat):
  p = mat.shape[1]-1
  x = np.zeros(p)
  for i in range(p-1,-1,-1):
    x[i] = mat[i,p]
    if i < p-1:
      for j in range(i+1, p):
        x[i] = x[i] - mat[i,j]*x[j]
    x[i] = x[i]/mat[i,i]
  return x

def GaussElim2(mat):
  p = mat.shape[0]
  for i in range(p):
    if abs(mat[i,i]) == 0 and i < p-1:
      mat = rowswap(mat, i, i+1)
    mat = rowscale(mat, i, 1/mat[i, i])
    if i < p - 1:
      for k in range(i + 1, p):
        a = -mat[k, i] / mat[i, i]
        mat = rowtransform(mat, i, k, a)

  return mat

def inv_mat(mat):
  p = mat.shape[0] ; E = np.diag([1]*p)
  invA = np.copy(mat)
  for i in range(p):
    temp = GaussElim2(np.c_[mat[:,0:p], E[:,i]])
    temp = backsub(temp)
    invA[:,i] = np.transpose(temp)
  return invA

print(inv_mat(A))
print(np.matmul(A, inv_mat(A)))


# Slide 10 - Matrix inversion in R
import numpy as np
from scipy.linalg import lu
A = np.array([2,-5,4,1,-2,1,1,-4,6], dtype=float).reshape(3,-1)
P, L, U = lu(A)
print(np.matmul(L,U)) ; print(np.matmul(np.transpose(P), A))
print(np.dot(np.dot(P,L),U)) ; print(A)


# Slide 11 - Matrix inversion in R : Least Squares
import numpy as np
from scipy import stats

x = np.arange(0, 11)
np.random.seed(333)
y = 3*x+4+np.random.normal(0, 0.3, len(x))

lm = stats.linregress(x, y)
print(lm)
print(f"intercept: {lm.intercept:.4f}")
print(f"slope: {lm.slope:.4f}")


# Slide 13 - Matrix inversion in R : Least Squares
import numpy as np
from scipy import stats


x = np.array([0.45, 0.08,-1.08, 0.92, 1.65, 0.53, 0.52,-2.15,-2.20,-0.32,
       -1.87,-0.16,-0.19, -0.98,-0.20, 0.67, 0.08, 0.38, 0.76,-0.78])
y = np.array([1.26, 0.58,-1.00, 1.07, 1.28,-0.33, 0.68,-2.22,-1.82,-1.17,
       -1.54, 0.35,-0.23,-1.53, 0.16, 0.91, 0.22, 0.44, 0.98,-0.98])
X = np.c_[np.ones(20), x] ; XX = np.matmul(np.transpose(X), X)
Xy = np.dot(np.transpose(X),y)
b = np.dot(np.linalg.solve(XX, np.diag([1]*(XX.shape[0]))), Xy)
print(b)
print(np.linalg.solve(np.dot(np.transpose(X), X), np.dot(np.transpose(X),y)))

lm = stats.linregress(x, y)
print(lm.intercept, lm.slope)


# Slide 15 - Least Squares : Polynomial Regression
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from functools import wraps

import time


def timing(f):
  def wrap(*args, **kwargs):
    time1 = time.time()
    ret = f(*args, **kwargs)
    time2 = time.time()
    print('{:s} function took {:.3f} ms'.format(f.__name__, (time2 - time1) * 1000.0))
    return ret
  return wrap

@timing
def polymat(x, m):
    mat = np.ones(len(x))[:, np.newaxis]
    for i in range(2, m + 2):
        mat = np.c_[mat, x ** (i - 1)]
    return mat


n = 20;
x = np.arange(1, n + 1) / n

x = np.arange(1, 5);
y = np.array([7.9, 15.1, 24.1, 34.8])
X = polymat(x, 2)
XX = np.dot(np.transpose(X), X);
Xy = np.dot(np.transpose(X), y)
b = np.dot(np.linalg.solve(XX, np.diag([1] * (XX.shape[0]))), Xy)
print(b)
print(np.linalg.solve(np.dot(np.transpose(X), X), np.dot(np.transpose(X), y)))

x2 = x ** 2
lm = LinearRegression()
lm.fit(pd.DataFrame(np.transpose(np.array([x, x2]))), y)
print(np.r_[lm.intercept_, lm.coef_])

@timing
def polymat2(x, m):
    mat = np.zeros(len(x) * (m + 1)).reshape(len(x), m + 1)
    for i in range(m + 1):
        mat[:, i] = x ** (i)
    return mat

@timing
def polymat3(x, m):
    return x[:, None] ** np.arange(m + 1)

polymat(x,8) ; polymat2(x,8) ; polymat3(x,8)

