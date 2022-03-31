# Slide 1 - Eigenvalues and Eigenvectors
import numpy as np
A = np.array([1,4,9,1]).reshape(2,2)
eigen = np.linalg.eig
print(eigen(A))
print(eigen(A)[0][0]*eigen(A)[1][:,0])
print(np.dot(A, eigen(A)[1][:,0]))

one = np.ones(2).reshape(2,1)
print(np.outer(one, eigen(A)[0]))
print(np.dot(A, eigen(A)[1]))

A = np.array([1,4,9,1]).reshape(2,2)
p2 = np.poly1d([1, -(A[0,0] + A[1,1]), A[0,0] * A[1,1] - A[0,1]*A[1,0]])
print(p2.roots)


# Slide 3 - Diagonizable Matrices : Linear Difference Equation
import numpy as np

def solvediff(A, k, x0, x1):
  eigen = np.linalg.eig
  eigenval = eigen(A)[0]
  diagmat = np.diag([eigenval[0]**k,eigenval[1]**k])
  eigenvec = eigen(A)[1]
  x1vec = np.array([x0,x1]).reshape(2,1)
  A = np.dot(eigenvec, diagmat)
  B = np.dot(A, np.linalg.solve(eigenvec, np.diag([1]*eigenvec.shape[0])))
  xk = np.dot(B, x1vec)
  return xk

def matrixpower(mat, pow):
  if pow == 0:
    return np.diag([1]*mat.shape[2])
  if pow == 1:
    return mat
  if pow > 1:
    return np.matmul(mat, matrixpower(mat, pow-1))

cmat = np.array([5,-6, 1,0]).reshape(2,2)
print(solvediff(cmat, 10,1,1))
print(2*2**11-3**11); print(2*2**10-3**10)
print(np.matmul(matrixpower(cmat,10), np.ones(2).reshape(2,1)))


# Slide 4 - Diagonizable Matrices : Power method
import numpy as np

A = np.array([1, 0.4, 0.2, 0.4, 1, 0.4, 0.2, 0.4, 1]).reshape(3, -1)

def vnorm(x):
    return np.sqrt(np.sum(x * x))

def power_method(A, y0, eps=1e-6, maxiter=100):
    y_old = y0
    steps = 1
    ind = 1
    while ind == 1:
        z_new = np.matmul(A, y_old)
        y_new = z_new / vnorm(z_new)
        if vnorm(abs(y_new) - abs(y_old)) <= eps:
            break
        y_old = y_new
        steps += 1
        if steps == maxiter:
            break
    lamb = vnorm(np.matmul(A, y_new))
    return [y_new, lamb, steps]

print(power_method(A, np.array([1, 0, 0])))

# Slide 6 - Markov Chain : Limiting Distribution
import numpy as np
import pandas as pd

def matrixpower(mat, pow):
  if pow == 0:
    return np.diag([1]*mat.shape[2])
  if pow == 1:
    return mat
  if pow > 1:
    return np.matmul(mat, matrixpower(mat, pow-1))


Pmat = np.array([0.1,0.2,0.4,0.3,
                 0.4,0.1,0.3,0.2,
                 0.3,0.3,0,0.4,
                 0.2,0.1,0.4,0.3]).reshape(4,-1)
Pmat = pd.DataFrame(Pmat)
states= ["F", "R", "S", "Y"]
Pmat.index = states ; Pmat.columns = states
print(Pmat)
print(matrixpower(Pmat,3)) ; print(matrixpower(Pmat,30)) ; print(matrixpower(Pmat,365))

# Slide 7 - Markov Chain : Limiting Distribution
import numpy as np
import pandas as pd


def matrixpower(mat, pow):
  if pow == 0:
    return np.diag([1] * mat.shape[2])
  if pow == 1:
    return mat
  if pow > 1:
    return np.matmul(mat, matrixpower(mat, pow - 1))


P_Ehren = np.array([0, 1, 0, 0, 0, 0, 1 / 5, 0, 4 / 5, 0, 0, 0, 0, 2 / 5, 0, 3 / 5, 0, 0,
                    0, 0, 3 / 5, 0, 2 / 5, 0, 0, 0, 0, 4 / 5, 0, 1 / 5, 0, 0, 0, 0, 1, 0]).reshape(6, 6)
P_Ehren = pd.DataFrame(P_Ehren)
states = ["0", "1", "2", "3", "4", "5"]
P_Ehren.index = states;
P_Ehren.columns = states
print(P_Ehren)
print(matrixpower(P_Ehren, 3));
print(matrixpower(P_Ehren, 30));
print(matrixpower(P_Ehren, 365))


# Slide 8 - Markov Chain : Stationary Distribution
import numpy as np
Wmat = np.array([1/2,1/4,1/4,1/2, 0,1/2,1/4,1/4,1/2]).reshape(3,3)
Rmat = np.ones(3*3).reshape(3,3)
zerovec = np.zeros(1*3).reshape(1,3)
Rmat[:,0:2] = Wmat[:, 0:2] - np.diag([1]*3)[:,0:2]
bvec = np.ones(1*3).reshape(1,3) ; bvec[:,0:2] = zerovec[:,0:2]
print(np.matmul(bvec, np.linalg.solve(Rmat, np.diag([1]*Rmat.shape[0]))))


# Slide 9 - Markov Chain : Stationary Distribution
import numpy as np

eigen = np.linalg.eig
Wmat = np.array([1/2,1/4,1/4,1/2, 0,1/2,1/4,1/4,1/2]).reshape(3,3)
eigenval = eigen(np.transpose(Wmat))[0]
eigenvec = eigen(np.transpose(Wmat))[1]
x = eigenvec[:, np.where((eigenval > 0.999) & (eigenval < 1.001))]
x = x/np.sum(x)
print(x.reshape(1,-1))


# Slide 11 - Markov Chain : Stationary Distribution
import numpy as np

def matrixpower(mat, pow):
  if pow == 0:
    return np.diag([1]*mat.shape[2])
  if pow == 1:
    return mat
  if pow > 1:
    return np.matmul(mat, matrixpower(mat, pow-1))

def ergodic_state(mat, frm, to):
  count = 0 ; ind = 0 ; mat_temp = np.copy(mat)
  while ind == 0 and count < 100:
    mat_temp = matrixpower(mat, count+1)
    if mat_temp[frm, to] > 0:
      ind = 1
    count += 1
  return count

P_Ehren = np.array([0, 1, 0, 0, 0, 0, 1 / 5, 0, 4 / 5, 0, 0, 0, 0, 2 / 5, 0, 3 / 5, 0, 0,
                    0, 0, 3 / 5, 0, 2 / 5, 0, 0, 0, 0, 4 / 5, 0, 1 / 5, 0, 0, 0, 0, 1, 0]).reshape(6, 6)
print(ergodic_state(P_Ehren, 1, 2))

def check_ergodic(mat):
  n = len(mat[1,:]); mat_temp = np.zeros(n*n).reshape(n,n)
  for i in range(n):
    for j in range(n):
      mat_temp[i,j] = ergodic_state(mat, i, j)
  return mat_temp

print(check_ergodic(P_Ehren))



# Slide 12 - Markov Chain : Stationary Distribution
import numpy as np

def vnorm(x):
  return np.sqrt(np.sum(x * x))

P_Ehren = np.array([0, 1, 0, 0, 0, 0, 1 / 5, 0, 4 / 5, 0, 0, 0, 0, 2 / 5, 0, 3 / 5, 0, 0,
                    0, 0, 3 / 5, 0, 2 / 5, 0, 0, 0, 0, 4 / 5, 0, 1 / 5, 0, 0, 0, 0, 1, 0]).reshape(6, 6)
e = np.ones(6) ; I = np.diag(e) ; E = np.ones(6*6).reshape(6,6)
print(np.linalg.solve(I+E-np.transpose(P_Ehren), e))
temp = I+E-P_Ehren
print(np.matmul(e, np.linalg.solve(temp, np.diag([1]*temp.shape[0]))))

Pmat = np.copy(P_Ehren); n =  len(Pmat[0,:]) ; pi_vec0 = np.array([1/n for i in range(n)])
for i in range(20):
  pi_vec = np.matmul(pi_vec0, Pmat)
  print(round(vnorm(pi_vec - pi_vec0),8))
  pi_vec0 = pi_vec
print(pi_vec)



# Slide 13 - Solution of Nonlinear Equations : Fixed-Point Iteration
import numpy as np
n = 50; x = np.ones(n) ; x[0] = 0.5 ; diff = 0.5 ; eps = 0.00005 ; k = 0
while diff > eps:
  k += 1
  x[k] = np.cos(x[k-1])
  diff = abs(x[k]-x[k-1])

print(k+1) ; print(x[k])

# Slide 15 - Solution of Nonlinear Equations : Fixed-Point Iteration
import numpy as np

def fixedpoint(ftn, x0, tol=1e-9, max_iter = 100):
  xold = x0 ; xnew = ftn(xold) ; iter = 1
  while abs(xnew-xold) > tol and iter < max_iter:
    xold = xnew ; xnew = ftn(xold)
    iter += 1
  if abs(xnew-xold) > tol:
    print("Algorithm failed to converge")
    return np.nan
  else:
    print("Algorithm converged")
    return round(xnew,4)

def g1(x):
  return np.exp(np.exp(-x))

print(fixedpoint(g1, 2))


