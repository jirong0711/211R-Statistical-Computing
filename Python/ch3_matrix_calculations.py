# Slide 1 - Matrix in R
import numpy as np
A = np.array([[1,2],[3,4]]) ; print(A)
B = np.arange(1,7).reshape(2,3) ; print(B)

# Slide 2 - Matrix in R
import numpy as np

A  = np.array([1,7,2,8]).reshape(2,2)
dA = np.diag(A); print(np.diag(dA)) ; print(np.diag([1]*2))
print(A.shape); print(np.transpose(A)); print(np.linalg.det(A)) ; print(np.linalg.solve(A, np.diag([1]*2)))


# Slide 3 - Matrix in R
import numpy as np

#method1
n = 4 ; h4 = np.zeros(n*n).reshape(n,n)
for i in range(n):
    for j in range(n):
        h4[i,j] = 1/(i+j+1)
        h4[j,i] = h4[i,j]
print(h4)

#method2
n = 4 ; h4 = np.zeros(n*n).reshape(n,n); j = np.arange(n)
for i in range(n):
    h4[i,:] = 1/(i+j+1)
print(h4)

#method3
print(1/np.vstack((np.arange(1,5), np.arange(2,6), np.arange(3,7), np.arange(4,8))))

#method4
temp = np.arange(n)+1
for i in range(1,n):
    temp = np.vstack((temp, np.arange(i, n+i)+1))
h4 = 1/temp
print(h4)


# Slide 4 - Matrix in R

import numpy as np
x = np.arange(1,5) ; y = np.arange(0.1, 0.4, 0.1)
print(np.outer(x,y))
print(np.divide.outer(x,y))
print(np.subtract.outer(x,y))
print(x[:, None]**y)

h4=1/np.vstack((np.arange(1,5), np.arange(2,6), np.arange(3,7), np.arange(4,8)))
print(np.inner(h4, h4+1))
print(np.matmul(np.transpose(h4), h4+1))
print(np.inner(x,y)) ; print(np.outer(x,y))
print(sum(x*y))  # inner product (a scalar value)


# Slide 5 - Matrix in R : Loops or Vectors?
import numpy as np
import time
h4=1/np.vstack((np.arange(1,5), np.arange(2,6), np.arange(3,7), np.arange(4,8)))
print(list(map(np.prod, h4))) ; print(list(map(np.prod, np.transpose(h4))))


matrixA = np.random.random(10**6).reshape(1000,-1)
rowsums = np.repeat(np.nan, matrixA.shape[0])
start1 = time.time()
for i in range(matrixA.shape[0]):
    s = 0
    for j in range(matrixA.shape[1]):
        s += matrixA[i,j]
    rowsums[i] = s
print(rowsums) ; print(time.time()-start1)

start2 = time.time()
print(list(map(sum, matrixA))) ; print(time.time()-start2)


# Slide 6 - Efficiency : Time
import numpy as np
import time

start1 = time.time()
print(np.cumsum(np.sin(np.random.random(10**7)))) ; print(time.time() - start1)

matrixA = np.random.random(10**6).reshape(1000,-1)
colmins = np.repeat(float("inf"), matrixA.shape[1])
start2 = time.time()
for j in range(1000):
    for i in range(1000):
        if colmins[j] > matrixA[i,j]:
            colmins[j] = matrixA[i,j]
print(colmins) ; print(time.time() - start2)

start3 = time.time()
for j in range(1000):
    colmins[j] = min(matrixA[:,j])
print(list(map(min, np.transpose(matrixA)))) ;  print(time.time() - start3)



# Slide 7 - Matrix in R : User-defined function
import numpy as np

def vec_crossprod(x,y):
    mat = np.vstack(((np.repeat(np.nan,3)),x,y)) ; temp = np.zeros(3)
    for i in range(3):
        idxs = list(range(3))
        idxs.pop(i)
        temp[i] = (-1)**(i+2) * np.linalg.det(mat[1:3,idxs])
    return temp
matA = np.array([3,2,1,2,4,-1,1,-1,0]).reshape(3,3) ; avec = matA[1,:] ; bvec = matA[2,:]
print(vec_crossprod(avec,bvec))


def cofac_det(mat):
    temp = np.zeros(len(mat[0,:]))
    for i in range(3):
        idxs = list(range(3))
        idxs.pop(i)
        temp[i] = (-1)**(i+2) * mat[0,i]* np.linalg.det(mat[1:3,idxs])
    return sum(temp)

print(cofac_det(matA)) ; print(np.linalg.det(matA))


# Slide 8 - Matrix in R : User-defined function
import numpy as np

A = np.array([1,1,0,1]).reshape(2,2)
def matrixpower0(A, pow):
    B = A
    for i in range(pow-1):
        A = np.matmul(A,B)
    return A
print(matrixpower0(A,2))

def matrixpower(mat, pow):
    if pow == 0:
        return np.diag([1]*mat.shape[0])
    if pow == 1:
        return mat
    if pow > 1:
        return np.matmul(mat, matrixpower(mat, pow-1))
print(matrixpower(A,2))

print(np.linalg.matrix_power(A,2))


# Slide 12 - Markov Chain : Multi-step Probabilities
import numpy as np
import pandas as pd

def matrixpower(mat, pow):
    if pow == 0:
        return np.diag([1]*mat.shape[0])
    if pow == 1:
        return mat
    if pow > 1:
        return np.matmul(mat, matrixpower(mat, pow-1))

Wmat = np.array([1/2,1/4,1/4,1/2, 0,1/2,1/4,1/4,1/2]).reshape(3,3)
print(np.dot(Wmat, np.repeat(1,3)))
pow = 4;  Wpowermat = matrixpower(Wmat, pow)
print(sum(Wpowermat[0,:]*np.array([1,0,0])))


def w_forecast1(Wmat, days, today):
    weather = np.zeros(days, dtype=int)
    weather[0] = today
    for j in range(1, days):
        weather[j] = np.random.choice([1,2,3],1, p=Wmat[weather[j-1]-1, :])
    return weather[days-1]
nreps = 10**4; res1 = [w_forecast1(Wmat,5,1) for _ in range(nreps)]
res1 = pd.Series(res1)
print(res1.value_counts().sort_index()/nreps)

def w_forecast2(Wpowermat, today):
    weather = np.random.choice([1,2,3],1, p = Wpowermat[today-1,:])
    return weather

res2 = [w_forecast2(Wpowermat,1) for _ in range(nreps)]
res2 = pd.Series(res2)
print(res2.value_counts().sort_index()/nreps)


# Slide 13 - Markov Chain : Gambler¡¯s Ruin
import numpy as np

def matrixpower(mat, pow):
    if pow == 0:
        return np.diag([1]*mat.shape[0])
    if pow == 1:
        return mat
    if pow > 1:
        return np.matmul(mat, matrixpower(mat, pow-1))

p = 0.6 ; q = 1-p ; N = 8 ; Tpmat = np.zeros((N+1)**2).reshape(N+1, -1)

for i in range(1,N):
    Tpmat[i,i-1] = q
    Tpmat[i,i+1] = p
Tpmat[0,0] = 1;  Tpmat[N,N] = 1
Tpmat4 = matrixpower(Tpmat, 4)
print(sum(np.arange(9)*Tpmat4[3,:]))

def g_fortune1(Tpmat, plays, init):
    fortune = np.zeros(plays+1, dtype=int)
    fortune[0] = init
    for j in range(1,plays+1):
        fortune[j] = np.random.choice(np.arange(9),1,p=Tpmat[fortune[j-1],:])
    return fortune[plays]
nreps=10**4; res1 = [g_fortune1(Tpmat, 4, 3) for _ in range(nreps)]
print(np.mean(res1))

def g_fortune2(Tpmat4, init):
    fortune = np.random.choice(np.arange(9),1,p=Tpmat4[init,:])
    return fortune
nreps=10**4; res2 = [g_fortune2(Tpmat4,3) for _ in range(nreps)]
print(np.mean(res2))


# Slide 15 - Markov chain and Simulation : Craps Game
import numpy as np


def craps_sim():
    roll = sum(np.random.choice(np.arange(1,7), 2))
    win = -10 ; roll2 = -10  # Assign negative value
    if (roll == 2) or (roll == 3) or (roll == 12):
        win = 0
        return win
    else:
        if (roll == 7) or (roll == 11):
            win = 1
            return win
        else:
            roll2 = 1
    while (roll2 != 7) and (roll2 != roll):
        roll2 = sum(np.random.choice(np.arange(1,7), 2))
        if roll2 == 7:
            win = 0
        if roll2 == roll:
            win = 1
    return win

craps_res = [craps_sim() for _ in range(10**4)]
print(np.mean(craps_res))



# Slide 17 - Markov chain and Simulation : Craps Game
import numpy as np
def matrixpower(mat, pow):
    if pow == 0:
        return np.diag([1]*mat.shape[0])
    if pow == 1:
        return mat
    if pow > 1:
        return np.matmul(mat, matrixpower(mat, pow-1))

state = ["C", "W", "L", 4, 5, 6, 8, 9, 10]
P_craps = np.zeros(len(state)**2).reshape(len(state),-1)
P_craps[0, 1:3] = [6+2, 1+2+1]
P_craps[0, 3:] = [3, 4, 5, 5, 4, 3]
P_craps[3:,1] = [3, 4, 5, 5, 4, 3]
P_craps[3:,2] = 6
P_craps = P_craps/36
np.fill_diagonal(P_craps, np.r_[0,1,1,1-sum(np.transpose(P_craps[3:,1:3]))])
P_craps = P_craps.round(3)
print(P_craps)
print(matrixpower(P_craps,100).round(3))




