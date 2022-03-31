# Slide 3 - Buffon¡¯s Needle Problem
import numpy as np

def buffon(D, L):
  x = np.random.uniform(0, D/2, 1)
  theta = np.random.uniform(0, np.pi/2, 1)
  count = (x <= L/2)*(np.sin(theta))
  return count
L = 1 ; D = 1 ; N=10**5
times = np.array([buffon(L,D) for i in range(N)])
print((2*L)/(D*np.mean(times)))


# Slide 4 - Hit-or-Miss Method (Integration)
import numpy as np

nsim = 10**6
x = np.random.random(nsim)
y = np.random.random(nsim)
print(4*np.sum(x**2+y**2 < 1)/nsim)


# Slide 7 - Monte Carlo Integration
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

x = np.random.exponential(1/2, 2000)
y = np.sqrt(1+x**2)/2
print(np.mean(y)) ; print(np.sqrt(np.var(y)/2000))

def ftn(x):
  return np.sqrt(1+x**2)*np.exp(-2*x)
print(quad(ftn, 0, np.inf))

x = np.random.normal(0,1,10000) ; w = x > 1.96
print(np.mean(w)) ; print(np.sqrt(np.var(w)/10000))

def ftn(x):
  return 1/np.sqrt(2*np.pi)*np.exp(-1/2*(x**2))
print(quad(ftn, 1.96, np.inf))
print(1-norm.cdf(1.96))


# Slide 8 - Random Number Generation
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

np.random.seed(123)
print(np.random.uniform(-3, 1, 10))
print(np.random.uniform(1, 3, 10))
print(np.random.random(10))

n = 10; N = 1000; M = np.zeros(N)
for i in range(N):
  u = np.random.random(n)
  M[i] = max(u)
plt.hist(M,  density=True)
plt.xlim([0,1])
plt.title("Max of 10 uniforms")
plt.ylabel("Density")
x = np.arange(0, 1, 0.001)
plt.plot(x, beta.pdf(x, 10, 1), color="blue")
plt.show()


# Slide 9 - Random Number Generation
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

N = 1000;
theta = np.random.random(N) * 2 * np.pi
pos_x = np.cos(theta)
plt.hist(pos_x, bins=20, density=True)
plt.xlim([-1, 1])
plt.ylim([0, 2])
plt.title("X-position on the circle")
print(np.percentile(pos_x, [0, 25, 50, 75, 100]))

x = np.arange(-1, 1, 0.001)
plt.plot(x, 1 / (np.pi * np.sqrt(1 - x ** 2)), color="red")
plt.show()


def h(x):
  return (np.cos(50*x)+np.sin(20*x))**2


x = h(np.random.random(10 ** 4))
estint = np.cumsum(x) / (np.arange(1, 10 ** 4 + 1))
esterr = np.sqrt(np.cumsum(x - estint) ** 2) / (np.arange(1, 10 ** 4 + 1))
print(quad(h, 0, 1))
plt.plot(estint, 'o', color="black")
plt.ylim(np.mean(x) + np.dot(20, [-esterr[10 ** 4 - 1], esterr[10 ** 4 - 1]]))
plt.plot(estint + 2 * esterr, 'o', color="gold")
plt.plot(estint - 2 * esterr, 'o', color="gold")
plt.show()


# Slide 10 - Random Number Generation : Discrete Random Variables
import numpy as np
from math import comb


def cdf_binom(x, n, p):
  Fx = 0
  for i in range(0, x+1):
    Fx += comb(n, i)*(p**i)*((1-p)**(n-i))
  return Fx

def sim_binom0(n,p):
  X = 0; U = np.random.random(1)
  while cdf_binom(X, n, p) < U:
    X += 1
  return X

print([sim_binom0(4, 0.3) for i in range(10)])
print(np.random.binomial(4, 0.3, 10))


# Slide 11 - Random Number Generation : Discrete Random Variables
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binom

def binom_sim(n,p):
  X = 0; px = (1-p)**n ; Fx = px ; U = np.random.random(1)
  while Fx < U:
    X += 1
    px *= p/(1-p)*(n-X+1)/X
    Fx += px
  return X

nreps=10**4; n=10;p=0.5;
binomX = pd.DataFrame([binom_sim(n,p) for i in range(nreps)])
print(binomX.value_counts().sort_index())
plt.hist(binomX, density=True, bins=10, color="grey", alpha=0.5)
plt.xticks(np.arange(0,10+1, 2))
plt.scatter(np.arange(0,n)+1, binom.pmf(np.arange(0, n), n, p), c="blue")

x = np.arange(0, n+1)
z = np.zeros(n+1)
y = binomX.value_counts().sort_index()/nreps
for i in range(n):
  plt.vlines(x[i]+1, ymin=z[i], ymax=y[i], colors="red")
plt.show()

X = 0
for i in range(n):
  U = np.random.random(1)
  if U < p:
    X += 1

X = sum(np.random.random(n) < p)



# Slide 12 - Random Number Generation : Discrete Random Variables
import numpy as np
import math

X = 1; success = False ; p=0.5
while success != 1:
  U = np.random.random(1)
  if U < p:
    success = True
  else:
    X += 1

def geo_sim(p):
  ntrials = math.ceil(1/p+10*(1-p)/p**2)
  x = np.min(np.where(np.random.random(ntrials) < p))
  return x+1

nreps = 10**3 ; p = 0.3 ;  x = np.array([geo_sim(p) for i in range(nreps)])
print(sum(x)/nreps) ; print(np.var(x)) ; print(1/p) ; print((1-p)/p**2)


# Slide 14 - The Inverse Transform Method
import numpy as np
import matplotlib.pyplot as plt

n = 10**3 ; u = np.random.random(n) ; x = u**(1/3)
plt.hist(x, density=True)
y = np.arange(0,1,0.001)
plt.plot(y,3*(y**2), color="red")
plt.show()

Nsim=10**4
U = np.random.random(Nsim)
X = -np.log(U)
Y = np.random.exponential(size=Nsim)

plt.subplot(1,2,1)
plt.hist(X, density=True)
plt.title("Exp from Inverse Transform")

plt.subplot(1,2,2)
plt.hist(Y, density=True)
plt.title("Exp from R")
plt.show()

print(np.percentile(X, [0, 25, 50, 75, 100]))
print(np.percentile(Y, [0, 25, 50, 75, 100]))


# Slide 15 - The Inverse Transform Method
import numpy as np
pmfs = np.array([0.2, 0.3, 0.1, 0.15, 0.05, 0.2])

def inv_discrete(pmfs):
  cumprobs = np.cumsum(pmfs)
  u = np.random.random(1)
  X = np.sum(u > cumprobs)
  return X

n = 20
print([inv_discrete(pmfs) for i in range(n)])
print(np.random.choice(np.arange(0, len(pmfs)),10, p=pmfs))
print([np.random.choice(np.arange(0, len(pmfs)),1, p=pmfs) for i in range(10)])


# Slide 16 - The Inverse Transform Method
import numpy as np
import math
n = 1000; p = 0.25 ; u = np.random.random(n)
X = list(map(math.ceil, np.log(1-u)/np.log(1-p)))


