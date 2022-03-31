# Slide 1 - Permutations and Combinations

from itertools import permutations
from math import exp, pi
print(list(permutations([1, 2, 3])))
print(list(permutations([1, exp(1), pi])))


# Slide 2 - Permutations and Combinations

from itertools import combinations
from math import comb
from numpy import diff

print(list(combinations([1, 2, 3], 2)))
print(comb(3,2))
print(diff([2,6,7])) ; print(diff(range(1,11)))


# Slide 3 - Permutations and Combinations
from numpy import diff, mean
from itertools import combinations
import numpy as np

def mingap(n,m,k):
    temp = np.array(list(map(lambda x: min(diff(x)), combinations(range(1,n+1), m))))
    return mean(temp == k)

print(mingap(5,3,2)) ; print(list(combinations(range(1,6),3)))
print(mingap(6,3,2)) ; print(mingap(7,3,2))


# Slide 4 - Random Sampling and Monte Carlo Simulation
import numpy as np
from numpy import diff,mean
from itertools import combinations

iter = 10**4; n=7; m=3; k=2
res = np.zeros(iter)
for i in range(iter):
    res[i] = min(diff(sorted(np.random.choice(np.arange(1,n+1), m, replace=False))))
print(mean(res == k))

def mingap(n,m,k):
    temp = np.array(list(map(lambda x: min(diff(x)), combinations(range(1,n+1), m))))
    return mean(temp == k)

print(mingap(7,3,2))



# Slide 5 - Random Sampling and Monte Carlo Simulation
import numpy as np

def sim_marble(n):
    bmarble = np.zeros(n)
    for i in range(n):
        firstmarble = np.random.choice([1,0], 1, p = [98/100, 2/100])
        pb = float(np.where(firstmarble == 1, 61/121, 60/121))
        bmarble[i] = np.random.choice([1,0], 1, p = [pb, 1-pb])
    return np.mean(bmarble)
n = 10**4 ; print(sim_marble(n))



# Slide 6 - Random Sampling and Monte Carlo Simulation
import numpy as np

bdays = np.random.choice(np.arange(1, 365 + 1), 23)
print(bdays)
print(np.bincount(bdays))

n = 10 ** 4
coincidebdays = np.array([max(np.bincount(np.random.choice(np.arange(1, 365 + 1), 23))) for _ in range(n)])
print(sum(coincidebdays >= 2) / n)


def rolling_dice(d):
    return np.random.choice(np.arange(1, 7), size=d)
tworolls = np.array([rolling_dice(2) for _ in range(10**4)])
print(np.mean(tworolls[:, 0] == tworolls[:, 1]))


# Slide 8 - Difference Equations : Matching Problem
import numpy as np
n = 100 ; probn = np.zeros(n) ; probn[0] = 0 ; probn[1] = 0.5
for k in range(2,n):
    probn[k] = (k/(k+1))*probn[k-1] + (1/(k+1))*probn[k-2]
print(probn)

n = 100 ; nreps = 10**4
nomatches = np.array([sum(np.random.choice(np.arange(n),n) == np.arange(n)) for _ in range(nreps)])
print(sum(nomatches == 0)/nreps)



# Slide 9 - Difference Equations : Gambler¡¯s Ruin
import numpy as np
N = 10; probn = np.zeros(N+1)
probn[0] = 0 ; probn[1] = 1/N
for n in range(1, N):
    probn[n+1] = 2*probn[n] - probn[n-1]
print(probn) ; print(1-probn)


# Slide 10 - Difference Equations : Gambler¡¯s Ruin
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

win = np.random.choice([-1,1],10)
cum_win = np.cumsum(win)+10 ; print(cum_win) ; print(cum_win[-2:])
total_sum = np.r_[10, cum_win]

ax = plt.figure().gca()
plt.plot(np.arange(11), total_sum, 'ro')
plt.xlabel("Index")
plt.ylabel("Total Sum")
plt.ylim([-1,20])
plt.yticks(np.linspace(0,20,5))
plt.axhline(y=10, color="black")
plt.show()


initial = 10
def Gambler_win(n):
    win = np.random.choice([-1, 1], 10)
    return sum(win)

Gambler_win_sim = [Gambler_win(10) for _ in range(1000)]
total_sum = pd.Series(Gambler_win_sim) + initial
print(total_sum.value_counts().sort_index())


# Slide 11 - Difference Equations : Gambler¡¯s Ruin
import numpy as np

def ruin(initial, N, p):
    total_sum = initial
    while (total_sum > 0) & (total_sum < N):
        win = np.random.choice([-1,1],1, p=[1-p,p])
        total_sum = total_sum + win
    return(np.where(total_sum == 0, 1, 0))
initial = 10
N = 20
p = 1/2
print(ruin(initial, N, p))

nreps = 10**4
ruin_sim = [ruin(initial, N, p) for _ in range(nreps)]
print(np.mean(ruin_sim)) ; print(sum(ruin_sim)/nreps)




# Slide 12 - Linear Congruential Sequence
import numpy as np

def LCG(n,m,a,c,x0):
    x = np.zeros(n) ; xn = x0
    for i in range(n):
        xn = (a*xn + c) % m
        x[i] = xn
    return x
print(LCG(10,8,5,1,0))


# Slide 13 - Linear Congruential Generator
import numpy as np
x = np.zeros(101)
for i in range(1, 101):
    x[i] = (21*x[i-1] + 31) % 100
print(x)

# Slide 14 - Linear Congruential Generator : Uniformity
import numpy as np
import pandas as pd
from math import trunc
from scipy.stats import chi2

def myrng(n,a,c,m,seed):
    x = np.zeros(n) ; x[0] = seed
    for i in range(n-1):
        x[i+1] = (a*x[i]+c) % m
    return x[:]/m
x = myrng(60,171,0,30269,27218)
print(x)

def rng_chisq_test(x,m):
    Obs = pd.Series(list(map(trunc, x*10))).value_counts().sort_index()
    Exv = len(x)*np.array([1 for _ in range(m)])/m
    chival = sum((Obs-Exv)**2/Exv) ; pval = 1-chi2.cdf(chival,m-1)
    value = {'test_stat': chival, 'p_value':pval, 'degf':m-1}
    return value
print(rng_chisq_test(x,10))


# Slide 15 - Linear Congruential Generator : Independence
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def myrng(n,a,c,m,seed):
    x = np.zeros(n) ; x[0] = seed
    for i in range(n-1):
        x[i+1] = (a*x[i]+c) % m
    return x[:]/m

def show(x):
    plt.scatter(x[1:], x[0:999], s=10)
    plt.show()
    pd.plotting.lag_plot(pd.Series(x), s=10)
    plt.show()
    pd.plotting.autocorrelation_plot(pd.Series(x))
    plt.show()

x = myrng(1000,171,0,30269,27218)
show(x)

x1 = myrng(1000,401,101,1024,0)
show(x1)

u = np.random.random(1000)
show(u)


# Slide 16 - Pseudo-Random Numbers in R
import numpy as np

np.random.seed(323)
print(np.random.random(2))

np.random.seed(323)
print(np.random.random(4))

state = np.random.get_state()
np.random.set_state(state)
print(np.random.random(2))



# Slide 17 - Monte Carlo Simulation : runif & replicate
import numpy as np

def sim_marble2(nb1, nb2, n1, n2):
    nb2 = np.where(np.random.random(1) < nb1/n1, nb2+1, nb2)
    bmarble = np.where(np.random.random(1) < nb2/n2, 1, 0)
    return bmarble
print(sim_marble2(98,60,100,120))

nreps=10**4
np.random.seed(4)
secondmarble = [sim_marble2(98,60,100,120) for _ in range(nreps)]
print(np.mean(secondmarble))


def longerpiece(k):
    breakpts = sorted(np.random.random(k-1))
    length = np.diff(np.r_[0,breakpts,1])
    return max(length)
np.random.seed(4)
longerpieces = np.array([longerpiece(2) for _ in range(nreps)])
print(np.mean(longerpieces)) ; print(np.mean(1-longerpieces < 0.2))