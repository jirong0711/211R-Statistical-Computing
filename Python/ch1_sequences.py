
# Slide 1 - Sequences
import numpy as np
def arith_mean(x):
    s = 0
    n = len(x)
    for i in range(n):
        s += x[i]
    return s/n

x = np.arange(0.2, 5.9, 1.0)
print(arith_mean(x)); print(sum(x)/len(x)); print(np.mean(x))


# Slide 2 - Recursive Sequences
import numpy as np

def Fibo_func1(n, x1, x2):
    temp1 = x1
    temp2 = x2
    for k in range(3,n+1):
        temp3 = temp1 + temp2
        temp1 = temp2
        temp2 = temp3
    return temp3
print(Fibo_func1(3,1,1)) ; print(Fibo_func1(10,1,1)) ; print(Fibo_func1(20,1,1))

phi = (1+np.sqrt(5))/2
n = np.arange(1,11)
Fn = (phi**n-(-phi)**(-n))/np.sqrt(5)
print(Fn)



# Slide 3 - Recursive Sequences
import numpy as np

n = 20
Fn = [0 for i in range(n)]
Fn[0] = 1 ; Fn[1] = 1
for i in range(2,n):
    Fn[i] = Fn[i-1] + Fn[i-2]
print(Fn)

Fn = [1,1] ; n = 1
while Fn[n] <= 100:
    n += 1
    Fn.append(Fn[n-1] + Fn[n-2])
print(n+1)


# Slide 4 - Recursive Sequences
import numpy as np
n = 50; Fn = np.zeros(n)
Fn[0] = 1 ; Fn[1] = 1
for i in range(2,n):
    Fn[i] = Fn[i-1] + Fn[i-2]
#print(Fn)
print(list(map(int,Fn)))

def Fibo_func2(n):
    if n == 1 or n == 2:
        return 1
    else:
        return Fibo_func2(n-1) + Fibo_func2(n-2)
print(Fibo_func2(30))


# Slide 5 - Factorial and Probability
import numpy as np
import math
import scipy.special

n = 6; n_factorial = 1
for i in range(1,n+1):
    n_factorial *= i
print(n_factorial)
print(np.prod(range(1,n+1)))

nfact = [0 for i in range(n)]
nfact[0] = 1
for i in range(n-1):
    nfact[i+1] = nfact[i]*(i+2)
print(nfact)

print(math.factorial(n))
print(scipy.special.factorial(range(1,n+1)))


# Slide 6 - Factorial and Probability
import numpy as np

n = 23; a = 1
for k in range(1,n+1):
    a *= (365-k+1)/365
print(1-a)

birthday = np.arange(365-n+1,365+1, dtype=np.float64)
print(1-math.prod(birthday)/365**n)

def pbday(n):
    return 1-math.factorial(365)/(math.factorial(365-n)*(365**n))
print(pbday(n))



# Slide 7 - Factorial and Probability
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def lfactorial(n):
    return sum(list(map(math.log, np.arange(1,n+1))))
def pbday(n):
    return 1-math.exp(lfactorial(365) - lfactorial(365-n)-n*math.log(365))
n = 23
print(pbday(n))
print(list(map(pbday, np.arange(20,31))))

nstudents = np.arange(2,51)
pbday_df = pd.DataFrame({"nstudents": nstudents, "prob_birthday": list(map(pbday, nstudents))})
print(pbday_df)

plt.plot(np.arange(1,101), list(map(pbday, np.arange(1,101))), color="red",linewidth=2)
plt.xlabel("Number of students")
plt.ylabel(r"$P(A)$")
plt.show()


# Slide 8 - Combination and Probabiltiy
from math import comb, factorial
print(comb(98,50)/comb(100,50))

def n_choose_r1(n,r):
    ncr = factorial(n)/(factorial(r)*factorial(n-r))
    return ncr
print(n_choose_r1(100,50))
print(comb(200,100)) ; print(n_choose_r1(200,100))


# Slide 9 - Combination and Probabiltiy
import numpy as np
from math import log,prod

def lfactorial(n):
    return sum(list(map(log, np.arange(1,n+1))))


def n_choose_r2(n,r):
    ncr = 10**(lfactorial(n)/log(10) - lfactorial(r)/log(10) - lfactorial(n-r)/log(10))
    return ncr
print(n_choose_r2(100,50))
print(n_choose_r2(200,100))

a = 1
for k in range(1,101):
    a *= (100+k)/k
print(a)

print(prod(range(101,201))/prod(range(1,101)))


# Slide 10 - Limit of a Sequence and Series
import numpy as np
n  = np.arange(1,10001)
print((1+1/n)**n)
print(np.exp(1))

n = 200
Fn = np.zeros(n)
Fn[0] = 1; Fn[1] = 1
for i in range(2,n):
    Fn[i] = Fn[i-1] + Fn[i-2]

print(Fn[2:(n-1)]/Fn[1:(n-2)])
print((1+np.sqrt(5))/2)


# Slide 11 - Limit of a Sequence and Series
import numpy as np
from scipy.special import factorial
from numpy import log, sqrt, pi, exp
n = np.arange(1,101, dtype=np.float64)
print(factorial(n)/n**n)

def stirling(n):
    return sqrt(2*pi*n)*(n**n)*exp(-n)
print(factorial(52)) ; print(stirling(52))
print(factorial(n)/stirling(n))

print(list(map(factorial, n))) ; print(list(map(stirling, n)))
#print(factorial(1000)/stirling(1000)) => return NaN

def lstirling(n):
    return log(sqrt(2*pi*n))+n*(log(n)-1)
def lfactorial(n):
    return sum(list(map(log, np.arange(1,n+1))))

print(exp(lfactorial(1000)-lstirling(1000)))


# Slide 12 - Limit of a Sequence and Series
import numpy as np
import matplotlib.pyplot as plt

upper=1000 ; n = np.arange(1,upper+1)
an = 1/n ; bn = 1/n**2

partials_a = np.cumsum(an)
partials_b = np.cumsum(bn)

plt.plot(n, partials_a, 'r--', linewidth=2)
plt.plot(n, partials_b, 'b', linewidth=2)
plt.xlabel("n")
plt.ylabel("Sequence of Partial Sums Values")
plt.legend((r'$a_n$',r'$b_n$'), loc=(0.8,0.5))
plt.show()

print(sum(bn)) ; print(np.pi**2/6)



# Slide 13 - Taylor Series
import numpy as np

def exponential(x):
    sum = 1
    temp = 1
    k = 1
    while abs(temp) > np.finfo(float).eps:
        temp *= x/k
        sum += temp
        k += 1
    return sum

print(round(exponential(1),16))
print(round(np.exp(1),16))


# Slide 14 - Taylor Polynomial
from sympy import * #for taking derivative
import numpy as np
from math import factorial
import matplotlib.pyplot as plt


def log1plusx(x, eps):
    k = 0;
    temp = 0;
    last_term = 10  # could be any large numbers
    while k == 0 or abs(last_term) > eps:
        k += 1
        last_term = (-1) ** (k + 1) * x ** k / k
        temp += last_term
    return temp

eps = 1e-12 ; x = 0.5
print(log1plusx(x, eps)) ; print(log(1 + x))


power = 50 ; coef = [cos(0)]
for k in range(1, power + 1):
    x = symbols("x")
    dcos = diff(cos(x), x, k)
    coef.append(dcos.subs(x, 0) / factorial(k))
Taylor_poly_cos = np.poly1d(coef[::-1])
x = np.linspace(-2*np.pi, 2*np.pi, len(coef)) ; y = Taylor_poly_cos(x)

plt.plot(x , y, color="black", linewidth = 3)
plt.plot(x , np.cos(x), 'r--', linewidth = 2)
plt.ylim((-1.5, 1.5))
plt.grid(True, which='both')
plt.axhline(y=0, color='black')
plt.ylabel("Taylor Polynomial for cos(x) with n="+str(power))
plt.show()



# Slide 15 - Limits, Series and Probability
import numpy as np
from numpy import sqrt, pi, exp
from scipy.stats import binom
from math import factorial

n = np.array([1,5,13,10**2,10**3,10**4])
print(list(map(lambda x:binom.pmf(k=x, n=2*x, p=0.5), n))) ; print(1/sqrt(pi*n))

n = 50 ; a = 0
for k in range(1,n+1):
  a += (-1)**(k+1)/factorial(k)
print(a) ; print(1-exp(-1))

def exp1(x):
  sum = 1 ; temp = 1
  for k in range(1,101):
    if abs(temp) <  np.finfo(float).eps:
      return sum
    else:
      temp *= x/k
      sum += temp
  return "Computation is incomplete within the limit 100"
print(1-exp1(-1))