__author__ = 'Hugh'
import numpy as np
import collections
from collections import defaultdict

def iteration(M, beta, r, constraint):
    r_prime = np.dot(r, np.transpose(np.dot(M, beta)))
    n = r.shape[0]
    A = np.array([(constraint - np.sum(r_prime)) / n] * n)
    r_prime = np.add(r_prime, A)
    r_prime = (r_prime / np.sum(r_prime)) * constraint
    return r_prime

def calculatePageRank(M, beta, constraint):
    # firstly make the columns normalised
    col_sums = M.sum(axis=0)
    M = M / col_sums[np.newaxis, :]

    # r
    number_of_nodes = M.shape[0]
    r0 = np.array([1.0 / number_of_nodes] * number_of_nodes)
    r0 = (r0 / np.sum(r0)) * constraint
    r = [r0]

    while True:
        r1 = iteration(M, beta, r[len(r) - 1], constraint)
        r.append(r1)
        if np.sum(np.absolute(np.subtract(r[len(r) - 1], r[len(r) - 2]))) < 0.0001:
            break

    return r

def primes(n):
    primfac = []
    d = 2
    while d*d <= n:
        while (n % d) == 0:
            primfac.append(d)  # supposing you want multiple factors repeated
            n /= d
        d += 1
    if n > 1:
       primfac.append(n)
    return list(set(primfac))

def convertIntoPrimeDivisorsAndCount(values):
    primes_count = defaultdict(int)
    for x in values:
        prime_list = primes(x)
        for prime in prime_list:
            primes_count[prime] += x

    return collections.OrderedDict(sorted(primes_count.items()))

print("Question 1")
M = np.array([[0.0, 1.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
M = np.transpose(M)
r = calculatePageRank(M, 0.7, 3.0)
print("done")

print("Question 2")
M = np.array([[0.0, 1.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
M = np.transpose(M)
r = calculatePageRank(M, 0.85, 2.0)
print("done")

print("Question 3")
M = np.array([[0.0, 1.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
M = np.transpose(M)
r = calculatePageRank(M, 1.0, 3.0)
print("done")

print("Question 4")
values = [15, 21, 24, 30, 49]
primes_count = convertIntoPrimeDivisorsAndCount(values)
print("done")
