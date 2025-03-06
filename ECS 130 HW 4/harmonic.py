import numpy as np
import scipy

def exactHarmonicNumber(n):
    return scipy.special.digamma(n + 1) - scipy.special.digamma(1)

def sumForward(nmax, dtype=np.float32):
    result = dtype(0)
    for n in range(1, nmax + 1):
        result += dtype(1) / dtype(n)
    return result

def sumReverse(nmax, dtype=np.float32):
    result = dtype(0)
    for n in range(nmax, 0, -1):
        result += dtype(1) / dtype(n)
    return result

def sumKahan(nmax, dtype=np.float32):
    result = dtype(0)

    # TODO (Problem 2): Implement Kahan summation
    c = dtype(0) #we want to create another 0 value to compensate for roundoff errors  
    for n in range(1, nmax+1): #using pseudo, we want to iterate thru this range 
        y = dtype(1)/dtype(n) - c  #this will subtract the compensation from inputted val
        t = result + y #updates the new partial sum temporarily 
        c = (t - result) - y
        result = t # update result so equal to new partial sum 
    return result

implementations = {
    'Forward': sumForward,
    'Reverse': sumReverse,
    'Kahan': sumKahan
}

powmax = 23 # Test up to n = 2^powmax
import sys
if len(sys.argv) > 1:
    powmax = int(sys.argv[1])

sizes = np.unique([int(np.round(f)) for f in np.logspace(1, powmax, 1 * powmax, base=2)])
exactHn = exactHarmonicNumber(sizes)
print('Ground truth calculated...')

data = {}
for i in implementations:
    Hn = []
    Hn_single = []
    for n in sizes:
        Hn.append(implementations[i](n, dtype=np.float64))
        Hn_single.append(implementations[i](n, dtype=np.float32))
    data[i] = (np.array(Hn), np.array(Hn_single))
    print(f'{i} finished...')

from matplotlib import pyplot as plt
for i in implementations:
    Hn, Hn_single = data[i]
    plt.loglog(sizes, np.abs(exactHn - Hn) / exactHn, label=f'{i} (double)')
    plt.loglog(sizes, np.abs(exactHn - Hn_single) / exactHn, label=f'{i} (single)')

plt.ylim(1e-16, 1)
plt.title(f'Relative error in the harmonic sum')
plt.xlabel('n')
plt.ylabel(r'Relative error')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('harmonic_relerrors.pdf')
plt.close()

# Note: the following unit tests will pass for even the naive sumForward
# implementation if sufficiently small values of `powmax` are used.
# So they should be run with a `powmax` of at least 15 to  verify
# the expected growth behavior of roundoff error.
import unittest
from relerror_unittest import RelerrorTestCase
class TestCases(RelerrorTestCase):
    def test_kahan_sum(self):
        worst_relerror_kahan_double = np.max(np.abs(data['Kahan'][0] - exactHn) / np.abs(exactHn))
        worst_relerror_kahan_single = np.max(np.abs(data['Kahan'][1] - exactHn) / np.abs(exactHn))
        self.   assertLessEqual(worst_relerror_kahan_single, 2e-7)
        self.assertGreaterEqual(worst_relerror_kahan_single, 1e-8)
        self.   assertLessEqual(worst_relerror_kahan_double, 1e-15)
        self.assertGreaterEqual(worst_relerror_kahan_double, 1e-17)

unittest.main(argv=['first-arg-is-ignored'], exit=False, verbosity=1)
