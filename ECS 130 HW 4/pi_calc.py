import numpy as np

def piApproxBad(n, dtype=np.float32):
    # TODO (Problem 3a): translate the numerically unstable algorithm for (sqrt(1 + t^2) - 1)/t
    # approximating pi from the pseudocode in the handout. You should append the
    # approximation obtained at each iteration to `approximations`, running `n`
    # iterations in total.
    # Be careful to do all arithmetic in the `dtype` number type!
    approximations = []
    t = dtype(1/np.sqrt(3)) #set t equal to 1 / root(3) 
    for i in range(1,n+1): #run thru set indices 
        t = (dtype(np.sqrt(dtype(1)+ t*t)) - dtype(1))/t 
        scale = dtype(6) * (dtype(2)** i)* t #I decided to scale pi with 6*2^i*a
        approximations.append(scale) #I appended the sclae and cut down the iterations 
    return approximations

def piApproxGood(n, dtype=np.float32):

    #In 3B, I computed that if we multiply with the hint then the updated form is t/(sqrt(1+t^2)+1)
    # TODO (Problem 3c): implement an improved version of the approximation t/(sqrt(1+t^2)+1) 
    # algorithm using the cancellation-avoiding formula you derived in 3b.
    # You should append the approximation obtained at each iteration to
    # `approximations`, running `n` iterations in total.
    # Again, be careful to do all arithmetic in the `dtype` number type!
    approximations = []
    t = dtype(1/np.sqrt(3)) #set t as the 1/root(3) 

    for i in range(1,n+1): #iterate thru indices
        t = t/(np.sqrt(dtype(1) + t*t) + dtype(1))
        sc = dtype(6) * (dtype(2)**i)*t #scale with the 6*2^i*t
        approximations.append(sc) 
    return approximations

from relerror_unittest import RelerrorTestCase, TestData
class TestCases(RelerrorTestCase):
    def test_pi_approx(self):
        for implementation in [piApproxBad, piApproxGood]:
            formula = 'bad' if implementation == piApproxBad else 'good'
            for dtype, approx_ground_truth in TestData(f'pi_approx_{formula}').data:
                approx = np.array(implementation(len(approx_ground_truth), dtype))
                if (len(approx) != len(approx_ground_truth)):
                    self.fail(f'piApprox{formula.title()}({len(approx_ground_truth)}, {dtype}) returned a list of length {len(approx)} instead of {len(approx_ground_truth)}')
                tol = 1e-7 if dtype == np.float32 else 1e-15
                gt = np.array(approx_ground_truth)
                # Replace ground-truth NaNs with 1 to ignore these entries (a single `NaN` will cause the full relative error calculation to return `NaN`)
                approx[np.isnan(gt)] = 1
                gt[np.isnan(gt)] = 1
                self.requireSame(approx, gt, msg=f'Unexpected values from piApprox{formula.title()}({len(approx_ground_truth)}, {dtype})', tol=tol)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    n = 30
    for dtname, dtype in {'single': np.float32, 'double': np.float64}.items():
        plt.semilogy(np.abs(np.array(piApproxGood(n, dtype)) - np.pi) / np.pi, label=f'Good ({dtname})')
        plt.semilogy(np.abs(np.array(piApproxBad(n, dtype)) - np.pi) / np.pi, label=f'Bad ({dtname})')
    plt.title(r'Relative error in the approximation of $\pi$')
    plt.xlabel('Number of iterations')
    plt.ylabel('Relative error')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig('pi_error.pdf')

    import unittest
    unittest.main(argv=['first-arg-is-ignored'], exit=False, verbosity=1)
