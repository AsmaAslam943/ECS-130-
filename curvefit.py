# ECS130 HW2 -- Curve Fitting
import numpy as np
import linear_systems, qr

def vandermonde_matrix(d, x):
    """ Construct a Vandermonde matrix for the sample coordinates in vector `x` and degree `d`. """

    # TODO (Problem 5): Implement the Vandermonde matrix construction
    x = np.asarray(x)

    numRows = len(x) # (We want the number of rows to equal that of X's points because x is the given parameter)
    numCols = d+1 # (The number of columns, we need to add +1 for the degree of the polynomial)
    V = np.zeros((numRows, numCols)) # We need to create a variable that will return the same value that we are asked to provide

    for i in range(numRows): #We need to iterate within numRows 
        for j in range(numCols): #We need another loop within numCols
            V[i,j] = x[i] ** j #We need to index within var[i,j] such that x[i]^j-th power exists 
    return V #We end up returning var 

def evaluate_polynomial(a, x):
    """ Evaluate the polynomial with coefficients `a` at the sample coordinates in vector `x`. """

    # TODO (Problem 6): Implement the polynomial evaluation
    y = np.zeros_like(x,dtype=float) #We need to create a resulting vector
    deg = len(a) - 1 #We need to create a degree variable that is LESS than the length of a

    for i in range(len(x)):
        y[i] = sum(a[j] * (x[i] ** j) for j in range(deg+1))
        
    return y

def curvefit_cholesky(x, y, d):
    """
    Fits a polynomial of degree `d` to the data points in `x` and `y` using the
    Cholesky factorization approach, returning the coefficients of the
    polynomial in a vector `a`.
    """

    # TODO (Problem 7): Implement the Cholesky factorization approach
    a = np.zeros((d + 1))
    x = np.asarray(x)
    y = np.asarray(y)


    v = vandermonde_matrix(d,x) #We need to create the Vandermonde matrix

    A = np.dot(v.T,v) #We initialize ATA iwth V^transpose and V
    b = np.dot(v.T,y)

    a = linear_systems.solve_cholesky(A,b)
    return a

def curvefit_qr(x, y, d, method="ModifiedGramSchmidt"):
    """
    Fits a polynomial of degree `d` to the data points in `x` and `y` using the
    QR factorization approach, returning the coefficients of the polynomial in
    a vector `a`.
    """
    A = vandermonde_matrix(d, x)
    return qr.least_squares(A, y, method)

approaches = { 'Cholesky': curvefit_cholesky }

# Create a curve fitting approach for each QR variant.
approaches.update({method: lambda x, y, d, m=method: curvefit_qr(x, y, d, m) for method in qr.methods})

from relerror_unittest import *
class TestCases(RelerrorTestCase):
    def test_vandermonde(self):
        # Test against numpy's vander function (Problem 5)
        for d in range(12):
            for i in range(100):
                numPts = np.random.randint(5, 20)
                x = np.random.normal(size=numPts)
                V_numpy = np.vander(x, d + 1, increasing=True)
                V = vandermonde_matrix(d, x)
                self.requireSame(V, V_numpy, msg='Error in Vandermonde matrix construction')

    def test_eval_polynomial(self):
        # Test against numpy's polynomial evaluation routines (Problem 6)
        for d in range(12):
            for i in range(100):
                a = np.random.normal(size=d + 1)
                x = np.random.normal(size=20)
                y = evaluate_polynomial(a, x)
                y_numpy = np.polynomial.polynomial.polyval(x, a)
                self.requireSame(y, y_numpy, msg="Error in polynomial evaluation")

    def synthetic_curvefit_testcase(self, degree, numpts):
        x = np.random.normal(size=numpts)
        a = np.random.normal(size=degree + 1)
        y = evaluate_polynomial(a, x)
        return x, y, a

    def do_test_curvefit(self, cfit, maxdegree, method_name, tol=1e-8):
        for d in range(maxdegree):
            for i in range(100):
                x, y, a = self.synthetic_curvefit_testcase(d, np.random.randint(d + 1, 30))
                self.requireSame(a, cfit(x, y, d), tol=tol, msg=f"Error in fit polynomial's coefficients ({method_name} approach)")

    def test_curvefit_cholesky(self):
        # Test Cholesky-based curve fitting on synthetic examples (Problem 7)
        self.do_test_curvefit(curvefit_cholesky, maxdegree=5, method_name='Cholesky') # Using too high a degree will run into stability issues.

    def test_curvefit_mgs(self):
        # Test MGS QR-based curve fitting on synthetic examples (Problem 10)
        cfit = lambda x, y, d: curvefit_qr(x, y, d, method='ModifiedGramSchmidt')
        self.do_test_curvefit(cfit, maxdegree=5, method_name='MGS QR')

    def test_curvefit_householder(self):
        # Test Householder QR-based curve fitting on synthetic examples (Problem 12)
        cfit = lambda x, y, d: curvefit_qr(x, y, d, method='Householder')
        self.do_test_curvefit(cfit, maxdegree=9, tol=1e-8, method_name='Householder QR') # We can go to higher degree here...

################################################################################
# Test each method on a collection of data sets by printing the calculated
# coefficients and plotting the resulting polynomials.
################################################################################
if __name__ == '__main__':
    import sys

    if ('-t' in sys.argv) or ('--test' in sys.argv):
        # Run the tests defined above.
        #np.random.seed(0)
        unittest.main(argv=[sys.argv[0]])
        sys.exit()

    # Parse command line arguments.
    # Example usage: python curvefit_solution.py 3 --example trig --method ModifiedGramSchmidt
    #            or: python curvefit_solution.py --test
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test', action='store_true', help='run the unit tests')
    parser.add_argument('degree',    type=int,               help='degree of polynomial to fit')
    parser.add_argument('--example', type=str, default=None, help='example data set to fit')
    parser.add_argument('--method',  type=str, default=None, help='method to use for fitting')
    args = parser.parse_args()

    if args.degree <= 0:
        print('A positive degree must be specified', file=sys.stderr)
        parser.print_usage(file=sys.stderr)
        sys.exit(1)

    import matplotlib, numpy as np
    matplotlib.use('Agg') # use a DISPLAY-free backend
    from matplotlib import pyplot as plt

    np.set_printoptions(linewidth=200, edgeitems=100)

    datasets = {
        'linear':           [np.linspace(0.4, 0.6, 25),    lambda x: x],
        'quadratic':        [np.linspace(0, 1,  25),       lambda x: 0.5 * x * x],
        'noisy_quadratic':  [np.linspace(0, 1,  15),       lambda x: 0.5 * x * x + np.random.normal(scale=1e-2, size=x.size)],
        'trig':             [np.linspace(0, 1,  25),       lambda x: np.cos(4 * x)],
        'trig_exp':         [np.linspace(0, 1, 100),       lambda x: np.exp(np.cos(4 * x))],
        'lecture_8':        [np.linspace(-0.25, 1.25, 11), lambda x: [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]]
    }

    examples = args.example.split(';') if args.example else datasets.keys()
    methods  = args.method.split(';')  if args.method  else approaches.keys()
    degree   = args.degree

    for name in examples:
        if name not in datasets:
            print(f"Invalid example name '{name}'; must be one of ", list(datasets.keys()))
            continue

        x, f = datasets[name]
        sample_data = np.column_stack([x, f(x)])

        print(f'Data set {name}:')

        plt.xlabel('x')
        plt.ylabel('y')

        plt.plot(*sample_data.T, 'x', markersize=5)

        for method_name in methods:
            if method_name not in approaches:
                print(f"Invalid method name '{method_name}'; must be one of ", list(approaches.keys()))
                continue

            fitter = approaches[method_name]
            eval_x = np.linspace(-0.25, 1.25, 1000)
            coeffs = fitter(*sample_data.T, degree)
            if np.isnan(coeffs).all(): continue # Skip unimplemented methods.
            eval_y = evaluate_polynomial(coeffs, eval_x)

            plt.plot(eval_x, eval_y, label=method_name)

            ground_truth = sample_data[:, 1]
            eval_y_sample = evaluate_polynomial(coeffs, sample_data[:, 0])
            print(f'  {method_name}:  {coeffs}; error = {np.linalg.norm(ground_truth - eval_y_sample) / np.linalg.norm(ground_truth)}')

        plt.grid()
        plt.legend()
        plt.title(f'Degree {degree} Polynomial Fit to Data Set {name}')
        plt.tight_layout()
        plt.savefig(f'result_{name}_deg_{degree}.pdf')
        plt.close()
