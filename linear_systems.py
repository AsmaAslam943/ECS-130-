# ECS130 HW2 -- Solving Linear Systems
import numpy as np, scipy

def lu(A):
    """ Construct an LU decomposition of `A` without pivoting. """
    m = A.shape[0]
    if A.shape[1] != m: raise Exception('A is not square!')
    L, U = np.identity(m), A.copy()
    U = A.copy()
    for k in range(m-1):
        for i in range(k+1,m):
            L[i,k] = U[i,k]/U[k,k]
            for j in range(k+1,m):
                U[i,j] -= L[i,k]*U[k,j]

    # TODO copy your solution to HW1 Problem 7

    return L, U

def forwardsub(L, b):
    """ Solve "L x = b" with a lower triangular matrix L. """
    m = L.shape[0]
    if L.shape[1] != m: raise Exception('L is not square!')
    if b.shape != (m,): raise Exception('b is not a correctly sized vector!')

    # TODO copy your solution to HW1 Problem 8
    x = b.copy()

    for i in range(m):
        sum_Lx = np.dot(L[i,:i],x[:i])
        x[i] = (b[i]-sum_Lx)/L[i,i]

    return x

def backsub(U, b):
    """ Solve "U x = b" with an upper triangular matrix U. """
    m = U.shape[0]
    if U.shape[1] != m: raise Exception('U is not square!')
    if b.shape != (m,): raise Exception('b is not a correctly sized vector!')

    # TODO copy your solution to HW1 Problem 9
    x = b.copy()
    for i in range(m-1,-1,-1):
        sum_Ux = np.dot(U[i,i+1:],x[i+1:])
        x[i] = (b[i] -sum_Ux)/U[i,i]
        
    return x

def solve(A, b):
    """ Solve "A x = b" using the LU decomposition approach """

    # TODO copy your solution to HW1 Problem 10
    # LU decomposition and then solve for `x` using its factors.
    L,U = lu(A) #I implemented the function from lu since it performs the LU decomposition on A
    y = forwardsub(L,b)
    B = backsub(U,y) 

    return B

def forwardsub_vectorized(L, b):
    """ Solves "L x = b" with a lower triangular matrix L. """

    # TODO (optional): Implement a vectorized version of `forwardsub`
    raise NotImplementedError('forwardsub_vectorized not implemented')

def cholesky(A):
    """ Constructs a Cholesky decomposition of symmetric positive definite matrix `A`. """
    m = A.shape[0]
    if A.shape[1] != m: raise Exception('A is not square!')

    L = A.copy()

    for k in range (m): #Using the code provided, I noticed how k iterates through range of 1 to m 
        alpha = np.sqrt(L[k,k]) 
        for i in range (k+1,m):
            beta = L[i,k] / L[k,k] #we need to use beta as multiplier for row i 
            for j in range(k+1,i): 
                L[i,j] -= beta*L[j,k] #we need to subtract beta*k from i-th row each tim to alter lower triangle 
        
        L[k:, k] /= alpha #k-th column is k-th column of matrix but we have to divide alpha 
        L[k,k+1:] = 0#overwrite as 0 
    # TODO (Problem 3): Implement the Cholesky factorization algorithm
    # https://julianpanetta.com/teaching/ECS130/07-Cholesky-deck.html#/cholesky-factorization-algorithm/3

    return L

def solve_cholesky(A, b):
    """ Solve "A x = b" for a positive definite matrix `A` using the Cholesky factorization approach """

    # TODO (Problem 4): Implement the Cholesky factorization approach to solving linear systems
    m = A.shape[0] #I noted that this was provided for us in problem 3 and utilized the indexing here
    if A.shape[1] != m: raise Exception("A is not square.")
    if b.shape != (m,): raise Exception("b is incorrectly sized")

    #In cholesky, notice that we want A = L*L(transpose) so we need to call on cholesky function in problem 3
    L = cholesky(A)
    Y = forwardsub(L,b)
    b = backsub(L.T,Y)
    
    return b

def syntheticFactors(m, order='F'):
    
    """ Generate a random unit lower triangular matrix L and upper triangular matrix U. """
    L = np.array(np.random.uniform(size=(m, m)), order=order)
    U = np.array(np.random.uniform(size=(m, m)), order=order)
    L[np.triu_indices(m, 1)] = 0
    L[np.diag_indices(m)] = 1
    U[np.tril_indices(m, -1)] = 0
    return L, U

def forwardsub_scipy(L, b):
    return scipy.linalg.solve_triangular(L, b, lower=True, check_finite=False)

forwardsub_implementations = {'loop': forwardsub,
                              'vectorized': forwardsub_vectorized,
                              'scipy': forwardsub_scipy}

from relerror_unittest import *
class TestCases(RelerrorTestCase):
    def test_lu_slides_example(self):
        A = np.array([[2, 1, 1, 0],
                      [4, 3, 3, 1],
                      [8, 7, 9, 5],
                      [6, 7, 9, 8]], dtype=float)
        L, U = lu(A)
        self.requireSame(np.tril(L), np.array([[1., 0., 0., 0.], [2., 1., 0., 0.], [4., 3., 1., 0.], [3., 4., 1., 1.]]))
        self.requireSame(np.triu(U), np.array([[2., 1., 1., 0.], [0., 1., 1., 1.], [0., 0., 2., 2.], [0., 0., 0., 2.]]))

    def test_lu_synthetic(self):
        for i in range(100):
            m = np.random.randint(1, 10)
            L, U = syntheticFactors(m)
            Lc, Uc = lu(L @ U)
            self.requireSame(np.tril(Lc), L, msg='Error in L factor')
            self.requireSame(np.triu(Uc), U, msg='Error in U factor')

    def test_forwardsub_synthetic(self):
        for i in range(100):
            m = np.random.randint(1, 10)
            L, U = syntheticFactors(m)
            x = np.random.normal(size=m)
            b = L @ x
            b_orig = b.copy()
            for name, fs in forwardsub_implementations.items():
                try:
                    self.requireSame(x, fs(L, b), msg=f'Error in forwardsub implementation "{name}"')
                    self.requireSame(b, b_orig, tol=0, msg=f'forwardsub implementation "{name}" should not modify b!')
                except NotImplementedError: # Skip implementations that have not been completed
                    pass

    def test_backsub_synthetic(self):
        for i in range(100):
            m = np.random.randint(1, 10)
            L, U = syntheticFactors(m)
            x = np.random.normal(size=m)
            b = U @ x
            b_orig = b.copy()
            self.requireSame(x, backsub(U, b), msg='Error in backsub')
            self.requireSame(b, b_orig, tol=0, msg='backsub should not modify b!')

    def test_solve_synthetic(self):
        for i in range(100):
            m = np.random.randint(1, 10)
            A = np.random.uniform(size=(m, m))
            x = np.random.normal(size=m)
            b = A @ x
            self.requireSame(x, solve(A, b), msg='Error in solve')

    def test_cholesky(self):
        for i in range(10):
            m = np.random.randint(1, 8)
            L, U = syntheticFactors(m)
            Lc = cholesky(L @ L.T)
            Uc = cholesky(U.T @ U).T
            self.requireSame(np.tril(Lc), L, msg='Error in L factor')
            self.requireSame(np.triu(Uc), U, msg='Error in L factor')

    def test_solve_cholesky(self):
        for i in range(10):
            m = np.random.randint(1, 8)
            L, U = syntheticFactors(m)

            x = np.random.normal(size=m)
            b_L = (L @ L.T) @ x
            b_U = (U.T @ U) @ x

            b_L_orig = b_L.copy()
            b_U_orig = b_U.copy()

            self.requireSame(x, solve_cholesky(L @ L.T, b_L), msg='Error in solve_cholesky')
            self.requireSame(x, solve_cholesky(U.T @ U, b_U), msg='Error in solve_cholesky')

            self.requireSame(b_L, b_L_orig, tol=0, msg='solve_cholesky should not modify b!')
            self.requireSame(b_U, b_U_orig, tol=0, msg='solve_cholesky should not modify b!')

if __name__ == '__main__':
    import sys
    if '-b' in sys.argv:
        # Benchmarking
        from matplotlib import pyplot as plt
        matplotlib.use('PDF')
        import timeit

        setup_command = lambda m, impl: f'from __main__ import syntheticFactors, forwardsub_implementations; import numpy as np; L, U = syntheticFactors({m}); b = np.random.normal(size={m}); forwardsub = forwardsub_implementations["{impl}"]'
        times = {}
        for impl in forwardsub_implementations:
            try:
                sizes = [16, 32, 64, 128, 200, 256, 400, 512, 756, 1024]
                if impl != 'loop': # Include larger matrices for the faster variants
                    sizes.extend([1600, 2048, 3000, 4096])
                time = [min(timeit.repeat(f'forwardsub(L, b)', setup_command(m, impl), number=10, repeat=2)) for m in sizes]
                plt.loglog(sizes, time,  '-s', ms=4, label=impl)
            except NotImplementedError: pass # Skip implementations that have not been completed

        plt.xlabel('Matrix size $m$')
        plt.ylabel('Time per substitution (s)')
        plt.title('Forward Substitution Benchmarks')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig('forwardsub_benchmarks.pdf')
        plt.close()

    # Run the tests defined above.
    np.random.seed(0)
    unittest.main()
