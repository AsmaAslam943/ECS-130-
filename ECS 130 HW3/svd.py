import eig as eig
import numpy as np

def svd(A, tol=1e-8, maxIterations=1000):
    """
    Construct the SVD of a potentially rectangular matrix A.
    """
    m, n = A.shape
    k = min(m, n)
    #U, sigma, V = np.eye(m, k), np.zeros(k), np.eye(n, k) # TODO: Replace these with the actual SVD.
    # TODO (Problem 5): compute the "thin" SVD of A by performing an eigenvalue decomposition
    # of either A^T A or A A^T depending on the shape of A.

    if n <= m: #we need to create a base case to find smaller  
        ATA = A.T @ A #finds ATA 
        sigma, V = eig.sorted_eigendecomposition(ATA, tol = tol, descending = True)
        sigma = np.sqrt(sigma) #we need to call on eig.sorted_decomposition so we can have the eigenvalues and eigenvectors  
        # in eig.sorted_decompostion, given parameters are A, tol=1e-8, descending = True so eigenvals are descending
        # gets sqrt of the eigenvalues for singular values in sigma  
        
        
        U = A @ V #initializing U 
        U = U/np.linalg.norm(U, axis=0) #normalization 
    else:
        AAT = A @ A.T #computes AAT 
        sigma, U = eig.sorted_eigendecomposition(AAT, tol = tol, descending = True)
        sigma = np.sqrt(sigma) #again, calling on eig.sorted_decomposition so we can get needed eigenvals and eigvecs  
        
        V = A.T @ U #this time we initialize V instead of U
        V /= np.linalg.norm(V, axis = 0) #normalization 
        
    return U, sigma, V


def relerror(a, b):
    """ Calculate the relative difference between vectors/matrices `a` and `b`. """
    return np.linalg.norm(a - b) / np.linalg.norm(b)

import unittest
class TestCases(unittest.TestCase):
    def requireSame(self, a, b, tol = 1e-6):
        self.assertLess(relerror(a, b), tol)

    def test_svd(self):
        # Generate a random matrix.
        for i in range(100):
            m = np.random.randint(1, 30)
            n = np.random.randint(1, 30)
            A = np.random.normal(size=(m, n))
            k = min(m, n)
            U, sigma, V = svd(A)
            # Check that U is orthogonal.
            self.requireSame(U.T @ U, np.eye(k))
            # Check that V is orthogonal.
            self.requireSame(V.T @ V, np.eye(k))
            # Check that U, sigma, V reconstruct A.
            self.requireSame(A, U @ np.diag(sigma) @ V.T)

import unittest
if __name__ == '__main__':
    # Run the unit tests defined above.
    unittest.main()
