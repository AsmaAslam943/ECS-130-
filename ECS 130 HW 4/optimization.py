# ECS 130 HW 4: Numerical Optimization
import numpy as np
import linear_systems
import eig
import test_functions

def line_search(f, g, x, d, alpha_bar = 1, c_1 = 1e-4):
    """
    Determine a step size `alpha` along `d` that satisfies the Armijo condition.
    """
    alpha = alpha_bar

    # TODO (Problem 4): Implement a backtracking line search
    # with the Armijo condition (using slope-damping parameter c_1)
    while f(x+alpha * d) > f(x) + c_1 * alpha * np.dot(g,d):
        alpha /= 2

    return alpha

def gradient_descent(f, grad, x0, tol, maxit, cb=None):
    """
    Run gradient descent on the function `f` with gradient `grad` starting at `x0`.
    The algorithm terminates when ||grad(x)|| < tol * ||grad(x0)|| or when maxit
    iterations have been performed.

    If `cb` is not None, it should be a function that takes three arguments:
    the iteration number, the current iterate, and the current gradient.
    """
    x = x0.copy()
    g_0 = np.linalg.norm(grad(x))

    # TODO (Problem 5): Implement gradient descent
    for i in range(maxit): #we need to reach the MAX iterations 
        g = grad(x)

        if cb is not None: #set this pre condition in every iteration 
            cb(i,x,g)

        g_normal = np.linalg.norm(g) #normalize g and ensure it is less than tolerance 
        if g_normal < tol * g_0:
            break
        alpha_bar = f.alpha_bar_gradient_descent() #call on alpha_bar_gradient_descent 
        alpha = line_search(f,g,x,-g,alpha_bar = alpha_bar) #implement alpha_bar in the alpha 


        x = x - alpha * g
 

    return x

def newton(f, grad, hess, x0, tol, maxit, cb=None):
    """
    Run Newton's method on the function `f` with gradient `grad` and Hessian `hess`
    starting at `x0`.
    """
    x = x0.copy()
    g_0 = np.linalg.norm(grad(x))

    # TODO (Problem 6): Implement Newton's method

    for i in range(maxit): #we iterate thru the maximum iterations 
        g = grad(x) #create a gradient and a hessian matrix 
        h = hess(x)

        L = linear_systems.cholesky(h) #I iimplemented cholesky and solve_cholesky from  linear_systems from my hw 2 to call on h and g 
        d = linear_systems.solve_cholesky(h, -g) 
        alpha = line_search(f,g, x,d) #the assignment sheet said not to call on alpha_bar in line_search so 
        x = x + alpha * d #update x every time 

        if cb is not None: # remembered to call on if cb is not None in every iteration 
            cb(i,x,g)
        g_norm = np.linalg.norm(g) # checked tolerance here 
        if g_norm < tol * g_0:
            break

    return x

def newton_eig(f, grad, hess, x0, tol, maxit, cb=None):
    """
    Run a modified Newton's method on the function `f` with gradient `grad` and
    Hessian `hess` starting at `x0`. The Hessian is modified by replacing its
    eigenvalues with their absolute values.
    """
    x = x0.copy()
    g_0 = np.linalg.norm(grad(x))

    # TODO (Problem 7): Implement Newton's method with a Hessian modification
    for i in range(maxit):
        g = grad(x)
        h = hess(x) #set up the gradient and the hessian matrix

        l, Q = eig.sorted_eigendecomposition(h)
        l = np.diag(np.abs(np.diag(l)))

        rhs = -g

        d = np.dot(Q, l, np.dot(Q.T, rhs))

        alpha = line_search(f, g, x, -d)
        x = x + alpha * d

        if cb is not None:
            cb(i,x,g)

        g_norm = np.linalg.norm(g)

        if g_norm < tol * g_0:
            break 
    
    return x

methods = {
    'gradient_descent': gradient_descent,
    'newton': newton,
    'newton_eig': newton_eig
}

def run_optimization(method, function, tol=1e-8, maxit=1000, verbose=True):
    f = test_functions.functions[function]()
    opt = methods[method]

    x0 = f.initial_guess()

    iterates = []
    residuals = []

    def cb(i, x, g):
        if verbose:
            print('Iteration', i)
            print('x =', x)
            print('f(x) =', f(x))
            print('||g|| =', np.linalg.norm(g))
        iterates.append(x.copy())
        residuals.append(np.linalg.norm(g))

    # Check if `opt` takes the `hess` argument
    if 'hess' in opt.__code__.co_varnames:
        x = opt(f, f.grad, f.hess, x0, tol, maxit, cb=cb)
    else: x = opt(f, f.grad, x0, tol, maxit, cb=cb)

    return iterates, residuals

from relerror_unittest import RelerrorTestCase, TestData
class TestCases(RelerrorTestCase):
    def test_linesearch(self):
        for d in TestData('linesearch').data:
            f, g, x, d, alpha_bar, c_1, alpha_ground_truth = d
            self.requireSame(line_search(f, g, x, d, alpha_bar, c_1), alpha_ground_truth, msg="Unexpected result from line_search")

    def test_optimization(self):
        test_data = TestData('opt')
        for method, function, iterates_gt, residuals_gt in test_data.data:
            iterates, residuals = run_optimization(method, function, verbose=False)

            iterates_gt  = np.array(iterates_gt)
            residuals_gt = np.array(residuals_gt)
            iterates     = np.array(iterates)
            residuals    = np.array(residuals)

            self.assertTrue( iterates.shape ==  iterates_gt.shape, msg=f'Unexpected  iterates array shape from  {method} on {function}: {iterates.shape} != {  iterates_gt.shape}')
            self.assertTrue(residuals.shape == residuals_gt.shape, msg=f'Unexpected residuals array shape from  {method} on {function}: {residuals.shape} != {residuals_gt.shape}')

            # Overwrite NaNs with 1 to ignore these entries (a single `NaN` will cause the full relative error calculation to return `NaN`)
            iterates   [np.isnan(iterates_gt)] = 1
            iterates_gt[np.isnan(iterates_gt)] = 1
            residuals   [np.isnan(residuals_gt)] = 1
            residuals_gt[np.isnan(residuals_gt)] = 1

            self.requireSame(iterates ,  iterates_gt, msg=f'Unexpected iterates from {method} on {function}')
            self.requireSame(residuals, residuals_gt, msg=f'Unexpected residuals from {method} on {function}')

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import sys

    import unittest
    unittest.main(argv=['first-arg-is-ignored'], exit=False, verbosity=1)

    # parse command line arguments
    # Usage: python optimization.py method function
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('method', choices=methods.keys())
    parser.add_argument('function', choices=test_functions.functions.keys())
    args = parser.parse_args()

    f = test_functions.functions[args.function]()
    iterates, residuals = run_optimization(args.method, args.function)

    # Plot the function
    plot_range = f.plot_range()
    x1 = np.linspace(*plot_range[0], 200)
    x2 = np.linspace(*plot_range[1], 200)
    X1, X2 = np.meshgrid(x1, x2)
    F = f([X1, X2])

    iterates = np.array(iterates)
    plt.contourf(X1, X2, F)
    plt.plot(iterates[:, 0], iterates[:, 1], 'k-o', ms=3, label=args.method)
    plt.legend()
    plt.scatter(f.ground_truth()[0], f.ground_truth()[1], marker='*', s=200, c='y')
    plt.tight_layout()
    plt.savefig(f'{args.function}_{args.method}.pdf')
    plt.close()

    rr = np.array(residuals) / residuals[0]
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.semilogy(rr)
    plt.xlabel('Iteration')
    plt.ylabel('Residual Reducation')
    plt.ylim([min(1e-9, min(rr)), max(1e0, max(rr))])
    plt.tight_layout()
    plt.grid()
    plt.savefig(f'residual_{args.function}_{args.method}.pdf')
    plt.close()

    plt.subplot(1, 2, 2)
    errors = np.linalg.norm(iterates - f.ground_truth(), axis=1)
    plt.semilogy(errors)
    plt.xlabel('Iteration')
    plt.ylim([min(1e-9, min(errors)), max(1e2, max(errors))])
    plt.ylabel('Error')
    plt.grid()

    plt.title(f'Convergence of {args.method} on {args.function}')
    plt.tight_layout()
    plt.savefig(f'convergence_{args.function}_{args.method}.pdf')
    plt.close()
