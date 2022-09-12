import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_file', help = "the path of the input file.", type = str, default = "HW1.csv")
parser.add_argument('-n', '--n', help = "the number of polynomial bases", type = int, default = 2)
parser.add_argument('-l', '--lambda_LSE', help = "the panalty term for regulated LSE", type = float, default = 0)
parser.add_argument('-t', '--iteration', help = "the iterations of Newton's method", type = int, default = 5)
args = parser.parse_args()

def design_matrix(x, n):
    A = np.zeros((len(x), n)) #initialize a (number of obs.) * (number of polynomial bases) matrix "A" with 0

    for i in range(len(x)):
        for j in range(n):
            # the (i, j)th element in "A" is the ith observation to the power (number of polynomial bases)-1-j
            A[i][j] = x[i] ** (n-1-j)

    return A

def LU_decomposition(A):
    assert len(A) == len(A[0]) # make sure that the matrix must be a square matrix
    n = len(A) # the order of the square matrix

    # Initialize the upper triangular matrix U, and lower triangular matrix L
    # "L" as an identity matrix and "U" as a copy of "A"
    L = np.identity(n)
    U = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            U[i][j] = A[i][j]

    for j in range(n-1):
        '''
        The current pivot cannot be 0.
        Otherwise, row switches would be needed and LU decomposition won't be executable.
        '''
        assert A[j][j] != 0

        # Do the LU decompostion
        for i in range(j+1, n):
            # For any column j, if The (i,j)th entry where i > j is not 0
            # then we need to do the gaussian elimination and record the process
            # So that we can reverse the sign of the process and store it into corresponding entry in "L"
            if A[i][j] != 0:
                r = U[i][j] / U[j][j]
                for k in range(n):
                    U[i][k] += -r * U[j][k]
                L[i][j] = r
    
    return L, U

def solve_via_LU(L, U, ATb):
    # First, solve L (Ux) = (ATb) to obtain (Ux) by forward substitution
    Ux = np.array([value for value in ATb])
    for i in range(len(L)):
        for j in range(i):
            Ux[i] -= L[i][j] * Ux[j]
    
    # Second, solve U x = Ux to obtain x by backward elimination
    x = np.array([value for value in Ux])
    for i in range(len(U)-1, -1, -1):
        for j in range(len(U)-1, i, -1):
            x[i] -= U[i][j] * x[j]
        x[i] /= U[i][i]

    return x

def rLSE(A, b):
    # The gram matrix A^TA
    ATA = np.matmul(A.T, A)

    # The transpose of A, i.e. A^T, multiply the vector b
    ATb = np.matmul(A.T, b)

    # The input argument lambda "lamb" times the identity matrix
    lambdaI = lamb * np.identity(ATA.shape[0])

    # The matrix A^TA + lambda*I
    ATA_lambdaI = np.add(ATA, lambdaI)

    '''Solve for (ATA+lambdaI)x = ATb:
    First, do the LU decomposition on the square matrix ATA+lambdaI, obtain matrixes L, U
    Second, solve the (LU)x = (ATb) by forward substitution, and then backward elimination
    Finally, obtain x, which is the rLSE parameters.
    '''
    # Get the LU decomposition of the matrix ATA+lambdaI
    L, U = LU_decomposition(ATA_lambdaI)
    
    # Obtain the rLSE parameters. by solving the equation (LU)x=(ATb)
    rlse = solve_via_LU(L, U, ATb)

    return rlse

def gradient(ATA, ATb, x):
    # gradient = 2ATAx - 2ATb
    grad = np.subtract(2 * np.matmul(ATA, x), 2 * ATb)
    return grad

def hessian(ATA):
    # hessian matrix = 2ATA
    return 2 * ATA    

def inverse(hess):
    # find the inverse of a square matrix
    # do the LU decomposition of the input hessian matrix
    L, U = LU_decomposition(hess)

    # check if the hessian matrix is invertible
    # by using the fact that det(hess) = det(L)det(U) = det(U), which must not be 0.
    det = 1
    for i in range(n):
        det *= U[i][i]
    assert det != 0

    # solve the inverse by: hess (hess_inv) = I, that is LU (hess_inv) = I
    inv = np.zeros((n, n), dtype = float) # initialize
    # an identity matrix
    I = np.identity(n)

    # column by column
    for j in range(n):
        inv[:, j] = solve_via_LU(L, U, I[:, j])
    
    return inv

def Newton(A, b, x):
    '''The iterative formula for Newton's method would look like this:
    x(n+1) = xn - (H(f(xn))^(-1) grad(f(xn))
    We will solve it in the fashion of LSE, set f(x) = ||Ax-b||^2
    Then, the gradient would be grad(f(xn)) = 2ATAx - 2ATb.
    The Hessian function would be H(f(xn)) = 2ATA.'''

    # Iteratively, obtain x from the above formula. Set 5 iteration (argument, default).
    # Get the gradient: grad(f(xn)) = 2ATAx - 2ATb

    # The gram matrix A^TA
    ATA = np.matmul(A.T, A)

    # The transpose of A, i.e. A^T, multiply the vector b
    ATb = np.matmul(A.T, b)
    
    hess = hessian(ATA)
    hess_inv = inverse(hess)
    for i in range(it):
        grad = gradient(ATA, ATb, x)
        # x(n+1) = xn - (H(f(xn))^(-1) grad(f(xn))
        x = np.subtract(x, np.matmul(hess_inv, grad))
    
    return x

def results_and_visual(n, A, b, rlse, newton):
    '''print out the results, including fitting line and total error, of rLSE method'''
    # fitting line
    fitting_line_rlse = ''
    for i in range(n):
        if i == n-1:
            fitting_line_rlse += str(rlse[i][0])
        else:
            fitting_line_rlse += str(rlse[i][0]) + " X^" + str(n-1-i) + " + "
    print("rLSE method:\nFitting Line: " + fitting_line_rlse)

    # total error
    total_error = 0
    estimated_rlse = np.matmul(A, rlse)
    for i in range(estimated_rlse.shape[0]):
        total_error += (estimated_rlse[i][0] - b[i][0]) ** 2
    print("Total error: " + str(total_error))
    
    '''print out the results, including fitting line and total error, of Newton's method'''
    # fitting line
    fitting_line_newton = ''
    for i in range(n):
        if i == n-1:
            fitting_line_newton += str(newton[i][0])
        else:
            fitting_line_newton += str(newton[i][0]) + " X^" + str(n-1-i) + " + "
    print("Newton's method:\nFitting Line: " + fitting_line_newton)

    # total error
    total_error = 0
    estimated_newton = np.matmul(A, newton)
    for i in range(estimated_newton.shape[0]):
        total_error += (estimated_newton[i][0] - b[i][0]) ** 2
    print("Total error: " + str(total_error))

    '''visualization and save the figure'''
    plt.subplot(211)
    plt.title("rLSE: n="+str(n)+", lambda="+str(lamb))
    plt.scatter(x, b, c = 'r')
    plt.plot(x, estimated_rlse)

    plt.subplot(212)
    plt.title("Newton: n="+str(n)+", iter="+str(it))
    plt.scatter(x, b, c = 'r')
    plt.plot(x, estimated_newton)
    plt.tight_layout() # to prevent the lower title overlays the upper figure.

    plt.savefig("HW1.png")
    plt.show()


# entry point of the program
if __name__ == "__main__":
    # Some global variables storing the set arguments
    n = args.n
    assert n > 0
    file = args.input_file
    lamb = args.lambda_LSE
    it = args.iteration
    assert it > 0

    # Store input data points in a 2d list "data"
    data = []
    with open(file, 'r') as f:
        csvobj = csv.reader(f)
        for row in csvobj:
            data.append([float(row[0]), float(row[1])])
    
    # turn "data" into a 2d numpy array
    data = np.array(data)

    # Set the value of vector x
    x = data[:, 0]

    # Set the value of vector b
    b = data[:, 1]
    # and reshape it as a column vector
    b = b.reshape((len(b), 1))

    # Set the design matrix A
    A = design_matrix(x, n)

    '''Regulated LSE method'''
    rlse = rLSE(A, x, b)

    '''Newton's method'''

    # We set the initial position x0 as the origin
    x0 = np.zeros((n, 1), dtype = float)

    newton = Newton(A, b, x0)

    '''results and visualization'''
    results_and_visual(n, A, b, rlse, newton)