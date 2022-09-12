'''No numpy version, all matrix operations are from scratch.'''

import argparse
import matplotlib.pyplot as plt

# argument settings.
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_file', help = "the path of the input file.", type = str, default = "HW1.txt")
parser.add_argument('-n', '--n', help = "the number of polynomial bases", type = int, default = 3)
parser.add_argument('-l', '--lambda_LSE', help = "the panalty term for regulated LSE", type = float, default = 0)
parser.add_argument('-t', '--iteration', help = "the iterations of Newton's method", type = int, default = 5)
args = parser.parse_args()

# m by n 2d zero matrix
def zeros(n):
    A = []

    # 2d matrixes at most
    if type(n) == tuple:
        m = n[0]
        n = n[1]
    else:
        m = n

    for i in range(m):
        temp = []
        for j in range(n):
            temp.append(0.0)
        A.append(temp)
    
    return A

# n by n 2d identity matrix
def identity(n):
    A = zeros(n)

    for i in range(n):
        A[i][i] = 1.0

    return A

# return the transpose of the input 2d matrix
def transpose(A):
    rows = len(A)
    cols = len(A[0])
    B = zeros((cols, rows))

    for i in range(rows):
        for j in range(cols):
            B[j][i] = A[i][j]

    return B

# matrix multiplication
def matmul(A, B):
    # the column number of the first matrix should be equal to the row number of the second matrix.
    assert len(A[0]) == len(B)

    C = zeros((len(A), len(B[0])))

    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                C[i][j] += A[i][k] * B[k][j]

    return C    

# matrix addition
def add(A, B):
    # the two matrixes should have the same size.
    assert len(A) == len(B)
    assert len(A[0]) == len(B[0])

    C = zeros((len(A), len(A[0])))

    for i in range(len(A)):
        for j in range(len(A[0])):
            C[i][j] = A[i][j] + B[i][j]
    
    return C

# matrix subtraction
def subtract(A, B):
    # the two matrixes should have the same size.
    assert len(A) == len(B)
    assert len(A[0]) == len(B[0])

    C = zeros((len(A), len(A[0])))

    for i in range(len(A)):
        for j in range(len(A[0])):
            C[i][j] = A[i][j] - B[i][j]
    
    return C

# scaler multiplication
def scaler_multiply(c, A):
    for i in range(len(A)):
        for j in range(len(A[0])):
            A[i][j] *= c
    
    return A

# return the design matrix A
def design_matrix(x, n):
    A = zeros((len(x), n)) #initialize a (number of obs.) * (number of polynomial bases) matrix "A" with 0
    for i in range(len(x)):
        for j in range(n):
            # the (i, j)th element in "A" is the ith observation to the power (number of polynomial bases)-1-j
            A[i][j] = x[i][0] ** (n-1-j)
    
    return A

# do the LU decomposition of the input matrix, and return an upper triangular and a lower triangular matrixes.
def LU_decomposition(A):
    assert len(A) == len(A[0]) # make sure that the matrix must be a square matrix
    n = len(A) # the order of the square matrix

    # Initialize the upper triangular matrix U, and lower triangular matrix L
    # "L" as an identity matrix and "U" as a copy of "A"
    L = identity(n)
    U = zeros(n)
    for i in range(n):
        for j in range(n):
            U[i][j] = A[i][j]

    for j in range(n-1):
        '''
        The current pivot cannot be 0.
        Otherwise, row switches would be needed and LU decomposition won't be executable.
        '''
        assert U[j][j] != 0

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

# solve Equations by LU decomposition
def solve_via_LU(L, U, ATb):
    # First, solve L (Ux) = (ATb) to obtain (Ux) by forward substitution
    Ux = zeros((len(ATb), 1))
    for i in range(len(ATb)):
        Ux[i][0] = ATb[i][0]
    
    for i in range(len(L)):
        for j in range(i):
            Ux[i][0] -= L[i][j] * Ux[j][0]
    
    # Second, solve U x = Ux to obtain x by backward elimination
    x = zeros((len(Ux), 1))
    for i in range(len(Ux)):
        x[i][0] = Ux[i][0]
    
    for i in range(len(U)-1, -1, -1):
        for j in range(len(U)-1, i, -1):
            x[i][0] -= U[i][j] * x[j][0]
        x[i][0] /= U[i][i]

    return x

# The whole process of regulated LSE method
def rLSE(A, b):
    # The gram matrix A^TA
    ATA = matmul(transpose(A), A)

    # The transpose of A, i.e. A^T, multiply the vector b
    ATb = matmul(transpose(A), b)

    # The input argument lambda "lamb" times the identity matrix
    lambdaI = scaler_multiply(lamb, identity(len(ATA)))

    # The matrix A^TA + lambda*I
    ATA_lambdaI = add(ATA, lambdaI)

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

# Get the gradient of ||Ax-b||^2
def gradient(ATA, ATb, x):
    # gradient = 2ATAx - 2ATb
    grad = subtract(scaler_multiply(2, matmul(ATA, x)), scaler_multiply(2, ATb))
    return grad

# Get the Hessian matrix of ||Ax-b||^2
def hessian(ATA):
    # hessian matrix = 2ATA
    return scaler_multiply(2, ATA)

# Get the inverse of some matrix, such as Hessian matrix, via solving LU decomposition
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
    inv = zeros(n) # initialize
    # an identity matrix
    I = identity(n)

    # column by column
    for j in range(n):
        temp = solve_via_LU(L, U, [[I[i][j]] for i in range(n)])
        for i in range(n):
            inv[i][j] = temp[i][0]
    
    return inv

# The whole process of Newton's method
def Newton(A, b, x):
    '''The iterative formula for Newton's method would look like this:
    x(n+1) = xn - (H(f(xn))^(-1) grad(f(xn))
    We will solve it in the fashion of LSE, set f(x) = ||Ax-b||^2
    Then, the gradient would be grad(f(xn)) = 2ATAx - 2ATb.
    The Hessian function would be H(f(xn)) = 2ATA.'''

    # Iteratively, obtain x from the above formula. Set 5 iteration (argument, default).
    # Get the gradient: grad(f(xn)) = 2ATAx - 2ATb

    # The gram matrix A^TA
    ATA = matmul(transpose(A), A)

    # The transpose of A, i.e. A^T, multiply the vector b
    ATb = matmul(transpose(A), b)
    
    hess = hessian(ATA)
    hess_inv = inverse(hess)

    # there is some weird behaviour. If we continuously iterate after convergented.
    # The result will start to change again, making the error bigger.
    # So, we need to set tolerence value to stop the iteration when needed.
    '''tolerence = 1.0e-10'''
    tol = 1.0e-10

    for i in range(it):
        grad = gradient(ATA, ATb, x)

        # x(n+1) = xn - (H(f(xn))^(-1) grad(f(xn))
        newx = subtract(x, matmul(hess_inv, grad))
        # if (x(n+1) - xn)^2 < tolerance value, stop iteration.
        if matmul(transpose(subtract(newx, x)), subtract(newx, x))[0][0] < tol:
            break
        x = newx
        
    return x

# The function that shows the result fitting lines and total errors of the 2 method,
# as well as shows the figure of observations and fitting lines, and saves the figure.
def results_and_visual(n, A, b, x, rlse, newton):
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
    estimated_rlse = matmul(A, rlse)
    for i in range(len(estimated_rlse)):
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
    estimated_newton = matmul(A, newton)
    for i in range(len(estimated_newton)):
        total_error += (estimated_newton[i][0] - b[i][0]) ** 2
    print("Total error: " + str(total_error))

    # reshape x, b, estimated_rlse, estimated_newton as a 1d vector
    x = [x[i][0] for i in range(len(A))]
    b = [b[i][0] for i in range(len(A))]
    estimated_rlse = [estimated_rlse[i][0] for i in range(len(A))]
    estimated_newton = [estimated_newton[i][0] for i in range(len(A))]
    
    '''visualization and save the figure'''
    # Use subplot to draw the results of the 2 method together.
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
        raw = f.readlines()
        for i in raw:
            temp = i.strip('\n').split(',')
            data.append([float(temp[0]), float(temp[1])])

    # Set the value of vector x
    x = [[a[0]] for a in data]

    # Set the value of vector b
    b = [[a[1]] for a in data]

    # Set the design matrix A
    A = design_matrix(x, n)

    '''Regulated LSE method'''
    rlse = rLSE(A, b)

    '''Newton's method'''

    # We set the initial position x0 as the origin
    x0 = zeros((n, 1))

    newton = Newton(A, b, x0)

    '''results and visualization'''
    results_and_visual(n, A, b, x, rlse, newton)