import numpy as np
cimport numpy as np
cimport cython
import time
import random
import os

def write_results(user_matrix, item_matrix, result, steps, error, filename):
    f = open(filename, "w")
    for line in user_matrix:
        s = (str(line)).strip('[]')
        f.write(s)
        f.write('\n')
    f.write('--------------------------------------------------\n')
    for line in item_matrix:
        s = (str(line)).strip('[]')
        f.write(s)
        f.write('\n')
    f.write('--------------------------------------------------\n')
    for line in result:
        s = (str(line)).strip('[]')
        f.write(s)
        f.write('\n')
    f.write("Used " + str(steps) + " steps\n")
    f.write("Error was " + str(error))

#The data type of the ratings, in this case it is integers
ctypedef np.int_t RTYPE
RTYPE_T = np.int #switch these
@cython.boundscheck(False) # turn of bounds-checking for entire function
def matrix_factorization(np.ndarray[RTYPE, ndim=2] R, int steps, int num_features, double delta, double beta, double error, filename="data.txt"):
    cdef int num_users, num_items
    cdef int step, i, j, k
    cdef double E, prev_e, E_ij
    cdef double temp1, temp2
    cdef double current_rating, dot_product, current_user, current_item, gradient_users, gradient_items

    num_users = len(R)
    num_items = len(R[0])

    cdef np.ndarray[np.float64_t, ndim=2] U = np.random.rand(num_users, num_features)
    cdef np.ndarray[np.float64_t, ndim=2] I = np.random.rand(num_items, num_features)

    #The factorization code
    for step in range(0, steps):
        for i in range(0, num_users):
            for j in range(0, num_items):
                current_rating = R[i,j]
                if current_rating > 0:
                    E_ij = current_rating
                    for k in range(0, num_features):
                        E_ij -= U[i,k] * I[j,k] #Calculate the error of the (i,j) element
                    for k in range(0, num_features):
                        current_user = U[i,k]
                        current_item = I[j,k]

                        gradient_users = 2 * E_ij * current_item - beta * current_user
                        gradient_items = 2 * E_ij * current_user - beta * current_item

                        U[i,k] = current_user + (gradient_users * delta)
                        I[j,k] = current_item + (gradient_items * delta)
                    E += E_ij**2
                    for k in range(0, num_features):
                        E += beta / 2 * (current_user**2 + current_item**2)
        #print E
        if abs(prev_E - E) < error:
            #Converting the numpy array to a list so that it writes one user per line
            result = np.dot(U, I.T).tolist()
            user_matrix = U.tolist()
            item_matrix = I.tolist()
            write_results(user_matrix, item_matrix, result, step, E, filename)
            return result

        #The loop will now exit but we don't want to reset the error so we can print it
        if step != steps - 1:
            prev_E = E
            E = 0

    result = np.dot(U, I.T).tolist()
    user_matrix = U.tolist()
    item_matrix = I.tolist()
    write_results(user_matrix, item_matrix, result, step, E, filename)
    return result

#Convert a file to a numpy array
def file_to_array(filename, num_users, num_items):
    cdef np.ndarray[RTYPE, ndim=2] ratings = np.zeros([num_users, num_items], RTYPE_T)
    cdef int user, item, rating

    f = open(filename, 'r')
    for line in f:
        split = line.split()

        user = int(split[0])
        item = int(split[1])
        rating = int(split[2])

        if item <= num_items and user <= num_users:
            ratings[user - 1][item - 1] = rating
    return ratings

#The main method to call
def test(num_users, num_items, num_features):
    test_array = np.array([
        [5, 3, 0, 1],
        [4, 0, 0, 1],
        [1, 1, 0, 5],
        [1, 0, 0, 4],
        [0, 1, 5, 4],
        ])
    print "The test array is\n", test_array
    print "The factorized array is\n", np.array(matrix_factorization(test_array, 10000, 2, .0002, .02, .001))
    print "The non-zero values in the test_array should be very close to the factorized array"

    dir = os.getcwd()
    train_file = os.path.join(dir, 'ml-100k/u1.base')
    test_file = os.path.join(dir, 'ml-100k/u1.test')

    train_ratings_matrix =  file_to_array(train_file, num_users, num_items)
    test_ratings_matrix = file_to_array(test_file, num_users, num_items)

    all_results = []
    for features in xrange(1, num_features + 1):
        factored_matrix = matrix_factorization(train_ratings_matrix, 50000, features, .001, .02, .001, "data/" + str(features) + "data.txt")
        all_results.append(factored_matrix)

    Errors = []
    for features in xrange(1, num_features + 1):
        factored_matrix = all_results[features - 1]
        E = 0
        for i in xrange(0, num_users):
            for j in xrange(0, num_items):
                if test_ratings_matrix[i][j] != 0:
                    E += (factored_matrix[i][j] - test_ratings_matrix[i][j])**2
        Errors.append(E)

    print "The errors are "
    for i in range(0, len(Errors)):
        print i+1, ": ", Errors[i]
