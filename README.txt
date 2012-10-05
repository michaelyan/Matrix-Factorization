This is my recommendation system project using a matrix factorization algorithm.

To run the program:

You will need cython and numpy installed in order to successfully run this program. Run the program with the commands:

python setup.py build_ext --inplace
python test.pyx

In this case I have piped stdout to result.txt. This files shows the error associated with each feature.

The data folder contains the results of the matrix factorization. __data.txt indicates the results for that many features. Within each file the first part of data is the user matrix, the second is the item matrix and the third is the ratings matrix, each separated by a series of dashes.

The ml-100k folder contains the MovieLens data for which I predicted ratings for.
