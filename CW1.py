import os
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import inv

def load_points_from_file(filename):
    """Loads 2d points from a csv called filename
    Args:
        filename : Path to .csv file
    Returns:
        (xs, ys) where xs and ys are a numpy array of the co-ordinates.
    """
    points = pd.read_csv(filename, header=None)
    return points[0].values, points[1].values


def view_data_segments(xs, ys):
    """Visualises the input file with each segment plotted in a different colour.
    Args:
        xs : List/array-like of x co-ordinates.
        ys : List/array-like of y co-ordinates.
    Returns:
        None
    """
    assert len(xs) == len(ys)
    assert len(xs) % 20 == 0
    len_data = len(xs)
    num_segments = len_data // 20
    colour = np.concatenate([[i] * 20 for i in range(num_segments)])
    plt.set_cmap('Dark2')
    plt.scatter(xs, ys, c=colour)
    plt.show()

#__________________________________________________________________________


def view_data_segments_withFittedLines(xs_mat, ys_mat, y_cal):
    """Visualises the input file with the final fitted lines plotted on top."""

    plt.scatter(xs, ys)

    for i in range (0,len(xs_mat)):
        plt.plot(xs_mat[i,:], y_cal[i,:],'r') #the line estimated
    plt.show()



def seperate_lineSegments(xs,ys):

    """Seperates signals into line segments.Every line segment is a different row.
    Args:
        xs : List/array-like of x co-ordinates.
        ys : List/array-like of y co-ordinates.
    Returns:
        xs_mat : List with every row a different line segment of x co-ordinates
        ys_mat : List with every row a different line segment of y co-ordinates
    """
    len_data = len(xs)
    num_segments = len_data // 20
    xs_mat = np.empty([num_segments, 20])
    ys_mat = np.empty([num_segments, 20])
    for i in range(num_segments):
        xs_mat[i] = xs[20*i:(20*(i+1))] #0 to 19 , 20 to 39
        ys_mat[i] = ys[20*i:(20*(i+1))]

    return xs_mat, ys_mat


def fit_foundParameters(order, W, x_test):

    #calculate estimated output y with fitted parameters
    y_cal = W[0] + W[1]*x_test
    for i in range(2, order+1): #1,2,..,order
        y_cal = y_cal + W[i]*(x_test**i)

    return y_cal



#finds parameters and calculates estimated output given the order
def lin_pol_parameter_fitting(order, xs, ys):


    """finds parameters and calculates estimated output given the order of the function e.g. y = a + bx + cx^2 + ... .
    Args:
        order : order of function that we want to fit (linear / polynomial)
        xs    : List/array-like of x co-ordinates.
        ys    : List/array-like of y co-ordinates.
    Returns:
        y_cal : Estimated y
        W     : Estimated parameters (a,b, ...)
    """


    #assuming y = a + bx + cx^2 + ...
    #finding matrix form of least squares: A = (X'.X)^(-1) . X' . Y

    n = len(xs)
    ones = np.ones((n,1))
    x2 = np.c_[ ones,xs ]
    for i in range(2, (order+1)): #1,2,..,order
        x2 = np.c_[ x2,xs**i ]


    #W =(X'.X)^-1 . (X'.Y)
    W = inv(x2.transpose() @ x2) @ (x2.transpose() @ ys)

    #calculate estimated output y
    y_cal = W[0] + W[1]*xs
    for i in range(2, order+1): #1,2,..,order
        y_cal = y_cal + W[i]*(xs**i)

    return y_cal, W





#finds parameters and calculates estimated output
def sin_parameter_fitting(xs, ys):

    """finds parameters and calculates estimated output to fit sinusoid.
    e.g. a + bsin(x)
    Args:
        xs    : List/array-like of x co-ordinates.
        ys    : List/array-like of y co-ordinates.
    Returns:
        y_cal : Estimated y
        W     : Estimated parameters (a,b)
    """

    n = len(xs)
    ones = np.ones((n,1))
    x2 = np.c_[ ones,np.sin(xs)]

    #calculates parameters W
    #A =(X'.X)^-1 . (X'.Y)
    W = inv(x2.transpose() @ x2) @ (x2.transpose() @ ys)

    #calculate estimated output y
    #y = a + b sin(x)
    y_cal = W[0] + (W[1]*np.sin(xs))


    return y_cal, W



def squared_error(y_cal, ys):

    """finds squared error between two entities
    Args:
        y_cal : List/array-like of y calculated
        ys    : List/array-like of y co-ordinates.
    Returns:
        error : squared error
    """
    error = np.sum ((ys - y_cal)**2)#.mean()

    return error

def validation_training_set (i,xs,ys,validation_range):

    #__________initialization of arrays_________
    X_training = np.empty(20-validation_range)
    Y_training = np.empty(20-validation_range)
    X_validation = np.empty(validation_range)
    Y_validation = np.empty(validation_range)
    training_set = np.empty(20-validation_range)
    #___________________________________________

    training_set = np.linspace(i*20,((i+1)*20)-1,(20-validation_range), dtype=int)

    #++++++++++++++++++++ create TRAINING & VALIDATION ++++++++++++++++++++
    # create TRAINING set
    j = 0
    for x in training_set:
        X_training[j] = xs[x]
        Y_training[j] = ys[x]
        j+=1

    # create VALIDATION set
    j = 0
    ind = np.array(range(i*20, ((i+1)*20)))
    for x in ind:
        if x not in training_set:
            X_validation[j] = xs[x]
            Y_validation[j] = ys[x]
            j+=1
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # NOW WE HAVE TRAINING AND VALIDATION SETS
    return X_training, Y_training, X_validation, Y_validation



def cross_Validation_bestOrder (X_training, Y_training, X_validation, Y_validation):


    error = np.empty([2])
    error_validation = np.empty([2])

    #~~~~~~~~~~~~~~~~~~~~ Estimates cross validation error for orders 1-2 ~~~~~~~~~~~~~~~~~~~~
    for i in range (1, 3):

        #Fitting in training set
        y_cal, W = lin_pol_parameter_fitting(i, X_training, Y_training) #finds parameters
        #we want to use this parameters to see what's the output of the new validation set inputs
        #and what is the error that they have according to their order

        y_cal_validation = fit_foundParameters(i, W, X_validation)
        #print (y_cal_validation)


        error[i-1]=squared_error(y_cal, Y_training) #training error
        error_validation[i-1]=squared_error(y_cal_validation, Y_validation) #validation error
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    cross_validation_best_order = np.argmin(error_validation)+1;
    return cross_validation_best_order;





# .................................... MAIN ..........................................
#FITS ONLY LINEAR, POLYNOMIAL AND UNKNOWN W/ CROSS VALIDATION


fileName = sys.argv[1]

xs, ys = load_points_from_file("train_data/" + fileName) #load data
#view_data_segments(xs, ys) #plot data


len_data = len(xs)
num_segments = len_data // 20
validation_range = 14 #range of test validation set
xs_mat, ys_mat = seperate_lineSegments(xs,ys) #creates 2d matrix (every row is the data of a different line)

#Initialization of arrays
error = np.empty([num_segments+1, 3]) #0:linear 1:pol 2:sin

y_lin = np.empty([num_segments+1, 20])
y_pol = np.empty([num_segments+1, 20])
y_sin = np.empty([num_segments+1, 20])
y_final = np.empty([num_segments+1, 20])
cross_validation_best_order = np.empty([num_segments],  dtype=int)

errorTotal = 0


for i in range(0,num_segments):

    # LINEAR
    y_lin[i,:], W_lin = lin_pol_parameter_fitting(1, xs_mat[i,:], ys_mat[i,:])
    error[i][0] = squared_error(y_lin[i,:], ys_mat[i,:])

    # POLYNOMIAL
    y_pol[i,:], W_pol = lin_pol_parameter_fitting(2, xs_mat[i,:], ys_mat[i,:])
    error[i][1] = squared_error(y_pol[i,:], ys_mat[i,:])

    # UNKNOWN
    y_sin[i,:], W_sin = sin_parameter_fitting(xs_mat[i,:], ys_mat[i,:])
    error[i][2] = squared_error(y_sin[i,:], ys_mat[i,:])

    index = np.argmin(error[i][:])
    if (index == 2):
        #print("sin")
        y_final[i,:] = y_sin[i,:]

    else:

        # ++++++++++++++++++++++++++++++++++++ cross validation ++++++++++++++++++++++++++++++++++++++++++++++

        #__________initialization of arrays_________
        X_training = np.empty(20-validation_range)
        Y_training = np.empty(20-validation_range)
        X_validation = np.empty(validation_range)
        Y_validation = np.empty(validation_range)
        #___________________________________________

        #CREATES TRAINING & VALIDATION SETS
        X_training, Y_training, X_validation, Y_validation = validation_training_set (i,xs,ys,validation_range)


        #FIT TRAINING SET WITH DIFFERENT ORDERS & FIND BEST ORDER
        cross_validation_best_order[i] = cross_Validation_bestOrder (X_training, Y_training, X_validation, Y_validation);
        #print (cross_validation_best_order[i])


        #VIEW FITTED LINE WITH BEST ORDER FROM CROSS VALIDATION
        y_final[i,:], W_bestPol = lin_pol_parameter_fitting(cross_validation_best_order[i], xs_mat[i,:], ys_mat[i,:]) #finds parameters
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


    errorTotal = errorTotal + squared_error(y_final[i,:], ys_mat[i,:])


print(errorTotal)


if (len(sys.argv)>2) :
    if (sys.argv[2] == "--plot"):
        view_data_segments_withFittedLines(xs_mat, ys_mat, y_final)
#print(cross_validation_best_order)
