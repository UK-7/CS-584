import tools
import svar
import numpy as np
from numpy.linalg import inv, pinv
import matplotlib.pyplot as plt
import math
import sys
import time

from sklearn.cross_validation import KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import euclidean

#
# Regression using pseudo inverse
# Input: Data Matrix and given labels
# Return: theta matrix - projection of Y on Z
#

def regress(Z, Y):
      Z_plus = pinv(Z)
      theta = np.dot(Z_plus, Y)
      return theta

#
# Linear Regression for multivarate data
# Input: inputFiles - List of input files for multivariate data
#        i - degree of required polynomial
# Returns: Mean Squared Error for each data set for the given polynomial degree
#
def polyRegressionKFold(inputFiles, deg=2):
      print "***************************"
      print "Degree: %s" % deg
      start_time = time.time()
      errors = []
      for File in inputFiles:
            print "___________________________"
            print "Data Set: %s" % File
            data = tools.readData(File)
            data = data[np.argsort(data[:,0])]
            X = data[:, :-1]
            Y = data[:, len(data[1,:]) - 1]
            kf = KFold(len(data), n_folds = 10, shuffle = True)
            TrainError = 0
            TestError = 0
            for train, test in kf:
                  pol = PolynomialFeatures(deg)
                  Z = pol.fit_transform(X[train]) 
                  Z_test = pol.fit_transform(X[test])     
                  theta = regress(Z, Y[train])
                  Y_hat = np.dot(Z, theta)
                  Y_hat_test = np.dot(Z_test, theta)
                  TrainError += mean_squared_error(Y[train], Y_hat)
                  TestError += mean_squared_error(Y[test], Y_hat_test)
            TestError /= len(kf)
            TrainError /= len(kf)
            errors.append([TestError, deg])
            print "---------------------------"
            print "Test Error: %s" % TestError
            print "Train Error: %s" % TrainError
      time_taken = start_time - time.time()
      print "Time Taken for primal: %s" % str(time_taken)
      return np.asarray(errors)
                  

# 
# Newton-Raphson method to iteratively calculate the parameter values
# Input: set of inputFiles
# Return: A parameter matrix theta calculated iteratively
#

def newtonRaphson(inputFiles):
      pol = PolynomialFeatures(2)
      errors = []
      for File  in inputFiles:
            data = tools.readData(File)
            X = data[:, :-1]
            Y = data[:, -1]
            kf = KFold(len(Y), n_folds = 10)
            trainError = 0
            testError = 0
            for train, test in kf:
                  Z = pol.fit_transform(X[train])
                  row, col = Z.shape
                  theta = np.empty(col, dtype='float')
                  meanDiff = 1.0
                  i = 1
                  #print "Theta iteration %s: \n%s" % ('0', str(theta))
                  while abs(meanDiff) > 1.0e-15 :
                        theta_new = recalculateTheta(theta, Z, Y[train])
                        diff = np.subtract(theta_new, theta)
                        meanDiff = np.mean(diff)
                        #print "Theta iteration %s: \n%s" % (str(i), str(theta_new))
                        #print "Diff: %s" % str(meanDiff)
                        theta = theta_new
                        i += 1
                  Z_test = pol.fit_transform(X[test])
                  Y_hat_test = np.dot(Z_test, theta)
                  Y_hat = np.dot(Z, theta)
                  trainError += tools.findError(Y_hat, Y[train])
                  testError += tools.findError(Y_hat_test, Y[test])
            trainError = trainError/len(kf)
            testError = testError/len(kf)
            iterative_error = [trainError, testError]
            errors. append(iterative_error)
      return np.asarray(errors)
      

# 
# theta_new = theta - inv(Z_trans*Z) (Z_trans(Y_hat - Y))
# Input: theta, Z and label Y
# Return: New value of parameters
#

def recalculateTheta(theta, Z, Y):
      Y_hat = np.dot(Z, theta)
      gradient = np.dot( \
                  inv(np.dot(np.transpose(Z), Z)), \
                  np.dot(np.transpose(Z), np.subtract(Y_hat, Y))\
                  )
      theta = np.subtract(theta, gradient)
      return theta

# 
# Returns normalized MSE for all second degree polynomial over all data
# Input: List of input files
# Return: List of errors for each input file
# 

def polyRegression(inputFiles):
      pol = PolynomialFeatures(2)
      errors = []
      for Files in inputFiles:
            data = tools.readData(Files)
            data = data[np.argsort(data[:, 0])]
            X = data[:, :-1]
            Y = data[:, -1]
            kf = KFold(len(Y), n_folds = 10)
            trainError = 0
            testError = 0
            for train, test in kf:
                  Z = pol.fit_transform(X[train])
                  theta = regress(Z, Y[train])
                  Y_hat = np.dot(Z, theta)
                  Z_test = pol.fit_transform(X[test])
                  Y_hat_test = np.dot(Z_test, theta)
                  trainError += tools.findError(Y_hat, Y[train])
                  testError += tools.findError(Y_hat_test, Y[test])
            testError = testError/len(kf)
            trainError = trainError/len(kf)
            explicit_error = [trainError, testError]
            errors.append(explicit_error)
      return np.asarray(errors)

#
# Compute theta matrix using the given alpha matrix
# Input: alpha matrix and data matrix X
# Return: theta
#

def computeTheta(alpha, X):
      theta = np.dot(np.transpose(X), alpha)
      return theta

#
# computeAlpha calcuates the value of apha using ridge regression with lamda
# Input: Grahm Matrix G, Label materix Y, and value of Lambda
# Return: alpha
#

def computeAlpha(G, Y, lamba):
      row, col = G.shape
      I = np.identity(row)
      alpha = np.dot(inv(G + lamba * I), Y)
      return alpha
      
#
# Gaussain Kernel Method returns the Grahm Matrix for given X and Sigma
# Input: Data matrix X and value of sigma
# Return: The final Grahm Matrix
#

def gaussian_function(X, sigma):
      G = [[0 for j in range(X.shape[0])] for j in range(X.shape[0])]
      for i in range(X.shape[0]):
            for k in range(X.shape[0]):
                  a = euclidean(X[i,:],X[k,:])/(2 * sigma * sigma)
                  G[i][k] = np.exp(-a)
      return np.asarray(G)
# 
# This method solve the regression using the dual problem.
# Input: List of input files
# Return: List of Normalized MSE for each data set
#

def dualProblem(inputFiles):
      errors = []
      start_time = time.time()
      for File in inputFiles:
            data = tools.readData(File)
            X = data[:, :-1]
            Y = data[:-1]
            kf = KFold(len(Y), n_folds = 10)
            trainError = 0
            testError = 0
            for train, test in kf:
                  G = gaussian_function(X[train], 0.05)
                  print "Done with G!"
                  alpha = computeAlpha(G, Y[train], 0.05)
                  theta = computeTheta(alpha, X[train])
                  Y_hat = np.dot(X[train], theta)
                  Y_hat_test = np.dot(X[test], theta)
                  trainError += tools.findError(Y_hat, Y[train])
                  testError += tools.findError(Y_hat_test, Y[test])
            trainError = trainError/len(kf)
            testError = testError/len(kf)
            error = [trainError, testError]
            errors.append(error)
      time_taken = start_time - time.time()
      print "Time Taken for all data sets: %s" % str(time_taken)
      return np.asarray(errors)

                        
#            
# Main Function
#

if __name__ == "__main__":
      inputFiles = ["mvar-set1.txt", "mvar-set2.txt",
                  "mvar-set3.txt", "mvar-set4.txt"]
      print "***************************"
      print "Multi Variate Regression"
      '''
      best = np.asarray([[sys.float_info.max, 0], [sys.float_info.max, 0], [sys.float_info.max, 0], [sys.float_info.max, 0]])
      k = 2;
      for k in range (2, 5):
            errors = polyRegressionKFold(inputFiles, deg=k)
            for i in range (0, len(errors)):
                  if (best[i,0] > errors[i,0]):
                        best[i,:] = errors[i,:]
      print "\nChosen Models:"
      for i in range (0,len(best)):
            print "Best for Dat Set %s: Degree %s" % (i, best[i,1])
            print "Error: %s\n" % best[i,0]
      '''
            
      iterative_er = newtonRaphson(inputFiles)
      explicit_er = polyRegression(inputFiles)
      i = 0
      print "\nError Comparisons:"
      print "__________________\n"
      print "File\t\tIterative\t\tExplicit"
      for f in inputFiles:
            print "%s:\t%s\t\t%s" % (f, iterative_er[i][1], explicit_er[i][1])
            i += 1
      
      d_error = dualProblem(["mvar-set1.txt"])
      p_error = polyRegressionKFold(["mvar-set1.txt"])
      print "Error comparison:"
      print "Dual Problem: %s" % d_error[0][0]
      print "Primal Problem: %s" % p_error[0][0]
      
