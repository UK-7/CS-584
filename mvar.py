import tools
import svar
import numpy as np
from numpy.linalg import inv, pinv
import matplotlib.pyplot as plt
import math
import sys

from sklearn.cross_validation import KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

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
            Z = pol.fit_transform(X)
            row, col = Z.shape
            theta = np.empty(col, dtype='float')
            meanDiff = 1.0
            i = 1
            print "Theta iteration %s: \n%s" % ('0', str(theta))
            while abs(meanDiff) > 1.0e-3 :
                  theta_new = recalculateTheta(theta, Z, Y)
                  diff = np.subtract(theta_new, theta)
                  meanDiff = np.mean(diff)
                  print "Theta iteration %s: \n%s" % (str(i), str(theta_new))
                  print "Diff: %s" % str(meanDiff)
                  theta = theta_new
                  i += 1
            Y_hat = np.dot(Z, theta)
            iterative_error = tools.findError(Y_hat, Y)
            errors. append(iterative_error)
      return errors
      

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
      



#            
# Main Function
#

if __name__ == "__main__":
      inputFiles = ["mvar-set1.txt", "mvar-set2.txt",
                  "mvar-set3.txt", "mvar-set4.txt"]
      print "***************************"
      print "Multi Variate Regression"
      best = np.asarray([[sys.float_info.max, 0], [sys.float_info.max, 0], [sys.float_info.max, 0], [sys.float_info.max, 0]])
      k = 2;
      '''
      for k in range (2, 5):
            errors = polyRegressionKFold(inputFiles, deg=k)
            for i in range (0, len(errors)):
                  if (best[i,0] > errors[i,0]):
                        best[i,:] = errors[i,:]
      '''
      print "\nChosen Models:"
      for i in range (0,len(best)):
            print "Best for Dat Set %s: Degree %s" % (i, best[i,1])
            print "Error: %s\n" % best[i,0]
            
      newtonRaphson(inputFiles)
