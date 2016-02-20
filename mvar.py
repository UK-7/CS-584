import tools
import svar
import numpy as np
from numpy.linalg import pinv
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
# Returns: NA. 3D plot if two variables
#
def polyRegressionKFold(inputFiles, deg=2):
      print "***************************"
      print "Degree: %s" % deg
      errors = []
      for File in inputFiles:
            print "___________________________"
            print "Data Set: %s" % File
            data = tools.readData(File)
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
                  Y_hat = np.dot(np.sort(Z, axis = 1), theta)
                  Y_hat_test = np.dot(np.sort(Z_test, axis = 1), theta)
                  TrainError += mean_squared_error(Y[train], Y_hat)
                  TestError += mean_squared_error(Y[test], Y_hat_test)
            TestError /= len(kf)
            TrainError /= len(kf)
            errors.append([TestError, deg])
            print "---------------------------"
            print "Test Error: %s" % TestError
            print "Train Error: %s" % TrainError
      return np.asarray(errors)
                  
            
# Main Function
#

if __name__ == "__main__":
      inputFiles = ["mvar-set1.txt", "mvar-set2.txt",
                  "mvar-set3.txt", "mvar-set4.txt"]
      print "***************************"
      print "Multi Variate Regression"
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
            
      
