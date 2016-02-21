import tools
import numpy as np
from numpy.linalg import pinv
import matplotlib.pyplot as plt
import math

from sklearn.cross_validation import KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
#
# Regression using pseudo inverse
# Input: Data Matrix and given labels
# Return: theta matrix - projection of Y on Z
#

def regress(Z, Y):
      Z_plus = pinv(Z)
      theta = np.dot(Z_plus, Y)
      Y_hat = np.dot(Z, theta)
      return theta

#
# Plot the curve based on given theta and specified degree
# Input: coefficient matrix theta and desired degree of polynomial
# Return: Y-hat coordinates of plotted data
#

def YHat(theta, X):
      X[np.argsort(X)]
      X_ = X
      Z = np.ones(len(X))
      for k in range(1, len(theta)):
            Z = np.column_stack((Z, X_))
            X_ = X_*X
      Y_hat = np.dot(Z, theta)
      return Y_hat

#
# Creates a scatter plot of data in files given in inputFiles
# Return: NA
# Input: inputFiles - list of data files
#

def plotData(inputFiles):
      i = 1;
      for File in inputFiles:
            data = tools.readData(File)
            plt.subplot(2, 2, i)
            plt.scatter(data[:, 0], data[:, 1], color="black")
            i = i+1
      plt.show()
      
#
# Linear Regression using sklearn LinearRegeression Package
# Input: Training data set X and labels Y
# Returns: NA. Prints training and testing errors.
#

def py_linearRegression(X, Y):
      regr = linear_model.LinearRegression(fit_intercept=False)
      
      kf = KFold(len(X), n_folds=10, shuffle=True)
      py_trainError=0
      py_testError=0
      for train, test in kf:
            regr.fit(tools.transposeHelper(X[train]), Y[train])
            py_trainError += tools.findError(regr.predict(
                              tools.transposeHelper(X[train])),
                              Y[train])
            py_testError += tools.findError(    
                              regr.predict(
                              tools.transposeHelper(X[test])), 
                              Y[test])
      py_testError  /= len(kf)
      py_trainError /= len(kf)
      print "---------------------------------"
      print "Python Functions:\n"
      print "Test Error: %s" % py_testError
      print "Train Error: %s" % py_trainError
            
       
#
# Linear Regression over 4 data sets with K-Fold validataion
# Input: List of files with the datasets 'inputFiles'. 
#        Maximum degree of polynomial 'i', 1 by default
# Return: error[training, testing]. Plot the model
#

def linearRegressionKFold(inputFiles, i=1):
      print "\nSingle Variable, Degree: %s" % i
      print "###########################"

      for File in inputFiles:
            print "==========================="
            print "Data Set %s" % File
            data = tools.readData(File)
            X = data[:, 0]
            Y = data[:, 1]
            kf = KFold(len(data), n_folds=10, shuffle=True)
            TrainError = 0
            TestError = 0
            for train, test in kf:
                  Z = tools.createZ(X[train], i)
                  theta = regress(Z, Y[train])
                  Y_hat = YHat(theta, X[train])
                  Y_hat_test = YHat(theta, X[test])
                  TrainError = TrainError + tools.findError(theta, Y[train])
                  TestError = TestError + tools.findError(theta, Y[test])  
            TestError = TestError / len(kf)
            TrainError = TrainError / len(kf)
            print "---------------------------"
            print "Test Error: %s" % TestError
            print "Train Error: %s" % TrainError
            py_linearRegression(X, Y)
      return TestError
            


# 
# Linear Regression over entire data set without K-Fold validatioan with plot
# Input: List of input files
# Returns: NA
#

def linearRegression(inputFiles, i = 1, quarters = 4, dataReduction = False):
      k = 1
      regr = linear_model.LinearRegression(fit_intercept=False)
      for File in inputFiles:
            data = tools.readData(File)
            data [np.argsort(data[:, 0])]
            limit = quarters * (len(data)/4)
            Z = tools.createZ(data[:, 0], i)
            theta = regress(Z, data[:, 1]) 
            Y_hat = YHat(theta, data[:, 0])
            plt.subplot(2,2,k)
            plt.scatter(data[:, 0], data[:, 1], color="green")
            X = data[:, 0]
            plt.plot(X, Y_hat, color="red", lw=3, label = "Original Method")
            k = k + 1
            if (dataReduction == False):
                  regr.fit(Z, data[:, 1])
                  #plt.plot(X, regr.predict(Z), color="blue", lw="1", label ="Python functions")
            else:
                  Z = tools.createZ(data[0:limit, 0], i)
                  theta = regress(Z, data[0:limit, 1])
                  Y_hat_small = YHat(theta, data[:, 0])
                  plt.plot(X, Y_hat_small, color="blue", lw = 1, label = "Reduced Data Set")
                  plt.title("Reduced Data %sn/4" % quarters)
      
      plt.suptitle("Single Variable Degree: %s" % i)
      plt.show()

#
# Main function.
#
       
if __name__ == "__main__":
      inputFiles = ["svar-set1.txt", "svar-set2.txt", 
                    "svar-set3.txt", "svar-set4.txt"]
      # Plot original data in a scatter plot
      #plotData(inputFiles)
      
      # Single Feature in various degrees
      for k in range (1,5):
            linearRegressionKFold(inputFiles, i=k)
            linearRegression(inputFiles, i=k)

      # Affect of reduced data
      for q in range (3, 1):
            linearRegression(inputFiles, i=2, quarters = q, dataReduction = True)
