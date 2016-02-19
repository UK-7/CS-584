import tools
import numpy as np
from numpy.linalg import pinv
import matplotlib.pyplot as plt
import math

from sklearn.cross_validation import KFold
from sklearn.preprocessing import PolynomialFeatures

#
# Create the Z matrix based on the specified degree of polynomial
# Input: data matrix of input vectors. Degree of polynomial i
# Return: Z matrix with added ones
#
def createZ(X, i=1):
      poly = PolynomialFeatures(i)
      X = transposeHelper(X)
      Z = poly.fit_transform(X)
      return Z

#
# Helper function to transpose the array or the matrix
# Input: X the data matrix or array
# Return: X transposed
#

def transposeHelper(X):
      if len(X.shape)>1:
            X=np.transpose(X)
      else:
            X=X.reshape(len(X),1)
      return X

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
      X_ = np.sort(X)
      X = np.sort(X)
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
# Find the errors between the given Y-Hat and the target Y
# Input: Y and Y-Hat
# Return: Normalized Mean Squared Error
#
def findError(Y_hat, Y):
      
      i = 0;
      for i in range(0, len(Y_hat)):
            error = ((Y_hat[i] - Y[i])**2)/(Y[i])**2
      error = error/len(Y_hat)
      return error

#
# Linear Regression using sklearn LinearRegeression Package
# Input: Training data set X and labels Y
# Returns: NA. Prints training and testing errors.
#

def 


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
                  Z = createZ(X[train], i)
                  theta = regress(Z, Y[train])
                  Y_hat = YHat(theta, X[train])
                  Y_hat_test = YHat(theta, X[test])
                  TrainError = TrainError + findError(theta, Y[train])
                  TestError = TestError + findError(theta, Y[test])  
            TestError = TestError / len(kf)
            TrainError = TrainError / len(kf)
            print "---------------------------"
            print "Theta: %s" % theta
            print "---------------------------"
            print "Test Error: %s" % TestError
            print "Train Error: %s" % TrainError
            


# 
# Linear Regression over entire data set without K-Fold validatioan with plot
# Input: List of input files
# Returns: NA
#

def linearRegression(inputFiles, i = 1):
      k = 1
      for File in inputFiles:
            data = tools.readData(File)
            Z = createZ(data[:, 0], i)
            theta = regress(Z, data[:, 1]) 
            Y_hat = YHat(theta, data[:, 0])
            plt.subplot(2,2,k)
            plt.scatter(data[:, 0], data[:, 1], color="green")
            X = np.sort(data[:, 0])
            plt.plot(X, Y_hat, color="red", lw=2)
            k = k + 1
      
      plt.suptitle("Single Variable Degree: %s" % i)
      plt.show()

#
# Main function.
#
       
if __name__ == "__main__":
      inputFiles = ["svar-set1.txt", "svar-set2.txt", 
                    "svar-set3.txt", "svar-set4.txt"]
      # Plot original data in a scatter plot
      plotData(inputFiles)
      
      # Single Feature first degree
      linearRegressionKFold(inputFiles)
      linearRegression(inputFiles)

      # Single Feature second degree
      linearRegressionKFold(inputFiles, i=2)
      linearRegression(inputFiles, i=2)

      # Single Feature thrid degree
      linearRegressionKFold(inputFiles, i=3)
      linearRegression(inputFiles, i=3)

      # Single Feature fourth degree
      linearRegressionKFold(inputFiles, i=4)
      linearRegression(inputFiles, i=4)
