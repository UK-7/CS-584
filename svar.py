import tools
import numpy as np
from numpy.linalg import pinv
import matplotlib.pyplot as plt
import math

#
# Create the Z matrix based on the specified degree of polynomial
#
def createZ(data, i):
      Z = np.ones(len(data))
      X = data[:, 0]
      x_new = X
      for k in range (0, i):
            x_new = x_new * X
            Z = np.column_stack((Z, x_new))
      
      return Z

#
# Strip the shuffled data set into 9/10 part training and 1/10 part test
#
def stripTestAndTraining(data):
      k = math.ceil(len(data)/10)
      training = data[k:,]
      test = data[0:k,]
      return test, training

#
# Regression using pseudo inverse
#
def regress(Z, Y):
      Z_plus = pinv(Z)
      theta = np.dot(Z_plus, Y)
      Y_hat = np.dot(Z, theta)
      return theta

#
# Plot the curve based on given theta
#

def plotTheta(theta, i):
      Z = np.ones(i)
      X = np.arange(i)
      for k in range(1, len(theta)):
            Z = np.column_stack((Z, X))
            X = X*X
      Y_hat = np.dot(Z, theta)
      return Y_hat
      

#
# Main function.
#
           
if __name__ == "__main__":
      inputFiles = ["svar-set1.txt", "svar-set2.txt", 
                    "svar-set3.txt", "svar-set4.txt"]
  
      i = 1;
      for File in inputFiles:
            data = tools.readData(File)
            plt.subplot(2, 2, i)
            plt.scatter(data[:, 0], data[:, 1], color="black")
            i = i+1
      plt.show()

      i = 1
      for File in inputFiles:
            data = tools.readData(File)
            np.random.shuffle(data)
            test, train = stripTestAndTraining(data)
            Z = createZ(train, 1)
            theta = regress(Z, train[:, 1])
            Y_hat = plotTheta(theta, 50)
            plt.subplot(2, 2, i)
            plt.scatter(data[:, 0], data[:, 1], color="red")
            plt.plot(Y_hat)
            i = i+1
      plt.show()
