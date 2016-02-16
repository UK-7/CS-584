import tools
import numpy as np
import matplotlib.pyplot as plt

#
# Create the Z matrix based on the specified degree of polynomial
#
def createZ(data, i):
      Z = np.ones(data.shape)
      X = data[:, 0]
      x_new = X
      for k in range (1 to i):
            x_new = x_new * X
            Z = np.append(Z, x_new)
      return Z

#
# Strip the shuffled data set into 9/10 part training and 1/10 part test
#
def stripTestAndTraining(data):
      k = math.ceil((data.shape)/10)
      training = data[k:,]
      test = data[0:k,]
      return test, training

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

      np.random.shuffle(data)
      test, train = stripTestAndTrain(data)
      Z = createZ(train, 1)

      
      
  
  
