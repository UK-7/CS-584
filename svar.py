
import tools
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
  inputFiles = ["svar-set1.txt", "svar-set2.txt", "svar-set3.txt", "svar-set4.txt"]
  
  i = 1;
  for File in inputFiles:
    data = tools.readData(File)
    plt.subplot(2, 2, i)
    plt.scatter(data[:, 0], data[:, 1], color="black")
    i = i+1
  plt.show()

  perm = np.random.permutation(data.shape)
  print(perm)
  
  
