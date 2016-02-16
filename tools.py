# ---
# Read data and return np_array

import numpy as np

def readData(dataFile):
    data=[]
    with open(dataFile) as f:
        for index, line in enumerate(f):
          if index >= 5:
            data.append(line.strip().split(" "))
        np_data = np.array(data, dtype="f")
    return np_data
