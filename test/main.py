
import numpy as np

data = np.array([1, -2, 3, 4])
print(data >= 0)

data = np.array(data >= 0, dtype=np.int)
print(data)