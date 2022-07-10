import numpy as np


a = np.array([[0,2,3], [3,4,5]])
b = np.where(a>3, 4, 0)
print(b)