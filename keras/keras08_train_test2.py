import numpy as np
from keras.models import Sequential
from keras.layers import Dense 

# 1. DATA 
x = np.array([1,2,3,4,5,6,7,8,9,10])  # (10, )
y = np.array(range(10))               # (10, )

# x, y 7:3 으로 나누기 

x_train = x[:7]     # [1 2 3 4 5 6 7] = x[:-3]
x_test = x[7:]      # [ 8  9 10]      = x[-3:]
y_train = x[:7]     # [1 2 3 4 5 6 7]
y_test = x[7:]      # [ 8  9 10]

print(x_train)
print(x_test)
print(y_train)
print(y_test)
