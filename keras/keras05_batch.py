import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

# 1.DATA
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])

# 2.model
model = Sequential()
model.add(Dense(10, input_dim =1))      # Dense(y, x)
model.add(Dense(20))    #하나의 layer
model.add(Dense(60))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

# 3.컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs=10, batch_size=2)

# 4. 평가, 예측
result = model.predict([6])
print("6의 예상 결과 : ", result)

"""
결과 :
6의 예상 결과 :  [[5.7134423]]
"""
