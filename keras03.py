import numpy as np
import tensorflow as tf

# 1. (정제된)DATA
x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 2, 3, 5, 4])

# 2. model 구성 y = wx + b
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(1, input_dim = 1))

# 3. 컴파일, 훈련 
model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs=290)

# 4.평가, 예측 
result = model.predict([6])
print('6의 예상 결과 : ', result)

