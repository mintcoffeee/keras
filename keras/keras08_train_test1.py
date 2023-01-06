import numpy as np
from keras.models import Sequential
from keras.layers import Dense 

# 1. DATA 
# x = np.array([1,2,3,4,5,6,7,8,9,10])  # (10, )
# y = np.array(range(10))               # (10, )
x_train = np.array([1,2,3,4,5,6,7])
x_test = np.array([8,9,10])
y_train = np.array(range(7))
y_test = np.array(range(7, 10))

# 2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim = 1))   
model.add(Dense(8))
model.add(Dense(6))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

# 3. 컴파일, 훈련 
model.compile(loss='mae', optimizer='adam')
model.fit(x_train, y_train, epochs=1000, batch_size=1)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
result = model.predict([11])
print("[11]의 예측 결과 : ", result)

"""
결과
model.add(Dense(10, input_dim = 1))   
model.add(Dense(8))
model.add(Dense(6))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

model.fit(x_train, y_train, epochs=1000, batch_size=1)

loss :  0.03443066403269768
[11]의 예측 결과 :  [[10.040348]]

"""


