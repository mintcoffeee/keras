import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# 1. DATA
x = np.array([range(10), range(21,31), range(201, 211)])
# print(range(10))    # range(0, 10) 0~9까지. 0부터 10-1까지
# print(x.shape)      # (3, 10)
y = np.array([[1,2,3,4,5,6,7,8,9,10],
              [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4]])


x = x.T
y = y.T
# 예측 [9, 30, 210] > [10, 1.4]

# 2. 모델구성
model = Sequential()
model.add(Dense(12, input_dim = 3))   
model.add(Dense(7))
model.add(Dense(8))
model.add(Dense(9))
model.add(Dense(7))
model.add(Dense(4))
model.add(Dense(2))

# 3. 컴파일, 훈련 
model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs=500, batch_size=1)

# 4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)
result = model.predict([[9, 30, 210]])
print("[9, 30, 210]의 예측 결과 : ", result)

"""
결과

model.add(Dense(12, input_dim = 3))   
model.add(Dense(7))
model.add(Dense(8))
model.add(Dense(9))
model.add(Dense(7))
model.add(Dense(4))
model.add(Dense(2))

model.fit(x, y, epochs=500, batch_size=1)

loss :  0.1808764487504959
[9, 30, 210]의 예측 결과 :  [[10.13283    1.5261254]]


"""


