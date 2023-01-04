import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# 1. DATA
x = np.array(range(10))     #(10, ), (10, 1)
y = np.array([[1,2,3,4,5,6,7,8,9,10],
              [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4],  # x와 y를 매칭하면서 훈련
              [9,8,7,6,5,4,3,2,1,0]])           
# x = 비트코인 가격(1개), y = 주가, 금리, 출산율(3개) > input(1개), output(3개) 시스템은 작동 하지만, 말도 안되는 값이 나온다.

y = y.T

# 2. 모델구성
model = Sequential()
model.add(Dense(12, input_dim = 1))   
model.add(Dense(7))
model.add(Dense(8))
model.add(Dense(9))
model.add(Dense(7))
model.add(Dense(3))

# 3. 컴파일, 훈련 
model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs=500, batch_size=1)

# 4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)
result = model.predict([9])
print("[9]의 예측 결과 : ", result)

"""
결과
model.add(Dense(12, input_dim = 1))   
model.add(Dense(7))
model.add(Dense(8))
model.add(Dense(9))
model.add(Dense(7))
model.add(Dense(3))

model.fit(x, y, epochs=500, batch_size=1)

loss :  0.0873289704322815
[9]의 예측 결과 :  [[10.182299   1.6971985 -0.0129894]]
"""


