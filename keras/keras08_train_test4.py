import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

# 1. DATA
x = np.array([range(10), range(21,31), range(201, 211)])
y = np.array([[1,2,3,4,5,6,7,8,9,10],
              [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4]])
x = x.T
y = y.T
# [실습] train_test_split를 이용하여
# 7:3으로 잘라서 모델 구현 / 소스 완성 
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7,
    random_state=123
)

print('x_train : ', x_train)
print('x_test : ', x_test)
print('y_train : ', y_train)
print('y_test : ', y_test)

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
model.fit(x_train, y_train, epochs=500, batch_size=1)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
result = model.predict([[9, 30, 210]])
print("[9, 30, 210]의 예측 결과 : ", result)

"""
결과
loss :  0.23179364204406738
[9, 30, 210]의 예측 결과 :  [[9.950726  1.5739204]]
"""


