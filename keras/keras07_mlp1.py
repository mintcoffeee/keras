import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# 1. DATA
x = np.array([[1,2,3,4,5,6,7,8,9,10],
              [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4]])
y = np.array([2,4,6,8,10,12,14,16,18,20])

print(x.shape)      # (2, 10) > (행, 렬)
print(y.shape)      # (10, )

x = x.T     # 전치하다. x의 행과 열을 바꾼다
print(x.shape)  # (10행, 2열)
# 열 = column, 피처, 특성

# 2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim =2))   # input_dim << 2열. input_dim : 열의 개수      
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

# 3.컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)   # batch_size가 1인데 터미널 앞에는 10/10....? >> 20/20이 아니다. [1, 1], [2, 1], [3, 1].... [10, 1.4]

# 4. 평가, 예측
loss = model.evaluate(x, y)     # 마지막 1/1 > evaluate 에도 배치 사이즈가 존재. defalut 값 = 32
print('loss : ', loss)
result = model.predict([[10, 1.4]])
print("[10, 1.4]의 예측값 : ", result)

"""
ex)
비트코인 가격(y) 예측
환율, 주가, 금리, 돼지고기 가격, 금 > input_dim = 5
"""

"""
결과 :
loss :  0.16264918446540833
[10, 1.4]의 예측값 :  19.826344
"""


