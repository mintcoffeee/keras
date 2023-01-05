# [실습]
# 1. train 0.7 이상
# 2. R2 : 0.8 이상 RMSE 사용

# import sklearn as sk
# print(sk.__version__)   # 1.1.3

from sklearn.datasets import load_boston

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

print(x)    
print(x.shape)  # (506, 13)
print(y)
print(y.shape)  # (506, )

print(datasets.feature_names)
# 컬럼명 : ['CRIM'(범죄율) 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B'(흑인) 'LSTAT']
print(datasets.DESCR)    # DESCR : decribe. 묘사하다.

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7, random_state=111
)

# 2. 모델구성
model = Sequential()
model.add(Dense(512, input_dim=13))
model.add(Dense(512))
model.add(Dense(256))
model.add(Dense(256))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1000, batch_size = 13)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss)

y_predict = model.predict(x_test)
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

"""
model = Sequential()
model.add(Dense(516, input_dim=13))
model.add(Dense(516))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(1))

model.fit(x_train, y_train, epochs=1000, batch_size = 13)

loss :  30.052589416503906
RMSE :  5.482024242627045
R2 :  0.6969786019931298
"""

