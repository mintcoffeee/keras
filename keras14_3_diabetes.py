# [과제 실습]
# R2 0.62 이상

from sklearn.datasets import load_diabetes

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# 1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x)
print(x.shape)  # (442, 10)
print(y)
print(y.shape)  # (442, )
print(datasets.feature_names)
# ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
print(datasets.DESCR)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size = 0.7, random_state=11
)

# 2. 모델구성
model = Sequential()
model.add(Dense(512, input_dim=10))
model.add(Dense(512))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(1))

# 3. 컴파일, 훈련 
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs = 500, batch_size=5)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE : ', RMSE(y_test, y_predict))

r2 =  r2_score(y_test, y_predict)
print('R2 : ', r2)

"""
model.add(Dense(256, input_dim=10))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(1))

model.fit(x_train, y_train, epochs = 300, batch_size=5)

loss :  3230.761962890625
RMSE :  56.83979289103339
R2 :  0.5510065815494369

model.add(Dense(512, input_dim=10))
model.add(Dense(512))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(1))

model.fit(x_train, y_train, epochs = 500, batch_size=5)

loss :  3198.679443359375
RMSE :  56.556867170561354
R2 :  0.5554652758297525
"""