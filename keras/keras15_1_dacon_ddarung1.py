import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 1. DATA
path = './_data/ddarung/'   # '.' : 현재폴터. '/' : 하단
train_csv = pd.read_csv(path + 'train.csv', index_col=0)        # index_col = 0 : '0'번째 컬럼은 index다. 데이터가 아니다.'
# train_csv = pd.read_csv('./_data/ddarung/train.csv', index_col = 0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'submission.csv', index_col=0)

print(train_csv)
print(train_csv.shape)  # (1459, 10) > count는 y 값임므로 제외, input_dim = 9
print(submission.shape) # (715, 1)


print(train_csv.columns)
# Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
#       dtype='object')
print(train_csv.info())
#  0   hour                    1459 non-null   int64  
#  1   hour_bef_temperature    1457 non-null   float64
#  2   hour_bef_precipitation  1457 non-null   float64
# 결측치 : 1459 - 1457 = 2개 데이터 빠짐
# 결측치 처리 방법
# 1. 결측치가 있는 데이터는 삭제한다.
print(test_csv.info())      # 이 값을 통해 predict 를 할 것이기 때문에 count 값은 필요 없다.
print(train_csv.describe())

x = train_csv.drop(['count'], axis = 1)     # pandas 에서 coulumn 하나 빼기, axis = 0 : 행적용, aixs = 1 : 열적용
print(x)    # [1459 rows x 9 columns]
y = train_csv['count']
print(y)
print(y.shape)      # (1459, )

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7, random_state=11
)
print(x_train.shape, x_test.shape)      # (1021, 9) (438, 9)
print(y_train.shape, y_test.shape)      # (1021,) (438,)

# 2. 모델구성
model = Sequential()
model.add(Dense(64, input_dim=9))
model.add(Dense(64))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')     # 평가지표가 RMSE기 때문에, 가급적 유사한 loss = mse 를 준다.
model.fit(x_train, y_train, epochs=10, batch_size=9)

# 4. 평가, 훈련
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
# test.csv : submission 하기 위한 데이터
# submission.csv : 제출용
# train.csv : train 과 test 용
print(y_predict)
# def RMSE(y_test, y_predict):
#     return np.sqrt(mean_squared_error(y_test, y_predict))
# print('RMSE : ', RMSE(y_test, y_predict))

# 결측치 다음에
#  제출용도
# y_submit = model.predict(test_csv)




