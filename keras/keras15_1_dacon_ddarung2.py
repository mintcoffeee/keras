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

print(test_csv.info())      # 이 값을 통해 predict 를 할 것이기 때문에 count 값은 필요 없다.
print(train_csv.describe())

##### 결측치 처리 1. 제거 #####
print(train_csv.isnull().sum())     # isnull : null 값 보여줌, sum : null 값 개수의 합
# 중단점 : 중단점 직전까지 디버깅
train_csv = train_csv.dropna()
print(train_csv.isnull().sum())
print(train_csv.shape)      # (1328, 10)

x = train_csv.drop(['count'], axis = 1)     # pandas 에서 coulumn 하나 빼기, axis = 0 : 행적용, aixs = 1 : 열적용
print(x)    # [1328 rows x 9 columns]
y = train_csv['count']
print(y)
print(y.shape)      # (1328, )

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7, random_state=11
)
print(x_train.shape, x_test.shape)      # (929, 9) (399, 9)
print(y_train.shape, y_test.shape)      # (929,) (399,)

# 2. 모델구성
model = Sequential()
model.add(Dense(64, input_dim=9))
model.add(Dense(64))
model.add(Dense(1))

# 3. 컴파일, 훈련
import time
model.compile(loss='mse', optimizer='adam')     # 평가지표가 RMSE기 때문에, 가급적 유사한 loss = mse 를 준다.
start = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=9)
end = time.time()


# 4. 평가, 훈련
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
# print(y_predict)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE : ', RMSE(y_test, y_predict))

print('걸린시간 : ', end - start)
# cpu 걸린시간 : 29.5874
# gpu 걸린시간 : 

#  제출용도
y_submit = model.predict(test_csv)
# print(y_submit)         # submission 의 count 제출용이기 때문에 삭제를 할 수 없다.
# print(y_submit.shape)   # (715, 1)

# to_csv()를 사용해서
# submission_0105.csv를 완성하시오!!

"""
내가 작업한거 
df = pd.DataFrame(y_submit)
df.to_csv(path + 'submission_0105.csv',
                sep = ',',
                na_rep = 'NaN')
"""
# print(submission)
submission['count'] = y_submit
# print(submission)
submission.to_csv(path + 'submission_0105.csv')

"""

RMSE :  51.32967701573189

"""



