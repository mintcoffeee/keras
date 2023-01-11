import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 1. 데이터
path = "./_data/ddarung/"
# path = "../_data/ddarung/" # vsc 2개 돌릴 때, keras > study 
# path = "c:/study/_data/ddarung/"
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'submission.csv', index_col=0)

print(train_csv.shape)      # (1459, 10)
print(train_csv.info())

# 결측치
print(train_csv.isnull().sum())
train_csv = train_csv.dropna()
print(train_csv.shape)      # (1328, 10)

x = train_csv.drop(['count'], axis=1)
# print(x)       # [1328 rows x 9 columns]
y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, test_size=0.3, random_state=333
)

### Scaling ####
scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler.fit(x_train)               # scaler에 대한 가중치생산
x_train = scaler.transform(x_train)     # 실질적인 값 변환
# x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

# 2. 모델 구성
model = Sequential()
model.add(Dense(128, input_shape=(9,), activation='linear'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam',
              metrics=['mae'])

from keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss',
                              mode='min',                
                              patience=50,                
                              restore_best_weights=True,
                              verbose=1)       
hist = model.fit(x_train, y_train, epochs=500, batch_size=18,
                 validation_split=0.3, verbose=1, callbacks=[earlyStopping])

# 4. 평가, 예측
mse, mae = model.evaluate(x_test, y_test)
print('mse : ', mse)
print('mae : ', mae)

y_predict = model.predict(x_test)
rmse = np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", rmse)


# 제출
y_submit = model.predict(test_csv)
submission['count'] = y_submit
submission.to_csv(path + "submission_0111_scaler.csv")


"""
loss :  2577.025390625
RMSE :  50.764412362513525

StandardScaling
mse :  1884.7247314453125
mae :  30.123701095581055
RMSE :  43.41341571651437

MinMaxScaler
mse :  1621.3255615234375
mae :  27.38595962524414
RMSE :  40.26568536876271
"""