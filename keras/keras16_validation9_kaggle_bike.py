import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 1. DATA
path = "./_data/bike/"
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)

print(train_csv.shape)      # (10886, 11)
print(train_csv.columns)
print(train_csv.info())
print(test_csv.info())

# 결측치
print(train_csv.isnull().sum())

x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
y = train_csv['count']

print(x.shape)      # (10886, 8)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    test_size=0.3, random_state=111
)

# 2. 모델
model = Sequential()
model.add(Dense(512, input_dim=8, activation='sigmoid'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=330, batch_size=8,
          validation_split=0.3)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
y_predict = model.predict(x_test)
rmse = np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", rmse)

# 제출
y_submit = model.predict(test_csv)
submission['count'] = y_submit
submission.to_csv(path + "submission_0108_val.csv")


"""
RMSE :  153.03832810565765


model.add(Dense(64, input_dim=8, activation='sigmoid'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))

model.fit(x_train, y_train, epochs=320, batch_size=10,
          validation_split=0.25)
          
loss :  22626.912109375
RMSE :  150.42246382205977


model.add(Dense(256, input_dim=8, activation='sigmoid'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))

model.fit(x_train, y_train, epochs=320, batch_size=8,
          validation_split=0.3)
          
loss :  21686.91015625
RMSE :  147.26476432576558
"""