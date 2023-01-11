import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 1. 데이터
# path = "./_data/bike/"
path = "../_data/bike/" 
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)

print(train_csv.shape)      # (10886, 11)
print(train_csv.info())

# 결측치
# print(train_csv.isnull().sum())   

x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
print(x)       # [10886 rows x 8 columns]
y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, test_size=0.3, random_state=333
)

### Scaling ####
# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)               # scaler에 대한 가중치생산
x_train = scaler.transform(x_train)     # 실질적인 값 변환
# x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)
# 2. 모델 구성
model = Sequential()
model.add(Dense(128, input_shape=(8,), activation='sigmoid'))
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
                              patience=30,                
                              restore_best_weights=True,
                              verbose=1)       
hist = model.fit(x_train, y_train, epochs=1000, batch_size=8,
                 validation_split=0.2, verbose=1, callbacks=[earlyStopping])

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
mse :  20982.228515625
RMSE :  144.85243953949185

StandardScaling
mse :  21429.439453125
mae :  107.59248352050781
RMSE :  146.3879700596804

MinMaxScaling
mse :  21421.458984375
mae :  107.41730499267578
RMSE :  146.3606992406514
"""