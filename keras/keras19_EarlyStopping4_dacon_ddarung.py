import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 1. 데이터
path = "./_data/ddarung/"
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
    x, y, shuffle=True, test_size=0.4, random_state=333
)

# 2. 모델 구성
model = Sequential()
model.add(Dense(64, input_shape=(9,), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss',
                              mode='min',                
                              patience=25,                
                              restore_best_weights=True,
                              verbose=1)       
hist = model.fit(x_train, y_train, epochs=300, batch_size=9,
                 validation_split=0.3, verbose=1, callbacks=[earlyStopping])

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ',loss)
y_predict = model.predict(x_test)
rmse = np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", rmse)
# print('==================================')
# print(hist)     
# print('=================================')
# print(hist.history)
# print('=================================')
# print(hist.history['loss'])
# print('=================================')
# print(hist.history['val_loss'])

# 제출
y_submit = model.predict(test_csv)
submission['count'] = y_submit
submission.to_csv(path + "submission_0109_early.csv")

import matplotlib.pylab as plt

plt.figure(figsize=(9,6))       
plt.plot(hist.history['loss'], c='red',
         marker='.', label='loss')      
plt.plot(hist.history['val_loss'], c='blue',
         marker='.', label='val_loss')
plt.grid()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('ddarung loss')
plt.legend()   
plt.show()


"""
loss :  2577.025390625
RMSE :  50.764412362513525
"""