import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 1. 데이터
path = "./_data/bike/"
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)

print(train_csv.shape)      # (10886, 11)
print(train_csv.info())

# 결측치
# print(train_csv.isnull().sum())
# train_csv = train_csv.dropna()
# print(train_csv.shape)      

x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
print(x)       # [10886 rows x 8 columns]
y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, test_size=0.2, random_state=333
)

# 2. 모델 구성
model = Sequential()
model.add(Dense(512, input_shape=(8,), activation='sigmoid'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss',
                              mode='min',                
                              patience=20,                
                              restore_best_weights=True,
                              verbose=1)       
hist = model.fit(x_train, y_train, epochs=300, batch_size=32,
                 validation_split=0.2, verbose=1, callbacks=[earlyStopping])

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
plt.title('bike loss')
plt.legend()   
plt.show()


"""
loss :  20982.228515625
RMSE :  144.85243953949185
"""