import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

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
model.add(Dense(64, input_shape=(8,)))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
hist = model.fit(x_train, y_train, epochs=300, batch_size=32,
                 validation_split=0.2, verbose=1)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ',loss)

print('==================================')
print(hist)     # <keras.callbacks.History object at 0x7fb51447f490>
print('=================================')
print(hist.history)
print('=================================')
print(hist.history['loss'])
print('=================================')
print(hist.history['val_loss'])

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




