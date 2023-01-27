import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv2D, Flatten
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# 1. 데이터
a = np.array(range(1,101))
x_predict = np.array(range(96,106)) 

timesteps = 5   # x는 4개 y는 1개 

def split_x(dataset, timesteps):
    aaa = []
    for i in range(len(dataset) - timesteps + 1):
        subset = dataset[i : (i + timesteps)]
        aaa.append(subset)
    return np.array(aaa)

b = split_x(a, timesteps)
# print(b)

x_pred = split_x(x_predict, 4)
# print(x_pred)
print(x_pred.shape)     # (7, 4)
x = b[:, :-1]
y = b[:,-1]
# print(x, y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=11, test_size=0.2  #train_size default 값 = 0.75
)

print(x_train.shape, y_train.shape)     # (76, 4) (76,)
print(x_test.shape, y_test.shape)       # (20, 4) (20,)

x_train = x_train.reshape(76, 2, 2, 1)
x_test = x_test.reshape(20, 2, 2, 1)
x_pred = x_pred.reshape(7, 2, 2, 1)


# 2. 모델구성
model = Sequential()
model.add(Conv2D(64, (2,2), activation='relu', padding='same', input_shape=(2, 2, 1)))
model.add(Conv2D(64, (2,2), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(1))
# model.summary()

# 3. 컴파일,훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss',
                   mode='min',
                   patience=100,
                   restore_best_weights=True,
                   verbose=1)
model.fit(x_train, y_train, validation_split=0.2, epochs=1000, batch_size=4, callbacks=[es])

# 4.평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

result = model.predict(x_pred)
print(result)

# LSTM, feature = 1
# loss :  0.0001897570036817342
# [[ 99.98978]
#  [100.9889 ]
#  [101.98791]
#  [102.98848]
#  [103.98911]
#  [104.98963]
#  [105.99002]]

# LSTM, feature = 2
# loss :  6.440608558477834e-05
# [[ 99.98592]
#  [100.98507]
#  [101.98423]
#  [102.98341]
#  [103.98263]
#  [104.98188]
#  [105.98116]]

# Conv2D
# loss :  5.684195457433816e-06
# [[100.00116 ]
#  [101.0013  ]
#  [102.0014  ]
#  [103.001526]
#  [104.00166 ]
#  [105.001854]
#  [106.00203 ]]