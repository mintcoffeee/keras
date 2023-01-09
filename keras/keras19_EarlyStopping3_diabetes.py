from sklearn.datasets import load_diabetes
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

# 1. 데이터
dataset = load_diabetes()
x = dataset.data
y = dataset.target 

print(x.shape, y.shape)     # (442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, test_size=0.3, random_state=333
)

# 2. 모델 구성
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(10,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
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
                 validation_split=0.3, verbose=1, callbacks=[earlyStopping])

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ',loss)

print('==================================')
print(hist)     
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
plt.title('diabetes loss')
plt.legend()   
plt.show()

"""
loss :  2702.078857421875
"""
