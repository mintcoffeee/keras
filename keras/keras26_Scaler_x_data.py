import numpy as np
from sklearn.datasets import load_boston
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 1. 데이터
dataset = load_boston()
x = dataset.data
y = dataset.target 
# print(x.shape, y.shape)     # (506, 13) (506,)

### Scaling ####
# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x)               # scaler에 대한 가중치생산
x = scaler.transform(x)     # 실질적인 값 변환
# prtin(x)
# print(type(x))      # <class 'numpy.ndarray'>
# print("최소값 : ", np.min(x))
# print("최대값 : ", np.max(x))

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, test_size=0.3, random_state=333
)

# 2. 모델 구성
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(13,)))
model.add(Dense(64, activation='sigmoid'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
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
model.fit(x_train, y_train, epochs=300, batch_size=13,
                 validation_split=0.3, verbose=1, callbacks=[earlyStopping])

# 4. 평가, 예측
mse, mae = model.evaluate(x_test, y_test)
print('mse : ', mse)
print('mae : ', mae)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('R2 : ', r2)


"""
MinMaxScaling
변환 전) R2 :  0.7337688290953932
변환 후) R2 :  0.7761110646887824
StandardScaling
R2 :  0.7988889132932151
"""

# 참고
# https://leehah0908.tistory.com/2
# 그래프 : https://mkjjo.github.io/python/2019/01/10/scaler.html