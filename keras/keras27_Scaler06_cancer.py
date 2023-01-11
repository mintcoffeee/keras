import numpy as np
from sklearn.datasets import load_breast_cancer
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score

# 1. 데이터
datasets = load_breast_cancer()
x = datasets['data']
y = datasets['target']
# print(x.shape, y.shape)     # (569, 30) (569,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, test_size=0.2, random_state=333
)

### Scaling ####
scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler.fit(x_train)               # scaler에 대한 가중치생산
x_train = scaler.transform(x_train)     # 실질적인 값 변환
# x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델구성
model = Sequential()
model.add(Dense(64, activation='linear', input_shape=(30,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))   # 이진분류일 때, activation = sigmoid
# sigmoid = 0 과 1을 출력하는 것이 아니라, 0 과 1 사이의 값을 출력하는 것이다.

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])       # 이진분류일 때, loss = 'binary_crossentropy'

from keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss',   # monitor : loss, val_loss, accuracy..다양한 값 있지만, 초보자인 시점에서는 val_loss값이 무난하고 좋다. 
                              mode='min',                       
                              patience=30,                
                              restore_best_weights=True,
                              verbose=1)

model.fit(x_train, y_train, epochs=1000, batch_size=15,
          validation_split=0.2,
          callbacks=[earlyStopping],
          verbose=1)

# 4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', accuracy)
# loss :  [0.18643629550933838, 0.9473684430122375]     #[binary_crossentropy, accuracy]. 94.7% 확률로 암을 예측할 수 있다.

y_predict = model.predict(x_test)
# 1. round 처리
y_predict = np.round(y_predict)
# print(y_predict)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_predict)
print('accuracy_score : ', acc)     # accuracy_score :  0.956140350877193




"""
loss :  0.16484998166561127
accuracy :  0.9561403393745422

StandardScaling
accuracy_score :  0.9649122807017544

MinMaxScaling
accuracy_score :  0.9736842105263158
"""
