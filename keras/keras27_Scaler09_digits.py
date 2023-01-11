import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 1. 데이터
datasets = load_digits()        # 이미지
x = datasets.data
y = datasets['target']
print(x.shape, y.shape)     # (1797, 64) (1797,)
print(np.unique(y, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180]))

y = to_categorical(y)
# print(y.shape)    # (1797, 10)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, test_size=0.2,
    stratify=y
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
model.add(Dense(512, activation='linear', input_shape=(64,)))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
earlyStopping = EarlyStopping(monitor="val_loss",
                              mode="min",
                              patience=25,
                              restore_best_weights=True,
                              verbose=1)
model.fit(x_train, y_train, epochs=300, batch_size=32,
          validation_split=0.3, verbose=1, callbacks=[earlyStopping])

# 4. 평가, 예측 
loss, accuracy = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', accuracy)

y_predict = model.predict(x_test)
# print(y_predict)
y_predict = np.argmax(y_predict, axis=1)
# print('y_pred : ', y_predict)
y_test = np.argmax(y_test, axis=1)
# print('y_test : ', y_test)

acc = accuracy_score(y_test, y_predict)
print('accuracy : ', acc )


"""
accuracy :  0.9833333333333333

MinMaxScaling
accuracy :  0.9777777777777777

Standard Scaling
accuracy :  0.95
"""
