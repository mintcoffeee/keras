import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target

# print(datasets.feature_names)
# ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids',
# 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']
print(x.shape, y.shape)     # (178, 13) (178,)
print(y)
print(np.unique(y))     # [0 1 2] : y는 0, 1, 2 만 있다.
print(np.unique(y, return_counts=True))     # [(array([0, 1, 2]), array([59, 71, 48]))

y = to_categorical(y)
# print(y.shape)    # (178, 3)

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
# model = Sequential()
# model.add(Dense(128, activation='linear', input_shape=(13,)))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(3, activation='softmax'))

# 2. 모델 구성(함수형)  (= 모델 구성(순차형))
input1  = Input(shape=(13,))
dense1 = Dense(128, activation='linear')(input1)
dense2 = Dense(64, activation='relu')(dense1)
dense3 = Dense(64, activation='relu')(dense2)
dense4 = Dense(32, activation='relu')(dense3)
dense5 = Dense(32, activation='relu')(dense4)
output1 = Dense(3, activation='softmax')(dense5)
model = Model(inputs=input1, outputs=output1)
# model.summary()
# Total params: 16,387
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
print('y_pred : ', y_predict)
y_test = np.argmax(y_test, axis=1)
print('y_test : ', y_test)

acc = accuracy_score(y_test, y_predict)
print('accuracy : ', acc )


"""
accuracy :  0.9722222222222222

Standard Scaling 
accuracy :  0.9722222222222222

MinMax Scaling
accuracy :  1.0
"""
