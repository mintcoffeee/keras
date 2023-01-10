import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import tensorflow as tf

# 1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets['target']
print(x.shape, y.shape)     # (581012, 54) (581012,)
print(np.unique(y, return_counts=True))
# (array([1, 2, 3, 4, 5, 6, 7], dtype=int32), array([211840, 283301,  35754,   2747,   9493,  17367,  20510]))

# print(datasets.feature_names)


########## One Hot Encoding #############
# 1.to_categorical
# y = to_categorical(y)
# print(y.shape)    # (581012, 8)
# array가 [1, 2, 3, 4, 5, 6, 7] 이므로 (581012, 7)이 나와야 한다. 따라서 to_categorical은 사용 불가
# np.delete 로 맨앞 0 컬럼 삭제 가능
# y = np.delete(y, 0, axis=1)
# print(y.shape)     # (581012, 7)

# 2. OneHotEncoder
print(y)
y = y.reshape(-1,1)
ohe = OneHotEncoder()
ohe.fit(y)
y = ohe.transform(y)        # ohe.fit_transform()
y = y.toarray()
# print(y)      
print(y.shape)      # (581012, 7)

# 3. pandas get_dummies
# y = pd.get_dummies(y)
# print(y.shape)    # (581012, 7)
# y_test = np.argmax() 에서 에러가 발생한다.    tf.argmax()에서는 정상 작동

# print(type(y))
# 힌트 : pandas 에서 numpy 형태로 변환필요  .values      .to_numpy()
#########################################

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, test_size=0.2,
    stratify=y
)

# 2. 모델구성
model = Sequential()
model.add(Dense(64, activation='linear', input_shape=(54,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(7, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
earlyStopping = EarlyStopping(monitor="val_loss",
                              mode="min",
                              patience=20,
                              restore_best_weights=True,
                              verbose=1)
model.fit(x_train, y_train, epochs=200, batch_size=54,
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
accuracy :  0.8133869177215735

"""