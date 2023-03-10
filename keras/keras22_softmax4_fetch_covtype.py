import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
import tensorflow as tf

# 1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets['target']
print(x.shape, y.shape)     # (581012, 54) (581012,)
print(np.unique(y, return_counts=True))
# (array([1, 2, 3, 4, 5, 6, 7], dtype=int32), array([211840, 283301,  35754,   2747,   9493,  17367,  20510]))

# print(datasets.feature_names)


# One Hot Encoding 
##########  1.to_categorical #############
# from keras.utils import to_categorical
# y = to_categorical(y)
# print(y.shape)    # (581012, 8)
# print(type(y))    # <class 'numpy.ndarray'>
# print(y[:10])
# print(np.unique(y[:,0], return_counts=True))    # (array([0.], dtype=float32), array([581012])), 맨 앞옆에 데이터가 0(쓸모 없는 데이터)으로 채워졌다. 
# print(np.unique(y[:,1], return_counts=True))    # (array([0., 1.], dtype=float32), array([369172, 211840]))
# array가 [1, 2, 3, 4, 5, 6, 7] 이므로 (581012, 7)이 나와야 한다. 따라서 to_categorical은 사용 불가
# np.delete 로 맨앞 0 컬럼 삭제 가능
# y = np.delete(y, 0, axis=1)
# print(y.shape)     # (581012, 7)
# print(y[:10])
# print(np.unique(y[:,0], return_counts=True))    # (array([0., 1.], dtype=float32), array([369172, 211840]))

########## 2.sklearn OneHotEncoder #############
print(y.shape)      # (581012,)
y = y.reshape(-1,1)
print(y.shape)      # (581012, 1)
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
# ohe.fit(y)
# y = ohe.transform(y)        # ohe.fit_transform()
y = ohe.fit_transform(y)
# print(type(y))      # <class 'scipy.sparse.csr.csr_matrix'> 
y = y.toarray()
# print(type(y))      # <class 'numpy.ndarray'>
# OneHotEncoder > fit, transform > toarray

########## 3.pandas get_dummies #############
# import pandas as pd
# 힌트 : pandas 에서 numpy 형태로 변환필요  .values      .to_numpy() 
# y_test = np.argmax() 에서 에러가 발생한다. tf.argmax()에서는 정상 작동
# numpy에서는 pandas 를 바로 받아들이지 못한다.
# y = pd.get_dummies(y)
# print(y[:10])
# print(type(y))      # <class 'pandas.core.frame.DataFrame'>, Index와 Header 자동 생성
# y = y.values
# y = y.to_numpy()
# print(type(y))      # <class 'numpy.ndarray'>
# print(y.shape)    # (581012, 7)
#############################################

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, test_size=0.2,random_state=11,
    stratify=y
)

# 2. 모델구성
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(54,)))
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
print('y_pred(예측값) : ', y_predict[:20])
y_test = np.argmax(y_test, axis=1)
print('y_test(원래값) : ', y_test[:20])

acc = accuracy_score(y_test, y_predict)
print('accuracy : ', acc )

"""
accuracy :  0.8133869177215735

"""