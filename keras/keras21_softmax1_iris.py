import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_boston
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

# 1. 데이터
datasets = load_iris()
# datasets = load_boston()
# print(datasets.DESCR)   # input 4개 output 1개.     pandas : .decribe() / .info()
# print(datasets.feature_names)   # pandas : .columns
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'] 

x = datasets.data
y = datasets['target']
# print(x)
# print(y)
# print(x.shape, y.shape)     # (150, 4) (150,)

##### One Hot Encoding  #####    # y클래스의 개수만큼 열이 늘어난다.
# 1. keras.utils의 to_categorical
from keras.utils import to_categorical
y = to_categorical(y)
print(y)
print(y.shape)      # (150, 3)

# 2. sklearn의 OneHotEncoder
# https://2-chae.github.io/category/1.ai/30
# from sklearn.preprocessing import OneHotEncoder
# y = y.reshape(-1,1)       # 1차원 배열에서 2차원 배열로 변형하는 코드. '-1'은 크기가 정해지지 않은 차원을 의미.
# ohe = OneHotEncoder()
# ohe.fit(y)            # fit_transform은 train에만 사용하고 test에는 학습된 인코더에 fit만 해야한다
# y = ohe.transform(y)
# y = y.toarray()       # List를 Array로 바꿔주는 메서드
# print(y)      
# print(y.shape)      

# 3. pandas
# y = pd.get_dummies(y)
# print(y)        # [150 rows x 3 columns]
# 문제점 : pandas.get_dummies는 train 데이터의 특성을 학습하지 않기 때문에 train 데이터에만 있고 
#        test 데이터에는 없는 카테고리를 test 데이터에서 원핫인코딩 된 칼럼으로 바꿔주지 않는다.
#        https://psystat.tistory.com/136

###################################

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True,            # shuffle = False 했을 때의 문제점 : y_test = [2 2 2 ... 2] , 
                                    # y_predict (ex)[1 2 0 1 ...0]) 값이랑 비교했을 때 대부분의 값이 일치하지 않을 수 있다.
    # random_state=11,
    test_size=0.1,  
    stratify=y      # 분류에서는 train 과 test 둘 중 한 곳으로 데이터가 치우치면 문제가 생길 수 있다. y데이터가 분류형 데이터일 때만 사용가능 하다.
)
print(y_train)
print(y_test)

# 2. 모델구성
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(4,)))
model.add(Dense(64, activation='sigmoid'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='linear'))
model.add(Dense(3, activation='softmax'))   # 다중 분류일 때 softmax

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',    # 다중 분류일 떄 loss='categorical_crossentropy'
              metrics=['accuracy'])
earlyStopping = EarlyStopping(monitor="val_loss",
                              mode="min",
                              patience=20,
                              restore_best_weights=True,
                              verbose=1)
model.fit(x_train, y_train, epochs=100, batch_size=32,
          validation_split=0.2, verbose=1, callbacks=[earlyStopping])

# 4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', accuracy)

# print(y_test[:5])
# y_predict = model.predict(x_test[:5])
# print(y_predict)

from sklearn.metrics import accuracy_score

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)    # 가장 큰 숫자의 위치값을 찾아내어 산출(?)
print('y_pred : ', y_predict)
y_test = np.argmax(y_test, axis=1)
print('y_test : ', y_test)

acc = accuracy_score(y_test, y_predict)
print('accuracy : ', acc)


"""
loss :  0.025238502770662308
accuracy :  1.0
"""