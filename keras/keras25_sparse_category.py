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
x = datasets.data
y = datasets['target']
# print(x)
# print(y)
# print(x.shape, y.shape)     # (150, 4) (150,)

##### One Hot Encoding  #####    # y클래스의 개수만큼 열이 늘어난다.
# 1. keras.utils의 to_categorical
# from keras.utils import to_categorical
# y = to_categorical(y)
# print(y)
# print(y.shape)      # (150, 3)

# 2. sklearn의 OneHotEncoder
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
###################################

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True,           
    random_state=11,
    test_size=0.1,  
    stratify=y     
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
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',    # 다중 분류일 떄 loss='categorical_crossentropy'
              metrics=['accuracy'])
earlyStopping = EarlyStopping(monitor="val_loss",
                              mode="min",
                              patience=20,
                              restore_best_weights=True,
                              verbose=1)
model.fit(x_train, y_train, epochs=300, batch_size=32,
          validation_split=0.2, verbose=1, callbacks=[earlyStopping])

# 4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', accuracy)

print(y_test[:5])
y_predict = model.predict(x_test[:5])
print(y_predict)

from sklearn.metrics import accuracy_score

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)    
print('y_pred : ', y_predict)
# y_test = np.argmax(y_test, axis=1)    # 위에서 onehot을 안했을 경우 사용하지 않는다.
print('y_test : ', y_test)

acc = accuracy_score(y_test, y_predict)
print('accuracy : ', acc)


"""
loss :  0.007831298746168613
accuracy :  1.0
"""