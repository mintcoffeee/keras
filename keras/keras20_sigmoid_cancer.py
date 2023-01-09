import numpy as np
from sklearn.datasets import load_breast_cancer
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

# 1. 데이터
datasets = load_breast_cancer()
# print(datasets)
# print(datasets.DESCR)
# print(datasets.feature_names)

x = datasets['data']
y = datasets['target']
# print(x.shape, y.shape)     # (569, 30) (569,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, test_size=0.2, random_state=333
)

# 2. 모델구성
model = Sequential()
model.add(Dense(64, activation='linear', input_shape=(30,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])       # 이진분류일 때, loss = 'binary_crossentropy'

from keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss',   # monitor : loss, val_loss, accuracy..다양한 값 있지만, 초보자인 시점에서는 val_loss값이 무난하고 좋다. 
                              mode='min',                
# earlyStopping = EarlyStopping(monitor='accuracy',  
#                               mode='max',            # monitor 가 accuracy 이면, mode 는 max 값 사용                   
                              patience=20,                
                              restore_best_weights=True,
                              verbose=1)

model.fit(x_train, y_train, epochs=1000, batch_size=15,
          validation_split=0.2,
          callbacks=[earlyStopping],
          verbose=1)

# 4. 평가, 예측
# loss = model.evaluate(x_test, y_test)
# print('loss, accuracy : ', loss)
loss, accuracy = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', accuracy)
# loss :  [0.18643629550933838, 0.9473684430122375]     #[binary_crossentropy, accuracy]. 94.7% 확률로 암을 예측할 수 있다.

y_predict = model.predict(x_test)

# print(y_predict[:10])   # 값이 실수형태. 0 or 1 이 아니다. -> 정수형으로 바궈야 한다.
# print(y_test[:10])
# [과제] 실수 -> 정수 변환해서 acc 값 프린트

y_predict = y_predict.flatten()     # 차원 펴주기
y_predict_int = np.where(y_predict > 0.5, 1 , 0) # 0.5보다크면 1, 작으면 0
# print(y_predict_int[:10])   # [1 0 1 1 0 1 1 1 0 1]

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_predict_int)
print('accuracy_score : ', acc)     # accuracy_score :  0.956140350877193




"""
loss :  0.16484998166561127
accuracy :  0.9561403393745422
"""
