import numpy as np
from keras.models import Sequential
from keras.layers import Dense 
from sklearn.model_selection import ShuffleSplit, train_test_split

# 1. DATA 
x = np.array([1,2,3,4,5,6,7,8,9,10])  # (10, )
y = np.array(range(10))               # (10, )

# x, y 7:3 으로 나누기 

# x_train = x[:7]     # [1 2 3 4 5 6 7] = x[:-3]
# x_test = x[7:]      # [ 8  9 10]      = x[-3:]
# y_train = x[:7]     # [1 2 3 4 5 6 7]
# y_test = x[7:]      # [ 8  9 10]

# [실습] train과 test를 섞어서 7:3 으로
# 힌트 : 사이킷런
"""
(내가 작업한거)
rs = ShuffleSplit(n_splits=1, train_size=0.7, test_size=0.3)
for i, (train_index, test_index) in enumerate(rs.split(x)):
    x_train = train_index
    x_test = test_index
for i, (train_index, test_index) in enumerate(rs.split(y)):
    y_train = train_index
    y_test = test_index
   
# ShuffleSplit 매 실행마다 데이터가 바뀐다.  

"""
# 선생님이 수업시간에 알려주신거
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7,
    # test_size=0.3,
    # shuffle=True,     # default = True
    random_state=123    # random_state 를 명시하지 않으면 실행할 때마다 데이터가 바뀐다. 아무런 의미 없는 값을 넣어도 상관없다.
)

print('x_train : ', x_train)
print('x_test : ', x_test)
print('y_train : ', y_train)
print('y_test : ', y_test)

# # 2. 모델구성
# model = Sequential()
# model.add(Dense(10, input_dim = 1))   
# model.add(Dense(8))
# model.add(Dense(6))
# model.add(Dense(5))
# model.add(Dense(3))
# model.add(Dense(1))

# # 3. 컴파일, 훈련 
# model.compile(loss='mae', optimizer='adam')
# model.fit(x_train, y_train, epochs=1000, batch_size=1)

# # 4. 평가, 예측
# loss = model.evaluate(x_test, y_test)
# print('loss : ', loss)
# result = model.predict([11])
# print("[11]의 예측 결과 : ", result)

