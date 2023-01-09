from sklearn.datasets import load_boston
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

# 1. 데이터
dataset = load_boston()
x = dataset.data
y = dataset.target 

print(x.shape, y.shape)     # (506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, test_size=0.2, random_state=333
)

# 2. 모델 구성
model = Sequential()
# model.add(Dense(128, input_dim=13))       # input_dim : 행과 열로 되어있는 데이터만 사용 가능 
model.add(Dense(128, input_shape=(13,)))     # (506, 13) input_shape = (13,)
# ex) input이 (100, 10, 5) 일 때 input_shape 사용, input_shape = (10, 5)
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(1))

# 3. 컴파일, 훈련
import time
model.compile(loss='mse', optimizer='adam')
start = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=32,
                 validation_split=0.2, verbose=2)
                # verbose = 0 : 모든게 안 보임
                # verbose = 1 : 모든게 다 보임(default)
                # verbose = 2 : progress bar 제거, 간략히 표현
                # verbose = 3 : epochs 값만 보인다.
                # verbose = 4 : 3 이상은 같음
                # verbose 값을 이용하여 시간을 단축 시ㅣㄹ 수 있으며, 보고싶은 값을 선택적으로 볼 수 있다. 
end = time.time()



# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ',loss)

print('걸린 시간 : ', end - start)

# 걸린 시간 :  4.518074989318848
# 걸린 시간 :  4.366379976272583