from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

# 1. DATA
x = np.array(range(1, 21))
y = np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20])

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7, shuffle=True, random_state=123
)

# 2. 모델구성
model = Sequential()
model.add(Dense(14, input_dim = 1))   
model.add(Dense(12))
model.add(Dense(10))
model.add(Dense(8))
model.add(Dense(1))

# 3. 컴파일, 훈련 
model.compile(loss='mse', optimizer='adam')     # mae : mean absolute eroor, mse : mean squared error
model.fit(x_train, y_train, epochs=500, batch_size=1)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

# mae : 3.167687177658081
# mse : 15.138476371765137

"""
데이터 범위 차에따른 mae, mse 결정 이유 근거
https://jysden.medium.com/%EC%96%B8%EC%A0%9C-mse-mae-rmse%EB%A5%BC-%EC%82%AC%EC%9A%A9%ED%95%98%EB%8A%94%EA%B0%80-c473bd831c62


MAE : 약간의 이상치가 있는 경우, 그 이상치의 영향을 적게 받으면서 모델을 만들고자 할 때, MAE 사용
      MAE는 이상치에 영향을 덜 받는다. 이는 이상치를 포함한 훈련 데이터에 적합하게 학습되어 unseen 데이터에 대해 낮은 성능을 보이게 하는 오버 피팅을 방지하는 데 도움이 될 수 있다.
MSE : MSE 경우에는 이상치에 민감하게 반응하여 학습하기 때문에, 손실 함수가 이상치에 의해 발생한 오차로부터 비교적 많은 영향을 받는다.
      오차값에 제곱을 취하기 때문에 오차가 0과 1 사이인 경우에, MSE에서 그 오차는 본래보다 더 작게 반영되고, 오차가 1보다 클 때는 본래보다 더 크게 반영된다.
RMSE : RMSE는 MSE 보다 이상치에 대해 상대적으로 둔감하다. 하지만 이는 MAE처럼 모든 error에 동일한 가중치를 주지 않고, 
      error가 크면 더 큰 가중치를 작으면 더 작은 가중치를 준다는 점에서, 여전히 이상치에 민감하다고 간주될 수 있다. 
      따라서 모델 학습 시 이상치에 가중치를 부여하고자 한다면, MSE에 루트를 씌운 RMSE를 채택하는 것은 적절하다.

"""
