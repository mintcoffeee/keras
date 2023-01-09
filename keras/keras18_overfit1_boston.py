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
# model.add(Dense(128, input_dim=13))
model.add(Dense(64, input_shape=(13,)))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
hist = model.fit(x_train, y_train, epochs=200, batch_size=32,
                 validation_split=0.2, verbose=1)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ',loss)

print('==================================')
print(hist)     # <keras.callbacks.History object at 0x7fb51447f490>
print('=================================')
print(hist.history)
# loss 와 val_loss 의 변화 값이 list 형태로 들어가있다. 리스트(list)란 원소들이 연속적으로 저장되는 형태의 자료형
# key(loss, val_loss) 와 value(숫자) 값 > dictionary 형태
print('=================================')
print(hist.history['loss'])
print('=================================')
print(hist.history['val_loss'])


import matplotlib.pylab as plt
from matplotlib import rc  

rc('font', family='AppleGothic') 			
plt.rcParams['axes.unicode_minus']

plt.figure(figsize=(9,6))       # inch
plt.plot(hist.history['loss'], c='red',
         marker='.', label='loss')      # c = color, marker : 점찍기
plt.plot(hist.history['val_loss'], c='blue',
         marker='.', label='val_loss')
plt.grid()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('보스턴 손실함수')
plt.legend()    # 자동으로 빈 자리에 생긴다
# plt.legend(loc='upper left')      # loc=location
plt.show()

# 평가 기준 val_loss : val_loss 가 들쑥날쑥하면 안좋은 데이터

# title 한글 변환 방법, matplotlib에서 한글 깨짐 찾기
