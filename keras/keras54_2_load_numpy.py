import numpy as np

# np.save('./_data/brain/brain_x_train.npy', arr=xy_train[0][0])
# np.save('./_data/brain/brain_y_train.npy', arr=xy_train[0][1])

# np.save('./_data/brain/brain_x_test.npy', arr=xy_test[0][0])
# np.save('./_data/brain/brain_y_test.npy', arr=xy_test[0][1])

x_train = np.load('./_data/brain/brain_x_train.npy')
y_train = np.load('./_data/brain/brain_y_train.npy')
x_test = np.load('./_data/brain/brain_x_test.npy')
y_test = np.load('./_data/brain/brain_y_test.npy')

print(x_train.shape, x_test.shape)  # (160, 200, 200, 1) (120, 200, 200, 1)
print(y_train.shape, y_test.shape)  # (160,) (120,)
# print(x_train[100])

# 2. 모델구성

# 2. 모델
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.callbacks import EarlyStopping

model = Sequential()
model.add(Conv2D(64, (3,3), activation='relu', input_shape=(200, 200, 1)))
model.add(MaxPooling2D())
model.add(Conv2D(256, (3,3), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(256, (3,3), activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))
model.summary()

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['acc'])
es = EarlyStopping(monitor='val_acc',
                   mode='max',
                   restore_best_weights=True,
                   verbose=1,
                   patience=30)
hist = model.fit(x_train, y_train,
                 batch_size=32,
                #  steps_per_epoch=16,     # steps_per_epoch = 훈련 샘플 수 / 배치 사이즈 : 1에포당 얼마나 걸을 걷이냐
                 epochs=10,
                 validation_data=(x_test, y_test),
                #  validation_steps=4,
                callbacks=[es],
                ) 

# 4. 평가, 예측
accuracy = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']     # epochs 값 만큼 나온다
val_loss = hist.history['val_loss']

print('loss : ', loss[-1])
print('val_loss : ', val_loss[-1])
print('accuracy : ', accuracy[-1])
print('val_acc : ', val_acc[-1])



