import numpy as np

# np.save('./_data/brain/brain_x_train.npy', arr=xy_train[0][0])
# np.save('./_data/brain/brain_y_train.npy', arr=xy_train[0][1])

# np.save('./_data/brain/brain_x_test.npy', arr=xy_test[0][0])
# np.save('./_data/brain/brain_y_test.npy', arr=xy_test[0][1])

x_train = np.load('C:/_data/dogs-vs-cats/train/dogs_cat_x_train.npy')
y_train = np.load('C:/_data/dogs-vs-cats/train/dogs_cat_y_train.npy')
# x_test = np.load('./_data/brain/brain_x_test.npy')
# y_test = np.load('./_data/brain/brain_y_test.npy')

print(x_train.shape, y_train.shape)  # (25000, 190, 190, 1) (25000,)
# x_train = np.reshape(x_train, (25000, 190, 190, 3))

# 2. 모델
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

model = Sequential()
model.add(Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(190, 190, 1)))
model.add(MaxPooling2D())
model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
model.add(Flatten())
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.4))
model.add(Dense(1, activation='sigmoid'))
model.summary()

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['acc'])
callbaks = [EarlyStopping(monitor='val_acc', mode='max', restore_best_weights=True,
                   verbose=1, patience=30),
            ReduceLROnPlateau(monitor='val_acc', patience=2, verbose=1, factor=0.5, min_lr=0.00001)] 
hist = model.fit(x_train, y_train,
                 batch_size=64,
                 epochs=10,
                validation_split=0.2,
                #  validation_steps=4,
                callbacks=[callbaks],
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


# loss :  0.039778999984264374
# val_loss :  0.7565954923629761
# accuracy :  0.9869499802589417
# val_acc :  0.8118000030517578

# loss :  0.04570339247584343
# val_loss :  0.5876166224479675
# accuracy :  0.9854999780654907
# val_acc :  0.8587999939918518