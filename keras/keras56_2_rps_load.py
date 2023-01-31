import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

X = np.load('C:/_data/rps/rps_x_train.npy')
y = np.load('C:/_data/rps/rps_y_train.npy')
print(X.shape, y.shape) # (2520, 200, 200, 3) (2520, 3)

# print(np.unique(y))
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=11
)

# 2. 모델
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

model = Sequential()
model.add(Conv2D(64, (3,3), padding='same', activation='relu', input_shape=(200, 200, 3)))
model.add(MaxPooling2D())
model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(256, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(256, (3,3), padding='same', activation='relu'))
model.add(Flatten())
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.4))
model.add(Dense(3, activation='softmax'))
model.summary()

# 3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
              metrics=['acc'])
call = [EarlyStopping(monitor='val_acc', mode='max', restore_best_weights=True,
                   verbose=1, patience=30),
            # ReduceLROnPlateau(monitor='val_acc', patience=2, verbose=1, factor=0.5, min_lr=0.00001)
            ] 
model.fit(x_train, y_train,
                 batch_size=32,
                 epochs=300,
                validation_split=0.25,
                #  validation_steps=4,
                callbacks=call,
                ) 

# 4. 평가, 예측
# accuracy = hist.history['acc']
# val_acc = hist.history['val_acc']
# loss = hist.history['loss']     # epochs 값 만큼 나온다
# val_loss = hist.history['val_loss']

# print('loss : ', loss[-1])
# print('val_loss : ', val_loss[-1])
# print('accuracy : ', accuracy[-1])
# print('val_acc : ', val_acc[-1])

# loss :  4.0051734373491854e-08
# val_loss :  1.0861867849598639e-06
# accuracy :  1.0
# val_acc :  1.0

result = model.evaluate(x_test, y_test)
print('loss : ', result[0])
print('acc : ', result[1])

# loss :  0.008380275219678879
# acc :  0.9968253970146179


