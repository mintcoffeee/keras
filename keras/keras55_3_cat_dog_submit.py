import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator

IMAGE_SIZE = (190,190)
BATCH_SIZE = 15000

datagen = ImageDataGenerator(
    rescale=1./255
)

xy_test1 = datagen.flow_from_directory(
    'C:/_data/dogs-vs-cats/test1/', 
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,   
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True,
    # Found 120 images belonging to 2 classes.
)
xy_test2 = datagen.flow_from_directory(
    'C:/_data/dogs-vs-cats/test2/', 
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,   
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True,
    # Found 120 images belonging to 2 classes.
)

# print(xy_test)
# print(xy_test[0][0].shape) # (12500, 190, 190, 1)

# test 파일 numpy 형태로 x_test을 저장한다.
# np.save('C:/_data/dogs-vs-cats/test1/dogs_cat_x_test.npy', arr=xy_test1[0][0])
np.save('C:/_data/dogs-vs-cats/test2/dogs_cat_x_test.npy', arr=xy_test2[0][0])

x_train = np.load('C:/_data/dogs-vs-cats/train/dogs_cat_x_train.npy')
y_train = np.load('C:/_data/dogs-vs-cats/train/dogs_cat_y_train.npy')
x_test = np.load('C:/_data/dogs-vs-cats/test1/dogs_cat_x_test.npy')

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
            # ReduceLROnPlateau(monitor='val_acc', patience=2, verbose=1, factor=0.5, min_lr=0.00001)
            ] 
model.fit(x_train, y_train,
                 batch_size=64,
                 epochs=20,
                validation_split=0.25,
                #  validation_steps=4,
                callbacks=[callbaks],
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

# submission 파일 생성
# y_predict = model.predict(x_test)
# # print(y_predict)
# y_predict = (y_predict > 0.5).astype(int)
# print(y_predict)

# submission_csv = pd.read_csv('C:/_data/dogs-vs-cats/sampleSubmission.csv', index_col=0)
# submission_csv['label'] = y_predict
# submission_csv.to_csv( 'C:/_data/dogs-vs-cats/sampleSubmission_0131.csv')

# 개 사진 1개, 고양이 사진 1개를 인터넷에서 잘라내서 뭔지 맞춰라!!
x_test2 = np.load('C:/_data/dogs-vs-cats/test2/dogs_cat_x_test.npy')
print(x_test2.shape)    # (3, 190, 190, 1)

y_predict = model.predict(x_test2)
y_predict = (y_predict > 0.5).astype(int)
print(y_predict)

# predict          실제
# [[1]:개         고양이
#  [1]:개           개
#  [1]]:개          개