import numpy as np
from keras.preprocessing.image import ImageDataGenerator

# 1. 데이터
train_datagen =ImageDataGenerator(  
    rescale=1./255,         # 이미지 Min Max scaling
    horizontal_flip=True,   # 수평 반전
    vertical_flip=True,     # 수직으로 반전
    width_shift_range=0.1,  # 옆으로 10% 이동
    height_shift_range=0.1,
    rotation_range=5,   # 회전
    zoom_range=1.2,     # 원래 그림의 20% 확대
    shear_range=0.7,     # 전단
    fill_mode='nearest'     # 수평으로 이동했을 때, 왼쪽 or 오른쪽 끝을 가까이 있는 값으로 채워라
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)


xy_train = train_datagen.flow_from_directory(   
    './_data/brain/train/',     
    target_size=(100, 100),     # 이미지의 크기 100*100
    batch_size=1000,            # 밑에서 fit 으로 훈련을 진행할 거기 때문에 데이터의 전체 숫자를 적는다
                                # 모르면 큰 값 기록
    class_mode='binary',
    color_mode='grayscale',      
    shuffle=True,   # 0과 1의 데이터를 적절히 섞는다.
    # Found 160 images belonging to 2 classes.
)

xy_test = test_datagen.flow_from_directory(    
    './_data/brain/test/',      
    target_size=(100, 100),     
    batch_size=1000,   
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True,
    # Found 120 images belonging to 2 classes.
)

# 2. 모델
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.callbacks import EarlyStopping

model = Sequential()
model.add(Conv2D(64, (3,3), activation='relu', input_shape=(100, 100, 1)))
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

# hist = model.fit_generator(xy_train, steps_per_epoch=16, epochs=5,    # steps_per_epoch = 훈련 샘플 수 / 배치 사이즈 : 1에포당 얼마나 걸을 걷이냐
#                     validation_data=xy_test,
#                     validation_steps=4,)   
es = EarlyStopping(monitor='val_acc',
                   mode='max',
                   restore_best_weights=True,
                   verbose=1,
                   patience=30)
hist = model.fit(xy_train[0][0], xy_train[0][1],
                 batch_size=1,
                #  steps_per_epoch=16,     # steps_per_epoch = 훈련 샘플 수 / 배치 사이즈 : 1에포당 얼마나 걸을 걷이냐
                 epochs=300,
                 validation_data=(xy_test[0][0], xy_test[0][1]),
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

# loss :  0.006500168237835169
# val_loss :  0.5363031029701233
# accuracy :  1.0
# val_acc :  0.8666666746139526
