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
    batch_size=5,   
    class_mode='binary',
    color_mode='grayscale',      
    shuffle=True,   # 0과 1의 데이터를 적절히 섞는다.
    # Found 160 images belonging to 2 classes.
)

xy_test = test_datagen.flow_from_directory(    
    './_data/brain/test/',      
    target_size=(100, 100),     
    batch_size=5,   
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
model.add(Conv2D(128, (3,3), activation='relu', input_shape=(100, 100, 1)))
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
hist = model.fit_generator(xy_train,
                           steps_per_epoch=160/5,  # steps_per_epoch = 훈련 샘플 수 / 배치 사이즈 : 1에포당 얼마나 걸을 걷이냐
                           epochs=300,    
                           validation_data=xy_test,
                           validation_steps=4,      # validation_steps <<<<< 찾아보기 
                           callbacks=[es])     

# 4. 평가, 예측
accuracy = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']     # epochs 값 만큼 나온다
val_loss = hist.history['val_loss']

print('loss : ', loss[-1])
print('val_loss : ', val_loss[-1])
print('accuracy : ', accuracy[-1])
print('val_acc : ', val_acc[-1])

# loss :  0.18722914159297943
# val_loss :  0.20249447226524353
# accuracy :  0.918749988079071
# val_acc :  0.949999988079071


# loss :  0.2615147531032562
# val_loss :  0.012746581807732582
# accuracy :  0.8999999761581421
# val_acc :  1.0

# 그림 그려서 확인해본다
# matplotlib
import matplotlib.pyplot as plt 

fig = plt.figure(figsize=(14, 7)) # 그림 사이즈 지정 (가로 14인치, 세로 7인치)
fig.suptitle('acc & loss')

ax1 = fig.add_subplot(2, 1, 1) # 서브플롯들을 2 x 1 배열로 배치 그중 첫번째
ax2 = fig.add_subplot(2, 1, 2)

ax1.plot(accuracy, c='red', label='acc')
ax1.plot(val_acc, c='blue', label='val_acc')

ax2.plot(loss, c='limegreen', label='loss')
ax2.plot(val_loss, c='violet', label='val_loss')
ax1.set_xlabel('epochs')
ax2.set_xlabel('epochs')
plt.grid()
plt.legend()
plt.show()
