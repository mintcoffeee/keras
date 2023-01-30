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
# ImageDataGenerator :
# 사진 이미지를 > 수치로 바꿔주는 역할
# 데이터를 수정해서 증폭
# 같은 데이터를 수정하지 않고 증폭하면 과적합 문제가 발생할 수 있다.

test_datagen = ImageDataGenerator(
    rescale=1./255
)
# test 데이터는 scaling만 한다.
# test 데이터는 증폭을 할 필요가 없다.
# test 데이터는 평가모델을 위해 쓰이기 때문에 데이터를 증폭시키지 않고 사용 -> 실제 데이터를 사용해야 한다.

xy_train = train_datagen.flow_from_directory(    # 폴더에 있는 이미지 데이터를 가져오겠다. / dirctory : 폴더
    './_data/brain/train/',      # 폴더를 인식, ad -> 0 , noraml -> 1
    # x = (160,150,150,1) = (N:데이터 개수, 150, 150(이미지 크기), 1(흑백))
    # y = (160,)
    # np.unique : [0 : 80개, 1 : 80개] 
    target_size=(100, 100),     # 이미지의 크기가 다르더라도, 동일하게 200 * 200 증폭 or 축소 시킨다.
    batch_size=10,   # 훈련전에 배치사이즈를 미리 분리한다.   
    # 파이토치는 데이터를 미리 분리
    class_mode='binary',
    color_mode='grayscale',      # 끝자리가 0
    shuffle=True,   # 0과 1의 데이터를 적절히 섞는다.
    # Found 160 images belonging to 2 classes.
)

xy_test = test_datagen.flow_from_directory(    
    './_data/brain/test/',      
    target_size=(100, 100),     
    batch_size=10,   
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True,
    # Found 120 images belonging to 2 classes.
)


# 2. 모델
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(64, (2,2), activation='relu', input_shape=(100, 100, 1)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['acc'])

hist = model.fit_generator(xy_train, steps_per_epoch=16, epochs=5,    # steps_per_epoch = 훈련 샘플 수 / 배치 사이즈 : 1에포당 얼마나 걸을 걷이냐
                    validation_data=xy_test,
                    validation_steps=4,)     # validation_steps <<<<< 찾아보기 

# 4. 평가, 예측
accuracy = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']     # epochs 값 만큼 나온다
val_loss = hist.history['val_loss']

print('loss : ', loss[-1])
print('val_loss : ', val_loss[-1])
print('accuracy : ', accuracy[-1])
print('val_acc : ', val_acc[-1])


# loss = model.evaluate(xy_test)
# print('loss : ', loss)

# loss :  [0.6931626796722412, 0.5]