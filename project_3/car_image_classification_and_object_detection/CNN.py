#현재 이미지 딥러닝 모델 생성하는 코드 이므로 영상 적용 작업 필요

from keras.models import Sequential
from keras.layers import Dropout, Activation, Dense
from keras.layers import Flatten, Conv2D, MaxPooling2D
from keras.models import load_model 
import cv2
import numpy as np
from numpy.compat.py3k import npy_load_module
from pratice_sample import car_count
import matplotlib.pyplot as plt

X_train, X_test, Y_train, Y_test = np.load('C:/work/project/car/img_data_2.npy', allow_pickle = True)


model = Sequential()

# model.add(Conv2D(32, (3, 3), activation='relu',border_mode='same',
#             input_shape=X_train.shape[1:]))
model.add(Conv2D(16, (3, 3), activation='relu',
            input_shape=X_train.shape[1:]))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(car_count,activation='softmax'))
model.summary()
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
hist = model.fit(X_train, Y_train, batch_size=32, epochs=100, validation_data=(X_test, Y_test))
model.save('car_model_binary_cross_5.h5')

test_loss, test_acc = model.evaluate(X_test,Y_test)
print(test_loss, test_acc)


import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')

acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
acc_ax.plot(hist.history['val_accuracy'], 'g', label='val acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuray')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()