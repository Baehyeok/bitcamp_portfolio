import cv2
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
from keras.models import load_model
import tensorflow as tf

car_folder = ['넥스트스파크','말리부','볼트','볼트 EV','스파크','임팔라','캡티바','크루즈','트랙스']

cars_path = 'C:/work/project/car/test'

image_w=28
image_h=28

#img=cv2.imread('C:/work/project/car/test/test_img/NextSpark_1.jpg')
img_array = np.fromfile('C:/work/project/car/test_img/Maliboo_2.jpg',np.uint8)
img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

img=cv2.resize(img,None,fx=image_w/img.shape[1], fy=image_h/img.shape[0])
img = img/256
img=np.expand_dims(img, 0)
print(img.shape)
img = np.array(img)
model = load_model('car_model_binary_cross_5.h5')
predict = model.predict(img)

print(np.argmax(predict[0]))
