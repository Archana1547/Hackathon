import pathlib
import pickle

import numpy as np
import tensorflow as tf
import dlib
import cv2
import SavingModel
from keras.models import load_model

image='test.jpeg'

Test = pathlib.Path(image)
testData=tf.keras.preprocessing.image.load_img(Test, target_size=(SavingModel.imgHeight, SavingModel.imgWidth))
imgArray=tf.keras.preprocessing.image.img_to_array(testData)
imgArray=tf.expand_dims(imgArray,0)


Savedmodel=load_model('model.h5')
predict=Savedmodel.predict(imgArray)
score = tf.nn.softmax(predict[0])

detector = dlib.get_frontal_face_detector()
imgD = cv2.imread(image)
imgD = cv2.resize(imgD, (0, 0), fx=0.25, fy=0.25)
gray = cv2.cvtColor(imgD, cv2.COLOR_BGR2GRAY)
rect = detector(gray, 0)


x1,y1,x2,y2,h,w=0,0,0,0,0,0
for i,d in enumerate(rect):
    x1, y1, x2, y2, h, w = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.height(), d.width()

msg= SavingModel.Class_names[np.argmax(
    score)] + " with a {:.2f} sureity ".format(100 * np.max(score))
cv2.rectangle(imgD, (x1, y1), (x2, y2), (255, 150, 67), 2)
cv2.putText(imgD, msg, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0, 0), 1)
cv2.imshow("Results", imgD)
cv2.waitKey(0)

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(SavingModel.Class_names[np.argmax(score)], 100 * np.max(score))
)
