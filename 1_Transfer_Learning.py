import cv2 as cv
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import models
from keras.models import load_model
import numpy as np
from keras.applications import VGG16

class_mapping = {
    0: 'dislike',
    1: 'exactly',
    2: 'five',
    3: 'left',
    4: 'like',
    5: 'three',
    6: 'two',
    7: 'zero'
}

images_size = 224

conv_base = VGG16(weights = None,
                  include_top = False,
                  input_shape=(images_size, images_size, 3))

conv_base.trainable = True

set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(8, activation='softmax'))

model.load_weights("./Model(1).keras")

cap = cv.VideoCapture(0)

while(True):
  _, frame = cap.read()
  image = cv.resize(frame, (images_size, images_size))
  image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
  image = np.reshape(image, (1, images_size, images_size, 3))
  answer = model.predict(image)
  answer = np.argmax(answer)
  answer = class_mapping[answer]
  cv.putText(frame, str(answer), (50,50), cv.FONT_HERSHEY_PLAIN, 2, (255,0,0))
  cv.imshow("video", frame)
  if cv.waitKey(1) == 27:
      break

cv.destroyAllWindows()
cap.release()