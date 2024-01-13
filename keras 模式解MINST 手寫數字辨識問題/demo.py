import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
from skimage import io, transform
from skimage.color import rgb2hsv

model = load_model('/content/drive/MyDrive/number/keras_model.h5')  #導入模型

from PIL import Image
im = Image.open("/content/drive/MyDrive/number/5.jpg").convert('L') #導入照片，並轉灰階
im = np.array(im)  #轉成array 
im_test = im.astype('float32')  #型態轉為float
im_test = im_test.reshape((1,28*28))
im_norm = im_test/255
plt.show() 
predictions = np.argmax(model.predict(im_test), axis=-1)
print("辨識為",predictions)
plt.imshow(im)
plt.show() 