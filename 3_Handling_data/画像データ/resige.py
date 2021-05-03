from PIL import Image
import cv2
import numpy as np
from matplotlib import pyplot as plt


image = np.array(Image.open("cat.jpg"))
height = image.shape[0]
width = image.shape[1]
simage = cv2.resize(image,(int(width*0.5),int(height*0.5)))

plt.subplot(1,2,1)
plt.imshow(image)
plt.subplot(1,2,2)
plt.imshow(simage)
plt.show()
