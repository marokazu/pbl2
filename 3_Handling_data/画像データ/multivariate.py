#pip install opencv-python

from PIL import Image
import cv2
import numpy as np
from matplotlib import pyplot as plt


image = np.array(Image.open("cat.jpg"))

plt.imshow(image)
plt.show()
