from PIL import Image 
import cv2
import numpy as np
from matplotlib import pyplot as plt


image = np.array(Image.open("cat.jpg"))
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

plt.imshow(gray)
plt.show()
cv2.imwrite('outimage.jpg', gray)
