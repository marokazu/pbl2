from PIL import Image
import cv2
import numpy as np
from matplotlib import pyplot as plt


image = np.array(Image.open("noise.jpg"))
fil = cv2.medianBlur(image, 9)


plt.imshow(fil)
plt.show()
cv2.imwrite('median.jpg', fil)
