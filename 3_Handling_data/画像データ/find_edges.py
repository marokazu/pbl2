from PIL import Image
import cv2
import numpy as np
from matplotlib import pyplot as plt

image = np.array(Image.open("cat.jpg"))
edge = cv2.Canny(image, 50, 110)

plt.imshow(edge)
plt.show()
cv2.imwrite("cat_edge.jpg", edge)


image_n = np.array(Image.open("noise.jpg"))
edge = cv2.Canny(image_n, 50, 110)

plt.imshow(edge)
plt.show()
cv2.imwrite("cat_edge_noise.jpg", edge)
