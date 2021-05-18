from PIL import Image
import cv2
import numpy as np
from matplotlib import pyplot as plt

image = np.array(Image.open("cat.jpg"))
kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]], np.float32)
dst = cv2.filter2D(image,-1,kernel)

plt.imshow(dst)
plt.show()
cv2.imwrite("cat_edge_con.jpg", dst)

image_n = np.array(Image.open("noise.jpg"))
dst = cv2.filter2D(image_n,-1,kernel)

plt.imshow(dst)
plt.show()
cv2.imwrite("cat_edge_con_noise.jpg", dst)