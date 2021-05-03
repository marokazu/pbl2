from PIL import Image
import cv2
import numpy as np
from matplotlib import pyplot as plt


image = np.array(Image.open("cat.jpg"))
alpha = 1.5#要調整
beta = 30.0#要調整
con = image*alpha+beta 
on = np.clip(con,0,255).astype(np.uint8)

cv2.imwrite('contrast.jpg', on)
plt.imshow(on)
plt.show()
