import cv2
import matplotlib as plt

image = cv2.imread('../L11 - Fun with filters/example.jpg')
imagergb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(imagergb)
plt.show()