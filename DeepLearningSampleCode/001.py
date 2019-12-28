import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
print('hello world')
x = np.array([1,2,3,4,5])
y = np.array([6,7,8,9,10])
z = x * y

x = np.arange(0,6,0.1)
y1 = np.sin(x)
y2 = np.cos(x)

plt.plot(x,y1,label = "sin")
plt.plot(x,y2, linestyle = "--",label = "cos")
plt.show()

img = imread("D:\Python_Test\Rose.jpg")
plt.imshow(img)
plt.show()