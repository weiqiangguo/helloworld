import numpy as np
import matplotlib.pyplot as plt
from SimpleNet01 import TwoLayerNet

x = np.arange(-1.0,1.0,0.02)
noise = np.random.randn(x.size)
y = 2 * np.power(x,2) + 1 + 0.1* noise
y1 = 2 * np.power(x,2) + 1

plt.scatter(x, y, label = "XXX")
plt.plot(x, y1, color= 'r', linestyle= "--",label = "yyy")
plt.xlim(-1.0, 1.0)
plt.show()

iters_num = 2000
learning_rate = 0.005
train_loss_list = []

reX = x.reshape(x.size,1)
reY = y.reshape(y.size,1)
network = TwoLayerNet(input_size=1, hidden_size=10,output_size=1)

for i in range(iters_num):
    grad = network.numerical_gradient(reX,reY)
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    if i % 5 == 0:
        loss = network.loss(reX, reY)
        train_loss_list.append(loss)
        # plot and show learning process
        plt.cla()
        plt.text(0.5, 1.0, 'Loss=%.4f' %loss)
        plt.text(0.5, 0.9, 'Iteration=%i' % i)
        plt.scatter(x, y)
        plt.plot(x, y1, 'g',linestyle = "--", lw=3)
        plt.plot(reX, network.predict(reX), 'r-', lw=3)
        plt.pause(0.02)

plt.pause(5)
plt.close()
plt.text(380.0, 20.0, 'Loss')
plt.plot(train_loss_list, color= 'g', label = "zzz")
plt.show()