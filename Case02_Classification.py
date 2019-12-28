import numpy as np
import matplotlib.pyplot as plt
from SimpleNet02 import TwoLayerNet
from mnist import load_mnist

def img_show(img, pauseTime = 5):
    plt.imshow(img,cmap=plt.cm.gray)
    if pauseTime > 0:
        plt.pause(pauseTime)
        plt.close()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, one_hot_label=True, normalize=True)

# 超参数#
iters_num = 5000
batch_size = 100
learning_rate = 0.1
# 平均每个epoch的重复次数
#iter_per_epoch = max(train_size / batch_size, 1)
iter_per_epoch = 100
train_size = x_train.shape[0]

train_loss_list = []
train_acc_list = []
test_acc_list = []

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

for i in range(iters_num):
    #获取mini-batch
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    # 计算梯度
    #grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)
    # 更新参数
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # 记录学习过程
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        loss = network.loss(x_batch, t_batch)

        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        train_loss_list.append(loss)
        print("train_acc=%.4f" %train_acc, "test_acc=%.4f " %test_acc)

        plt.cla()
        plt.subplot(211);plt.plot(train_loss_list, 'r', lw=1);plt.xlim(0, iters_num/iter_per_epoch);
        plt.ylim(0, 3);plt.xlabel("train_loss")
        plt.subplot(223);plt.plot(train_acc_list, 'g', lw=2);plt.xlim(0, iters_num/iter_per_epoch);
        plt.ylim(0, 1);plt.xlabel("train_acc")
        plt.subplot(224);plt.plot(test_acc_list, 'b', lw=2);plt.xlim(0, iters_num/iter_per_epoch);
        plt.ylim(0, 1);plt.xlabel("test_acc")
        plt.text(30, 0.1, 'Iteration=%i' % i)
        plt.tight_layout()
        plt.pause(0.01)

plt.show()

#直观的验证网络的预测能力
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
for i in range(10):
    value = network.predict(x_test[i])
    print(np.argmax(value))
    img = x_test[i].reshape(28, 28)
    img_show(img)
