import sys,os
import numpy as np
sys.path.append(os.pardir)
from DL.deep_learning.common.functions import *
from DL.deep_learning.common.gradient import numerical_gradient
from DL.deep_learning.dataset.mnist import load_mnist

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size,
                 weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.rand(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.rand(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self,x):
        W1, W2 = self.params['W1'],self.params['W2']
        b1, b2 = self.params['b1'],self.params['b2']
        a1 = np.dot(x,W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1,W2) + b2
        y = softmax(a2)
        return y

    def loss(self,x,t):
        y = self.predict(x)

        return cross_entropy_error(y,t)

    def accuracy(self,x,t):
        y = self.predict(x)
        y = np.argmax(y,axis=1)
        t = np.argmax(t,axis=1)

        accuracy = np.sum(y==t)/float(x.shape[0])
        return accuracy

    def numerical_gradient(self,x,t):
        loss_W = lambda W: self.loss(x,t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W,self.params['W1'])
        grads['W2'] = numerical_gradient(loss_W,self.params['W2'])
        grads['b1'] = numerical_gradient(loss_W,self.params['b1'])
        grads['b2'] = numerical_gradient(loss_W,self.params['b2'])

        return grads

(x_train, t_train), (x_test, t_test) = \
                load_mnist(normalize = True,one_hot_label = True)

train_loss_list = []
train_acc_list = []
test_acc_list = []

print(x_train.shape)
print(x_test.shape)

# hyper parameters
iters_num= 10000
train_size = x_train.shape[0]
print(train_size)
batch_size = 100
learning_late= 0.1
net = TwoLayerNet(input_size=784,hidden_size=100, output_size=10)

# 1에폭당 반복 수
iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    # 미니배치 획득
    # 0~ train_size = 60000중에 batch_size = 100개 고름
    batch_mask = np.random.choice(train_size,batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 기울기 계산
    grad = net.numerical_gradient(x_batch, t_batch)

    # 매개 변수 갱신
    for key in ('W1','b1','W2','b2'):
        net.params[key] -= learning_late * grad[key]

    # 학습 경과 기록
    loss = net.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0 :
        train_acc = net.accuracy(x_train,t_train)
        test_acc = net.accuracy(x_test,t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        print("train acc, test acc : " + str(train_acc) + "," + str(test_acc))
