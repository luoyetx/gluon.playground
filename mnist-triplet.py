# coding: utf-8
# pylint: disable=invalid-name, no-member

from __future__ import print_function
import argparse
import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon as g
from mxnet.gluon import nn
import matplotlib.pyplot as plt
import seaborn as sns


ctx = mx.gpu(0)
lr = 0.01
epoches = 100
margin = 0.2
seed = 0
np.random.seed(seed)
mx.random.seed(seed)

### Data Part

class MNIST(g.data.vision.MNIST):

    def __init__(self, root, train=True, transform=None, num_per_batch=10):
        super(MNIST, self).__init__(root, train, transform)
        self._data = self._data.asnumpy()
        self.num_per_batch = num_per_batch
        self._label_lst = [[] for _ in range(10)]
        for idx, label in enumerate(self._label):
            self._label_lst[label].append(idx)
        self._label_lst = [np.array(_) for _ in self._label_lst]
        self._label_idx = [0 for _ in range(10)]

    def __getitem__(self, idx):
        data = np.zeros((10*self.num_per_batch, 1, 28, 28), dtype=np.float32)
        label = np.zeros(10*self.num_per_batch, dtype=np.float32)
        for i in range(10):
            label_idx = self._label_idx[i]
            label_lst = self._label_lst[i]
            if label_idx + self.num_per_batch <= len(label_lst):
                b, e = label_idx, label_idx + self.num_per_batch
                idxs = label_lst[range(b, e)]
            else:
                b1, e1 = label_idx, len(label_lst)
                b2, e2 = 0, self.num_per_batch - (e1 - b1)
                idxs = label_lst[range(b1, e1) + range(b2, e2)]
            data[i*self.num_per_batch: (i+1)*self.num_per_batch], _ = self._transform(self._data[idxs], None)
            label[i*self.num_per_batch: (i+1)*self.num_per_batch] = i
            label_idx += self.num_per_batch
            if label_idx >= len(label_lst):
                label_idx = 0
                np.random.shuffle(label_lst)
            self._label_idx[i] = label_idx
            self._label_lst[i] = label_lst
        return data, label

    def __len__(self):
        l = super(MNIST, self).__len__()
        l /= 10 * self.num_per_batch
        return l


def transform(data, label):
    if len(data.shape) == 3:
        data = data.reshape((1, 28, 28)).astype(np.float32)
        data = (data - 128) / 128
    else:
        num = data.shape[0]
        data = data.reshape((num, 1, 28, 28)).astype(np.float32)
        data = (data - 128) / 128
    return data, label


train_loader = g.data.DataLoader(MNIST(root='./data', train=True, transform=transform), batch_size=1, shuffle=False, last_batch='discard')
test_loader = g.data.DataLoader(MNIST(root='./data', train=False, transform=transform), batch_size=1, shuffle=False)


### Model Part

class Net(g.HybridBlock):

    def __init__(self):
        super(Net, self).__init__()
        with self.name_scope():
            self.conv1_1 = nn.Conv2D(20, 3, 1, 1)
            self.bn1_1 = nn.BatchNorm()
            self.conv1_2 = nn.Conv2D(20, 3, 1, 1)
            self.bn1_2 = nn.BatchNorm()
            self.pool1 = nn.MaxPool2D(2, 2)
            self.conv2_1 = nn.Conv2D(40, 3, 1, 1)
            self.bn2_1 = nn.BatchNorm()
            self.conv2_2 = nn.Conv2D(40, 3, 1, 1)
            self.bn2_2 = nn.BatchNorm()
            self.pool2 = nn.MaxPool2D(2, 2)
            self.fc3_1 = nn.Dense(64)
            self.fc3_2 = nn.Dense(2)
            self.bn3 = nn.BatchNorm()

    def hybrid_forward(self, F, x):
        x = self.bn1_1(F.relu(self.conv1_1(x)))
        x = self.bn1_2(F.relu(self.conv1_2(x)))
        x = self.pool1(x)
        x = self.bn2_1(F.relu(self.conv2_1(x)))
        x = self.bn2_2(F.relu(self.conv2_2(x)))
        x = self.pool2(x)
        x = self.fc3_2(F.relu(self.fc3_1(x)))
        x = self.bn3(x)
        return x


net = Net()
net.initialize(mx.init.Normal(), ctx=ctx)
criterion = g.loss.TripletLoss(margin=margin)
trainer = g.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr, 'momentum': 0.9, 'wd': 5e-4})


### Plot Part

def plot_mnist_test(epoch, train=False):
    net = Net()
    net.load_params('./tmp/net-%04d.params'%epoch, ctx=ctx)
    test_loader = g.data.DataLoader(g.data.vision.MNIST(root='./data', train=train, transform=transform), batch_size=32, shuffle=False)
    label_embedding = [[] for _ in range(10)]
    for i, (data, label) in enumerate(test_loader):
        data = data.as_in_context(ctx)
        embedding = net(data)
        for idx, label_idx in enumerate(label):
            label_embedding[label_idx.asscalar()].append(embedding[idx].asnumpy())
    label_embedding = [np.array(_) for _ in label_embedding]
    plt.figure()
    palette = np.array(sns.color_palette('hls', 10))
    for i in range(10):
        plt.scatter(label_embedding[i][:, 0], label_embedding[i][:, 1], c=palette[i], s=10)
    plt.show()


### Train part

class LossRecorder(object):

    def __init__(self, moving_len=100):
        self.moving_len = moving_len
        self.cur_loss = 0
        self.counter = 0
        self.losses = []

    def push(self, loss):
        self.cur_loss = self.cur_loss + loss
        self.counter += 1
        rv = None
        if self.counter == self.moving_len:
            self.losses.append(self.cur_loss / self.moving_len)
            self.cur_loss = 0
            self.counter = 0
            rv = self.losses[-1]
        return rv

    @property
    def loss(self):
        return np.array(self.losses)


def calc_label_index(label):
    label = label.asnumpy()
    label_idx = [[] for _ in range(10)]
    for i, digit in enumerate(label):
        label_idx[int(digit)].append(i)
    neg_idx = [[] for _ in range(10)]
    for i in range(10):
        for j in range(10):
            if i != j:
                neg_idx[i] += label_idx[j]
    label_idx = [np.array(_) for _ in label_idx]
    neg_idx = [np.array(_) for _ in neg_idx]

    return label_idx, neg_idx


def semihard_sample(diss, label, margin):
    label_idx, neg_idx = calc_label_index(label)
    diss = diss.asnumpy()
    anchor_idx = []
    positive_idx = []
    negative_idx = []
    for k in range(10):
        n = len(label_idx[k])
        for i in range(n):
            for j in range(n):
                if i != j:
                    ii = label_idx[k][i]
                    jj = label_idx[k][j]
                    dmin, dmax = diss[ii, jj], diss[ii, jj] + margin
                    neg_dis = diss[ii][neg_idx[k]]
                    ind = np.where(neg_dis < dmax)[0]
                    ind = neg_idx[k][ind]
                    if len(ind) != 0:
                        anchor_idx.append(ii)
                        positive_idx.append(jj)
                        negative_idx.append(np.random.choice(ind))
    return nd.array(anchor_idx, ctx=ctx), nd.array(positive_idx, ctx=ctx), nd.array(negative_idx, ctx=ctx)


def random_sample(embedding, label):
    label_idx, neg_idx = calc_label_index(label)
    anchor_idx = []
    positive_idx = []
    negative_idx = []
    for k in range(10):
        n = len(label_idx[k])
        for i in range(n):
            for j in range(n):
                if i != j:
                    anchor_idx.append(label_idx[k][i])
                    positive_idx.append(label_idx[k][j])
                    negative_idx.append(np.random.choice(neg_idx[k]))
    return nd.array(anchor_idx, ctx=ctx), nd.array(positive_idx, ctx=ctx), nd.array(negative_idx, ctx=ctx)


train_recorder = LossRecorder(moving_len=100)
test_recorder = LossRecorder(moving_len=1)

for epoch in range(epoches):
    print('[epoch %d] lr %f' % (epoch + 1, trainer.learning_rate))
    for i, (data, label) in enumerate(train_loader):
        data = data[0].as_in_context(ctx)
        label = label[0].as_in_context(ctx)
        valid = True
        with autograd.record():
            embedding = net(data)
            with autograd.pause():
                x = embedding
                n = embedding.shape[0]
                x2 = nd.sum(nd.square(x), axis=1)
                xi = x2.reshape((n, 1)).tile(reps=(1, n))
                xj = x2.reshape((1, n)).tile(reps=(n, 1))
                diss = xi + xj - 2 * nd.dot(x, x.T)
                anchor_idx, positive_idx, negative_idx = semihard_sample(diss, label, margin)
            if anchor_idx.size > 0:
                anchor = embedding.take(anchor_idx)
                positive = embedding.take(positive_idx)
                negative = embedding.take(negative_idx)
                loss = criterion(anchor, positive, negative)
            else:
                valid = False
                loss = None
        if valid:
            loss.backward()
            trainer.step(loss.shape[0])
            rv = train_recorder.push(loss.mean().asscalar())
            if rv is not None:
                print('[epoch %d][iter %d] loss: %f' %  (epoch + 1, i + 1, rv))
    test_loss = 0
    test_counter = 0
    for i, (data, label) in enumerate(test_loader):
        data = data[0].as_in_context(ctx)
        label = label[0].as_in_context(ctx)
        embedding = net(data)
        anchor_idx, positive_idx, negative_idx = random_sample(embedding, label)
        anchor = embedding.take(anchor_idx)
        positive = embedding.take(positive_idx)
        negative = embedding.take(negative_idx)
        loss = criterion(anchor, positive, negative)
        test_loss += loss.mean().asscalar()
        test_counter += 1
    test_loss /= test_counter
    test_recorder.push(test_loss)
    print('[epoch %d][test] loss: %f' % (epoch + 1, test_loss))
    trainer.set_learning_rate(trainer.learning_rate * 0.98)
    net.save_params('./tmp/net-%04d.params'%(epoch + 1))
    if epoch % 10 == 9:
        plot_mnist_test(epoch+1, True)
        plot_mnist_test(epoch+1, False)
