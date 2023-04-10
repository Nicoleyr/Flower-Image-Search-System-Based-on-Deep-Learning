# encoding:utf-8
from scipy.io import loadmat as load
import numpy as np
import matplotlib.pyplot as plt

def reformat(samples, labels):
    #  改变数据的形状
    #（图片高，图片宽，通道数，图片数） -》 （图片数，图片宽，图片高，通道数）
    #（   0,    1,    2,      3）  -》 （   3，   0,    1,      2）
    new = np.transpose(samples,(3,0,1,2)).astype(np.float32)

    #labels 变成 one-hot encoding
    labels = np.array([x[0] for x in labels])
    one_hot_labels = []

    for num in labels:
        one_hot = [0.0]*10
        if num ==10:
            one_hot[0]=1.0
        else:
            one_hot[num]=1.0
        one_hot_labels.append(one_hot)
    labels = np.array(one_hot_labels)
    return new, labels


def normalize(samples):
    # (R+G+B)/3
    # 0~255 -> -1.0~1.0
    a = np.add.reduce(samples, keepdims=True, axis=3)
    a = a/3.0
    return a/128.0-1.0


def distribution(labels, name):
    # 查看一下每个label的分布，再画个统计图
    # keys:
    # 0
    # 1
    # 2
    # ...
    # 9
    count = {}
    for label in labels:
        key = 0 if label[0] == 10 else label[0]
        if key in count:
            count[key] += 1
        else:
            count[key] = 1
    x = []
    y = []
    for k, v in count.items():
        # print(k, v)
        x.append(k)
        y.append(v)

    y_pos = np.arange(len(x))
    plt.bar(y_pos, y, align='center', alpha=0.5)
    plt.xticks(y_pos, x)
    plt.ylabel('Count')
    plt.title(name + ' Label Distribution')
    plt.show()


def inspect(dataset, labels, i):
    print(labels[i])
    plt.imshow(dataset[i].squeeze())
    plt.show()


train = load('../../2/data/train_32x32.mat')
test = load('../../2/data/test_32x32.mat')

train_samples = train['X']
train_labels = train['y']

test_samples = test['X']
test_labels = test['y']

n_train_samples, _train_labels = reformat(train_samples,train_labels)
n_test_samples, _test_labels = reformat(test_samples, test_labels)

_train_samples = normalize(n_train_samples)
_test_samples = normalize(n_test_samples)



num_labels = 10
image_size = 32
num_channels = 1



if __name__ == '__main__':
    _train_samples = normalize(_train_samples)
    inspect(_train_samples, _train_labels, 100)
    distribution(train_labels, 'Train Labels')

