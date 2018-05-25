import tensorflow as tf
import torch
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
import numpy as np

if __name__ == '__main__':

    # torch.manual_seed(12)
    #
    # a = Variable(torch.randn([5, 4]), requires_grad=True)
    # b = Variable(torch.randn([5, 4]) + 1, requires_grad=True)
    # y = Variable(torch.LongTensor(np.asarray([0, 1, 1, 0, 1]))).view(-1, 1)
    # y = y.float()
    # print(a, b, y)
    # # print(torch.sum(torch.pow(a - b, 2), dim=1))
    # euclidean_distance = F.pairwise_distance(a, b)
    # # print(1, euclidean_distance)
    # #
    # # print(2, torch.pow(euclidean_distance, 2))
    # # print(3, 2.0 - euclidean_distance)
    # # print(4, y)
    # # print(5, y * torch.pow(euclidean_distance, 2))
    # # print(5.1, torch.clamp(2.0 - euclidean_distance, min=0.0))
    # # print(6, (1 - y) * torch.pow(torch.clamp(2.0 - euclidean_distance, min=0.0), 2))
    # loss_contrastive = y * torch.pow(euclidean_distance, 2) + (1 - y) * torch.pow(torch.clamp(2.0 - euclidean_distance, min=0.0), 2)
    # print(7, loss_contrastive)
    # print(8, torch.mean(loss_contrastive))

    # np.random.seed(12)
    # for i in range(10):
    #     print(np.random.permutation(10))

    ones = tf.ones([3, 1, 1])
    left = tf.constant(np.reshape(np.asarray(range(18)), [3, 2, 3]), dtype=tf.float32)
    right = tf.constant(np.reshape(np.asarray(range(6)), [2, 3]), dtype=tf.float32)
    sess = tf.Session()
    print(sess.run(ones))
    print(sess.run(left))
    print(sess.run(right))
    print(sess.run(ones * right))
    print(sess.run(tf.shape(ones * right)))
    print(sess.run(left + ones * right))
    print(sess.run(tf.transpose(left, [1, 0, 2])))
