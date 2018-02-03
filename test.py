import tensorflow as tf
import numpy as np
import pandas as pd
import collections

# tmp = np.reshape(np.asarray(range(24)), [2, 3, 4])
# W = tf.Variable(tf.constant(1, shape=[4, 1], dtype=tf.float32))
#
# # tmp = np.reshape(np.asarray(range(24)), [3, 2, 4])
# # W = tf.Variable(tf.constant(1, shape=[4, 4], dtype=tf.float32))
# img = tf.Variable(tf.random_normal([2, 3]))
#
# tmp_a = tf.constant(tmp, dtype=tf.float32)
#
# sess = tf.Session()
# init = tf.global_variables_initializer()
# sess.run(init)
# # print tmp
# print sess.run(tmp_a)
# print sess.run(tf.shape(tmp_a))
# tmp_b = tf.unstack(tf.transpose(tmp_a, [1, 0, 2]))
# print sess.run(tmp_b)
# print sess.run(tf.shape(tmp_b))
# tmp_c = tf.transpose(tf.stack(tmp_b), [1, 0, 2])
# print sess.run(tmp_c)
# print sess.run(tf.reduce_mean(tmp_c, axis=1))
'''
tmp_a = tf.reshape(tmp_a, [-1, 4])
print sess.run(tmp_a)
print sess.run(tf.shape(tmp_a))

logits = tf.matmul(tmp_a, W)
print sess.run(logits)
print sess.run(tf.shape(logits))

logits = tf.reshape(logits, [-1, 3])
print sess.run(logits)
print sess.run(tf.shape(logits))

signals = tf.nn.softmax(logits)
print sess.run(signals)
print sess.run(tf.shape(signals))

signals = tf.reshape(signals, [-1])
print sess.run(signals)
print sess.run(tf.shape(signals))

result = tf.multiply(tf.transpose(tmp_a), signals)
print sess.run(result)
print sess.run(tf.shape(result))

result = tf.transpose(result)
print sess.run(result)
print sess.run(tf.shape(result))
'''

# mean_a = tf.reduce_mean(tmp_a, axis=1)
# print sess.run(mean_a)
# print sess.run(tf.shape(mean_a))
#
# tmp_a = tf.reshape(tmp_a, [-1, 4])
# print sess.run(tmp_a)
# print sess.run(tf.shape(tmp_a))
#
# mean_a = tf.transpose(mean_a)
# print sess.run(mean_a)
# print sess.run(tf.shape(mean_a))
#
# mediumResult = tf.matmul(tmp_a, W)
# print sess.run(mediumResult)
# print sess.run(tf.shape(mediumResult))
#
# result = tf.matmul(mediumResult, mean_a)
# print sess.run(result)
# print sess.run(tf.shape(result))
#
# result = tf.reshape(result, [-1, 2, 3])
# print sess.run(result)
# print sess.run(tf.shape(result))
#
# result = tf.transpose(result, [0, 2, 1])
# print sess.run(result)
# print sess.run(tf.shape(result))
#
# result = tf.reshape(result, [-1, 2])
# print sess.run(result)
# print sess.run(tf.shape(result))
#
# # result = tf.gather(result, [0, 4, 8])
# result = tf.gather(result, [i for i in range(3**2) if i % (3+1) == 0])
# print sess.run(result)
# print sess.run(tf.shape(result))

# tmp_a = tf.reshape(tmp_a, [-1, 4])
# print sess.run(tmp_a)
# print sess.run(tf.shape(tmp_a))
#
# tmp_b = np.reshape(np.asarray(range(24, 48)), [3, 2, 4])
# tmp_b = tf.constant(tmp_b, dtype=tf.float32)
# print sess.run(tmp_b)
# print sess.run(tf.shape(tmp_b))
#
# tmp_b = tf.reshape(tmp_b, [-1, 4])
# print sess.run(tmp_b)
# print sess.run(tf.shape(tmp_b))
#
# tmp_b_T = tf.transpose(tmp_b)
# print sess.run(tmp_b_T)
# print sess.run(tf.shape(tmp_b_T))
#
# dot_product = tf.matmul(tmp_a, tmp_b_T)
# print sess.run(dot_product)
# print sess.run(tf.shape(dot_product))
#
# dot_product = tf.reshape(dot_product, [-1, 2, 6])
# print sess.run(dot_product)
# print sess.run(tf.shape(dot_product))
#
# dot_product = tf.reduce_sum(dot_product, axis=1)
# print sess.run(dot_product)
# print sess.run(tf.shape(dot_product))
#
# dot_product = tf.reshape(dot_product, [-1, 2])
# print sess.run(dot_product)
# print sess.run(tf.shape(dot_product))
#
# dot_product = tf.gather(dot_product, [i for i in range(3**2) if i % (3+1) == 0])
# print sess.run(dot_product)
# print sess.run(tf.shape(dot_product))
#
# dot_product = tf.transpose(dot_product)
# print sess.run(dot_product)
# print sess.run(tf.shape(dot_product))
#
# dot_product = tf.reshape(dot_product, [-1, 2, 6])
# print sess.run(dot_product)
# print sess.run(tf.shape(dot_product))
#
# dot_product = tf.reduce_sum(dot_product, axis=1)
# print sess.run(dot_product)
# print sess.run(tf.shape(dot_product))
#
# dot_product = tf.reshape(dot_product, [-1, 2])
# print sess.run(dot_product)
# print sess.run(tf.shape(dot_product))

# tmp_a = tf.unstack(tmp_a)
# print sess.run(tmp_a)
# print sess.run(tf.shape(tmp_a))
#
# tmp_b = np.reshape(np.asarray(range(24, 48)), [3, 2, 4])
# tmp_b = tf.constant(tmp_b, dtype=tf.float32)
# tmp_b = tf.unstack(tmp_b)
# print sess.run(tmp_b)
# print sess.run(tf.shape(tmp_b))
#
# result = [tf.matmul(a, tf.transpose(b)) for a, b in zip(tmp_a, tmp_b)]
# print sess.run(result)
# print sess.run(tf.shape(result))
#
# result = tf.stack(result)
# print sess.run(result)
# print sess.run(tf.shape(result))
#
# result = tf.reduce_mean(result, axis=1)
# print sess.run(result)
# print sess.run(tf.shape(result))

# axis = list(range(len(img.get_shape()) - 1))
# mean, variance = tf.nn.moments(img, axis)
# print img.get_shape()
# print sess.run(img)
# print sess.run(mean)
# print sess.run(variance)

# task = 'disagree_agree'
task = 'match_unmatch'
# qrPair_df = pd.read_csv('./data/qrPair.csv')
qrPair_df = pd.read_csv('./data/qrPair_%s.csv' % task)
labelList = [str(label) for label in qrPair_df['%s' % task].values]
# labelList = [str(label) for label in qrPair_df['attacking_respectful'].values]
# labelList = [str(label) for label in qrPair_df['emotion_fact'].values]
# labelList = [str(label) for label in qrPair_df['nasty_nice'].values]
# labelList = [str(label) for label in qrPair_df['topic'].values]

from sklearn.model_selection import train_test_split
_, _, y_train, y_test = train_test_split([0] * len(labelList), labelList, test_size=0.2, random_state=12)
# _, _, y_train, y_valid = train_test_split([0] * len(y_train), y_train, test_size=0.2, random_state=12)

print collections.Counter(y_train).most_common(), collections.Counter(y_train).most_common()[0][1] / float(len(y_train))
# print collections.Counter(y_valid).most_common(), collections.Counter(y_valid).most_common()[0][1] / float(len(y_valid))
print collections.Counter(y_test).most_common(), collections.Counter(y_test).most_common()[0][1] / float(len(y_test))
#
# a=682.0
# b=848.0
# c=215.0
# d=328.0
#
# def average_f1_score(a, b, c, d):
#     precision_1 = a / b
#     recall_1 = a / (a+d-c)
#     precision_2 = c / d
#     recall_2 = c / (c+b-a)
#     result = precision_1 * recall_1 / (precision_1 + recall_1) + precision_2 * recall_2 / (precision_2 + recall_2)
#     return result
#
# print average_f1_score(a, b, c, d)