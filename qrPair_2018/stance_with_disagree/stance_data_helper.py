import pandas as pd
import collections
import numpy as np
from itertools import chain
import pickle

def read_csv(data_type, topic):
    qrPair_df = pd.read_csv('../data/%s.csv' % data_type) # data on server
    if topic != None:
        qrPair_df = qrPair_df[qrPair_df.topic == topic]

    qrPair_df['quoteText'] = qrPair_df['quoteText'].apply(lambda text: [word for word in str(text).split() if word != ''])
    qrPair_df['responseText'] = qrPair_df['responseText'].apply(lambda text: [word for word in str(text).split() if word != ''])

    print(topic, collections.Counter(list(qrPair_df.text_disagree_agree.values)).most_common(), len(qrPair_df))
    return qrPair_df

def data_save(data_type, topic, max_len, wordFilter=True):

    df_data = read_csv(data_type, topic)
    allTextList = np.concatenate((df_data['quoteText'].values, df_data['responseText'].values), axis=0)
    wordList = list(chain(*allTextList))

    if not wordFilter:
        wordSet = [word for (word, count) in collections.Counter(wordList).most_common()]
    else:
        wordSet = [word for (word, count) in collections.Counter(wordList).most_common() if count > 2]
        wordSet.append('UNK')

        def processUNK(text):
            tmp = []
            for word in text:
                if word in wordSet:
                    tmp.append(word)
                else:
                    tmp.append('UNK')
            return tmp

        df_data['quoteText'] = df_data['quoteText'].apply(processUNK)
        df_data['responseText'] = df_data['responseText'].apply(processUNK)

    print('vocabulary size = {}'.format(len(wordSet)))

    wordIDSet = range(1, len(wordSet) + 1)

    label = ['0', '1']
    labelIDSet = range(len(label))

    word2id = pd.Series(wordIDSet, index=wordSet)
    id2word = pd.Series(wordSet, index=wordIDSet)
    label2id = pd.Series(labelIDSet, index=label)
    id2label = pd.Series(label, index=labelIDSet)

    def X_padding(words):
        ids = list(word2id[words])
        if len(ids) >= max_len:
            return ids[:max_len]
        ids.extend([0] * (max_len - len(ids)))
        return ids

    def y_padding(label):
        id = label2id[label]
        return id

    df_data['X_quote'] = df_data['quoteText'].apply(X_padding)
    df_data['X_response'] = df_data['responseText'].apply(X_padding)

    df_data['y_qstance'] = df_data['quote_stance'].apply(y_padding)
    df_data['y_rstance'] = df_data['response_stance'].apply(y_padding)

    df_data['y_pair'] = df_data['text_disagree_agree'].apply(y_padding)

    X = df_data.loc[:, ['X_quote', 'X_response']].as_matrix()
    y = df_data.loc[:, ['y_qstance', 'y_rstance', 'y_pair']].as_matrix()

    with open('./data/%s_%s_%s.pkl' % (data_type, topic, max_len), 'wb') as outp:
        pickle.dump(X, outp)
        pickle.dump(y, outp)
        pickle.dump(word2id, outp)
        pickle.dump(id2word, outp)
        pickle.dump(label2id, outp)
        pickle.dump(id2label, outp)
    print('** Finished saving the %s_%s_%s data.' % (data_type, topic, max_len))

class BatchGenerator(object):
    def __init__(self, X, y, shuffle=False):

        if type(X) != np.ndarray:
            X = np.asarray(X)
        if type(y) != np.ndarray:
            y = np.asarray(y)

        self._X = X
        self._y = y
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._number_examples = self._X.shape[0]
        self._shuffle = shuffle
        if self._shuffle:
            new_index = np.random.permutation(self._number_examples)
            self._X = self._X[new_index]
            self._y = self._y[new_index]

    @property
    def X(self):
        return self._X

    @property
    def y(self):
        return self._y

    @property
    def num_examples(self):
        return self._number_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._number_examples:
            self._epochs_completed += 1
            if self._shuffle:
                new_index = np.random.permutation(self._number_examples)
                self._X = self._X[new_index]
                self._y = self._y[new_index]
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._number_examples
        end = self._index_in_epoch

        X = self._X[start:end]
        y = self._y[start:end]
        X_q_batch, X_r_batch = zip(*X)

        return np.asarray(X_q_batch), np.asarray(X_r_batch), y

# def data_save(data_type, topic, max_len, wordFilter=True):
#
#     df_data = read_csv(data_type, topic)
#     allTextList = np.concatenate((df_data['quoteText'].values, df_data['responseText'].values), axis=0)
#     wordList = list(chain(*allTextList))
#
#     if not wordFilter:
#         wordSet = [word for (word, count) in collections.Counter(wordList).most_common()]
#     else:
#         wordSet = [word for (word, count) in collections.Counter(wordList).most_common() if count > 2]
#         wordSet.append('UNK')
#
#         def processUNK(text):
#             tmp = []
#             for word in text:
#                 if word in wordSet:
#                     tmp.append(word)
#                 else:
#                     tmp.append('UNK')
#             return tmp
#
#         df_data['quoteText'] = df_data['quoteText'].apply(processUNK)
#         df_data['responseText'] = df_data['responseText'].apply(processUNK)
#
#     print('vocabulary size = {}'.format(len(wordSet)))
#
#     wordIDSet = range(1, len(wordSet) + 1)
#
#     label = ['0', '1']
#     labelIDSet = range(len(label))
#
#     word2id = pd.Series(wordIDSet, index=wordSet)
#     id2word = pd.Series(wordSet, index=wordIDSet)
#     label2id = pd.Series(labelIDSet, index=label)
#     id2label = pd.Series(label, index=labelIDSet)
#
#     def X_padding(words):
#         ids = list(word2id[words])
#         if len(ids) >= max_len:
#             return ids[:max_len]
#         ids.extend([0] * (max_len - len(ids)))
#         return ids
#
#     def y_padding(label):
#         id = label2id[label]
#         return id
#
#     df_data['X_quote'] = df_data['quoteText'].apply(X_padding)
#     df_data['X_response'] = df_data['responseText'].apply(X_padding)
#
#     df_data['y_qstance'] = df_data['quote_stance'].apply(y_padding)
#     df_data['y_rstance'] = df_data['response_stance'].apply(y_padding)
#
#     X = np.concatenate((df_data['X_quote'].as_matrix(), df_data['X_response'].as_matrix()), axis=0)
#     y = np.concatenate((df_data['y_qstance'].as_matrix(), df_data['y_rstance'].as_matrix()), axis=0)
#
#     with open('./data/%s_%s_%s.pkl' % (data_type, topic, max_len), 'wb') as outp:
#         pickle.dump(X, outp)
#         pickle.dump(y, outp)
#         pickle.dump(word2id, outp)
#         pickle.dump(id2word, outp)
#         pickle.dump(label2id, outp)
#         pickle.dump(id2label, outp)
#     print('** Finished saving the %s_%s_%s data.' % (data_type, topic, max_len))
#
# class BatchGenerator(object):
#     def __init__(self, X, y, shuffle=False):
#
#         if type(X) != np.ndarray:
#             X = np.asarray(X)
#         if type(y) != np.ndarray:
#             y = np.asarray(y)
#
#         self._X = X
#         self._y = y
#         self._epochs_completed = 0
#         self._index_in_epoch = 0
#         self._number_examples = self._X.shape[0]
#         self._shuffle = shuffle
#         if self._shuffle:
#             new_index = np.random.permutation(self._number_examples)
#             self._X = self._X[new_index]
#             self._y = self._y[new_index]
#
#     @property
#     def X(self):
#         return self._X
#
#     @property
#     def y(self):
#         return self._y
#
#     @property
#     def num_examples(self):
#         return self._number_examples
#
#     @property
#     def epochs_completed(self):
#         return self._epochs_completed
#
#     def next_batch(self, batch_size):
#         start = self._index_in_epoch
#         self._index_in_epoch += batch_size
#         if self._index_in_epoch > self._number_examples:
#             self._epochs_completed += 1
#             if self._shuffle:
#                 new_index = np.random.permutation(self._number_examples)
#                 self._X = self._X[new_index]
#                 self._y = self._y[new_index]
#             start = 0
#             self._index_in_epoch = batch_size
#             assert batch_size <= self._number_examples
#         end = self._index_in_epoch
#
#         X = self._X[start:end]
#         y = self._y[start:end]
#         X_batch = [np.asarray(i) for i in X]
#
#         return np.asarray(X_batch), y

if __name__ == '__main__':

    # read_csv(data_type='iac_stance', topic='abortion')
    for topic in ['evolution', 'abortion', 'gun control', 'gay marriage', 'existence of God']:
        data_save(data_type='iac_stance',
                  topic=topic,
                  max_len=64)

    # with open('./data/iac_stance_evolution_64.pkl', 'rb') as fr:
    #     X = pickle.load(fr)
    #     y = pickle.load(fr)
    #     word2id = pickle.load(fr)
    #     id2word = pickle.load(fr)
    #     label2id = pickle.load(fr)
    #     id2label = pickle.load(fr)
    #
    # np.random.seed(12)
    # new_index = np.random.permutation(len(X))
    # X = X[new_index]
    # y = y[new_index]
    #
    # tr_batch_size = 32
    # max_epoch = 100
    #
    # data_train = BatchGenerator(X, y, shuffle=True)
    #
    # tr_batch_num = int(data_train.y.shape[0] / tr_batch_size)
    #
    # for epoch in range(1, max_epoch + 1):
    #     for batch in range(tr_batch_num):
    #         # X_q_batch, X_r_batch, y_batch = data_train.next_batch(tr_batch_size)
    #         X_batch, y_batch = data_train.next_batch(tr_batch_size)