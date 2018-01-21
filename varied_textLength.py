import pandas as pd
import numpy as np
from itertools import chain
import collections
import pickle
from sklearn.model_selection import train_test_split

def dataSave():
    task = 'disagree_agree'
    qrPair_df = pd.read_csv('./data/qrPair_%s.csv' % task)

    quoteTextList = [[word for word in str(text).split() if word != ''] for text in qrPair_df['quoteText'].values]
    responseTextList = [[word for word in str(text).split() if word != ''] for text in qrPair_df['responseText'].values]
    labelList = [str(label) for label in qrPair_df['disagree_agree'].values]
    df_data = pd.DataFrame({'quoteText': quoteTextList, 'responseText': responseTextList, 'label': labelList}, index=range(len(quoteTextList)))

    allTextList = np.concatenate((df_data['quoteText'].values, df_data['responseText'].values), axis=0)

    wordList = list(chain(*allTextList))

    wordSet = [word for (word, count) in collections.Counter(wordList).most_common()] # 27589 42487

    print 'vocabulary size = {}'.format(len(wordSet))

    idSet = range(1, len(wordSet) + 1)

    label = ['0', '1']
    labelIDSet = range(len(label))

    word2id = pd.Series(idSet, index=wordSet)
    id2word = pd.Series(wordSet, index=idSet)
    label2id = pd.Series(labelIDSet, index=label)
    id2label = pd.Series(label, index=labelIDSet)

    max_len = 150
    def X_padding(words):
        ids = list(word2id[words])
        if len(ids) >= max_len:
            return ids[:max_len]
        # ids.extend([0]*(max_len-len(ids)))
        return ids

    def y_padding(label):
        id = label2id[label]
        return id

    df_data['X_quote'] = df_data['quoteText'].apply(X_padding)
    df_data['X_response'] = df_data['responseText'].apply(X_padding)
    df_data['y'] = df_data['label'].apply(y_padding)

    X_quote = np.asarray(list(df_data['X_quote'].values))
    X_response = np.asarray(list(df_data['X_response'].values))
    y = np.asarray(list(df_data['y'].values))

    with open('./data/data_variedLen_%s.pkl' % task, 'wb') as outp:
        pickle.dump(X_quote, outp)
        pickle.dump(X_response, outp)
        pickle.dump(y, outp)
        pickle.dump(word2id, outp)
        pickle.dump(id2word, outp)
        pickle.dump(label2id, outp)
        pickle.dump(id2label, outp)
    print '** Finished saving the data.'

def textLenDistribution():
    df_data['quote_len'] = df_data['quoteText'].apply(lambda x: len(x))
    df_data['response_len'] = df_data['responseText'].apply(lambda x: len(x))

    import matplotlib.pyplot as plt
    df_data['quote_len'].hist(bins=100)
    plt.xlim(0, 300)
    plt.xlabel('sentence_length')
    plt.ylabel('sentence_num')
    plt.title('Distribution of the Length of Sentence')
    plt.show()

    df_data['response_len'].hist(bins=100)
    plt.xlim(0, 300)
    plt.xlabel('sentence_length')
    plt.ylabel('sentence_num')
    plt.title('Distribution of the Length of Sentence')
    plt.show()

# def splitTrainAndTest():
#     task = 'disagree_agree'
#     with open('./data/data_variedLen_%s.pkl' % task, 'rb') as fr:
# 
#         X_quote = pickle.load(fr)
#         X_response = pickle.load(fr)
#         y = pickle.load(fr)
#         _ = pickle.load(fr)
#         _ = pickle.load(fr)
#         _ = pickle.load(fr)
#         _ = pickle.load(fr)
# 
#     print 'X_quote.shape={}, X_response.shape={}, y.shape={}'.format(X_quote.shape, X_response.shape, y.shape)
# 
#     X_QR = zip(X_quote, X_response)
# 
#     X_QR_train, X_QR_test, y_train, y_test = train_test_split(X_QR, y, test_size=0.2, random_state=12)
# 
#     X_quote_train = np.asarray([pair[0] for pair in X_QR_train])
#     X_response_train = np.asarray([pair[1] for pair in X_QR_train])
# 
#     X_quote_test = np.asarray([pair[0] for pair in X_QR_test])
#     X_response_test = np.asarray([pair[1] for pair in X_QR_test])
# 
#     print 'X_quote_train.shape={}, X_response_train.shape={}, y_train.shape={};\nX_quote_test.shape={}, X_response_test.shape={}, y_test.shape={}'.format(X_quote_train.shape, X_response_train.shape, y_train.shape, X_quote_test.shape, X_response_test.shape, y_test.shape)
# 
#     return X_quote_train, X_response_train, y_train, X_quote_test, X_response_test, y_test

def splitTrainAndTest():
    task = 'disagree_agree'
    max_len = 150
    with open('./data/data_maxLen_%s_%s.pkl' % (max_len, task), 'rb') as fr:
        X_quote = pickle.load(fr)
        X_response = pickle.load(fr)
        y = pickle.load(fr)
        word2id = pickle.load(fr)
        id2word = pickle.load(fr)
        label2id = pickle.load(fr)
        id2label = pickle.load(fr)

    print 'X_quote.shape={}, X_response.shape={}, y.shape={}'.format(X_quote.shape, X_response.shape, y.shape)

    X_QR = np.hstack((X_quote, X_response))

    X_QR_train, X_QR_test, y_train, y_test = train_test_split(X_QR, y, test_size=0.2, random_state=12)

    X_quote_train = X_QR_train[:, :max_len]
    X_response_train = X_QR_train[:, max_len:]

    X_quote_test = X_QR_test[:, :max_len]
    X_response_test = X_QR_test[:, max_len:]

    print 'X_quote_train.shape={}, X_response_train.shape={}, y_train.shape={};\nX_quote_test.shape={}, X_response_test.shape={}, y_test.shape={}'.format(X_quote_train.shape, X_response_train.shape, y_train.shape, X_quote_test.shape, X_response_test.shape, y_test.shape)
    
    return X_quote_train, X_response_train, y_train, X_quote_test, X_response_test, y_test


class BatchGenerator(object):
    def __init__(self, X_quote, X_response, y, shuffle=False):

        if type(X_quote) != np.ndarray:
            X_quote = np.asarray(X_quote)
        if type(X_response) != np.ndarray:
            X_response = np.asarray(X_response)
        if type(y) != np.ndarray:
            y = np.asarray(y)

        self._X_quote = X_quote
        self._X_response = X_response
        self._y = y
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._number_examples = self._X_quote.shape[0]
        self._shuffle = shuffle
        if self._shuffle:
            new_index = np.random.permutation(self._number_examples)
            self._X_quote = self._X_quote[new_index]
            self._X_response = self._X_response[new_index]
            self._y = self._y[new_index]

    @property
    def X_quote(self):
        return self._X_quote

    @property
    def X_response(self):
        return self._X_response

    @property
    def y(self):
        return self._y

    @property
    def num_examples(self):
        return self._number_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    # def next_batch(self, batch_size):
    #     start = self._index_in_epoch
    #     self._index_in_epoch += batch_size
    #     if self._index_in_epoch > self._number_examples:
    #         self._epochs_completed += 1
    #         if self._shuffle:
    #             new_index = np.random.permutation(self._number_examples)
    #             self._X_quote = self._X_quote[new_index]
    #             self._X_response = self._X_response[new_index]
    #             self._y = self._y[new_index]
    #         start = 0
    #         self._index_in_epoch = batch_size
    #         assert batch_size <= self._number_examples
    #     end = self._index_in_epoch
    #     return self._X_quote[start:end], self._X_response[start:end], self._y[start:end]
    
    def next_batch(self, batch_size):
        start = self._index_in_epoch
        if start >= self._number_examples:
            self._epochs_completed += 1
            if self._shuffle:
                new_index = np.random.permutation(self._number_examples)
                self._X_quote = self._X_quote[new_index]
                self._X_response = self._X_response[new_index]
                self._y = self._y[new_index]
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._number_examples
        else:
            self._index_in_epoch += batch_size
        end = min(self._index_in_epoch, self.num_examples)
        return self._X_quote[start:end], self._X_response[start:end], self._y[start:end]

X_quote_train, X_response_train, y_train, X_quote_test, X_response_test, y_test = splitTrainAndTest()
print 'Creating the data generator ...'
# data_train = BatchGenerator(X_quote_train, X_response_train, y_train, shuffle=True)
data_train = BatchGenerator(X_quote_train, X_response_train, y_train, shuffle=False)
data_test = BatchGenerator(X_quote_test, X_response_test, y_test, shuffle=False)
print 'Finished creating the generator.'

def getDataSet():
    return data_train, data_test

if __name__ == '__main__':
    # dataSave()
    data_train, data_test = getDataSet()

    tr_batch_size = 32
    max_epoch = 30
    max_max_epoch = 100
    display_num = 5
    tr_batch_num = int(data_train.y.shape[0] / tr_batch_size) + 1
    display_batch = int(tr_batch_num / display_num)

    for epoch in range(1, max_max_epoch+1):
        print 'EPOCH {}'.format(epoch)
        for batch in xrange(tr_batch_num):
            if batch == tr_batch_num - 1:
                print batch
            X_q_batch, X_r_batch, y_batch = data_train.next_batch(tr_batch_size)

            if (batch + 1) % display_batch == 0:
                data_size = data_test.y.shape[0]
                _batch_size = data_size
                batch_num = int(data_size / _batch_size)
                for i in xrange(batch_num):
                    X_q_batch, X_r_batch, y_batch = data_test.next_batch(_batch_size)
