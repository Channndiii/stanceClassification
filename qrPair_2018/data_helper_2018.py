import pandas as pd
import re
import numpy as np
from itertools import chain
import collections
import pickle
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
'''
def read_csv(task):
    qrPair_df = pd.read_csv('./data/qrPair_2018.csv')

    def labelFilter(row):
        label = row['%s' % task]
        if label > -2.0 and label <= -1.0:
            filterLabel = True
        else:
            filterLabel = False
        row['filterLabel'] = filterLabel
        return row
    
    qrPair_df = qrPair_df.apply(labelFilter, axis=1)
    qrPair_df = qrPair_df[qrPair_df.filterLabel == False]
    
    quoteTextList = [[word for word in str(text).split() if word != ''] for text in qrPair_df['quoteText'].values]
    responseTextList = [[word for word in str(text).split() if word != ''] for text in qrPair_df['responseText'].values]
    
    labelList = []
    for label in qrPair_df['%s' % task].values:
        if label <= -2.0:
            labelList.append('0')
        if label >= 1.0:
            labelList.append('2')
        if label > -1.0 and label < 1.0:
            labelList.append('1')
    df_data = pd.DataFrame({'quoteText': quoteTextList, 'responseText': responseTextList, 'label': labelList}, index=range(len(quoteTextList)))
    
    print(collections.Counter(list(df_data.label.values)).most_common(), len(df_data))
    return df_data

def data_save(task, max_len, wordFilter):

    df_data = read_csv(task)
    allTextList = np.concatenate((df_data['quoteText'].values, df_data['responseText'].values), axis=0)
    wordList = list(chain(*allTextList))
    
    if not wordFilter:
        wordSet = [word for (word, count) in collections.Counter(wordList).most_common()]
    else:
        wordSet = [word for (word, count) in collections.Counter(wordList).most_common() if count > 2]
        wordSet.append('unkWord')
        def processUNK(text):
            tmp = []
            for word in text:
                if word in wordSet:
                    tmp.append(word)
                else:
                    tmp.append('unkWord')
            return tmp
        df_data['quoteText'] = df_data['quoteText'].apply(processUNK)
        df_data['responseText'] = df_data['responseText'].apply(processUNK)
    
    print('vocabulary size = {}'.format(len(wordSet)))

    wordIDSet = range(1, len(wordSet) + 1)

    label = ['0', '1', '2']
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
    df_data['y'] = df_data['label'].apply(y_padding)

    X = np.asarray(list(zip(list(df_data['X_quote'].values), list(df_data['X_response'].values))))
    y = np.asarray(list(df_data['y'].values))

    with open('./data/data_%s.pkl' % (task), 'wb') as outp:
        pickle.dump(X, outp)
        pickle.dump(y, outp)
        pickle.dump(word2id, outp)
        pickle.dump(id2word, outp)
        pickle.dump(label2id, outp)
        pickle.dump(id2label, outp)
    print('** Finished saving the %s data.' % task)

def splitTrainAndTest(task, resampleFlag):
    with open('./data/data_%s.pkl' % (task), 'rb') as fr:
        X = pickle.load(fr)
        y = pickle.load(fr)
        word2id = pickle.load(fr)
        id2word = pickle.load(fr)
        label2id = pickle.load(fr)
        id2label = pickle.load(fr)
    
    print('X.shape={}, y.shape={}'.format(X.shape, y.shape))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)
    
    def train_resample(data):
        data_0 = [d for d in data if d[1] == 0]
        data_1 = [d for d in data if d[1] == 1]
        data_2 = [d for d in data if d[1] == 2]
        n_samples = max([len(data_0), len(data_1), len(data_2)])
        result = []
        for data in [data_0, data_1, data_2]:
            if len(data) < n_samples:
                data = resample(data, replace=True, n_samples=n_samples, random_state=12)
            result.extend(data)
        return result
    
    if resampleFlag:
        X_train, y_train = list(zip(*train_resample(list(zip(X_train, y_train)))))
        X_train, y_train = np.asarray(X_train), np.asarray(y_train)

    print('X_train.shape={}, y_train.shape={};\nX_test.shape={}, y_test.shape={}'.format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))

    return X_train, X_test, y_train, y_test, word2id

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

def getDataSet(task, resampleFlag):
    X_train, X_test, y_train, y_test, word2id = splitTrainAndTest(task, resampleFlag)
    print('Creating the data generator ...')
    data_train = BatchGenerator(X_train, y_train, shuffle=True)
    data_test = BatchGenerator(X_test, y_test, shuffle=False)
    print('Finished creating the generator.')
    return data_train, data_test, word2id
'''

def read_csv(task, topic):
    qrPair_df = pd.read_csv('./data/qrPair_2018.csv') # data on server

    # [('evolution', 3985), ('abortion', 2174), ('gun control', 1925), ('gay marriage', 920), ('existence of God', 546), ('healthcare', 116), ('climate change', 94), ('death penalty', 86), ('marijuana legalization', 80), ('communism vs. capitalism', 56)]
    if topic != None:
        qrPair_df = qrPair_df[qrPair_df.topic == topic]

    def labelFilter(row):
        label = row['%s' % task]
        # if label <= -1.0:
        #     row['label'] = '0'
        # elif label >= 1.0:
        #     row['label'] = '1'
        # else:
        #     row['label'] = '2'
        if label <= -2.0:
            row['label'] = '0'
        elif label > 0.0:
            row['label'] = '1'
        else:
            row['label'] = False
        return row
    qrPair_df = qrPair_df.apply(labelFilter, axis=1)
    qrPair_df = qrPair_df[qrPair_df.label != False]

    quoteTextList = [[word for word in str(text).split() if word != ''] for text in qrPair_df['quoteText'].values]
    responseTextList = [[word for word in str(text).split() if word != ''] for text in qrPair_df['responseText'].values]
    labelList = [label for label in qrPair_df['label'].values]

    df_data = pd.DataFrame({'quoteText': quoteTextList, 'responseText': responseTextList, 'label': labelList}, index=range(len(quoteTextList)))

    print(collections.Counter(list(df_data.label.values)).most_common(), len(df_data))
    return df_data

# def read_csv(task, topic):
#     qrPair_df = pd.read_csv('./data/debatepedia_agreement_dataset.csv', header=None, usecols=range(7), names=['pair_id', 'relation', 'topic', 'debate_title', 'debate_subtitle', 'Snippet_A', 'Snippet_B']) # data on server
#
#     if topic != None:
#         qrPair_df = qrPair_df[qrPair_df.topic == topic]
#
#     def cleanText(text):
#         text = str(text)
#         text = ' '.join([word for word in text.strip().split()])
#         text = re.sub(r'[^\x00-\x7f]', '', text)
#         text = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", text)
#         text = re.sub(r"\'s", " \'s", text)
#         text = re.sub(r"\'ve", " \'ve", text)
#         text = re.sub(r"n\'t", " n\'t", text)
#         text = re.sub(r"\'re", " \'re", text)
#         text = re.sub(r"\'d", " \'d", text)
#         text = re.sub(r"\'ll", " \'ll", text)
#         text = re.sub(r",", " , ", text)
#         text = re.sub(r"!", " ! ", text)
#         text = re.sub(r"\(", " \( ", text)
#         text = re.sub(r"\)", " \) ", text)
#         text = re.sub(r"\?", " \? ", text)
#         text = re.sub(r"\s{2,}", " ", text)
#         return text.strip().lower()
#     qrPair_df['quoteText'] = qrPair_df['Snippet_A'].apply(cleanText)
#     qrPair_df['responseText'] = qrPair_df['Snippet_B'].apply(cleanText)
#
#     def labelFilter(row):
#         label = row['relation']
#         if label == 'disagreement':
#             row['label'] = '0'
#         else:
#             row['label'] = '1'
#         return row
#     qrPair_df = qrPair_df.apply(labelFilter, axis=1)
#
#     quoteTextList = [[word for word in str(text).split() if word != ''] for text in qrPair_df['quoteText'].values]
#     responseTextList = [[word for word in str(text).split() if word != ''] for text in qrPair_df['responseText'].values]
#     labelList = [label for label in qrPair_df['label'].values]
#
#     df_data = pd.DataFrame({'quoteText': quoteTextList, 'responseText': responseTextList, 'label': labelList}, index=range(len(quoteTextList)))
#
#     print(collections.Counter(list(df_data.label.values)).most_common(), len(df_data))
#     return df_data

def data_save(task, topic, max_len, wordFilter):
    df_data = read_csv(task, topic)
    allTextList = np.concatenate((df_data['quoteText'].values, df_data['responseText'].values), axis=0)
    wordList = list(chain(*allTextList))

    if not wordFilter:
        wordSet = [word for (word, count) in collections.Counter(wordList).most_common()]
    else:
        wordSet = [word for (word, count) in collections.Counter(wordList).most_common() if count > 2]
        wordSet.append('unkWord')

        def processUNK(text):
            tmp = []
            for word in text:
                if word in wordSet:
                    tmp.append(word)
                else:
                    tmp.append('unkWord')
            return tmp

        df_data['quoteText'] = df_data['quoteText'].apply(processUNK)
        df_data['responseText'] = df_data['responseText'].apply(processUNK)

    print('vocabulary size = {}'.format(len(wordSet)))

    wordIDSet = range(1, len(wordSet) + 1)

    label = ['0', '1', '2']
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
    df_data['y'] = df_data['label'].apply(y_padding)

    X = np.asarray(list(zip(list(df_data['X_quote'].values), list(df_data['X_response'].values))))
    y = np.asarray(list(df_data['y'].values))

    with open('./data/data_%s_%s_%s.pkl' % (task, topic, max_len), 'wb') as outp:
        pickle.dump(X, outp)
        pickle.dump(y, outp)
        pickle.dump(word2id, outp)
        pickle.dump(id2word, outp)
        pickle.dump(label2id, outp)
        pickle.dump(id2label, outp)
    print('** Finished saving the %s_%s_%s data.' % (task, topic, max_len))

def splitTrainAndTest(task, topic, max_len, resampleFlag):
    with open('./data/data_%s_%s_%s.pkl' % (task, topic, max_len), 'rb') as fr:
        X = pickle.load(fr)
        y = pickle.load(fr)
        word2id = pickle.load(fr)
        id2word = pickle.load(fr)
        label2id = pickle.load(fr)
        id2label = pickle.load(fr)

    print('X.shape={}, y.shape={}'.format(X.shape, y.shape))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

    def train_resample(data):
        data_0 = [d for d in data if d[1] == 0]
        data_1 = [d for d in data if d[1] == 1]
        data_2 = [d for d in data if d[1] == 2]
        n_samples = max([len(data_0), len(data_1), len(data_2)])
        result = []
        for data in [data_0, data_1, data_2]:
            if len(data) < n_samples:
                data = resample(data, replace=True, n_samples=n_samples, random_state=12)
            result.extend(data)
        return result

    if resampleFlag:
        X_train, y_train = list(zip(*train_resample(list(zip(X_train, y_train)))))
        X_train, y_train = np.asarray(X_train), np.asarray(y_train)

    print('X_train.shape={}, y_train.shape={};\nX_test.shape={}, y_test.shape={}'.format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))

    return X_train, X_test, y_train, y_test, word2id

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

def getDataSet(task, topic, max_len, resampleFlag):
    X_train, X_test, y_train, y_test, word2id = splitTrainAndTest(task, topic, max_len, resampleFlag)
    print('Creating the data generator ...')
    data_train = BatchGenerator(X_train, y_train, shuffle=True)
    data_test = BatchGenerator(X_test, y_test, shuffle=False)
    print('Finished creating the generator.')
    return data_train, data_test, word2id

if __name__ == '__main__':

    # read_csv(task='disagree_agree', topic=None)
    # read_csv(task='debatepedia', topic=None)

    # data_save(task='disagree_agree', topic='evolution', max_len=64, wordFilter=True)
    data_save(task='disagree_agree', topic=None, max_len=64, wordFilter=True)
    # data_save(task='debatepedia', topic=None, max_len=64, wordFilter=True)

    # data_train, data_test, vocabulary_size = getDataSet(task='disagree_agree')

    # tr_batch_size = 32
    # max_epoch = 30
    # max_max_epoch = 100
    # display_num = 5

    # tr_batch_num = int(data_train.y.shape[0] / tr_batch_size)
    # display_batch = int(tr_batch_num / display_num)

    # for epoch in range(1, max_max_epoch + 1):
    #     for batch in xrange(tr_batch_num):
    #         X_q_batch, X_r_batch, y_batch = data_train.next_batch(tr_batch_size)




