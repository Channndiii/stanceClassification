import pandas as pd
import numpy as np
from itertools import chain
import collections
import pickle
from sklearn.model_selection import train_test_split

def buildDataSet(sampleRate, posNegRate):
    qrText_df = pd.read_csv('./data/unsupervisedQRTextSample.csv')
    responseTextList = list(qrText_df.responseText.values)

    def sampleAndShuffle(row):
        tmp1 = np.random.random_sample()
        if tmp1 < sampleRate:
            filterLabel = False
            tmp2 = np.random.random_sample()
            if tmp2 > posNegRate:
                match_unmatch = 1
            else:
                shuffleIndex = np.random.permutation(len(responseTextList))[0]
                row['responseText'] = responseTextList[shuffleIndex]
                match_unmatch = 0
        else:
            filterLabel = True
            match_unmatch = 1
        row['filterLabel'] = filterLabel
        row['match_unmatch'] = match_unmatch
        return row

    qrText_df = qrText_df.apply(sampleAndShuffle, axis=1)
    qrText_df = qrText_df[qrText_df.filterLabel == False]
    print len(qrText_df)
    qrText_df.to_csv('./data/qrPair_match_unmatch.csv', index=None, columns=['quoteTextID', 'responseTextID', 'quoteText', 'responseText', 'match_unmatch'])

def read_csv(task):
    qrPair_df = pd.read_csv('./data/qrPair_%s.csv' % task)

    quoteTextList = [[word for word in str(text).split() if word != ''] for text in qrPair_df['quoteText'].values]
    responseTextList = [[word for word in str(text).split() if word != ''] for text in qrPair_df['responseText'].values]
    labelList = [str(label) for label in qrPair_df['%s' % task].values]
    df_data = pd.DataFrame({'quoteText': quoteTextList, 'responseText': responseTextList, 'label': labelList}, index=range(len(quoteTextList)))

    return df_data

def data_save(main_task, sup_task, max_len, topWord):

    main_task_df = read_csv(main_task)
    sup_task_df = read_csv(sup_task)

    allTextList = np.concatenate((main_task_df['quoteText'].values, main_task_df['responseText'].values,
                                  sup_task_df['quoteText'].values, sup_task_df['responseText'].values), axis=0)

    wordList = list(chain(*allTextList))

    wordSet = [word for (word, count) in collections.Counter(wordList).most_common()[:topWord]]
    wordSet.append('unkWord')
    print 'vocabulary size = {}'.format(len(wordSet))

    idSet = range(1, len(wordSet) + 1)

    label = ['0', '1']
    labelIDSet = range(len(label))

    word2id = pd.Series(idSet, index=wordSet)
    id2word = pd.Series(wordSet, index=idSet)
    label2id = pd.Series(labelIDSet, index=label)
    id2label = pd.Series(label, index=labelIDSet)

    def processUNK(text):
        tmp = []
        for word in text:
            if word in wordSet:
                tmp.append(word)
            else:
                tmp.append('unkWord')
        return tmp

    def X_padding(words):
        ids = list(word2id[words])
        if len(ids) >= max_len:
            return ids[:max_len]
        ids.extend([0] * (max_len - len(ids)))
        return ids

    def y_padding(label):
        id = label2id[label]
        return id

    for task, df_data in [(main_task, main_task_df), (sup_task, sup_task_df)]:

        df_data['quoteText'] = df_data['quoteText'].apply(processUNK)
        df_data['responseText'] = df_data['responseText'].apply(processUNK)

        df_data['X_quote'] = df_data['quoteText'].apply(X_padding)
        df_data['X_response'] = df_data['responseText'].apply(X_padding)
        df_data['y'] = df_data['label'].apply(y_padding)

        X_quote = np.asarray(list(df_data['X_quote'].values))
        X_response = np.asarray(list(df_data['X_response'].values))
        y = np.asarray(list(df_data['y'].values))

        with open('./data/data_%s_%s_%s.pkl' % (task, max_len, topWord), 'wb') as outp:
            pickle.dump(X_quote, outp)
            pickle.dump(X_response, outp)
            pickle.dump(y, outp)
            pickle.dump(word2id, outp)
            pickle.dump(id2word, outp)
            pickle.dump(label2id, outp)
            pickle.dump(id2label, outp)
        print '** Finished saving the %s data.' % task

def splitTrainAndTest(max_len, topWord, task):
    
    with open('./data/data_%s_%s_%s.pkl' % (task, max_len, topWord), 'rb') as fr:
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

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._number_examples:
            self._epochs_completed += 1
            if self._shuffle:
                new_index = np.random.permutation(self._number_examples)
                self._X_quote = self._X_quote[new_index]
                self._X_response = self._X_response[new_index]
                self._y = self._y[new_index]
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._number_examples
        end = self._index_in_epoch
        return self._X_quote[start:end], self._X_response[start:end], self._y[start:end]

def getDataSet(task):
    X_quote_train, X_response_train, y_train, X_quote_test, X_response_test, y_test = splitTrainAndTest(max_len=150, topWord=30000, task=task)
    print 'Creating the data generator ...'
    data_train = BatchGenerator(X_quote_train, X_response_train, y_train, shuffle=True)
    data_test = BatchGenerator(X_quote_test, X_response_test, y_test, shuffle=False)
    print 'Finished creating the generator.'
    vocabulary_size = 30001 + 1
    return data_train, data_test, vocabulary_size

if __name__ == '__main__':
    # buildDataSet(sampleRate=0.06, posNegRate=0.5)
    data_save(main_task='disagree_agree', sup_task='match_unmatch', max_len=150, topWord=30000)