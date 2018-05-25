import pandas as pd
import copy
import numpy as np
from itertools import chain
import collections
import pickle
from sklearn.model_selection import train_test_split

def get_unlabeledResponse():
    qrText_df = pd.read_csv('./data/unsupervisedQRTextSample.csv')

    def rowFilter(row):
        quoteText = str(row[2])
        responseText = str(row[3])
        if quoteText != 'nan' and responseText != 'nan':
            filterLabel = False
        else:
            filterLabel = True
        row['filterLabel'] = filterLabel
        return row

    qrText_df = qrText_df.apply(rowFilter, axis=1)
    qrText_df = qrText_df[qrText_df.filterLabel == False]
    print('Number of Total Unlabeled Pairs: {}'.format(len(qrText_df)))

    responseList = list(qrText_df.responseText.values)
    return responseList

def buildDataSet(source):
    if source == 'qrPair_2018':
        qrPair_df = pd.read_csv('./data/qrPair_2018.csv')  # data on server

        def labelFilter(row):
            label = row['disagree_agree']
            # Right Now
            if label <= -2.0:
                row['label'] = '0'
            elif label > 0.0:
                row['label'] = '1'
            else:
                row['label'] = False
            # Right Now
            return row

        qrPair_df = qrPair_df.apply(labelFilter, axis=1)
        qrPair_df = qrPair_df[qrPair_df.label != False]
    if source == 'debatePedia':
        pass

    quoteTextList = [[word for word in str(text).split() if word != ''] for text in qrPair_df['quoteText'].values]
    responseTextList = [[word for word in str(text).split() if word != ''] for text in qrPair_df['responseText'].values]
    labelList = [label for label in qrPair_df['label'].values]
    topicList = [topic for topic in qrPair_df['topic'].values]

    df_data = pd.DataFrame({'topic': topicList, 'quoteText': quoteTextList, 'responseText': responseTextList, 'label': labelList}, index=range(len(quoteTextList)))

    main_task_df = copy.deepcopy(df_data)
    sup_task_pos_df = copy.deepcopy(df_data)
    sup_task_neg_df = copy.deepcopy(df_data)

    sup_task_pos_df['label'] = sup_task_pos_df['label'].apply(lambda x: '1')
    unlabelResponseList = [[word for word in str(text).split() if word != ''] for text in get_unlabeledResponse()]

    def negativeSampling(row):
        shuffleIndex = np.random.choice(len(unlabelResponseList))
        row['responseText'] = unlabelResponseList[shuffleIndex]
        match_unmatch = '0'
        row['label'] = match_unmatch
        return row

    sup_task_neg_df = sup_task_neg_df.apply(negativeSampling, axis=1)
    sup_task_df = pd.concat([sup_task_pos_df, sup_task_neg_df], ignore_index=True)

    print('Main Task', collections.Counter(list(main_task_df.label.values)).most_common(), len(main_task_df))
    print('Sup Task', collections.Counter(list(sup_task_df.label.values)).most_common(), len(sup_task_df))
    return main_task_df, sup_task_df

def data_save(source, wordFilter, max_len):

    main_task_df, sup_task_df = buildDataSet(source)

    allTextList = np.concatenate((sup_task_df['quoteText'].values, sup_task_df['responseText'].values), axis=0)
    wordList = list(chain(*allTextList))

    if not wordFilter:
        wordSet = [word for (word, count) in collections.Counter(wordList).most_common()]
    else:
        wordSet = [word for (word, count) in collections.Counter(wordList).most_common() if count > 2]
        wordSet.append('UNK')
    print('vocabulary size = {}'.format(len(wordSet)))

    wordIDSet = range(1, len(wordSet) + 1)

    label = ['0', '1']
    labelIDSet = range(len(label))

    word2id = pd.Series(wordIDSet, index=wordSet)
    id2word = pd.Series(wordSet, index=wordIDSet)
    label2id = pd.Series(labelIDSet, index=label)
    id2label = pd.Series(label, index=labelIDSet)

    def processUNK(text):
        tmp = []
        for word in text:
            if word in wordSet:
                tmp.append(word)
            else:
                tmp.append('UNK')
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

    for task, df_data in [('main', main_task_df), ('sup', sup_task_df)]:

        df_data['quoteText'] = df_data['quoteText'].apply(processUNK)
        df_data['responseText'] = df_data['responseText'].apply(processUNK)

        df_data['X_quote'] = df_data['quoteText'].apply(X_padding)
        df_data['X_response'] = df_data['responseText'].apply(X_padding)
        df_data['y'] = df_data['label'].apply(y_padding)

        X = np.asarray(list(zip(list(df_data['X_quote'].values), list(df_data['X_response'].values))))
        y = np.asarray(list(df_data['y'].values))

        name = 'iac' if source == 'qrPair_2018' else 'dp'
        with open('./data/%s_%s_%s.pkl' % (name, task, max_len), 'wb') as outp:
            pickle.dump(X, outp)
            pickle.dump(y, outp)
            pickle.dump(word2id, outp)
            pickle.dump(id2word, outp)
            pickle.dump(label2id, outp)
            pickle.dump(id2label, outp)
        print('** Finished saving the %s_%s data.' % (name, task))

def splitTrainAndTest(name, task, max_len):
    with open('./data/%s_%s_%s.pkl' % (name, task, max_len), 'rb') as fr:
        X = pickle.load(fr)
        y = pickle.load(fr)
        word2id = pickle.load(fr)
        id2word = pickle.load(fr)
        label2id = pickle.load(fr)
        id2label = pickle.load(fr)

    print('X.shape={}, y.shape={}'.format(X.shape, y.shape))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)
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

            if self._number_examples >= 4700:
                print('**EPOCH {} Completed'.format(self._epochs_completed))

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

def getDataSet(name, task, max_len):
    X_train, X_test, y_train, y_test, word2id = splitTrainAndTest(name, task, max_len)
    print('Creating the data generator ...')
    data_train = BatchGenerator(X_train, y_train, shuffle=True)
    data_test = BatchGenerator(X_test, y_test, shuffle=False)
    print('Finished creating the generator.')
    return data_train, data_test, word2id

if __name__ == '__main__':

    data_save(source='qrPair_2018', wordFilter=True, max_len=64)




