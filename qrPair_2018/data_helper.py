import pickle
import numpy as np
from sklearn.model_selection import train_test_split

task = 'disagree_agree'
max_len = 150
# with open('./data/data_maxLen_%s.pkl' % str(max_len), 'rb') as fr:
with open('./data/data_maxLen_%s_%s.pkl' % (max_len, task), 'rb') as fr:
    X_quote = pickle.load(fr)
    X_response = pickle.load(fr)
    y = pickle.load(fr)
    word2id = pickle.load(fr)
    id2word = pickle.load(fr)
    label2id = pickle.load(fr)
    id2label = pickle.load(fr)

print('X_quote.shape={}, X_response.shape={}, y.shape={}'.format(X_quote.shape, X_response.shape, y.shape))

X_QR = np.hstack((X_quote, X_response))

X_QR_train, X_QR_test, y_train, y_test = train_test_split(X_QR, y, test_size=0.2, random_state=12)
# X_QR_train, X_QR_valid, y_train, y_valid = train_test_split(X_QR_train, y_train, test_size=0.2, random_state=12)

X_quote_train = X_QR_train[:, :max_len]
X_response_train = X_QR_train[:, max_len:]

# X_quote_valid = X_QR_valid[:, :max_len]
# X_response_valid = X_QR_valid[:, max_len:]

X_quote_test = X_QR_test[:, :max_len]
X_response_test = X_QR_test[:, max_len:]

# print 'X_quote_train.shape={}, X_response_train.shape={}, y_train.shape={};\nX_quote_valid.shape={}, X_response_valid.shape={}, y_valid.shape={};\nX_quote_test.shape={}, X_response_test.shape={}, y_test.shape={}'.format(X_quote_train.shape, X_response_train.shape, y_train.shape, X_quote_valid.shape, X_response_valid.shape, y_valid.shape, X_quote_test.shape, X_response_test.shape, y_test.shape)
print('X_quote_train.shape={}, X_response_train.shape={}, y_train.shape={};\nX_quote_test.shape={}, X_response_test.shape={}, y_test.shape={}'.format(X_quote_train.shape, X_response_train.shape, y_train.shape, X_quote_test.shape, X_response_test.shape, y_test.shape))

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

print('Creating the data generator ...')
data_train = BatchGenerator(X_quote_train, X_response_train, y_train, shuffle=True)
# data_valid = BatchGenerator(X_quote_valid, X_response_valid, y_valid, shuffle=False)
data_test = BatchGenerator(X_quote_test, X_response_test, y_test, shuffle=False)
print('Finished creating the generator.')

def getDataSet():
    # return data_train, data_valid, data_test
    return data_train, data_test

def loadEmbedding(word2id, targetFileName, embeddingFileName='/home/chandi/Downloads/glove.840B.300d.txt', embeddingSize=300):

    vocabularySize = len(word2id.index) + 1
    embeddingMatrix = np.asarray(np.random.uniform(-0.01, 0.01, size=(vocabularySize, embeddingSize)), dtype=np.float32)
    hit = 0
    with open(embeddingFileName, 'r') as fr:
        lines = fr.readlines()
        for line in lines:
            line = line.strip().split(' ')
            word = line[0].decode('utf-8')
            embedding = line[1:]
            if word in word2id.index:
                wordIndex = word2id[word]
                embeddingArray = np.fromstring('\n'.join(embedding), dtype=np.float32, sep='\n')
                embeddingMatrix[wordIndex] = embeddingArray
                hit += 1
    hitRate = float(hit) / vocabularySize
    print('PreTrain Embedding hitRate={}'.format(hitRate))
    # return embeddingMatrix
    with open(targetFileName, 'wb') as fw:
        pickle.dump(embeddingMatrix, fw)

def getWordEmbedding(targetFileName):
    with open(targetFileName, 'rb') as fr:
        embeddingMatrix = pickle.load(fr)
    return embeddingMatrix

def getRandomWordEmbedding(vocabularySize, embeddingSize):
    # embeddingMatrix = np.asarray(np.random.uniform(-0.01, 0.01, size=(vocabularySize, embeddingSize)), dtype=np.float32)
    embeddingMatrix = np.asarray(np.random.uniform(-1.0, 1.0, size=(vocabularySize, embeddingSize)), dtype=np.float32)
    return embeddingMatrix

if __name__ == '__main__':
    loadEmbedding(word2id, targetFileName='./data/embeddingMatrix.300d.pkl')