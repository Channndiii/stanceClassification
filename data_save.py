import pandas as pd
import numpy as np
from itertools import chain
import collections
import pickle

task = 'disagree_agree'
# qrPair_df = pd.read_csv('./data/qrPair.csv')
qrPair_df = pd.read_csv('./data/qrPair_%s.csv' % task)

quoteTextList = [[word for word in str(text).split() if word != ''] for text in qrPair_df['quoteText'].values]
responseTextList = [[word for word in str(text).split() if word != ''] for text in qrPair_df['responseText'].values]
labelList = [str(label) for label in qrPair_df['disagree_agree'].values]
df_data = pd.DataFrame({'quoteText': quoteTextList, 'responseText': responseTextList, 'label': labelList}, index=range(len(quoteTextList)))

allTextList = np.concatenate((df_data['quoteText'].values, df_data['responseText'].values), axis=0)

wordList = list(chain(*allTextList))

# topWord = 10000
# wordSet = [word for (word, count) in collections.Counter(wordList).most_common()[:topWord]]
# def processUNK(text):
#     tmp = []
#     for word in text:
#         if word in wordSet:
#             tmp.append(word)
#         else:
#             tmp.append('unkWord')
#     return tmp
#
# df_data['quoteText'] = df_data['quoteText'].apply(processUNK)
# df_data['responseText'] = df_data['responseText'].apply(processUNK)
# wordSet.append('unkWord')

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
    ids.extend([0]*(max_len-len(ids)))
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

# with open('./data/data_maxLen_%s.pkl' % str(max_len), 'wb') as outp:
with open('./data/data_maxLen_%s_%s.pkl' % (max_len, task), 'wb') as outp:
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