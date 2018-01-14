import pandas as pd
import numpy as np
from itertools import chain
import collections
import pickle
import tensorflow as tf
import time
import data_helper

def data_save():
    qrPair_df = pd.read_csv('./data/qrPair.csv')

    quoteTextList = [[word for word in text.split() if word != ''] for text in qrPair_df['quoteText'].values]
    responseTextList = [[word for word in text.split() if word != ''] for text in qrPair_df['responseText'].values]
    multiLabelList = zip(qrPair_df['disagree_agree'].values, qrPair_df['attacking_respectful'].values, qrPair_df['emotion_fact'].values, qrPair_df['nasty_nice'].values)
    labelList = [[str(label) for label in multiLabel] for multiLabel in multiLabelList]


    df_data = pd.DataFrame({'quoteText': quoteTextList, 'responseText': responseTextList, 'label': labelList}, index=range(len(quoteTextList)))

    allTextList = np.concatenate((df_data['quoteText'].values, df_data['responseText'].values), axis=0)

    wordList = list(chain(*allTextList))
    topWord = 10000
    wordSet = [word for (word, count) in collections.Counter(wordList).most_common()[:topWord]]

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

    wordSet.append('unkWord')
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
        ids.extend([0] * (max_len - len(ids)))
        return ids

    def y_padding(labels):
        ids = list(label2id[labels])
        return ids

    df_data['X_quote'] = df_data['quoteText'].apply(X_padding)
    df_data['X_response'] = df_data['responseText'].apply(X_padding)
    df_data['y'] = df_data['label'].apply(y_padding)

    X_quote = np.asarray(list(df_data['X_quote'].values))
    X_response = np.asarray(list(df_data['X_response'].values))
    y = np.asarray(list(df_data['y'].values))

    with open('./data/data_maxLen_%s_multiTask.pkl' % str(max_len), 'wb') as outp:
        pickle.dump(X_quote, outp)
        pickle.dump(X_response, outp)
        pickle.dump(y, outp)
        pickle.dump(word2id, outp)
        pickle.dump(id2word, outp)
        pickle.dump(label2id, outp)
        pickle.dump(id2label, outp)
    print '** Finished saving the data.'

class BiLSTMForMultiTask(object):
    def __init__(self, wordEmbedding, textLength, embedding_size, hidden_size=128, layer_num=2, class_num=2, max_grad_norm=5.0):

        self.wordEmbedding = wordEmbedding
        self.textLength = textLength
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.class_num = class_num
        self.max_grad_norm = max_grad_norm

        self.X_q_inputs = tf.placeholder(tf.int32, [None, self.textLength], name='X_q_inputs')
        self.X_r_inputs = tf.placeholder(tf.int32, [None, self.textLength], name='X_r_inputs')
        self.y_inputs = tf.placeholder(tf.int32, [None, 4], name='y_inputs')
        self.batch_size = tf.placeholder(tf.int32)
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.lr = tf.placeholder(tf.float32)

        with tf.variable_scope('embedding'):

            # word_embedding = tf.Variable(initial_value=self.wordEmbedding, name='word_embedding')
            word_embedding = tf.Variable(initial_value=self.wordEmbedding, name='word_embedding', trainable=False)
            quote_inputs = tf.nn.embedding_lookup(word_embedding, self.X_q_inputs, name='quote_inputs')
            response_inputs = tf.nn.embedding_lookup(word_embedding, self.X_r_inputs, name='response_inputs')

        outputs = []
        for inputs in [quote_inputs, response_inputs]:
            with tf.variable_scope('BiLSTM-%s' % inputs.name.split('/')[-1][:-2]):

                lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size, forget_bias=1.0, state_is_tuple=True)
                lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size, forget_bias=1.0, state_is_tuple=True)

                lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_fw_cell, input_keep_prob=1.0, output_keep_prob=self.dropout_keep_prob)
                lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_bw_cell, input_keep_prob=1.0, output_keep_prob=self.dropout_keep_prob)

                mtcell_fw = tf.contrib.rnn.MultiRNNCell([lstm_fw_cell] * self.layer_num, state_is_tuple=True)
                mtcell_bw = tf.contrib.rnn.MultiRNNCell([lstm_bw_cell] * self.layer_num, state_is_tuple=True)

                init_state_fw = mtcell_fw.zero_state(self.batch_size, dtype=tf.float32)
                init_state_bw = mtcell_bw.zero_state(self.batch_size, dtype=tf.float32)

                inputs = tf.unstack(tf.transpose(inputs, [1, 0, 2]))
                output, _, _ = tf.contrib.rnn.static_bidirectional_rnn(mtcell_fw, mtcell_bw, inputs, initial_state_fw=init_state_fw, initial_state_bw=init_state_bw, dtype=tf.float32) # tf.shape(output) = textLength * [batch_size, hidden_size *2]
                outputs.append(output[-1])

        outputs_concat = tf.concat(outputs, axis=1, name='concat-quote-response')
        accuracy = []
        cost = []
        for index, label in enumerate(['disagree_agree', 'attacking_respectful', 'emotion_fact', 'nasty_nice']):
            with tf.variable_scope('outputs-%s' % label):
                W_out = tf.Variable(tf.random_normal([self.hidden_size * 2 * 2, self.class_num]), name='W_out-%s' % label)
                b_out = tf.Variable(tf.constant(0.1, shape=[self.class_num]), name='b_out-%s' % label)
                logits = tf.matmul(outputs_concat, W_out) + b_out

                _correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), tf.reshape(self.y_inputs[:, index], [-1]))
                _accuracy = tf.reduce_mean(tf.cast(_correct_prediction, tf.float32))
                _cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.reshape(self.y_inputs[:, index], [-1])))

                accuracy.append(_accuracy)
                cost.append(_cost)

        self.accuracy = accuracy
        self.cost = tf.reduce_mean(cost, axis=0) * tf.constant(4.0, dtype=tf.float32)

        tvars = tf.trainable_variables()
        print 'trainable_variables:'
        for x in tvars:
            print x.name
        grads = tf.gradients(self.cost, tvars)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=tf.contrib.framework.get_or_create_global_step())
        self.saver = tf.train.Saver(max_to_keep=10)

def test_epoch(sess, model, dataset):
    fetches = [model.accuracy, model.cost]
    _y = dataset.y
    data_size = _y.shape[0]
    _batch_size = data_size
    batch_num = int(data_size / _batch_size)
    _accs = np.zeros(shape=[4], dtype=np.float32)
    _costs = 0.0
    for i in xrange(batch_num):
        X_q_batch, X_r_batch, y_batch = dataset.next_batch(_batch_size)
        feed_dict = {
            model.X_q_inputs: X_q_batch,
            model.X_r_inputs: X_r_batch,
            model.y_inputs: y_batch,
            model.batch_size: _batch_size,
            model.dropout_keep_prob: 1.0,
            model.lr: 1e-3,
        }
        _acc, _cost = sess.run(fetches, feed_dict)
        _accs += _acc
        _costs += _cost
    mean_acc = _accs / batch_num
    mean_cost = _costs / batch_num
    return mean_acc, mean_cost

def TrainAndTest():
    max_len = 150
    data_train, data_valid, data_test = data_helper.getDataSet()
    # wordEmbedding = data_helper.getWordEmbedding(targetFileName='./data/embeddingMatrix.300d.pkl')
    wordEmbedding = data_helper.getRandomWordEmbedding(vocabularySize=10002, embeddingSize=300)

    model = BiLSTMForMultiTask(wordEmbedding, textLength=max_len, embedding_size=300)

    config = tf.ConfigProto()
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    sess.run(init)

    model_save_path = './ckpt/bilstmForMulitTask/bilstm.ckpt'
    tr_batch_size = 128
    decay = 0.85
    max_epoch = 5
    max_max_epoch = 12
    # max_epoch = 12
    # max_max_epoch = 25
    display_num = 5
    tr_batch_num = int(data_train.y.shape[0] / tr_batch_size)
    display_batch = int(tr_batch_num / display_num)

    for epoch in xrange(max_max_epoch):
        _lr = 1e-3
        if epoch > max_epoch:
            _lr = _lr * ((decay) ** (epoch - max_epoch))
        print 'EPOCH {}, lr={}'.format(epoch + 1, _lr)

        start_time = time.time()
        _accs = np.zeros(shape=[4], dtype=np.float32)
        _costs = 0.0
        show_accs = np.zeros(shape=[4], dtype=np.float32)
        show_costs = 0.0

        for batch in xrange(tr_batch_num):
            fetches = [model.accuracy, model.cost, model.train_op]
            X_q_batch, X_r_batch, y_batch = data_train.next_batch(tr_batch_size)
            feed_dict = {
                model.X_q_inputs: X_q_batch,
                model.X_r_inputs: X_r_batch,
                model.y_inputs: y_batch,
                model.batch_size: tr_batch_size,
                model.dropout_keep_prob: 0.5,
                model.lr: _lr,
            }

            _acc, _cost, _ = sess.run(fetches, feed_dict)
            _accs += _acc
            _costs += _cost
            show_accs += _acc
            show_costs += _cost
            if (batch + 1) % display_batch == 0:
                valid_acc, valid_cost = test_epoch(sess, model, data_valid)

                disagree_agree_acc = show_accs[0] / display_batch
                attacking_respectful_acc = show_accs[1] / display_batch
                emotion_fact_acc = show_accs[2] / display_batch
                nasty_nice_acc = show_accs[3] / display_batch

                print '\ttraining acc={}, {}, {}, {}, cost={};  valid acc={}, {}, {}, {}, cost={} '.format(disagree_agree_acc, attacking_respectful_acc, emotion_fact_acc, nasty_nice_acc, show_costs / display_batch, valid_acc[0], valid_acc[1], valid_acc[2], valid_acc[3], valid_cost)

                show_accs = np.zeros(shape=[4], dtype=np.float32)
                show_costs = 0.0

        mean_acc = _accs / tr_batch_num
        mean_cost = _costs / tr_batch_num

        if (epoch + 1) % 3 == 0:
            save_path = model.saver.save(sess, model_save_path, global_step=(epoch + 1))
            print 'The save path is ', save_path
        print '\ttraining {}, acc={}, {}, {}, {}, cost={}'.format(data_train.y.shape[0], mean_acc[0], mean_acc[1], mean_acc[2], mean_acc[3], mean_cost)
        print 'Epoch training {}, acc={}, {}, {}, {}, cost={}, speed={} s/epoch\n'.format(data_train.y.shape[0], mean_acc[0], mean_acc[1], mean_acc[2], mean_acc[3], mean_cost, time.time() - start_time)
    print '**TEST RESULT:'
    test_acc, test_cost = test_epoch(sess, model, data_test)
    print '**Test {}, acc={}, {}, {}, {}, cost={}'.format(data_test.y.shape[0], test_acc[0], test_acc[1], test_acc[2], test_acc[3], test_cost)

if __name__ == '__main__':
    TrainAndTest()