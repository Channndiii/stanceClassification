import tensorflow as tf
import data_helper
import time
import numpy as np
from sklearn.metrics import confusion_matrix

class TextBiLSTM(object):
    def __init__(self, wordEmbedding, textLength, vocabulary_size, embedding_size, hidden_size=128, layer_num=2, class_num=2, max_grad_norm=5.0):

        self.wordEmbedding = wordEmbedding
        self.textLength = textLength
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.class_num = class_num
        self.max_grad_norm = max_grad_norm

        self.X_q_inputs = tf.placeholder(tf.int32, [None, self.textLength], name='X_q_inputs')
        self.X_r_inputs = tf.placeholder(tf.int32, [None, self.textLength], name='X_r_inputs')
        self.y_inputs = tf.placeholder(tf.int32, [None], name='y_inputs')
        self.batch_size = tf.placeholder(tf.int32)
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.lr = tf.placeholder(tf.float32)

        with tf.variable_scope('embedding'):

            # word_embedding = tf.Variable(initial_value=self.wordEmbedding, name='word_embedding')
            # word_embedding = tf.Variable(initial_value=self.wordEmbedding, name='word_embedding', trainable=False)
            word_embedding = tf.get_variable('word_embedding', [self.vocabulary_size, self.embedding_size], dtype=tf.float32)
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

                # inputs = batch_normalize(inputs, out_size=self.embedding_size)

                _inputs = tf.unstack(tf.transpose(inputs, [1, 0, 2]))
                output, _, _ = tf.contrib.rnn.static_bidirectional_rnn(mtcell_fw, mtcell_bw, _inputs, initial_state_fw=init_state_fw, initial_state_bw=init_state_bw, dtype=tf.float32) # tf.shape(output) = textLength * [batch_size, hidden_size *2]
                # outputs.append(output[-1])
                
                # output = output[-1]
                output = tf.transpose(tf.stack(output), [1, 0, 2])
                # output = tf.reduce_mean(output, axis=1)

                # with tf.variable_scope('AttentionFromInputs'):
                #     attention_inputs = tf.reshape(inputs, [-1, self.embedding_size])
                # 
                #     W = tf.Variable(tf.random_normal([self.embedding_size, 1]), name='W')
                #     b = tf.Variable(tf.constant(0.1, shape=[1]), name='b')
                # 
                #     attention_logits = tf.reshape(tf.matmul(attention_inputs, W) + b, [-1, self.textLength])
                #     attention_signals = tf.reshape(tf.nn.softmax(attention_logits), [-1])
                # 
                #     _output = tf.transpose(tf.reshape(output, [-1, self.hidden_size * 2]))
                #     output = tf.transpose(tf.multiply(_output, attention_signals))
                #     output = tf.reshape(output, [-1, self.textLength, self.hidden_size * 2])
                #     output = tf.reduce_mean(output, axis=1)
                
                with tf.variable_scope('AttentionFromLSTM_output'):
                    attention_inputs = tf.reshape(output, [-1, self.hidden_size * 2])

                    W = tf.Variable(tf.random_normal([self.hidden_size * 2, 1]), name='W')
                    b = tf.Variable(tf.constant(0.1, shape=[1]), name='b')

                    attention_logits = tf.reshape(tf.matmul(attention_inputs, W) + b, [-1, self.textLength])
                    # attention_signals = tf.reshape(tf.nn.softmax(attention_logits), [-1])
                    attention_signals = tf.reshape(tf.nn.softmax(tf.nn.tanh(attention_logits)), [-1])

                    _output = tf.transpose(tf.reshape(output, [-1, self.hidden_size * 2]))
                    output = tf.transpose(tf.multiply(_output, attention_signals))
                    output = tf.reshape(output, [-1, self.textLength, self.hidden_size * 2])
                    output = tf.reduce_mean(output, axis=1)
                
                # with tf.variable_scope('AMFromWE_transitionMatrix'):
                #     mean_inputs = tf.transpose(tf.reduce_mean(inputs, axis=1))
                #     transitionMatrix = tf.Variable(tf.random_normal([self.embedding_size, self.embedding_size]), name='transitionMatrix')
                #     
                #     mediumResult = tf.matmul(tf.matmul(tf.reshape(inputs, [-1, self.embedding_size]), transitionMatrix), mean_inputs)
                #     mediumResult = tf.transpose(tf.reshape(mediumResult, [-1, self.textLength, self.batch_size]), [0, 2, 1])
                #     mediumResult = tf.reshape(mediumResult, [-1, self.textLength])
                #     
                #     attention_logits = tf.gather(mediumResult, [i for i in range(32**2) if i % (32+1) == 0])
                #     # attention_signals = tf.reshape(tf.nn.softmax(attention_logits), [-1])
                #     attention_signals = tf.reshape(tf.nn.softmax(tf.nn.tanh(attention_logits)), [-1])
                # 
                #     _output = tf.transpose(tf.reshape(output, [-1, self.hidden_size * 2]))
                #     output = tf.transpose(tf.multiply(_output, attention_signals))
                #     output = tf.reshape(output, [-1, self.textLength, self.hidden_size * 2])
                #     output = tf.reduce_mean(output, axis=1)

                # with tf.variable_scope('AMFromLO_transitionMatrix'):
                #     mean_output = tf.transpose(tf.reduce_mean(output, axis=1))
                #     transitionMatrix = tf.Variable(tf.random_normal([self.hidden_size * 2, self.hidden_size * 2]), name='transitionMatrix')
                # 
                #     mediumResult = tf.matmul(tf.matmul(tf.reshape(output, [-1, self.hidden_size*2]), transitionMatrix), mean_output)
                #     mediumResult = tf.transpose(tf.reshape(mediumResult, [-1, self.textLength, self.batch_size]), [0, 2, 1])
                #     mediumResult = tf.reshape(mediumResult, [-1, self.textLength])
                # 
                #     attention_logits = tf.gather(mediumResult, [i for i in range(32 ** 2) if i % (32 + 1) == 0])
                #     # attention_signals = tf.reshape(tf.nn.softmax(attention_logits), [-1])
                #     attention_signals = tf.reshape(tf.nn.softmax(tf.nn.tanh(attention_logits)), [-1])
                # 
                #     _output = tf.transpose(tf.reshape(output, [-1, self.hidden_size * 2]))
                #     output = tf.transpose(tf.multiply(_output, attention_signals))
                #     output = tf.reshape(output, [-1, self.textLength, self.hidden_size * 2])
                #     output = tf.reduce_mean(output, axis=1)
                
                with tf.variable_scope('output-batch_normalize'):
                    output = batch_normalize(output, out_size=self.hidden_size * 2)
                    outputs.append(output)

                # with tf.variable_scope('bidirectional_rnn'):
                #     outputs_fw = list()
                #     state_fw = init_state_fw
                #     with tf.variable_scope('fw'):
                #         for timestep in range(self.textLength):
                #             if timestep > 0:
                #                 tf.get_variable_scope().reuse_variables()
                #             (output_fw, state_fw) = mtcell_fw(inputs[:, timestep, :], state_fw)
                #             outputs_fw.append(output_fw)
                #
                #     outputs_bw = list()
                #     state_bw = init_state_bw
                #     with tf.variable_scope('bw'):
                #         inputs = tf.reverse(inputs, [1])
                #         for timestep in range(self.textLength):
                #             if timestep > 0:
                #                 tf.get_variable_scope().reuse_variables()
                #             (output_bw, state_bw) = mtcell_bw(inputs[:, timestep, :], state_bw)
                #             outputs_bw.append(output_bw)
                #
                #     outputs_bw = tf.reverse(outputs_bw, [0])
                #
                #     output = tf.concat([outputs_fw, outputs_bw], 2)
                #     output = tf.unstack(output)
                #     outputs.append(output[-1])

        with tf.variable_scope('outputs'):
            outputs_concat = tf.concat(outputs, axis=1, name='concat-quote-response')
            outputs_concat = batch_normalize(outputs_concat, out_size=self.hidden_size * 2 * 2)
            W_out = tf.Variable(tf.random_normal([self.hidden_size * 2 * 2, self.class_num]), name='W_out')
            b_out = tf.Variable(tf.constant(0.1, shape=[self.class_num]), name='b_out')
            self.logits = tf.matmul(outputs_concat, W_out) + b_out

        self.prediction = tf.cast(tf.argmax(self.logits, 1), tf.int32)
        self.correct_prediction = tf.equal(tf.cast(tf.argmax(self.logits, 1), tf.int32), tf.reshape(self.y_inputs, [-1]))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        
        self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=tf.reshape(self.y_inputs, [-1])))

        tvars = tf.trainable_variables()
        print 'trainable_variables:'
        for x in tvars:
            print x.name
        # grads = tf.gradients(self.cost, tvars)
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), self.max_grad_norm)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=tf.contrib.framework.get_or_create_global_step())
        self.saver = tf.train.Saver(max_to_keep=10)

def batch_normalize(x, out_size):
    fc_mean, fc_var = tf.nn.moments(
        x,
        axes=[0]
        # axes = list(range(len(x.get_shape()) - 1))
    )
    scale = tf.Variable(tf.ones([out_size]), name='scale')
    shift = tf.Variable(tf.zeros([out_size]), name='shift')
    epsilon = 0.001

    ema = tf.train.ExponentialMovingAverage(decay=0.5)

    def mean_var_with_update():
        ema_apply_op = ema.apply([fc_mean, fc_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(fc_mean), tf.identity(fc_var)

    mean, var = mean_var_with_update()

    x = tf.nn.batch_normalization(x, mean, var, shift, scale, epsilon)
    return x

def test_epoch(sess, model, dataset):
    fetches = [model.prediction, model.accuracy, model.cost]
    _y = dataset.y
    data_size = _y.shape[0]
    _batch_size = data_size
    # _batch_size = 32
    batch_num = int(data_size / _batch_size)
    _accs = 0.0
    _costs = 0.0
    _pred = []
    _true = []
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
        _pred, _acc, _cost = sess.run(fetches, feed_dict)
        _true = y_batch
        # pred, _acc, _cost = sess.run(fetches, feed_dict)
        # if i == 0:
        #     _pred = pred
        #     _true = y_batch
        # else:
        #     _pred = np.concatenate((_pred, pred))
        #     _true = np.concatenate((_true, y_batch))
        _accs += _acc
        _costs += _cost
    mean_acc = _accs / batch_num
    mean_cost = _costs / batch_num
    cm = confusion_matrix(y_true=_true, y_pred=_pred)
    return mean_acc, mean_cost, cm

if __name__ == '__main__':

    max_len = 150
    # data_train, data_valid, data_test = data_helper.getDataSet()
    data_train, data_test = data_helper.getDataSet()
    # wordEmbedding = data_helper.getWordEmbedding(targetFileName='./data/embeddingMatrix.300d.pkl')
    wordEmbedding = data_helper.getRandomWordEmbedding(vocabularySize=10002, embeddingSize=300)

    # model = TextBiLSTM(wordEmbedding, textLength=max_len, vocabulary_size=10002, embedding_size=128)
    model = TextBiLSTM(wordEmbedding, textLength=max_len, vocabulary_size=27590, embedding_size=300)
    # model = TextBiLSTM(wordEmbedding, textLength=max_len, vocabulary_size=42488, embedding_size=128)

    config = tf.ConfigProto()
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    sess.run(init)

    model_save_path = './ckpt/bilstmModel/bilstm.ckpt'
    # tr_batch_size = 128
    tr_batch_size = 32
    decay = 0.85
    # max_epoch = 5
    # max_max_epoch = 12
    # max_epoch = 15
    # max_max_epoch = 30
    max_epoch = 60
    max_max_epoch = 100
    display_num = 5
    tr_batch_num = int(data_train.y.shape[0] / tr_batch_size)
    display_batch = int(tr_batch_num / display_num)

    for epoch in xrange(max_max_epoch):
        _lr = 1e-3
        # if epoch > max_epoch:
        #     _lr = _lr * ((decay) ** (epoch - max_epoch))
        print 'EPOCH {}, lr={}'.format(epoch + 1, _lr)

        start_time = time.time()
        _accs = 0.0
        _costs = 0.0
        show_accs = 0.0
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
                # valid_acc, valid_cost, valid_cm = test_epoch(sess, model, data_valid)
                valid_acc, valid_cost, valid_cm = test_epoch(sess, model, data_test)
                print '\ttraining acc={}, cost={};  valid acc={}, cost={}, confusion_matrix=[{}/{}, {}/{}] '.format(show_accs / display_batch, show_costs / display_batch, valid_acc, valid_cost, valid_cm[0][0], valid_cm[0][0]+valid_cm[1][0], valid_cm[1][1], valid_cm[0][1]+valid_cm[1][1])
                show_accs = 0.0
                show_costs = 0.0

        mean_acc = _accs / tr_batch_num
        mean_cost = _costs / tr_batch_num

        if (epoch + 1) % 3 == 0:
            save_path = model.saver.save(sess, model_save_path, global_step=(epoch + 1))
            print 'The save path is ', save_path
        # print '\ttraining {}, acc={}, cost={}'.format(data_train.y.shape[0], mean_acc, mean_cost)
        # print 'Epoch training {}, acc={}, cost={}, speed={} s/epoch\n'.format(data_train.y.shape[0], mean_acc, mean_cost, time.time() - start_time)
        print 'Epoch training {}, acc={}, cost={}, speed={} s/epoch'.format(data_train.y.shape[0], mean_acc, mean_cost, time.time() - start_time)
        test_acc, test_cost, test_cm = test_epoch(sess, model, data_test)
        print '**Test {}, acc={}, cost={}, confusion_matrix=[{}/{}, {}/{}]\n'.format(data_test.y.shape[0], test_acc, test_cost, test_cm[0][0], test_cm[0][0]+test_cm[1][0], test_cm[1][1], test_cm[0][1]+test_cm[1][1])
    # print '**TEST RESULT:'
    # test_acc, test_cost, test_cm = test_epoch(sess, model, data_test)
    # print '**Test {}, acc={}, cost={}, confusion_matrix=[{}/{}, {}/{}] '.format(data_test.y.shape[0], test_acc, test_cost, test_cm[0][0], test_cm[0][0]+test_cm[1][0], test_cm[1][1], test_cm[0][1]+test_cm[1][1])