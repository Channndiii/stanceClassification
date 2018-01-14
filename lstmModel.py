import tensorflow as tf
import data_helper
import time

class TextLSTM(object):
    def __init__(self, wordEmbedding, textLength, embedding_size, hidden_size=128, class_num=2):

        self.wordEmbedding = wordEmbedding
        self.textLength = textLength
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.class_num = class_num

        self.X_q_inputs = tf.placeholder(tf.int32, [None, self.textLength], name='X_q_inputs')
        self.X_r_inputs = tf.placeholder(tf.int32, [None, self.textLength], name='X_r_inputs')
        self.y_inputs = tf.placeholder(tf.int32, [None], name='y_inputs')
        self.batch_size = tf.placeholder(tf.int32)
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.lr = tf.placeholder(tf.float32)

        with tf.variable_scope('embedding'):

            word_embedding = tf.Variable(initial_value=self.wordEmbedding, name='word_embedding')
            # word_embedding = tf.Variable(initial_value=self.wordEmbedding, name='word_embedding', trainable=False)
            quote_inputs = tf.nn.embedding_lookup(word_embedding, self.X_q_inputs, name='quote_inputs')
            response_inputs = tf.nn.embedding_lookup(word_embedding, self.X_r_inputs, name='response_inputs')

        # regularizer = tf.contrib.layers.l1_regularizer(0.1)
        regularizer = tf.contrib.layers.l2_regularizer(1e-2)

        outputs = []
        for inputs in [quote_inputs, response_inputs]:
            with tf.variable_scope('LSTM-%s' % inputs.name.split('/')[-1][:-2]):
                # inputs = tf.reshape(inputs, [-1, self.embedding_size])
                # W_in = tf.Variable(tf.random_normal([self.embedding_size, self.hidden_size]), name='W_in')
                # b_in = tf.Variable(tf.constant(0.1, shape=[self.hidden_size]), name='b_in')
                # inputs = tf.matmul(inputs, W_in) + b_in
                # inputs = tf.reshape(inputs, [-1, self.textLength, self.hidden_size])

                lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size)
                lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=self.dropout_keep_prob)
                init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
                output, _ = tf.nn.dynamic_rnn(lstm_cell, inputs, initial_state=init_state, time_major=False)
                output = tf.unstack(tf.transpose(output, [1, 0, 2]))
                outputs.append(output[-1])

        # with tf.variable_scope('outputs'):
        with tf.variable_scope('outputs', regularizer=regularizer):
            outputs_concat = tf.concat(outputs, axis=1, name='concat-quote-response')
            W_out = tf.Variable(tf.random_normal([self.hidden_size * 2, self.class_num]), name='W_out')
            b_out = tf.Variable(tf.constant(0.1, shape=[self.class_num]), name='b_out')
            self.logits = tf.matmul(outputs_concat, W_out) + b_out

        self.correct_prediction = tf.equal(tf.cast(tf.argmax(self.logits, 1), tf.int32), tf.reshape(self.y_inputs, [-1]))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=tf.reshape(self.y_inputs, [-1]))) + regularization_loss

        tvars = tf.trainable_variables()
        print 'trainable_variables:'
        for x in tvars:
            print x.name
        grads = tf.gradients(self.cost, tvars)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=tf.contrib.framework.get_or_create_global_step())
        self.saver = tf.train.Saver(max_to_keep=10)
        print 'Finished creating the model.'

def test_epoch(sess, model, dataset):
    fetches = [model.accuracy, model.cost]
    _y = dataset.y
    data_size = _y.shape[0]
    _batch_size = data_size
    batch_num = int(data_size / _batch_size)
    _accs = 0.0
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

if __name__ == '__main__':

    max_len = 150
    data_train, data_valid, data_test = data_helper.getDataSet()
    # wordEmbedding = data_helper.getWordEmbedding(targetFileName='./data/embeddingMatrix.300d.pkl')
    wordEmbedding = data_helper.getRandomWordEmbedding(vocabularySize=10002, embeddingSize=300)

    model = TextLSTM(wordEmbedding, textLength=max_len, embedding_size=300)

    config = tf.ConfigProto()
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    sess.run(init)

    model_save_path = './ckpt/lstmModel/lstm.ckpt'
    tr_batch_size = 128
    decay = 0.85
    # max_epoch = 5
    # max_max_epoch = 12
    # max_epoch = 12
    # max_max_epoch = 25
    max_epoch = 120
    max_max_epoch = 200
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
                valid_acc, valid_cost = test_epoch(sess, model, data_valid)
                print '\ttraining acc={}, cost={};  valid acc={}, cost={} '.format(show_accs / display_batch,
                                                                                   show_costs / display_batch,
                                                                                   valid_acc, valid_cost)
                show_accs = 0.0
                show_costs = 0.0

        mean_acc = _accs / tr_batch_num
        mean_cost = _costs / tr_batch_num

        if (epoch + 1) % 3 == 0:
            save_path = model.saver.save(sess, model_save_path, global_step=(epoch + 1))
            print 'The save path is ', save_path
        print '\ttraining {}, acc={}, cost={}'.format(data_train.y.shape[0], mean_acc, mean_cost)
        print 'Epoch training {}, acc={}, cost={}, speed={} s/epoch\n'.format(data_train.y.shape[0], mean_acc,
                                                                              mean_cost, time.time() - start_time)
    print '**TEST RESULT:'
    test_acc, test_cost = test_epoch(sess, model, data_test)
    print '**Test {}, acc={}, cost={}'.format(data_test.y.shape[0], test_acc, test_cost)