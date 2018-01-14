import tensorflow as tf
import data_helper
import time

def add_layer(inputs, in_size, out_size, activation_function, name):

    with tf.variable_scope(name):
        Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='weights')
        biases = tf.Variable(tf.constant(0.1, shape=[out_size]), name='biases')
        Wx_plus_b = tf.matmul(inputs, Weights) + biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs

class nbowModel(object):

    def __init__(self, wordEmbedding, textLength, embedding_size, class_num=2):

        self.wordEmbedding = wordEmbedding
        self.textLength = textLength
        self.embedding_size = embedding_size
        self.class_num = class_num

        self.X_q_inputs = tf.placeholder(tf.int32, [None, self.textLength], name='X_q_inputs')
        self.X_r_inputs = tf.placeholder(tf.int32, [None, self.textLength], name='X_r_inputs')
        self.y_inputs = tf.placeholder(tf.int32, [None], name='y_inputs')
        self.lr = tf.placeholder(tf.float32)

        with tf.variable_scope('embedding'):
            # word_embedding = tf.Variable(initial_value=self.wordEmbedding, name='word_embedding')
            word_embedding = tf.Variable(initial_value=self.wordEmbedding, name='word_embedding', trainable=False)
            quote_inputs = tf.nn.embedding_lookup(word_embedding, self.X_q_inputs)
            response_inputs = tf.nn.embedding_lookup(word_embedding, self.X_r_inputs)
            self.quote_inputs = tf.reshape(tf.reduce_mean(quote_inputs, axis=1), [-1, self.embedding_size])
            self.response_inputs = tf.reshape(tf.reduce_mean(response_inputs, axis=1), [-1, self.embedding_size])

        with tf.variable_scope('full-connect-quote'):
            quote_hidden_layer_outputs1 = add_layer(self.quote_inputs, self.embedding_size, 256, activation_function=tf.nn.relu, name='h1')
            quote_hidden_layer_outputs2 = add_layer(quote_hidden_layer_outputs1, 256, 128, activation_function=tf.nn.relu, name='h2')

        with tf.variable_scope('full-connect-response'):
            response_hidden_layer_outputs1 = add_layer(self.response_inputs, self.embedding_size, 256, activation_function=tf.nn.relu, name='h1')
            response_hidden_layer_outputs2 = add_layer(response_hidden_layer_outputs1, 256, 128, activation_function=tf.nn.relu, name='h2')

        outputs_concat = tf.concat([quote_hidden_layer_outputs2, response_hidden_layer_outputs2], axis=1, name='concat-quote-response')

        self.logits = add_layer(outputs_concat, 128 * 2, 2, activation_function=None, name='logits')

        self.correct_prediction = tf.equal(tf.cast(tf.argmax(self.logits, 1), tf.int32), tf.reshape(self.y_inputs, [-1]))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(self.y_inputs, [-1]), logits=self.logits))
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
            model.lr: 1e-4,
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

    model = nbowModel(wordEmbedding, textLength=max_len, embedding_size=300)

    config = tf.ConfigProto()
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    sess.run(init)

    model_save_path = './ckpt/nbowModel/nbow.ckpt'
    tr_batch_size = 128
    decay = 0.85
    # max_epoch = 5
    # max_max_epoch = 12
    max_epoch = 60
    max_max_epoch = 100
    display_num = 5
    tr_batch_num = int(data_train.y.shape[0] / tr_batch_size)
    display_batch = int(tr_batch_num / display_num)

    for epoch in xrange(max_max_epoch):
        _lr = 1e-4
        if epoch > max_epoch:
            _lr = _lr * ((decay) ** (epoch - max_epoch))
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
        print 'Epoch training {}, acc={}, cost={}, speed={} s/epoch\n'.format(data_train.y.shape[0], mean_acc, mean_cost, time.time() - start_time)
    print '**TEST RESULT:'
    test_acc, test_cost = test_epoch(sess, model, data_test)
    print '**Test {}, acc={}, cost={}'.format(data_test.y.shape[0], test_acc, test_cost)