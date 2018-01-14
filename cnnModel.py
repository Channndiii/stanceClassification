import tensorflow as tf
import data_helper
import time

class TextCNN(object):

    def __init__(self, wordEmbedding, textLength, embedding_size, filter_sizes=[2, 3, 4], num_filters=128, class_num=2, l2_reg_lambda=0.0):

        self.wordEmbedding = wordEmbedding
        self.textLength = textLength
        self.embedding_size = embedding_size
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.class_num = class_num
        
        # Placeholders for input, output and dropout
        self.X_q_inputs = tf.placeholder(tf.int32, [None, self.textLength], name='X_q_inputs')
        self.X_r_inputs = tf.placeholder(tf.int32, [None, self.textLength], name='X_r_inputs')
        self.y_inputs = tf.placeholder(tf.int32, [None], name='y_inputs')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.lr = tf.placeholder(tf.float32)

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            # word_embedding = tf.Variable(initial_value=self.wordEmbedding, name='word_embedding')
            word_embedding = tf.Variable(initial_value=self.wordEmbedding, name='word_embedding', trainable=False)
            quote_inputs = tf.nn.embedding_lookup(word_embedding, self.X_q_inputs)
            response_inputs = tf.nn.embedding_lookup(word_embedding, self.X_r_inputs)
            quote_inputs = tf.expand_dims(quote_inputs, -1, name='quote_inputs')
            response_inputs = tf.expand_dims(response_inputs, -1, name='response_inputs')

        # Create a convolution + maxpool layer for each filter size
        num_filters_total = self.num_filters * len(self.filter_sizes)
        outputs = []
        for inputs in [quote_inputs, response_inputs]:
            pooled_outputs = []
            for i, filter_size in enumerate(self.filter_sizes):
                with tf.name_scope('conv-maxpool-%s' % inputs.name.split('/')[-1][:-2]+'-'+str(filter_size)):
                    # Convolution Layer
                    filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                    b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name='b')
                    conv = tf.nn.conv2d(
                        inputs,
                        W,
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name='conv')
                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, self.textLength - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name='pool')
                    pooled_outputs.append(pooled)
    
            # Combine all the pooled features
            self.h_pool = tf.concat(pooled_outputs, axis=3)
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
    
            # Add dropout
            with tf.name_scope('dropout-%s' % inputs.name.split('/')[-1][:-2]):
                self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
                outputs.append(self.h_drop)
        
        outputs_concat = tf.concat(outputs, axis=1, name='concat-quote-response')

        # Final (unnormalized) scores and predictions
        with tf.name_scope('output'):
            W = tf.get_variable(
                'W',
                shape=[num_filters_total * 2, class_num],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[class_num]), name='b')
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(outputs_concat, W, b, name='scores')
            self.predictions = tf.cast(tf.argmax(self.scores, 1), tf.int32, name='prediction')

        # Calculate mean cross-entropy loss
        with tf.name_scope('loss'):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.scores, labels=tf.reshape(self.y_inputs, [-1]))
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predictions, tf.reshape(self.y_inputs, [-1]))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='accuracy')

        tvars = tf.trainable_variables()
        print 'trainable_variables:'
        for x in tvars:
            print x.name
        grads = tf.gradients(self.loss, tvars)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=tf.contrib.framework.get_or_create_global_step())
        self.saver = tf.train.Saver(max_to_keep=10)
        print 'Finished creating the model.'

def test_epoch(sess, model, dataset):
    fetches = [model.accuracy, model.loss]
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

    model = TextCNN(wordEmbedding, textLength=max_len, embedding_size=300)

    config = tf.ConfigProto()
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    sess.run(init)

    model_save_path = './ckpt/cnnModel/cnn.ckpt'
    tr_batch_size = 128
    decay = 0.85
    # max_epoch = 5
    # max_max_epoch = 12
    max_epoch = 12
    max_max_epoch = 25
    display_num = 5
    tr_batch_num = int(data_train.y.shape[0] / tr_batch_size)
    display_batch = int(tr_batch_num / display_num)

    for epoch in xrange(max_max_epoch):
        _lr = 1e-3
        if epoch > max_epoch:
            _lr = _lr * ((decay) ** (epoch - max_epoch))
        print 'EPOCH {}, lr={}'.format(epoch + 1, _lr)

        start_time = time.time()
        _accs = 0.0
        _costs = 0.0
        show_accs = 0.0
        show_costs = 0.0

        for batch in xrange(tr_batch_num):
            fetches = [model.accuracy, model.loss, model.train_op]
            X_q_batch, X_r_batch, y_batch = data_train.next_batch(tr_batch_size)
            feed_dict = {
                model.X_q_inputs: X_q_batch,
                model.X_r_inputs: X_r_batch,
                model.y_inputs: y_batch,
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
        print 'Epoch training {}, acc={}, cost={}, speed={} s/epoch\n'.format(data_train.y.shape[0], mean_acc, mean_cost, time.time() - start_time)
    print '**TEST RESULT:'
    test_acc, test_cost = test_epoch(sess, model, data_test)
    print '**Test {}, acc={}, cost={}'.format(data_test.y.shape[0], test_acc, test_cost)