import tensorflow as tf
from tensorflow.python.training.moving_averages import assign_moving_average
import multiTask_data_helper
import os
import time
from sklearn.metrics import confusion_matrix

class TextBiLSTM(object):
    def __init__(self, wordEmbedding, textLength, vocabulary_size,
                 embedding_size, hidden_size, layer_num,
                 class_num, do_BN, attention_mechanism, max_grad_norm=5.0):

        self.wordEmbedding = wordEmbedding
        self.textLength = textLength
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.class_num = class_num
        self.do_BN = do_BN
        self.attention_mechanism = attention_mechanism
        self.max_grad_norm = max_grad_norm

        self.X_q_inputs = tf.placeholder(tf.int32, [None, self.textLength], name='X_q_inputs')
        self.X_r_inputs = tf.placeholder(tf.int32, [None, self.textLength], name='X_r_inputs')
        self.y_inputs = tf.placeholder(tf.int32, [None], name='y_inputs')
        self.batch_size = tf.placeholder(tf.int32, [])
        self.task = tf.placeholder(tf.bool, name='taskFlag')
        self.train = tf.placeholder(tf.bool, name='train')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.lr = tf.placeholder(tf.float32)

        with tf.variable_scope('embedding'):
            if self.wordEmbedding == None:
                word_embedding = tf.get_variable('word_embedding', [self.vocabulary_size, self.embedding_size], dtype=tf.float32)
            else:
                word_embedding = tf.Variable(initial_value=self.wordEmbedding, name='word_embedding')
            quote_inputs = tf.nn.embedding_lookup(word_embedding, self.X_q_inputs, name='quote_inputs')
            response_inputs = tf.nn.embedding_lookup(word_embedding, self.X_r_inputs, name='response_inputs')

        if self.do_BN:
            quote_inputs = self.batch_normalize(quote_inputs, self.train, name='quote_inputs')
            response_inputs = self.batch_normalize(response_inputs, self.train, name='response_inputs')

        quote_outputs = self.bilstm(quote_inputs, name='quote')
        response_outputs = self.bilstm(response_inputs, name='response')

        if self.attention_mechanism != None:
            ActFunc = self.attention_mechanism['ActFunc']
            if self.attention_mechanism['Type'] == 'self_attention':
                quote_outputs = self.attention_layer(bilstm_outputs=quote_outputs, attention_source=quote_outputs, ActFunc=ActFunc, name='quote')
                response_outputs = self.attention_layer(bilstm_outputs=response_outputs, attention_source=response_outputs, ActFunc=ActFunc, name='response')
            if self.attention_mechanism['Type'] == 'cross_attention':
                _quote_outputs = self.attention_layer(bilstm_outputs=quote_outputs, attention_source=response_outputs, ActFunc=ActFunc, name='quote')
                _response_outputs = self.attention_layer(bilstm_outputs=response_outputs, attention_source=quote_outputs, ActFunc=ActFunc, name='response')
                quote_outputs = _quote_outputs
                response_outputs = _response_outputs
        else:
            quote_outputs = tf.reduce_mean(quote_outputs, axis=1)
            response_outputs = tf.reduce_mean(response_outputs, axis=1)

        if self.do_BN:
            quote_outputs = self.batch_normalize(quote_outputs, self.train, name='quote_outputs')
            response_outputs = self.batch_normalize(response_outputs, self.train, name='response_outputs')

        outputs = [quote_outputs, response_outputs]

        with tf.variable_scope('outputs'):
            outputs_concat = tf.concat(outputs, axis=1, name='concat-quote-response')
            if self.do_BN:
                outputs_concat = self.batch_normalize(outputs_concat, self.train, name='concat_outputs')
            
            W_main = tf.Variable(tf.random_normal([self.hidden_size * 2 * 2, self.class_num]), name='W_main')
            b_main = tf.Variable(tf.constant(0.1, shape=[self.class_num]), name='b_main') 
            
            W_sup = tf.Variable(tf.random_normal([self.hidden_size * 2 * 2, self.class_num]), name='W_sup')
            b_sup = tf.Variable(tf.constant(0.1, shape=[self.class_num]), name='b_sup')
            
            self.logits = tf.cond(self.task, lambda: tf.matmul(outputs_concat, W_main) + b_main, lambda: tf.matmul(outputs_concat, W_sup) + b_sup)

        self.prediction = tf.cast(tf.argmax(self.logits, 1), tf.int32)
        self.correct_prediction = tf.equal(tf.cast(tf.argmax(self.logits, 1), tf.int32), tf.reshape(self.y_inputs, [-1]))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        
        self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=tf.reshape(self.y_inputs, [-1])))

        tvars = tf.trainable_variables()
        print('trainable_variables:')
        for x in tvars:
            print(x.name)
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), self.max_grad_norm)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=tf.contrib.framework.get_or_create_global_step())
        self.saver = tf.train.Saver(max_to_keep=10)

    def lstm_cell(self):
        cell = tf.contrib.rnn.LSTMCell(self.hidden_size, reuse=tf.get_variable_scope().reuse)
        return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.dropout_keep_prob)

    def bilstm(self, inputs, name):
        with tf.variable_scope('BiLSTM-%s' % name):
            mtcell_fw = tf.contrib.rnn.MultiRNNCell([self.lstm_cell() for _ in range(self.layer_num)], state_is_tuple=True)
            mtcell_bw = tf.contrib.rnn.MultiRNNCell([self.lstm_cell() for _ in range(self.layer_num)], state_is_tuple=True)

            init_state_fw = mtcell_fw.zero_state(self.batch_size, dtype=tf.float32)
            init_state_bw = mtcell_bw.zero_state(self.batch_size, dtype=tf.float32)

            inputs = tf.unstack(tf.transpose(inputs, [1, 0, 2]))
            outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(mtcell_fw, mtcell_bw, inputs, initial_state_fw=init_state_fw, initial_state_bw=init_state_bw, dtype=tf.float32)  # tf.shape(output) = textLength * [batch_size, hidden_size *2]
            outputs = tf.transpose(tf.stack(outputs), [1, 0, 2])
            
            return outputs
    
    def attention_layer(self, bilstm_outputs, attention_source, ActFunc, name):
        with tf.variable_scope('Attention-Layer-%s' % name):
            attention_inputs = tf.reshape(attention_source, [-1, self.hidden_size * 2])
            W = tf.Variable(tf.random_normal([self.hidden_size * 2, 1]), name='W')
            b = tf.Variable(tf.constant(0.1, shape=[1]), name='b')

            attention_logits = tf.reshape(tf.matmul(attention_inputs, W) + b, [-1, self.textLength])
            attention_signals = tf.reshape(tf.nn.softmax(ActFunc(attention_logits)), [-1, self.textLength, 1])

            outputs = attention_signals * bilstm_outputs
            outputs = tf.reduce_mean(outputs, axis=1)
            return outputs

    # def batch_normalize(self, x, out_size):
    #     fc_mean, fc_var = tf.nn.moments(
    #         x,
    #         axes=[0]
    #         # axes = list(range(len(x.get_shape()) - 1))
    #     )
    #     scale = tf.Variable(tf.ones([out_size]), name='scale')
    #     shift = tf.Variable(tf.zeros([out_size]), name='shift')
    #     epsilon = 0.001
    #
    #     ema = tf.train.ExponentialMovingAverage(decay=0.5)
    #
    #     def mean_var_with_update():
    #         ema_apply_op = ema.apply([fc_mean, fc_var])
    #         with tf.control_dependencies([ema_apply_op]):
    #             return tf.identity(fc_mean), tf.identity(fc_var)
    #
    #     mean, var = mean_var_with_update()
    #
    #     x = tf.nn.batch_normalization(x, mean, var, shift, scale, epsilon)
    #     return x

    def batch_normalize(self, x, train, name, eps=1e-05, decay=0.9, affine=True):
        with tf.variable_scope('BatchNorm-%s' % name):
            params_shape = x.get_shape()[-1:]
            moving_mean = tf.get_variable('mean', params_shape, initializer=tf.zeros_initializer, trainable=False)
            moving_variance = tf.get_variable('variance', params_shape, initializer=tf.ones_initializer, trainable=False)

            def mean_var_with_update():
                mean, variance = tf.nn.moments(x, axes=list(range(len(x.get_shape())-1)), name='moments')
                with tf.control_dependencies([assign_moving_average(moving_mean, mean, decay),
                                              assign_moving_average(moving_variance, variance, decay)]):
                    return tf.identity(mean), tf.identity(variance)

            mean, variance = tf.cond(train, mean_var_with_update, lambda: (moving_mean, moving_variance))
            if affine:
                beta = tf.get_variable('beta', params_shape, initializer=tf.zeros_initializer)
                gamma = tf.get_variable('gamma', params_shape, initializer=tf.ones_initializer)
                x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, eps)
            else:
                x = tf.nn.batch_normalization(x, mean, variance, None, None, eps)
            return x

def train_epoch(sess, model, data_train, taskFlag):
    
    tr_batch_num = int(data_train.y.shape[0] / tr_batch_size)
    display_batch = int(tr_batch_num / display_num)
    
    start_time = time.time()
    _accs = 0.0
    _costs = 0.0
    show_accs = 0.0
    show_costs = 0.0

    for batch in range(tr_batch_num):
        fetches = [model.accuracy, model.cost, model.train_op]
        X_q_batch, X_r_batch, y_batch = data_train.next_batch(tr_batch_size)
        feed_dict = {
            model.X_q_inputs: X_q_batch,
            model.X_r_inputs: X_r_batch,
            model.y_inputs: y_batch,
            model.batch_size: tr_batch_size,
            model.task: taskFlag,
            model.train: True,
            model.dropout_keep_prob: 0.5,
            model.lr: _lr,
        }

        _acc, _cost, _ = sess.run(fetches, feed_dict)
        _accs += _acc
        _costs += _cost
        show_accs += _acc
        show_costs += _cost
        if (batch + 1) % display_batch == 0:
            valid_acc, valid_cost, valid_cm = test_epoch(sess, model, main_test)
            print('\ttraining acc={:.6f}, cost={:.6f};  valid acc={:.6f}, cost={:.6f}, confusion_matrix=[{}/{}, {}/{}] '.format(show_accs / display_batch, show_costs / display_batch, valid_acc, valid_cost, valid_cm[0][0], valid_cm[0][0] + valid_cm[1][0], valid_cm[1][1], valid_cm[0][1] + valid_cm[1][1]))
            show_accs = 0.0
            show_costs = 0.0

    mean_acc = _accs / tr_batch_num
    mean_cost = _costs / tr_batch_num

    # if (epoch + 1) % 3 == 0:
    #     save_path = model.saver.save(sess, model_save_path, global_step=(epoch + 1))
    #     print 'The save path is ', save_path
    print('Epoch training {}, acc={:.6f}, cost={:.6f}, speed={:.6f} s/epoch'.format(data_train.y.shape[0], mean_acc, mean_cost, time.time() - start_time))
    test_acc, test_cost, test_cm = test_epoch(sess, model, main_test)
    print('**Test {}, acc={:.6f}, cost={:.6f}, confusion_matrix=[{}/{}, {}/{}]\n'.format(main_test.y.shape[0], test_acc,
                                                                                 test_cost, test_cm[0][0],
                                                                                 test_cm[0][0] + test_cm[1][0],
                                                                                 test_cm[1][1],
                                                                                 test_cm[0][1] + test_cm[1][1]))

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
    for i in range(batch_num):
        X_q_batch, X_r_batch, y_batch = dataset.next_batch(_batch_size)
        feed_dict = {
            model.X_q_inputs: X_q_batch,
            model.X_r_inputs: X_r_batch,
            model.y_inputs: y_batch,
            model.batch_size: _batch_size,
            model.task: True,
            model.train: False,
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

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    print('Using GPU: {}...'.format(os.environ['CUDA_VISIBLE_DEVICES']))

    # attention_mechanism_config = None # gpu: 0
    attention_mechanism_config = {'Type': 'self_attention', 'ActFunc': tf.nn.tanh}  # gpu: 1
    # attention_mechanism_config = {'Type': 'cross_attention', 'ActFunc': tf.nn.tanh} # gpu: 2

    max_len = 150
    main_train, main_test, vocabulary_size = multiTask_data_helper.getDataSet(task='disagree_agree')
    sup_train, sup_test, vocabulary_size = multiTask_data_helper.getDataSet(task='match_unmatch')

    model = TextBiLSTM(wordEmbedding=None, textLength=max_len, vocabulary_size=vocabulary_size,
                       embedding_size=300, hidden_size=128, layer_num=2,
                       class_num=2, do_BN=True, attention_mechanism=attention_mechanism_config)

    config = tf.ConfigProto()
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    sess.run(init)

    model_save_path = './ckpt/bilstmModel/bilstm.ckpt'
    tr_batch_size = 32
    decay = 0.85
    max_epoch = 30
    max_max_epoch = 100
    display_num = 5
    
    _lr = 1e-3
    for epoch in range(1, max_max_epoch+1):
        if epoch > max_epoch:
            _lr = _lr * 0.97
        
        if epoch % 3 != 0:
            print('EPOCH {}, lr={}, training Main Task'.format(epoch, _lr))
            train_epoch(sess, model, main_train, taskFlag=True)
        else:
            print('EPOCH {}, lr={}, training Sup Task'.format(epoch, _lr))
            train_epoch(sess, model, sup_train, taskFlag=False)