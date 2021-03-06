import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import data_helper_2018
import numpy as np
import time
from sklearn.metrics import confusion_matrix
import os

class TextBiLSTM(nn.Module):
    def __init__(self, config):
        super(TextBiLSTM, self).__init__()
        self.word2id = config['word2id']
        self.pretrain_emb = config['pretrain_emb']
        self.vocabulary_size = len(self.word2id) + 1
        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']
        self.layer_num = config['layer_num']
        self.do_BN = config['do_BN']

        self.dropout = nn.Dropout(config['dropout'])
        self.embedding = nn.Embedding(num_embeddings=self.vocabulary_size, embedding_dim=self.embedding_size)
        self.init_embeddings()
        self.embedding.weight.data[0] = 0

        if self.pretrain_emb:
            self.load_pretrained_embeddings()

        if self.do_BN:
            self.quote_input_BN = nn.BatchNorm1d(num_features=self.embedding_size)
            self.response_input_BN = nn.BatchNorm1d(num_features=self.embedding_size)

        self.quote_bilstm = nn.LSTM(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.layer_num,
            dropout=config['dropout'],
            bidirectional=True,
            batch_first=True
        )

        self.response_bilstm = nn.LSTM(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.layer_num,
            dropout=config['dropout'],
            bidirectional=True,
            batch_first=True
        )

    def forward(self, X_q_inputs, X_r_inputs, q_init_state, r_init_state):

        quote_inputs = self.dropout(self.embedding(X_q_inputs))
        response_inputs = self.dropout(self.embedding(X_r_inputs))

        if self.do_BN:
            quote_inputs = quote_inputs.transpose(1, 2).contiguous()
            quote_inputs = self.quote_input_BN(quote_inputs)
            quote_inputs = quote_inputs.transpose(1, 2)
            response_inputs = response_inputs.transpose(1, 2).contiguous()
            response_inputs = self.response_input_BN(response_inputs)
            response_inputs = response_inputs.transpose(1, 2)

        quote_outputs, _ = self.quote_bilstm(quote_inputs, q_init_state)
        response_outputs, _ = self.response_bilstm(response_inputs, r_init_state)

        return quote_outputs, response_outputs

    def init_embeddings(self, init_range=0.1):
        self.embedding.weight.data.uniform_(-init_range, init_range)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.layer_num * 2, batch_size, self.hidden_size).zero_()),
                Variable(weight.new(self.layer_num * 2, batch_size, self.hidden_size).zero_()))

    def load_pretrained_embeddings(self, embeddingFileName='/home/chandi/Downloads/glove.840B.300d.txt'):
        hit = 0
        with open(embeddingFileName, 'r') as fr:
            lines = fr.readlines()
            for line in lines:
                line = line.strip().split(' ')
                word = line[0].decode('utf-8')
                embedding = line[1:]
                if word in self.word2id.index:
                    wordIndex = self.word2id[word]
                    embeddingArray = np.fromstring('\n'.join(embedding), dtype=np.float32, sep='\n')
                    self.embedding.weight.data[wordIndex] = embeddingArray
                    hit += 1
        hitRate = float(hit) / self.vocabularySize
        print 'PreTrain Embedding hitRate={}'.format(hitRate)

class AttentionModel(nn.Module):
    def __init__(self, config):
        super(AttentionModel, self).__init__()
        self.bilstm = TextBiLSTM(config)
        self.dropout = nn.Dropout(config['dropout'])
        self.attention_mechanism = config['attention_mechanism']
        self.hidden_size = config['hidden_size']
        self.do_BN = config['do_BN']

        if self.do_BN:
            self.quote_output_BN = nn.BatchNorm1d(num_features=self.hidden_size * 2)
            self.response_output_BN = nn.BatchNorm1d(num_features=self.hidden_size * 2)

        if self.attention_mechanism != None:
            self.quote_attention_layer = nn.Linear(in_features=self.hidden_size * 2, out_features=1)
            self.response_attention_layer = nn.Linear(in_features=self.hidden_size * 2, out_features=1)
            init_range = 0.1
            self.quote_attention_layer.weight.data.uniform_(-init_range, init_range)
            self.quote_attention_layer.bias.data.fill_(init_range)
            self.response_attention_layer.weight.data.uniform_(-init_range, init_range)
            self.response_attention_layer.bias.data.fill_(init_range)

    def forward(self, X_q_inputs, X_r_inputs, q_init_state, r_init_state):
        quote_outputs, response_outputs = self.bilstm.forward(X_q_inputs, X_r_inputs, q_init_state, r_init_state)

        if self.attention_mechanism != None:
            ActFunc = self.attention_mechanism['ActFunc']
            if self.attention_mechanism['Type'] == 'self_attention':
                quote_outputs = self.attention_layer(self.quote_attention_layer, bilstm_outputs=quote_outputs,
                                                     attention_source=quote_outputs, ActFunc=ActFunc)
                response_outputs = self.attention_layer(self.response_attention_layer, bilstm_outputs=response_outputs,
                                                        attention_source=response_outputs, ActFunc=ActFunc)
            if self.attention_mechanism['Type'] == 'cross_attention':
                _quote_outputs = self.attention_layer(self.quote_attention_layer, bilstm_outputs=quote_outputs,
                                                      attention_source=response_outputs, ActFunc=ActFunc)
                _response_outputs = self.attention_layer(self.response_attention_layer, bilstm_outputs=response_outputs,
                                                         attention_source=quote_outputs, ActFunc=ActFunc)
                quote_outputs = _quote_outputs
                response_outputs = _response_outputs
        else:
            quote_outputs = torch.mean(quote_outputs, dim=1)
            response_outputs = torch.mean(response_outputs, dim=1)
        if self.do_BN:
            quote_outputs = self.quote_output_BN(quote_outputs)
            response_outputs = self.response_output_BN(response_outputs)
        return quote_outputs, response_outputs

    def attention_layer(self, layer, bilstm_outputs, attention_source, ActFunc):

        # attention_inputs = attention_source.contiguous().view(-1, self.hidden_size * 2)  # [32, 150, 256] -> [32*150, 256]
        # attention_logits = ActFunc(layer(self.dropout(attention_inputs)))  # [32*150, 256] -> [32*150, 1]
        # attention_logits = attention_logits.view(-1, attention_source.size()[1], 1)  # [32*150, 1] -> [32, 150, 1]
        # attention_logits = attention_logits.transpose(1, 2).contiguous()  # [32, 150, 1] -> [32, 1, 150]
        # attention_logits = attention_logits.view(-1, attention_source.size()[1])  # [32, 1, 150] -> [32*1, 150]
        # attention_signals = F.softmax(attention_logits, dim=1).view(-1, 1, attention_source.size()[1])  # [32*1, 150] -> [32, 1, 150]
        #
        # return torch.bmm(attention_signals, bilstm_outputs).view(-1, bilstm_outputs.size()[2])

        attention_inputs = attention_source.contiguous().view(-1, self.hidden_size * 2)
        attention_logits = layer(attention_inputs).view(-1, attention_source.size()[1])
        attention_signals = F.softmax(ActFunc(attention_logits), dim=1).view(-1, attention_source.size()[1], 1)
        outputs = attention_signals * bilstm_outputs
        outputs = torch.sum(outputs, dim=1)
        # outputs = torch.mean(outputs, dim=1)
        return outputs

    def init_hidden(self, batch_size):
        return self.bilstm.init_hidden(batch_size)

class Classifier(nn.Module):
    def __init__(self, config):
        super(Classifier, self).__init__()
        self.model = AttentionModel(config)
        self.dropout = nn.Dropout(config['dropout'])
        self.hidden_size = config['hidden_size']
        self.do_BN = config['do_BN']
        self.class_num = config['class_num']
        self.out = nn.Linear(in_features=self.hidden_size * 2 * 2, out_features=self.class_num)
        self.init_weights()

        if self.do_BN:
            self.concat_output_BN = nn.BatchNorm1d(num_features=self.hidden_size * 2 * 2)

    def init_weights(self, init_range=0.1):
        self.out.weight.data.uniform_(-init_range, init_range)
        self.out.bias.data.fill_(init_range)

    def forward(self, X_q_inputs, X_r_inputs, q_init_state, r_init_state):
        quote_outputs, response_outputs = self.model.forward(X_q_inputs, X_r_inputs, q_init_state, r_init_state)
        concat_outputs = torch.cat((quote_outputs, response_outputs), dim=1)
        if self.do_BN:
            concat_outputs = self.concat_output_BN(concat_outputs)
        output = self.out(self.dropout(concat_outputs))
        return output

    def init_hidden(self, batch_size):
        return self.model.init_hidden(batch_size)

def train_epoch(model, data_train):
    tr_batch_num = int(data_train.y.shape[0] / tr_batch_size)
    display_batch = int(tr_batch_num / display_num)

    start_time = time.time()
    _accs = 0.0
    _costs = 0.0
    show_accs = 0.0
    show_costs = 0.0

    for batch in xrange(tr_batch_num):

        X_q_batch, X_r_batch, y_batch = data_train.next_batch(tr_batch_size)

        X_q_batch = Variable(torch.LongTensor(X_q_batch)).cuda() if USE_GPU else Variable(torch.LongTensor(X_q_batch))
        X_r_batch = Variable(torch.LongTensor(X_r_batch)).cuda() if USE_GPU else Variable(torch.LongTensor(X_r_batch))
        y_batch = Variable(torch.LongTensor(y_batch)).cuda() if USE_GPU else Variable(torch.LongTensor(y_batch))

        q_init_state = model.init_hidden(y_batch.size()[0])
        r_init_state = model.init_hidden(y_batch.size()[0])

        output = model(X_q_batch, X_r_batch, q_init_state, r_init_state)
        _cost = loss_func(output, y_batch)
        y_pred = torch.max(output, 1)[1]

        if USE_GPU:
            _acc = sum(y_pred.data.cpu() == y_batch.data.cpu()) / float(y_batch.size(0))
        else:
            _acc = sum(y_pred.data == y_batch.data) / float(y_batch.size(0))

        _accs += _acc
        _costs += _cost.data[0]
        show_accs += _acc
        show_costs += _cost.data[0]

        optimizer.zero_grad()
        _cost.backward()
        nn.utils.clip_grad_norm(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()

        if (batch + 1) % display_batch == 0:
            model.eval()
            valid_acc, valid_cost, valid_cm = test_epoch(model, data_test)
            print '\ttraining acc={}, cost={};  valid acc={}, cost={}, confusion_matrix=[{}/{}, {}/{}] '.format(
                show_accs / display_batch, show_costs / display_batch, valid_acc, valid_cost, valid_cm[0][0],
                valid_cm[0][0] + valid_cm[1][0], valid_cm[1][1], valid_cm[0][1] + valid_cm[1][1])
            show_accs = 0.0
            show_costs = 0.0
            model.train()

    mean_acc = _accs / tr_batch_num
    mean_cost = _costs / tr_batch_num

    model.eval()
    print 'Epoch training {}, acc={}, cost={}, speed={} s/epoch'.format(data_train.y.shape[0], mean_acc, mean_cost, time.time() - start_time)
    test_acc, test_cost, test_cm = test_epoch(model, data_test)
    print '**Test {}, acc={}, cost={}, confusion_matrix=[{}/{}, {}/{}]\n'.format(data_test.y.shape[0], test_acc,
                                                                                 test_cost, test_cm[0][0],
                                                                                 test_cm[0][0] + test_cm[1][0],
                                                                                 test_cm[1][1],
                                                                                 test_cm[0][1] + test_cm[1][1])
    model.train()

def test_epoch(model, dataset):
    data_size = len(dataset.y)
    _batch_size = data_size
    batch_num = int(data_size / _batch_size)
    _accs = 0.0
    _costs = 0.0
    _pred = []
    _true = []
    for i in xrange(batch_num):

        X_q_batch, X_r_batch, y_batch = dataset.next_batch(_batch_size)

        X_q_batch = Variable(torch.LongTensor(X_q_batch), volatile=True).cuda() if USE_GPU else Variable(torch.LongTensor(X_q_batch), volatile=True)
        X_r_batch = Variable(torch.LongTensor(X_r_batch), volatile=True).cuda() if USE_GPU else Variable(torch.LongTensor(X_r_batch), volatile=True)

        y_batch = Variable(torch.LongTensor(y_batch), volatile=True).cuda() if USE_GPU else Variable(torch.LongTensor(y_batch), volatile=True)

        q_init_state = model.init_hidden(y_batch.size()[0])
        r_init_state = model.init_hidden(y_batch.size()[0])

        test_output = model(X_q_batch, X_r_batch, q_init_state, r_init_state)
        _cost = loss_func(test_output, y_batch)

        y_pred = torch.max(test_output, 1)[1]
        if USE_GPU:
            _acc = sum(y_pred.data.cpu() == y_batch.data.cpu()) / float(y_batch.size(0))
        else:
            _acc = sum(y_pred.data == y_batch.data) / float(y_batch.size(0))
        if i == 0:
            if USE_GPU:
                _pred = y_pred.data.cpu().numpy()
                _true = y_batch.data.cpu().numpy()
            else:
                _pred = y_pred.data.numpy()
                _true = y_batch.data.numpy()
        else:
            _pred = np.concatenate((_pred, y_pred.data.numpy()))
            _true = np.concatenate((_true, y_batch.data.numpy()))
        _accs += _acc
        _costs += _cost.data[0]
    mean_acc = _accs / batch_num
    mean_cost = _costs / batch_num
    cm = confusion_matrix(y_true=_true, y_pred=_pred)
    return mean_acc, mean_cost, cm

'''
class TextBiLSTM(nn.Module):
    def __init__(self, config):
        super(TextBiLSTM, self).__init__()
        self.word2id = config['word2id']
        self.pretrain_emb = config['pretrain_emb']
        self.vocabulary_size = len(self.word2id) + 1
        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']
        self.layer_num = config['layer_num']
        self.do_BN = config['do_BN']

        self.dropout = nn.Dropout(config['dropout'])
        self.embedding = nn.Embedding(num_embeddings=self.vocabulary_size, embedding_dim=self.embedding_size)
        self.init_embeddings()
        self.embedding.weight.data[0] = 0

        if self.pretrain_emb:
            self.load_pretrained_embeddings()

        if self.do_BN:
            self.quote_input_BN = nn.BatchNorm1d(num_features=self.embedding_size)
            self.response_input_BN = nn.BatchNorm1d(num_features=self.embedding_size)

        self.quote_bilstm = nn.LSTM(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.layer_num,
            dropout=config['dropout'],
            bidirectional=True,
            batch_first=True
        )

        self.response_bilstm = nn.LSTM(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.layer_num,
            dropout=config['dropout'],
            bidirectional=True,
            batch_first=True
        )

    def forward(self, X_q_inputs, X_q_mask, X_r_inputs, X_r_mask, q_init_state, r_init_state):

        quote_inputs = self.dropout(self.embedding(X_q_inputs))
        response_inputs = self.dropout(self.embedding(X_r_inputs))

        if self.do_BN:
            quote_inputs = quote_inputs.transpose(1, 2).contiguous()
            quote_inputs = self.quote_input_BN(quote_inputs)
            quote_inputs = quote_inputs.transpose(1, 2)
            response_inputs = response_inputs.transpose(1, 2).contiguous()
            response_inputs = self.response_input_BN(response_inputs)
            response_inputs = response_inputs.transpose(1, 2)

        quote_outputs, _ = self.quote_bilstm(quote_inputs, q_init_state)
        response_outputs, _ = self.response_bilstm(response_inputs, r_init_state)

        quote_outputs = torch.mul(quote_outputs, X_q_mask[:, :, None])
        response_outputs = torch.mul(response_outputs, X_r_mask[:, :, None])

        return quote_outputs, response_outputs

    def init_embeddings(self, init_range=0.1):
        self.embedding.weight.data.uniform_(-init_range, init_range)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.layer_num *2, batch_size, self.hidden_size).zero_()),
                Variable(weight.new(self.layer_num *2, batch_size, self.hidden_size).zero_()))

    def load_pretrained_embeddings(self, embeddingFileName='/home/chandi/Downloads/glove.840B.300d.txt'):
        hit = 0
        with open(embeddingFileName, 'r') as fr:
            lines = fr.readlines()
            for line in lines:
                line = line.strip().split(' ')
                word = line[0].decode('utf-8')
                embedding = line[1:]
                if word in self.word2id.index:
                    wordIndex = self.word2id[word]
                    embeddingArray = np.fromstring('\n'.join(embedding), dtype=np.float32, sep='\n')
                    self.embedding.weight.data[wordIndex] = embeddingArray
                    hit += 1
        hitRate = float(hit) / self.vocabularySize
        print 'PreTrain Embedding hitRate={}'.format(hitRate)
class AttentionModel(nn.Module):
    def __init__(self, config):
        super(AttentionModel, self).__init__()
        self.bilstm = TextBiLSTM(config)
        self.dropout = nn.Dropout(config['dropout'])
        self.attention_mechanism = config['attention_mechanism']
        self.hidden_size = config['hidden_size']
        self.do_BN = config['do_BN']

        if self.do_BN:
            self.quote_output_BN = nn.BatchNorm1d(num_features=self.hidden_size * 2)
            self.response_output_BN = nn.BatchNorm1d(num_features=self.hidden_size * 2)

        if self.attention_mechanism != None:
            self.quote_attention_layer = nn.Linear(in_features=self.hidden_size * 2, out_features=1)
            self.response_attention_layer = nn.Linear(in_features=self.hidden_size * 2, out_features=1)
            init_range=0.1
            self.quote_attention_layer.weight.data.uniform_(-init_range, init_range)
            self.quote_attention_layer.bias.data.fill_(init_range)
            self.response_attention_layer.weight.data.uniform_(-init_range, init_range)
            self.response_attention_layer.bias.data.fill_(init_range)

    def forward(self, X_q_inputs, X_q_mask, X_r_inputs, X_r_mask, q_init_state, r_init_state):
        quote_outputs, response_outputs = self.bilstm.forward(X_q_inputs, X_q_mask, X_r_inputs, X_r_mask, q_init_state, r_init_state)

        if self.attention_mechanism != None:
            ActFunc = self.attention_mechanism['ActFunc']
            if self.attention_mechanism['Type'] == 'self_attention':
                quote_outputs = self.attention_layer(self.quote_attention_layer, bilstm_outputs=quote_outputs, attention_source=quote_outputs, ActFunc=ActFunc)
                response_outputs = self.attention_layer(self.response_attention_layer, bilstm_outputs=response_outputs, attention_source=response_outputs, ActFunc=ActFunc)
            if self.attention_mechanism['Type'] == 'cross_attention':
                _quote_outputs = self.attention_layer(self.quote_attention_layer, bilstm_outputs=quote_outputs, attention_source=response_outputs, ActFunc=ActFunc)
                _response_outputs = self.attention_layer(self.response_attention_layer, bilstm_outputs=response_outputs, attention_source=quote_outputs, ActFunc=ActFunc)
                quote_outputs = _quote_outputs
                response_outputs = _response_outputs
        else:
            quote_outputs = torch.mean(quote_outputs, dim=1)
            response_outputs = torch.mean(response_outputs, dim=1)
        if self.do_BN:
            quote_outputs = self.quote_output_BN(quote_outputs)
            response_outputs = self.response_output_BN(response_outputs)
        return quote_outputs, response_outputs

    def attention_layer(self, layer, bilstm_outputs, attention_source, ActFunc):

        attention_inputs = attention_source.contiguous().view(-1, self.hidden_size*2) # [32, 150, 256] -> [32*150, 256]
        attention_logits = ActFunc(layer(self.dropout(attention_inputs))) # [32*150, 256] -> [32*150, 1]
        attention_logits = attention_logits.view(-1, attention_source.size()[1], 1) # [32*150, 1] -> [32, 150, 1]
        attention_logits = attention_logits.transpose(1, 2).contiguous() # [32, 150, 1] -> [32, 1, 150]
        attention_logits = attention_logits.view(-1, attention_source.size()[1]) # [32, 1, 150] -> [32*1, 150]
        attention_signals = F.softmax(attention_logits, dim=1).view(-1, 1, attention_source.size()[1]) # [32*1, 150] -> [32, 1, 150]

        return torch.bmm(attention_signals, bilstm_outputs).view(-1, bilstm_outputs.size()[2])

    def init_hidden(self, batch_size):
        return self.bilstm.init_hidden(batch_size)
class Classifier(nn.Module):
    def __init__(self, config):
        super(Classifier, self).__init__()
        self.model = AttentionModel(config)
        self.dropout = nn.Dropout(config['dropout'])
        self.hidden_size = config['hidden_size']
        self.do_BN = config['do_BN']
        self.class_num = config['class_num']
        self.out = nn.Linear(in_features=self.hidden_size*2*2, out_features=self.class_num)
        self.init_weights()

        if self.do_BN:
            self.concat_output_BN = nn.BatchNorm1d(num_features=self.hidden_size * 2 * 2)

    def init_weights(self, init_range=0.1):
        self.out.weight.data.uniform_(-init_range, init_range)
        self.out.bias.data.fill_(init_range)

    def forward(self, X_q_inputs, X_q_mask, X_r_inputs, X_r_mask, q_init_state, r_init_state):
        quote_outputs, response_outputs = self.model.forward(X_q_inputs, X_q_mask, X_r_inputs, X_r_mask, q_init_state, r_init_state)
        concat_outputs = torch.cat((quote_outputs, response_outputs), dim=1)
        if self.do_BN:
            concat_outputs = self.concat_output_BN(concat_outputs)
        output = self.out(self.dropout(concat_outputs))
        return output

    def init_hidden(self, batch_size):
        return self.model.init_hidden(batch_size)
def train_epoch(model, data_train):

    tr_batch_num = int(len(data_train.y) / tr_batch_size)
    display_batch = int(tr_batch_num / display_num)

    start_time = time.time()
    _accs = 0.0
    _costs = 0.0
    show_accs = 0.0
    show_costs = 0.0

    for batch in xrange(tr_batch_num):

        X_q_batch, X_q_mask, X_r_batch, X_r_mask, y_batch = data_train.next_batch(tr_batch_size)

        X_q_batch = Variable(torch.LongTensor(X_q_batch)).cuda() if USE_GPU else Variable(torch.LongTensor(X_q_batch))
        X_r_batch = Variable(torch.LongTensor(X_r_batch)).cuda() if USE_GPU else Variable(torch.LongTensor(X_r_batch))

        X_q_mask = Variable(torch.FloatTensor(X_q_mask)).cuda() if USE_GPU else Variable(torch.FloatTensor(X_q_mask))
        X_r_mask = Variable(torch.FloatTensor(X_r_mask)).cuda() if USE_GPU else Variable(torch.FloatTensor(X_r_mask))

        y_batch = Variable(torch.LongTensor(y_batch)).cuda() if USE_GPU else Variable(torch.LongTensor(y_batch))

        q_init_state = model.init_hidden(y_batch.size()[0])
        r_init_state = model.init_hidden(y_batch.size()[0])

        output = model(X_q_batch, X_q_mask, X_r_batch, X_r_mask, q_init_state, r_init_state)
        _cost = loss_func(output, y_batch)
        y_pred = torch.max(output, 1)[1]

        if USE_GPU:
            _acc = sum(y_pred.data.cpu() == y_batch.data.cpu()) / float(y_batch.size(0))
        else:
            _acc = sum(y_pred.data == y_batch.data) / float(y_batch.size(0))

        _accs += _acc
        _costs += _cost.data[0]
        show_accs += _acc
        show_costs += _cost.data[0]

        optimizer.zero_grad()
        _cost.backward()
        nn.utils.clip_grad_norm(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()

        if (batch + 1) % display_batch == 0:
            model.eval()
            valid_acc, valid_cost, valid_cm = test_epoch(model, data_test)
            print '\ttraining acc={}, cost={};  valid acc={}, cost={}, confusion_matrix=[{}/{}, {}/{}] '.format(
                show_accs / display_batch, show_costs / display_batch, valid_acc, valid_cost, valid_cm[0][0],
                valid_cm[0][0] + valid_cm[1][0], valid_cm[1][1], valid_cm[0][1] + valid_cm[1][1])
            show_accs = 0.0
            show_costs = 0.0
            model.train()

    mean_acc = _accs / tr_batch_num
    mean_cost = _costs / tr_batch_num

    model.eval()
    print 'Epoch training {}, acc={}, cost={}, speed={} s/epoch'.format(len(data_train.y), mean_acc, mean_cost, time.time() - start_time)
    test_acc, test_cost, test_cm = test_epoch(model, data_test)
    print '**Test {}, acc={}, cost={}, confusion_matrix=[{}/{}, {}/{}]\n'.format(len(data_test.y), test_acc,
                                                                                 test_cost, test_cm[0][0],
                                                                                 test_cm[0][0] + test_cm[1][0],
                                                                                 test_cm[1][1],
                                                                                 test_cm[0][1] + test_cm[1][1])
    model.train()
def test_epoch(model, dataset):
    data_size = len(dataset.y)
    _batch_size = data_size
    batch_num = int(data_size / _batch_size)
    _accs = 0.0
    _costs = 0.0
    _pred = []
    _true = []
    for i in xrange(batch_num):

        X_q_batch, X_q_mask, X_r_batch, X_r_mask, y_batch = dataset.next_batch(_batch_size)

        X_q_batch = Variable(torch.LongTensor(X_q_batch), volatile=True).cuda() if USE_GPU else Variable(torch.LongTensor(X_q_batch), volatile=True)
        X_r_batch = Variable(torch.LongTensor(X_r_batch), volatile=True).cuda() if USE_GPU else Variable(torch.LongTensor(X_r_batch), volatile=True)

        X_q_mask = Variable(torch.FloatTensor(X_q_mask), volatile=True).cuda() if USE_GPU else Variable(torch.FloatTensor(X_q_mask), volatile=True)
        X_r_mask = Variable(torch.FloatTensor(X_r_mask), volatile=True).cuda() if USE_GPU else Variable(torch.FloatTensor(X_r_mask), volatile=True)

        y_batch = Variable(torch.LongTensor(y_batch), volatile=True).cuda() if USE_GPU else Variable(torch.LongTensor(y_batch), volatile=True)

        q_init_state = model.init_hidden(y_batch.size()[0])
        r_init_state = model.init_hidden(y_batch.size()[0])

        test_output = model(X_q_batch, X_q_mask, X_r_batch, X_r_mask, q_init_state, r_init_state)
        _cost = loss_func(test_output, y_batch)

        y_pred = torch.max(test_output, 1)[1]
        if USE_GPU:
            _acc = sum(y_pred.data.cpu() == y_batch.data.cpu()) / float(y_batch.size(0))
        else:
            _acc = sum(y_pred.data == y_batch.data) / float(y_batch.size(0))
        if i == 0:
            if USE_GPU:
                _pred = y_pred.data.cpu().numpy()
                _true = y_batch.data.cpu().numpy()
            else:
                _pred = y_pred.data.numpy()
                _true = y_batch.data.numpy()
        else:
            _pred = np.concatenate((_pred, y_pred.data.numpy()))
            _true = np.concatenate((_true, y_batch.data.numpy()))
        _accs += _acc
        _costs += _cost.data[0]
    mean_acc = _accs / batch_num
    mean_cost = _costs / batch_num
    cm = confusion_matrix(y_true=_true, y_pred=_pred)
    return mean_acc, mean_cost, cm
'''
if __name__ == '__main__':

    USE_GPU = False
    if USE_GPU:
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        print 'Using GPU: {}...'.format(os.environ['CUDA_VISIBLE_DEVICES'])
    
    # attention_mechanism_config = None # gpu: 0
    attention_mechanism_config = {'Type': 'self_attention', 'ActFunc': F.tanh} # gpu: 1
    # attention_mechanism_config = {'Type': 'cross_attention', 'ActFunc': F.tanh} # gpu: 2
    
    data_train, data_test, word2id = data_helper_2018.getDataSet(task='disagree_agree')

    config = {
        'word2id': word2id, 'pretrain_emb': False, 'class_num': 2,
        'embedding_size': 300,  'hidden_size': 128,  'layer_num': 2, 'dropout': 0.5,
        'do_BN': True,  'attention_mechanism': attention_mechanism_config}
    
    model = Classifier(config)

    max_grad_norm = 5.0

    for name, para in model.named_parameters():
        if para.requires_grad == True:
            print name
    if USE_GPU:
        model.cuda()

    lr = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()

    tr_batch_size = 32
    max_epoch = 30
    max_max_epoch = 100
    display_num = 5

    for epoch in range(1, max_max_epoch+1):
        if epoch > max_epoch:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.97
        lr = optimizer.param_groups[0]['lr']
        print 'EPOCH {}, lr={}'.format(epoch, lr)
        train_epoch(model, data_train)







