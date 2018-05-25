import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import data_helper_2018
import numpy as np
import time
from sklearn.metrics import classification_report
import os
import pickle

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
        # self.init_embeddings()
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
            dropout=config['lstm_dropout'],
            bidirectional=True,
            batch_first=True
        )

        self.response_bilstm = nn.LSTM(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.layer_num,
            dropout=config['lstm_dropout'],
            bidirectional=True,
            batch_first=True
        )

    def forward(self, X_q_inputs, X_r_inputs, q_init_state, r_init_state):

        # quote_inputs = self.dropout(self.embedding(X_q_inputs))
        # response_inputs = self.dropout(self.embedding(X_r_inputs))
        quote_inputs = self.embedding(X_q_inputs)
        response_inputs = self.embedding(X_r_inputs)

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

    def load_pretrained_embeddings(self, embeddingFileName='../qrPair/data/glove.840B.300d.txt'):
        hit = 0
        with open(embeddingFileName, 'r') as fr:
            lines = fr.readlines()
            for line in lines:
                line = line.strip().split(' ')
                word = line[0]
                embedding = line[1:]
                if word in self.word2id.index:
                    wordIndex = self.word2id[word]
                    embeddingArray = np.fromstring('\n'.join(embedding), dtype=np.float32, sep='\n')
                    self.embedding.weight.data[wordIndex].copy_(torch.from_numpy(embeddingArray))
                    hit += 1
        hitRate = float(hit) / self.vocabulary_size
        print('PreTrain Embedding hitRate={}'.format(hitRate))
        fw.write('PreTrain Embedding hitRate={}\n'.format(hitRate))

class AttentionModel(nn.Module):
    def __init__(self, config):
        super(AttentionModel, self).__init__()
        self.bilstm = TextBiLSTM(config)
        self.dropout = nn.Dropout(config['dropout'])
        self.attention_mechanism = config['attention_mechanism']
        self.hidden_size = config['hidden_size']
        self.do_BN = config['do_BN']

        if self.do_BN:
            if self.attention_mechanism['Type'] != 'both_concat':
                self.quote_output_BN = nn.BatchNorm1d(num_features=self.hidden_size * 2)
                self.response_output_BN = nn.BatchNorm1d(num_features=self.hidden_size * 2)
            else:
                self.quote_output_BN = nn.BatchNorm1d(num_features=self.hidden_size * 2 * 2)
                self.response_output_BN = nn.BatchNorm1d(num_features=self.hidden_size * 2 * 2)

        if self.attention_mechanism['Type'] != 'None':

            if self.attention_mechanism['Type'][:4] == 'both':
                self.quote2quote_attention_layer = nn.Linear(in_features=self.hidden_size * 2, out_features=1)
                self.response2quote_attention_layer = nn.Linear(in_features=self.hidden_size * 2, out_features=1)
                self.response2response_attention_layer = nn.Linear(in_features=self.hidden_size * 2, out_features=1)
                self.quote2response_attention_layer = nn.Linear(in_features=self.hidden_size * 2, out_features=1)
            elif self.attention_mechanism['Type'] == 'share':
                self.share_attention_layer = nn.Linear(in_features=self.hidden_size * 2, out_features=1)
            else:
                self.quote_attention_layer = nn.Linear(in_features=self.hidden_size * 2, out_features=1)
                self.response_attention_layer = nn.Linear(in_features=self.hidden_size * 2, out_features=1)
            # init_range = 0.1
            # self.quote_attention_layer.weight.data.uniform_(-init_range, init_range)
            # self.quote_attention_layer.bias.data.fill_(init_range)
            # self.response_attention_layer.weight.data.uniform_(-init_range, init_range)
            # self.response_attention_layer.bias.data.fill_(init_range)

    def forward(self, X_q_inputs, X_r_inputs, q_init_state, r_init_state):
        quote_outputs, response_outputs = self.bilstm.forward(X_q_inputs, X_r_inputs, q_init_state, r_init_state)

        if self.attention_mechanism['Type'] != 'None':
            ActFunc = self.attention_mechanism['ActFunc']
            if self.attention_mechanism['Type'] == 'self_attention':
                quote_outputs = self.attention_layer(self.quote_attention_layer, bilstm_outputs=quote_outputs, attention_source=quote_outputs, ActFunc=ActFunc)
                response_outputs = self.attention_layer(self.response_attention_layer, bilstm_outputs=response_outputs, attention_source=response_outputs, ActFunc=ActFunc)
            if self.attention_mechanism['Type'] == 'cross_attention':
                _quote_outputs = self.attention_layer(self.quote_attention_layer, bilstm_outputs=quote_outputs, attention_source=response_outputs, ActFunc=ActFunc)
                _response_outputs = self.attention_layer(self.response_attention_layer, bilstm_outputs=response_outputs, attention_source=quote_outputs, ActFunc=ActFunc)
                quote_outputs = _quote_outputs
                response_outputs = _response_outputs
            if self.attention_mechanism['Type'] == 'both_sum':
                _quote_outputs = self.attention_layer(self.quote2quote_attention_layer, bilstm_outputs=quote_outputs, attention_source=quote_outputs, ActFunc=ActFunc) + \
                                 self.attention_layer(self.response2quote_attention_layer, bilstm_outputs=quote_outputs, attention_source=response_outputs, ActFunc=ActFunc)
                _response_outputs = self.attention_layer(self.response2response_attention_layer, bilstm_outputs=response_outputs, attention_source=response_outputs, ActFunc=ActFunc) + \
                                    self.attention_layer(self.quote2response_attention_layer, bilstm_outputs=response_outputs, attention_source=quote_outputs, ActFunc=ActFunc)
                quote_outputs = _quote_outputs
                response_outputs = _response_outputs
            if self.attention_mechanism['Type'] == 'both_concat':
                _quote_outputs = torch.cat((self.attention_layer(self.quote2quote_attention_layer, bilstm_outputs=quote_outputs, attention_source=quote_outputs, ActFunc=ActFunc),
                                            self.attention_layer(self.response2quote_attention_layer, bilstm_outputs=quote_outputs, attention_source=response_outputs, ActFunc=ActFunc)), dim=1)
                _response_outputs = torch.cat((self.attention_layer(self.response2response_attention_layer, bilstm_outputs=response_outputs, attention_source=response_outputs, ActFunc=ActFunc), 
                                               self.attention_layer(self.quote2response_attention_layer, bilstm_outputs=response_outputs, attention_source=quote_outputs, ActFunc=ActFunc)), dim=1)
                quote_outputs = _quote_outputs
                response_outputs = _response_outputs
            if self.attention_mechanism['Type'] == 'share':
                _quote_outputs = self.attention_layer(self.share_attention_layer, bilstm_outputs=quote_outputs, attention_source=response_outputs, ActFunc=ActFunc)
                _response_outputs = self.attention_layer(self.share_attention_layer, bilstm_outputs=response_outputs, attention_source=quote_outputs, ActFunc=ActFunc)
                quote_outputs = _quote_outputs
                response_outputs = _response_outputs
        else:
            quote_outputs = torch.sum(quote_outputs, dim=1)
            response_outputs = torch.sum(response_outputs, dim=1)
            # quote_outputs = torch.mean(quote_outputs, dim=1)
            # response_outputs = torch.mean(response_outputs, dim=1)
        if self.do_BN:
            quote_outputs = self.quote_output_BN(quote_outputs)
            response_outputs = self.response_output_BN(response_outputs)
        return quote_outputs, response_outputs

    def attention_layer(self, layer, bilstm_outputs, attention_source, ActFunc):

        attention_inputs = attention_source.contiguous().view(-1, self.hidden_size * 2)  # [32, 150, 256] -> [32*150, 256]
        # attention_logits = ActFunc(layer(self.dropout(attention_inputs)))  # [32*150, 256] -> [32*150, 1]
        attention_logits = ActFunc(layer(attention_inputs))  # [32*150, 256] -> [32*150, 1]
        attention_logits = attention_logits.view(-1, attention_source.size()[1], 1)  # [32*150, 1] -> [32, 150, 1]
        attention_logits = attention_logits.transpose(1, 2).contiguous()  # [32, 150, 1] -> [32, 1, 150]
        attention_logits = attention_logits.view(-1, attention_source.size()[1])  # [32, 1, 150] -> [32*1, 150]
        attention_signals = F.softmax(attention_logits, dim=1).view(-1, 1, attention_source.size()[1])  # [32*1, 150] -> [32, 1, 150]

        return torch.bmm(attention_signals, bilstm_outputs).view(-1, bilstm_outputs.size()[2])
    
        # attention_inputs = attention_source.contiguous().view(-1, self.hidden_size * 2)
        # # attention_logits = layer(self.dropout(attention_inputs)).view(-1, attention_source.size()[1])
        # attention_logits = layer(attention_inputs).view(-1, attention_source.size()[1])
        # attention_signals = F.softmax(ActFunc(attention_logits), dim=1).view(-1, attention_source.size()[1], 1)
        # outputs = attention_signals * bilstm_outputs
        # outputs = torch.sum(outputs, dim=1)
        # # outputs = torch.mean(outputs, dim=1)
        # return outputs

    def init_hidden(self, batch_size):
        return self.bilstm.init_hidden(batch_size)

class Classifier(nn.Module):
    def __init__(self, config):
        super(Classifier, self).__init__()
        self.model = AttentionModel(config)
        self.dropout = nn.Dropout(config['dropout'])
        self.hidden_size = config['hidden_size']
        self.attention_mechanism = config['attention_mechanism']
        self.do_BN = config['do_BN']
        self.class_num = config['class_num']
        
        if self.attention_mechanism['Type'] != 'both_concat':
            self.out = nn.Linear(in_features=self.hidden_size * 2 * 2, out_features=self.class_num)
        else:
            self.out = nn.Linear(in_features=self.hidden_size * 2 * 2 * 2, out_features=self.class_num)

        # self.init_weights()

        if self.do_BN:
            # self.concat_output_BN = nn.BatchNorm1d(num_features=self.hidden_size * 2 * 2)
            if self.attention_mechanism['Type'] != 'both_concat':
                self.concat_output_BN = nn.BatchNorm1d(num_features=self.hidden_size * 2 * 2)
            else:
                self.concat_output_BN = nn.BatchNorm1d(num_features=self.hidden_size * 2 * 2 * 2)

    def init_weights(self, init_range=0.1):
        self.out.weight.data.uniform_(-init_range, init_range)
        self.out.bias.data.fill_(init_range)

    def forward(self, X_q_inputs, X_r_inputs, q_init_state, r_init_state):
        quote_outputs, response_outputs = self.model.forward(X_q_inputs, X_r_inputs, q_init_state, r_init_state)
        concat_outputs = torch.cat((quote_outputs, response_outputs), dim=1)

        # Siamese
        # response_outputs_siamese, quote_outputs_siamese = self.model.forward(X_r_inputs, X_q_inputs, r_init_state, q_init_state)
        # concat_outputs_siamese = torch.cat((quote_outputs_siamese, response_outputs_siamese), dim=1)
        # concat_outputs = (concat_outputs + concat_outputs_siamese) / 2
        # Siamese

        if self.do_BN:
            concat_outputs = self.concat_output_BN(concat_outputs)
        output = self.out(self.dropout(concat_outputs))
        return output

    def init_hidden(self, batch_size):
        return self.model.init_hidden(batch_size)

def train_epoch(model, data_train):
    model.train()
    
    tr_batch_num = int(data_train.y.shape[0] / tr_batch_size)
    display_batch = int(tr_batch_num / display_num)

    start_time = time.time()
    _accs = 0.0
    _costs = 0.0
    show_accs = 0.0
    show_costs = 0.0

    for batch in range(tr_batch_num):
        model.train()

        X_q_batch, X_r_batch, y_batch = data_train.next_batch(tr_batch_size)

        X_q_batch = Variable(torch.LongTensor(X_q_batch)).cuda() if USE_GPU else Variable(torch.LongTensor(X_q_batch))
        X_r_batch = Variable(torch.LongTensor(X_r_batch)).cuda() if USE_GPU else Variable(torch.LongTensor(X_r_batch))

        y_batch = Variable(torch.LongTensor(y_batch)).cuda() if USE_GPU else Variable(torch.LongTensor(y_batch))

        q_init_state = model.init_hidden(y_batch.size()[0])
        r_init_state = model.init_hidden(y_batch.size()[0])

        output = model(X_q_batch, X_r_batch, q_init_state, r_init_state)
        _cost = loss_func(output, y_batch)
        y_pred = torch.max(output, 1)[1]

        _acc = sum(y_pred.data.cpu() == y_batch.data.cpu()) / float(y_batch.size(0)) if USE_GPU else sum(y_pred.data == y_batch.data) / float(y_batch.size(0))

        _accs += _acc
        _costs += _cost.data[0]
        show_accs += _acc
        show_costs += _cost.data[0]

        optimizer.zero_grad()
        _cost.backward()
        nn.utils.clip_grad_norm(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()

        if (batch + 1) % display_batch == 0:
            valid_acc, valid_cost, _ = test_epoch(model, data_test)

            print('\ttraining acc={:.6f}, cost={:.6f};  valid acc={:.6f}, cost={:.6f}'.format(show_accs / display_batch, show_costs / display_batch, valid_acc, valid_cost))
            fw.write('\ttraining acc={:.6f}, cost={:.6f};  valid acc={:.6f}, cost={:.6f}\n'.format(show_accs / display_batch, show_costs / display_batch, valid_acc, valid_cost))
            show_accs = 0.0
            show_costs = 0.0

    mean_acc = _accs / tr_batch_num
    mean_cost = _costs / tr_batch_num

    print('Epoch training {}, acc={:.6f}, cost={:.6f}, speed={:.6f} s/epoch'.format(data_train.y.shape[0], mean_acc, mean_cost, time.time() - start_time))
    fw.write('Epoch training {}, acc={:.6f}, cost={:.6f}, speed={:.6f} s/epoch\n'.format(data_train.y.shape[0], mean_acc, mean_cost, time.time() - start_time))

    test_acc, test_cost, test_cr = test_epoch(model, data_test)

    print('**Test {}, acc={:.6f}, cost={:.6f}\n'.format(data_test.y.shape[0], test_acc, test_cost))
    fw.write('**Test {}, acc={:.6f}, cost={:.6f}\n'.format(data_test.y.shape[0], test_acc, test_cost))

    print(test_cr)
    fw.write(test_cr)
    fw.write('\n')

def test_epoch(model, dataset):
    model.eval()
    data_size = len(dataset.y)
    _batch_size = data_size
    # _batch_size = int(data_size / 2)
    batch_num = int(data_size / _batch_size)
    _accs = 0.0
    _costs = 0.0
    _pred = []
    _true = []
    for i in range(batch_num):

        X_q_batch, X_r_batch, y_batch = dataset.next_batch(_batch_size)

        X_q_batch = Variable(torch.LongTensor(X_q_batch), volatile=True).cuda() if USE_GPU else Variable(torch.LongTensor(X_q_batch), volatile=True)
        X_r_batch = Variable(torch.LongTensor(X_r_batch), volatile=True).cuda() if USE_GPU else Variable(torch.LongTensor(X_r_batch), volatile=True)

        y_batch = Variable(torch.LongTensor(y_batch), volatile=True).cuda() if USE_GPU else Variable(torch.LongTensor(y_batch), volatile=True)

        q_init_state = model.init_hidden(y_batch.size()[0])
        r_init_state = model.init_hidden(y_batch.size()[0])

        test_output = model(X_q_batch, X_r_batch, q_init_state, r_init_state)
        _cost = loss_func(test_output, y_batch)

        y_pred = torch.max(test_output, 1)[1]
        _acc = sum(y_pred.data.cpu() == y_batch.data.cpu()) / float(y_batch.size(0)) if USE_GPU else sum(y_pred.data == y_batch.data) / float(y_batch.size(0))

        if i == 0:
            _pred = y_pred.data.cpu().numpy() if USE_GPU else y_pred.data.numpy()
            _true = y_batch.data.cpu().numpy() if USE_GPU else y_batch.data.numpy()
        else:
            _pred = np.concatenate((_pred, y_pred.data.cpu().numpy())) if USE_GPU else np.concatenate((_pred, y_pred.data.numpy()))
            _true = np.concatenate((_true, y_batch.data.cpu().numpy())) if USE_GPU else np.concatenate((_true, y_batch.data.numpy()))
        _accs += _acc
        _costs += _cost.data[0]
    mean_acc = _accs / batch_num
    mean_cost = _costs / batch_num
    cr = classification_report(y_true=_true, y_pred=_pred, target_names=['disagree', 'agree'], digits=4)
    return mean_acc, mean_cost, cr

if __name__ == '__main__':

    USE_GPU = True
    with open('./data/data_disagree_agree_None_64.pkl', 'rb') as fr:
    # with open('./data/data_debatepedia_None_64.pkl', 'rb') as fr:
        X = pickle.load(fr)
        y = pickle.load(fr)
        word2id = pickle.load(fr)
        id2word = pickle.load(fr)
        label2id = pickle.load(fr)
        id2label = pickle.load(fr)

    np.random.seed(12)
    new_index = np.random.permutation(len(X))
    X = X[new_index]
    y = y[new_index]

    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5)

    # for type in ['None', 'share']: # gpu:0
    # for type in ['self_attention', 'cross_attention']: # gpu:1
    # for type in ['both_sum', 'both_concat']: # gpu:2
    for type in ['cross_attention']: # gpu:3

        attention_mechanism_config = {'Type': type, 'ActFunc': F.tanh}
        config = {
            'word2id': word2id, 'pretrain_emb': True, 'class_num': 2,
            'embedding_size': 300, 'hidden_size': 128, 'layer_num': 2, 'dropout': 0.3, 'lstm_dropout': 0.5,
            'do_BN': True, 'attention_mechanism': attention_mechanism_config}

        for i, (train_index, test_index) in enumerate(kf.split(X)):
            # if i != 4:
            #     continue
            # with open('./result_log/iac_%s_%s.txt' % (type, i), 'w') as fw:
            # with open('./result_log/debatepedia_%s_%s.txt' % (type, i), 'w') as fw:
            # with open('./result_log/iac_shuffle_%s_%s.txt' % (type, i), 'w') as fw:
            # with open('./result_log/iac_siamese_%s_%s.txt' % (type, i), 'w') as fw:
            with open('./result_log/iac_shuffle_19_%s_%s.txt' % (type, i), 'w') as fw:
                if USE_GPU:
                    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
                    print('Using GPU: {}...'.format(os.environ['CUDA_VISIBLE_DEVICES']))
                    fw.write('Using GPU: {}...\n'.format(os.environ['CUDA_VISIBLE_DEVICES']))

                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                print('X_train.shape={}, y_train.shape={};\nX_test.shape={}, y_test.shape={}'.format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))
                fw.write('X_train.shape={}, y_train.shape={};\nX_test.shape={}, y_test.shape={}\n'.format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))

                print('Creating the data generator ...')
                fw.write('Creating the data generator ...\n')

                data_train = data_helper_2018.BatchGenerator(X_train, y_train, shuffle=True)
                data_test = data_helper_2018.BatchGenerator(X_test, y_test, shuffle=False)

                print('Finished creating the generator.')
                fw.write('Finished creating the generator.\n')

                model = Classifier(config)

                max_grad_norm = 0.1

                for name, para in model.named_parameters():
                    if para.requires_grad == True:
                        print(name)
                        fw.write(name + '\n')
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
                    print('EPOCH {}, lr={}'.format(epoch, lr))
                    fw.write('EPOCH {}, lr={}\n'.format(epoch, lr))
                    train_epoch(model, data_train)







