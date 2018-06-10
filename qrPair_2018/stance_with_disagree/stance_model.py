import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import stance_data_helper
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import os
import pickle
from sklearn.model_selection import KFold
import time

torch.manual_seed(12)

class TextBiLSTM(nn.Module):
    def __init__(self, config):
        super(TextBiLSTM, self).__init__()
        self.word2id = config['word2id']
        self.pretrain_emb = config['pretrain_emb']
        self.vocabulary_size = len(self.word2id) + 1
        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']
        self.layer_num = config['layer_num']

        self.drop_prob = config['drop_prob']
        self.embedding = nn.Embedding(num_embeddings=self.vocabulary_size, embedding_dim=self.embedding_size)
        # self.dropout = nn.Dropout(self.drop_prob)
        # self.init_embeddings()
        self.embedding.weight.data[0] = 0

        if self.pretrain_emb:
            self.load_pretrained_embeddings()

        self.bilstm = nn.LSTM(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.layer_num,
            dropout=config['lstm_drop_prob'],
            bidirectional=True,
            batch_first=True
        )


    def forward(self, X_q_inputs, X_r_inputs, q_init_state, r_init_state):

        # quote_inputs = self.dropout(self.embedding(X_q_inputs))
        # response_inputs = self.dropout(self.embedding(X_r_inputs))
        quote_inputs = self.embedding(X_q_inputs)
        response_inputs = self.embedding(X_r_inputs)

        quote_outputs, _ = self.bilstm(quote_inputs, q_init_state)
        response_outputs, _ = self.bilstm(response_inputs, r_init_state)

        return quote_outputs, response_outputs

    def init_embeddings(self, init_range=0.1):
        self.embedding.weight.data.uniform_(-init_range, init_range)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.layer_num * 2, batch_size, self.hidden_size).zero_()),
                Variable(weight.new(self.layer_num * 2, batch_size, self.hidden_size).zero_()))

    def load_pretrained_embeddings(self, embeddingFileName='../../qrPair/data/glove.840B.300d.txt'):
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
        self.drop_prob = config['drop_prob']
        self.attention_mechanism = config['attention_mechanism']
        self.hidden_size = config['hidden_size']

        if self.attention_mechanism['Type'] != 'None':

            if self.attention_mechanism['Type'] == 'single':
                attention_type_detail = self.attention_mechanism['TypeDetail']
                self.build_attention_layer(attention_type_detail)
            elif self.attention_mechanism['Type'] == 'share':
                self.share_attention_layer = nn.Sequential(
                    nn.Linear(in_features=self.hidden_size * 2, out_features=self.hidden_size),
                    nn.Tanh(),
                    nn.Dropout(self.drop_prob),
                    nn.Linear(in_features=self.hidden_size, out_features=1)
                )
            else:
                detail = self.attention_mechanism['TypeDetail']
                self_attention_type_detail = detail.split('-')[0]
                cross_attention_type_detail = detail.split('-')[1]

                self.build_attention_layer(self_attention_type_detail)
                self.build_attention_layer(cross_attention_type_detail)


    def build_attention_layer(self, attention_type_detail):

        if attention_type_detail.find('dot') == -1:
            quote_attention_layer = nn.Sequential(
                nn.Linear(in_features=self.hidden_size * 2, out_features=self.hidden_size),
                nn.Tanh(),
                nn.Dropout(self.drop_prob),
                nn.Linear(in_features=self.hidden_size, out_features=1)
            )
            response_attention_layer = nn.Sequential(
                nn.Linear(in_features=self.hidden_size * 2, out_features=self.hidden_size),
                nn.Tanh(),
                nn.Dropout(self.drop_prob),
                nn.Linear(in_features=self.hidden_size, out_features=1)
            )
            if attention_type_detail == 'self':
                setattr(self, 'q2q_attention_layer', quote_attention_layer)
                setattr(self, 'r2r_attention_layer', response_attention_layer)
            else:
                setattr(self, 'r2q_attention_layer', quote_attention_layer)
                setattr(self, 'q2r_attention_layer', response_attention_layer)
        else:
            quote_attention_layer = nn.Sequential(
                nn.Linear(in_features=64, out_features=1),
                nn.Tanh(),
                nn.Dropout(self.drop_prob),
            )
            response_attention_layer = nn.Sequential(
                nn.Linear(in_features=64, out_features=1),
                nn.Tanh(),
                nn.Dropout(self.drop_prob),
            )
            if attention_type_detail == 'self_dot':
                setattr(self, 'self_dot_q2q_attention_layer', quote_attention_layer)
                setattr(self, 'self_dot_r2r_attention_layer', response_attention_layer)
            else:
                setattr(self, '%s_r2q_attention_layer' % attention_type_detail, quote_attention_layer)
                setattr(self, '%s_q2r_attention_layer' % attention_type_detail, response_attention_layer)

    def forward(self, X_q_inputs, X_r_inputs, q_init_state, r_init_state):
        quote_outputs, response_outputs = self.bilstm.forward(X_q_inputs, X_r_inputs, q_init_state, r_init_state)

        if self.attention_mechanism['Type'] != 'None':
            if self.attention_mechanism['Type'] == 'share':
                _quote_outputs = self.attention_layer(
                    layer=self.share_attention_layer,
                    bilstm_outputs=quote_outputs,
                    attention_source=response_outputs
                )
                _response_outputs = self.attention_layer(
                    layer=self.share_attention_layer,
                    bilstm_outputs=response_outputs,
                    attention_source=quote_outputs
                )
                quote_outputs = _quote_outputs
                response_outputs = _response_outputs
            elif self.attention_mechanism['Type'] == 'single':
                attention_type_detail = self.attention_mechanism['TypeDetail']
                _quote_outputs, _response_outputs = self.get_attention_result(
                    attention_type_detail=attention_type_detail,
                    quote_outputs=quote_outputs,
                    response_outputs=response_outputs
                )
                quote_outputs = _quote_outputs
                response_outputs = _response_outputs
            else:
                detail = self.attention_mechanism['TypeDetail']
                self_attention_type_detail = detail.split('-')[0]
                cross_attention_type_detail = detail.split('-')[1]

                self_quote_outputs, self_response_outputs = self.get_attention_result(
                    attention_type_detail=self_attention_type_detail,
                    quote_outputs=quote_outputs,
                    response_outputs=response_outputs
                )

                cross_quote_outputs, cross_response_outputs = self.get_attention_result(
                    attention_type_detail=cross_attention_type_detail,
                    quote_outputs=quote_outputs,
                    response_outputs=response_outputs
                )

                if self.attention_mechanism['Type'] == 'sum':
                    quote_outputs = self_quote_outputs + cross_quote_outputs
                    response_outputs = self_response_outputs + cross_response_outputs
                elif self.attention_mechanism['Type'] == 'concat':
                    quote_outputs = torch.cat((self_quote_outputs, cross_quote_outputs), dim=1)
                    response_outputs = torch.cat((self_response_outputs, cross_response_outputs), dim=1)
                else:
                    quote_outputs = torch.cat(
                        (self_quote_outputs, cross_quote_outputs, self_quote_outputs + cross_quote_outputs), dim=1)
                    response_outputs = torch.cat(
                        (self_response_outputs, cross_response_outputs, self_response_outputs + cross_response_outputs), dim=1)
        else:
            quote_outputs = torch.sum(quote_outputs, dim=1)
            response_outputs = torch.sum(response_outputs, dim=1)
        return quote_outputs, response_outputs

    def attention_layer(self, layer, bilstm_outputs, attention_source):

        attention_inputs = attention_source.contiguous().view(-1, self.hidden_size * 2)  # [32, 64, 256] -> [32*64, 256]
        attention_logits = layer(attention_inputs)  # [32*64, 256] -> [32*64, 128] ->[32*64, 1]
        attention_logits = attention_logits.view(-1, attention_source.size()[1], 1)  # [32*64, 1] -> [32, 64, 1]
        attention_logits = attention_logits.transpose(1, 2).contiguous()  # [32, 64, 1] -> [32, 1, 64]

        attention_signals = F.softmax(attention_logits, dim=2) # [32, 1, 64]

        return torch.bmm(attention_signals, bilstm_outputs).view(-1, bilstm_outputs.size()[2]) # [32, 1, 64] * [32, 64, 256] -> [32, 1, 256] -> [32, 256]

    def self_dot_attention_layer(self, layer, bilstm_outputs):

        size = bilstm_outputs.size()
        bilstm_outputs_T = bilstm_outputs.transpose(1, 2).contiguous()
        attention_logits = F.tanh(torch.bmm(bilstm_outputs, bilstm_outputs_T))

        attention_logits = layer(attention_logits.view(-1, size[1]))
        attention_signals = F.softmax(attention_logits.view(-1, size[1], 1).transpose(1, 2), dim=2)

        return torch.bmm(attention_signals, bilstm_outputs).view(-1, size[2])

    def cross_dot_attention_layer(self, attention_type_detail, q2r_layer, r2q_layer, quote_outputs, response_outputs):

        size = quote_outputs.size()
        response_outputs_T = response_outputs.transpose(1, 2).contiguous()
        attention_logits = F.tanh(torch.bmm(quote_outputs, response_outputs_T))

        # version 1
        if attention_type_detail == 'cross_dot_v1':
            quote_attention_logits = r2q_layer(attention_logits.view(-1, size[1]))
            response2quote_attention = F.softmax(quote_attention_logits.view(-1, size[1], 1).transpose(1, 2), dim=2)

            response_attention_logits = q2r_layer(attention_logits.transpose(1, 2).contiguous().view(-1, size[1]))
            quote2response_attention = F.softmax(response_attention_logits.view(-1, size[1], 1).transpose(1, 2), dim=2)

        # version 2
        else:
            quote_attention_logits = q2r_layer(attention_logits.view(-1, size[1]))
            quote2response_attention = F.softmax(quote_attention_logits.view(-1, size[1], 1).transpose(1, 2), dim=2)

            response_attention_logits = r2q_layer(attention_logits.transpose(1, 2).contiguous().view(-1, size[1]))
            response2quote_attention = F.softmax(response_attention_logits.view(-1, size[1], 1).transpose(1, 2), dim=2)

        return torch.bmm(response2quote_attention, quote_outputs).view(-1, size[2]), torch.bmm(quote2response_attention, response_outputs).view(-1, size[2])

    def get_attention_result(self, attention_type_detail, quote_outputs, response_outputs):

        if attention_type_detail == 'self':
            _quote_outputs = self.attention_layer(
                self.q2q_attention_layer,
                bilstm_outputs=quote_outputs,
                attention_source=quote_outputs
            )
            _response_outputs = self.attention_layer(
                self.r2r_attention_layer,
                bilstm_outputs=response_outputs,
                attention_source=response_outputs
            )
        elif attention_type_detail == 'cross':
            _quote_outputs = self.attention_layer(
                self.r2q_attention_layer,
                bilstm_outputs=quote_outputs,
                attention_source=response_outputs
            )
            _response_outputs = self.attention_layer(
                self.q2r_attention_layer,
                bilstm_outputs=response_outputs,
                attention_source=quote_outputs
            )
        elif attention_type_detail == 'self_dot':
            _quote_outputs = self.self_dot_attention_layer(
                layer=self.self_dot_q2q_attention_layer,
                bilstm_outputs=quote_outputs
            )
            _response_outputs = self.self_dot_attention_layer(
                layer=self.self_dot_r2r_attention_layer,
                bilstm_outputs=response_outputs
            )
        else:
            _quote_outputs, _response_outputs = self.cross_dot_attention_layer(
                attention_type_detail=attention_type_detail,
                q2r_layer=getattr(self, '%s_q2r_attention_layer' % attention_type_detail),
                r2q_layer=getattr(self, '%s_r2q_attention_layer' % attention_type_detail),
                quote_outputs=quote_outputs,
                response_outputs=response_outputs
            )
        return _quote_outputs, _response_outputs

    def init_hidden(self, batch_size):
        return self.bilstm.init_hidden(batch_size)

class Classifier(nn.Module):
    def __init__(self, config):
        super(Classifier, self).__init__()
        self.model = AttentionModel(config)
        self.drop_prob = config['drop_prob']
        self.hidden_size = config['hidden_size']
        self.multi_task = config['multi_task']
        self.attention_mechanism = config['attention_mechanism']
        self.out_actfuc = config['out_actfuc']
        self.class_num = config['class_num']

        if self.multi_task:
            if self.attention_mechanism['Type'] == 'concat':
                self.pair_out = nn.Sequential(
                    nn.Linear(in_features=self.hidden_size * 2 * 2 * 2, out_features=self.hidden_size * 2),
                    self.out_actfuc,
                    nn.Dropout(self.drop_prob),
                    nn.Linear(in_features=self.hidden_size * 2, out_features=self.class_num)
                )
            elif self.attention_mechanism['Type'] == 'hybrid':
                self.pair_out = nn.Sequential(
                    nn.Linear(in_features=self.hidden_size * 2 * 3 * 2, out_features=self.hidden_size * 2),
                    self.out_actfuc,
                    nn.Dropout(self.drop_prob),
                    nn.Linear(in_features=self.hidden_size * 2, out_features=self.class_num)
                )
            else:
                self.pair_out = nn.Sequential(
                    nn.Linear(in_features=self.hidden_size * 2 * 2, out_features=self.hidden_size * 2),
                    self.out_actfuc,
                    nn.Dropout(self.drop_prob),
                    nn.Linear(in_features=self.hidden_size * 2, out_features=self.class_num)
                )

        self.single_out = nn.Sequential(
            nn.Linear(in_features=self.hidden_size * 2 * 3, out_features=self.hidden_size * 2),
            self.out_actfuc,
            nn.Dropout(self.drop_prob),
            nn.Linear(in_features=self.hidden_size * 2, out_features=self.class_num)
            )

        # self.init_weights()

    def init_weights(self, init_range=0.1):
        # self.out.weight.data.uniform_(-init_range, init_range)
        # self.out.bias.data.fill_(init_range)
        for name, param in self.named_parameters():
            if name.find('weight') != -1:
                if name.find('embedding') == -1:
                    nn.init.xavier_normal(param.data)
            elif name.find('bias') != -1:
                param.data.uniform_(-init_range, init_range)
            else:
                continue

    def forward(self, X_q_inputs, X_r_inputs, q_init_state, r_init_state):

        quote_outputs, response_outputs = self.model.forward(X_q_inputs, X_r_inputs, q_init_state, r_init_state)
        concat_outputs = torch.cat((quote_outputs, response_outputs), dim=1)

        quote_output = self.single_out(quote_outputs)
        response_output = self.single_out(response_outputs)
        if self.multi_task:
            pair_output = self.pair_out(concat_outputs)
            return quote_output, response_output, pair_output
        else:
            return quote_output, response_output

    def init_hidden(self, batch_size):
        return self.model.init_hidden(batch_size)

def train_batch(model, data_train):
    model.train()

    X_q_batch, X_r_batch, y_batch = data_train.next_batch(tr_batch_size)

    X_q_batch = Variable(torch.LongTensor(X_q_batch)).cuda() if USE_GPU else Variable(torch.LongTensor(X_q_batch))
    X_r_batch = Variable(torch.LongTensor(X_r_batch)).cuda() if USE_GPU else Variable(torch.LongTensor(X_r_batch))

    y_q_batch = y_batch[:, 0]
    y_r_batch = y_batch[:, 1]

    y_q_batch = Variable(torch.LongTensor(y_q_batch)).cuda() if USE_GPU else Variable(torch.LongTensor(y_q_batch))
    y_r_batch = Variable(torch.LongTensor(y_r_batch)).cuda() if USE_GPU else Variable(torch.LongTensor(y_r_batch))



    q_init_state = model.init_hidden(y_batch.shape[0])
    r_init_state = model.init_hidden(y_batch.shape[0])


    if model.multi_task:
        y_pair_batch = y_batch[:, 2]
        y_pair_batch = Variable(torch.LongTensor(y_pair_batch)).cuda() if USE_GPU else Variable(torch.LongTensor(y_pair_batch))
        quote_output, response_output, pair_output = model(X_q_batch, X_r_batch, q_init_state, r_init_state)
        cost = loss_func(quote_output, y_q_batch) + loss_func(response_output, y_r_batch) + loss_func(pair_output, y_pair_batch)
        y_pair_pred = torch.max(pair_output, 1)[1]
        pair_acc = sum(y_pair_pred.data.cpu() == y_pair_batch.data.cpu()) / float(y_pair_batch.size(0)) if USE_GPU else sum(y_pair_pred.data == y_pair_batch.data) / float(y_pair_batch.size(0))

    else:
        quote_output, response_output = model(X_q_batch, X_r_batch, q_init_state, r_init_state)
        cost = loss_func(quote_output, y_q_batch) + loss_func(response_output, y_r_batch)

    y_q_pred = torch.max(quote_output, 1)[1]
    y_r_pred = torch.max(response_output, 1)[1]

    stance_acc = (sum(y_q_pred.data.cpu() == y_q_batch.data.cpu()) + sum(y_r_pred.data.cpu() == y_r_batch.data.cpu())) / (2 * float(y_batch.shape[0])) if USE_GPU else (sum(y_q_pred.data == y_q_batch.data) + sum(y_r_pred.data == y_r_batch.data)) / (2 * float(y_batch.shape[0]))

    optimizer.zero_grad()
    cost.backward()
    nn.utils.clip_grad_norm(model.parameters(), max_norm=max_grad_norm)
    optimizer.step()

    if model.multi_task:
        valid_stance_acc, valid_pair_acc, valid_cost, con_f1, pro_f1, disagree_f1, agree_f1 = test_epoch(model, data_test)
        average_pair_f1 = (disagree_f1 + agree_f1) / 2
        average_stance_f1 = (con_f1 + pro_f1) / 2
        log = 'BATCH {} Training stance_acc={:.6f}, pair_acc={:.6f}, cost={:.6f} || Test stance_acc={:.6f}, pair_acc={:.6f}, cost={:.6f}, con_f1={:.6f}, pro_f1={:.6f}, average_stance_f1={:.6f}, disagree_f1={:.6f}, agree_f1={:.6f}, average_pair_f1={:.6f}'.format(batch, stance_acc, pair_acc, cost.data[0], valid_stance_acc, valid_pair_acc, valid_cost, con_f1, pro_f1, average_stance_f1, disagree_f1, agree_f1, average_pair_f1)

    else:
        valid_stance_acc, valid_cost, con_f1, pro_f1 = test_epoch(model, data_test)
        average_stance_f1 = (con_f1 + pro_f1) / 2
        log = 'BATCH {} Training stance_acc={:.6f}, cost={:.6f} || Test stance_acc={:.6f}, cost={:.6f}, con_f1={:.6f}, pro_f1={:.6f}, average_stance_f1={:.6f}'.format(batch, stance_acc, cost.data[0], valid_stance_acc, valid_cost, con_f1, pro_f1, average_stance_f1)

    print(log)
    fw.write(log + '\n')
    return average_stance_f1

def test_epoch(model, dataset):
    model.eval()
    data_size = len(dataset.y)
    _batch_size = data_size

    X_q_batch, X_r_batch, y_batch = dataset.next_batch(_batch_size)

    X_q_batch = Variable(torch.LongTensor(X_q_batch), volatile=True).cuda() if USE_GPU else Variable(torch.LongTensor(X_q_batch), volatile=True)
    X_r_batch = Variable(torch.LongTensor(X_r_batch), volatile=True).cuda() if USE_GPU else Variable(torch.LongTensor(X_r_batch), volatile=True)

    y_q_batch = y_batch[:, 0]
    y_r_batch = y_batch[:, 1]

    y_q_batch = Variable(torch.LongTensor(y_q_batch), volatile=True).cuda() if USE_GPU else Variable(torch.LongTensor(y_q_batch), volatile=True)
    y_r_batch = Variable(torch.LongTensor(y_r_batch), volatile=True).cuda() if USE_GPU else Variable(torch.LongTensor(y_r_batch), volatile=True)

    q_init_state = model.init_hidden(y_batch.shape[0])
    r_init_state = model.init_hidden(y_batch.shape[0])

    if model.multi_task:
        y_pair_batch = y_batch[:, 2]
        y_pair_batch = Variable(torch.LongTensor(y_pair_batch), volatile=True).cuda() if USE_GPU else Variable(torch.LongTensor(y_pair_batch), volatile=True)
        quote_output, response_output, pair_output = model(X_q_batch, X_r_batch, q_init_state, r_init_state)
        cost = loss_func(quote_output, y_q_batch) + loss_func(response_output, y_r_batch) + loss_func(pair_output, y_pair_batch)
        y_pair_pred = torch.max(pair_output, 1)[1]
        pair_acc = sum(y_pair_pred.data.cpu() == y_pair_batch.data.cpu()) / float(y_pair_batch.size(0)) if USE_GPU else sum(y_pair_pred.data == y_pair_batch.data) / float(y_pair_batch.size(0))
        _, _, pair_f1, _ = precision_recall_fscore_support(y_true=y_pair_batch.data.cpu().numpy(), y_pred=y_pair_pred.data.cpu().numpy())

    else:
        quote_output, response_output = model(X_q_batch, X_r_batch, q_init_state, r_init_state)
        cost = loss_func(quote_output, y_q_batch) + loss_func(response_output, y_r_batch)

    y_q_pred = torch.max(quote_output, 1)[1]
    y_r_pred = torch.max(response_output, 1)[1]

    stance_acc = (sum(y_q_pred.data.cpu() == y_q_batch.data.cpu()) + sum(y_r_pred.data.cpu() == y_r_batch.data.cpu())) / (2 * float(y_batch.shape[0])) if USE_GPU else (sum(y_q_pred.data == y_q_batch.data) + sum(y_r_pred.data == y_r_batch.data)) / (2 * float(y_batch.shape[0]))

    stance_true = torch.cat((y_q_batch, y_r_batch), dim=0).data.cpu().numpy()
    stance_pred = torch.cat((y_q_pred, y_r_pred), dim=0).data.cpu().numpy()

    _, _, stance_f1, _ = precision_recall_fscore_support(y_true=stance_true, y_pred=stance_pred)

    if model.multi_task:
        return stance_acc, pair_acc, cost.data[0], stance_f1[0], stance_f1[1], pair_f1[0], pair_f1[1]
    else:
        return stance_acc, cost.data[0], stance_f1[0], stance_f1[1]


if __name__ == '__main__':

    log_time = time.strftime('%Y-%m-%d %H:%M:%S')
    USE_GPU = True
    data_type = 'iac_stance'

    for topic in ['evolution', 'abortion', 'gun control', 'gay marriage', 'existence of God']:

        with open('./data/%s_%s_64.pkl' % (data_type, topic), 'rb') as fr:
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


        kf = KFold(n_splits=5)

        # for type in ['None', 'share']: # gpu:0
        # for type in ['self_attention', 'cross_attention']: # gpu:1
        # for type in ['both_sum', 'both_concat']: # gpu:2
        # for setting in ['cross_attention, 0']:
        # for setting in ['None, 1']:
        # for setting in ['both_sum, 3']:
        # for setting in ['both_concat, 0']:
        # for setting in ['both_hybrid, 3']:
        # for setting in ['share, 3']:
        # for setting in ['self_dot, 1']:
        # for setting in ['self_attention, 2']:
        # for setting in ['cross_dot, 3']:
        # for setting in ['hybrid, 0']:
        # for setting in ['sum, self, cross, 0']:

        # for setting in ['single, cross_dot_v1, 0']:
        # for setting in ['hybrid, self, cross, 0']:
        # settings = []
        # gpu_index = 0
        # for type in ['None', 'single', 'sum', 'concat', 'hybrid']:
        #     if type == 'None':
        #         setting = 'None' + ', ' + str(gpu_index)
        #         settings.append(setting)
        #         gpu_index += 1
        #         if gpu_index == 4:
        #             gpu_index = gpu_index % 4
        #
        #     elif type == 'single':
        #         for detail in ['self', 'cross', 'self_dot', 'cross_dot_v1', 'cross_dot_v2']:
        #             setting = type + ', ' + detail + ', ' + str(gpu_index)
        #             settings.append(setting)
        #             gpu_index += 1
        #             if gpu_index == 4:
        #                 gpu_index = gpu_index % 4
        #     else:
        #         for self_detail in ['self', 'self_dot']:
        #             for cross_detail in ['cross', 'cross_dot_v1', 'cross_dot_v2']:
        #                 setting = type + ', ' + self_detail + ', ' + cross_detail + ', ' + str(gpu_index)
        #                 settings.append(setting)
        #                 gpu_index += 1
        #                 if gpu_index == 4:
        #                     gpu_index = gpu_index % 4

        # settings = ['None, 0', 'single, self, 0', 'single, cross, 0', 'single, self_dot, 0', 'single, cross_dot_v1, 0', 'single, cross_dot_v2, 0']
        # settings = ['sum, self, cross, 1', 'sum, self, cross_dot_v1, 1', 'sum, self, cross_dot_v2, 1', 'sum, self_dot, cross, 1', 'sum, self_dot, cross_dot_v1, 1', 'sum, self_dot, cross_dot_v2, 1']
        # settings = ['concat, self, cross, 2', 'concat, self, cross_dot_v1, 2', 'concat, self, cross_dot_v2, 2', 'concat, self_dot, cross, 2', 'concat, self_dot, cross_dot_v1, 2', 'concat, self_dot, cross_dot_v2, 2']
        # settings = ['hybrid, self, cross, 3', 'hybrid, self, cross_dot_v1, 3', 'hybrid, self, cross_dot_v2, 3', 'hybrid, self_dot, cross, 3', 'hybrid, self_dot, cross_dot_v1, 3', 'hybrid, self_dot, cross_dot_v2, 3']

        # settings = ['None, 0', 'single, self_dot, 0']
        # settings = ['single, cross_dot_v2, 1', 'sum, self_dot, cross_dot_v2, 1']
        # settings = ['concat, self_dot, cross_dot_v2, 2', 'hybrid, self_dot, cross_dot_v2, 2']
        settings = ['hybrid, self_dot, cross_dot_v2, 1']

        for setting in settings:

            print(setting)
            type = setting.split(',')[0]
            gpu = setting.split(',')[-1].strip()

            if type == 'None':
                assert len(setting.split(',')) == 2
                type_detail = None
            elif type == 'share':
                assert len(setting.split(',')) == 3
                type_detail = setting.split(',')[1].strip()
            elif type == 'single':
                assert len(setting.split(',')) == 3
                type_detail = setting.split(',')[1].strip()
            else:
                assert len(setting.split(',')) == 4
                self_attention_type_detail = setting.split(',')[1].strip()
                cross_attention_type_detail = setting.split(',')[2].strip()
                type_detail = self_attention_type_detail + '-' + cross_attention_type_detail

            attention_mechanism_config = {'Type': type, 'TypeDetail': type_detail, 'ActFunc': F.tanh}
            config = {
                'word2id': word2id, 'pretrain_emb': True, 'class_num': 2,
                'embedding_size': 300, 'hidden_size': 128, 'layer_num': 1,
                'out_actfuc': nn.Tanh(), 'drop_prob': 0.5, 'lstm_drop_prob': 0.5,
                'multi_task': True, 'attention_mechanism': attention_mechanism_config}

            KFold_BEST = []
            for i, (train_index, test_index) in enumerate(kf.split(X)):
                with open('./result_log/%s_%s_%s_%s_%s.txt' % (data_type, topic, type, type_detail, i), 'w') as fw:
                    if USE_GPU:
                        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
                        print('Using GPU: {}...'.format(os.environ['CUDA_VISIBLE_DEVICES']))
                        fw.write('Using GPU: {}...\n'.format(os.environ['CUDA_VISIBLE_DEVICES']))

                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]

                    print('X_train.shape={}, y_train.shape={};\nX_test.shape={}, y_test.shape={}'.format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))
                    fw.write('X_train.shape={}, y_train.shape={};\nX_test.shape={}, y_test.shape={}\n'.format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))

                    print('Creating the data generator ...')
                    fw.write('Creating the data generator ...\n')

                    data_train = stance_data_helper.BatchGenerator(X_train, y_train, shuffle=True)
                    data_test = stance_data_helper.BatchGenerator(X_test, y_test, shuffle=False)

                    model = Classifier(config)

                    max_grad_norm = 0.5

                    for name, para in model.named_parameters():
                        if para.requires_grad == True:
                            print(name)
                            fw.write(name + '\n')
                    if USE_GPU:
                        model.cuda()

                    lr = 1e-3
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
                    loss_func = nn.CrossEntropyLoss()

                    tr_batch_size = 32
                    max_epoch = 30
                    max_max_epoch = 100
                    display_num = 5

                    best_result = None
                    best_batch = None
                    EARLY_STOP = 10 * len(X_train) // tr_batch_size
                    stop_counter = 0

                    # Iteration Batch by Batch
                    for batch in range(1, 50000 + 1):
                        current_result = train_batch(model, data_train)

                        if not best_result or current_result > best_result:
                            best_result = current_result
                            best_batch = batch
                            # torch.save(model.state_dict(), './data/model_params.pkl')
                            stop_counter = 0
                        else:
                            if EARLY_STOP != 0:
                                stop_counter += 1
                        if stop_counter == EARLY_STOP:
                            print('BEST_BATCH={}, BEST_RESULT={:.6f}'.format(best_batch, best_result))
                            fw.write('BEST_BATCH={}, BEST_RESULT={:.6f}\n'.format(best_batch, best_result))
                            KFold_BEST.append((i+1, best_batch, best_result))
                            break

            with open('./result_log/%s_%s_%s_result_summary.txt' % (data_type, topic, config['multi_task']), 'a') as fw:
                print('{} {} {}'.format(type, type_detail, log_time))
                fw.write('{} {}\n'.format(type, type_detail))
                for item in KFold_BEST:
                    print(item)
                    fw.write('{}\n'.format(item))
                print('Average={:.6f}'.format(sum([f1[2] for f1 in KFold_BEST]) / len(KFold_BEST)))
                fw.write('Average={:.6f}\n'.format(sum([f1[2] for f1 in KFold_BEST]) / len(KFold_BEST)))