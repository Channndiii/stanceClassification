import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import data_helper_2018
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import KFold
import os
import pandas as pd
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

        self.drop_prob = config['drop_prob']
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
            dropout=config['lstm_drop_prob'],
            bidirectional=True,
            batch_first=True
        )

        self.response_bilstm = nn.LSTM(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.layer_num,
            dropout=config['lstm_drop_prob'],
            bidirectional=True,
            batch_first=True
        )

    def forward(self, X_q_inputs, X_r_inputs, q_init_state, r_init_state):

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

class AttentionModel(nn.Module):
    def __init__(self, config):
        super(AttentionModel, self).__init__()
        self.bilstm = TextBiLSTM(config)
        self.drop_prob = config['drop_prob']
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
        q_attention = None
        r_attention = None
        self_q_logits = None
        self_r_logits = None
        cross_qr_logits = None
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
                _quote_outputs, _response_outputs, q_attention, r_attention, q_logits, r_logits = self.get_attention_result(
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

                self_quote_outputs, self_response_outputs, self_q_attention, self_r_attention, self_q_logits, self_r_logits = self.get_attention_result(
                    attention_type_detail=self_attention_type_detail,
                    quote_outputs=quote_outputs,
                    response_outputs=response_outputs
                )

                cross_quote_outputs, cross_response_outputs, cross_q_attention, cross_r_attention, cross_qr_logits, _ = self.get_attention_result(
                    attention_type_detail=cross_attention_type_detail,
                    quote_outputs=quote_outputs,
                    response_outputs=response_outputs
                )

                q_attention = (self_q_attention, cross_q_attention)
                r_attention = (self_r_attention, cross_r_attention)

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
        if self.do_BN:
            quote_outputs = self.quote_output_BN(quote_outputs)
            response_outputs = self.response_output_BN(response_outputs)
        return quote_outputs, response_outputs, q_attention, r_attention, self_q_logits, self_r_logits, cross_qr_logits

    def attention_layer(self, layer, bilstm_outputs, attention_source):

        attention_inputs = attention_source.contiguous().view(-1, self.hidden_size * 2)  # [32, 64, 256] -> [32*64, 256]
        attention_logits = layer(attention_inputs)  # [32*64, 256] -> [32*64, 128] ->[32*64, 1]
        attention_logits = attention_logits.view(-1, attention_source.size()[1], 1)  # [32*64, 1] -> [32, 64, 1]
        attention_logits = attention_logits.transpose(1, 2).contiguous()  # [32, 64, 1] -> [32, 1, 64]

        attention_signals = F.softmax(attention_logits, dim=2) # [32, 1, 64]

        return torch.bmm(attention_signals, bilstm_outputs).view(-1, bilstm_outputs.size()[2]), attention_signals # [32, 1, 64] * [32, 64, 256] -> [32, 1, 256] -> [32, 256]

    def self_dot_attention_layer(self, layer, bilstm_outputs):

        size = bilstm_outputs.size()
        bilstm_outputs_T = bilstm_outputs.transpose(1, 2).contiguous()
        attention_logits = F.tanh(torch.bmm(bilstm_outputs, bilstm_outputs_T))
        attention_logits_visual = attention_logits.view(-1, size[1])

        attention_logits = layer(attention_logits.view(-1, size[1]))
        attention_signals = F.softmax(attention_logits.view(-1, size[1], 1).transpose(1, 2), dim=2)

        return torch.bmm(attention_signals, bilstm_outputs).view(-1, size[2]), attention_signals, attention_logits_visual

    def cross_dot_attention_layer(self, attention_type_detail, q2r_layer, r2q_layer, quote_outputs, response_outputs):

        size = quote_outputs.size()
        response_outputs_T = response_outputs.transpose(1, 2).contiguous()
        attention_logits = F.tanh(torch.bmm(quote_outputs, response_outputs_T))
        attention_logits_visual = attention_logits.view(-1, size[1])

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

        return torch.bmm(response2quote_attention, quote_outputs).view(-1, size[2]), torch.bmm(quote2response_attention, response_outputs).view(-1, size[2]), response2quote_attention, quote2response_attention, attention_logits_visual, attention_logits_visual

    def get_attention_result(self, attention_type_detail, quote_outputs, response_outputs):

        q_logits = None
        r_logits = None

        if attention_type_detail == 'self':
            _quote_outputs, q_attention = self.attention_layer(
                self.q2q_attention_layer,
                bilstm_outputs=quote_outputs,
                attention_source=quote_outputs
            )
            _response_outputs, r_attention = self.attention_layer(
                self.r2r_attention_layer,
                bilstm_outputs=response_outputs,
                attention_source=response_outputs
            )
        elif attention_type_detail == 'cross':
            _quote_outputs, q_attention = self.attention_layer(
                self.r2q_attention_layer,
                bilstm_outputs=quote_outputs,
                attention_source=response_outputs
            )
            _response_outputs, r_attention = self.attention_layer(
                self.q2r_attention_layer,
                bilstm_outputs=response_outputs,
                attention_source=quote_outputs
            )
        elif attention_type_detail == 'self_dot':
            _quote_outputs, q_attention, q_logits = self.self_dot_attention_layer(
                layer=self.self_dot_q2q_attention_layer,
                bilstm_outputs=quote_outputs
            )
            _response_outputs, r_attention, r_logits = self.self_dot_attention_layer(
                layer=self.self_dot_r2r_attention_layer,
                bilstm_outputs=response_outputs
            )
        else:
            _quote_outputs, _response_outputs, q_attention, r_attention, q_logits, r_logits = self.cross_dot_attention_layer(
                attention_type_detail=attention_type_detail,
                q2r_layer=getattr(self, '%s_q2r_attention_layer' % attention_type_detail),
                r2q_layer=getattr(self, '%s_r2q_attention_layer' % attention_type_detail),
                quote_outputs=quote_outputs,
                response_outputs=response_outputs
            )
        return _quote_outputs, _response_outputs, q_attention, r_attention, q_logits, r_logits

    def init_hidden(self, batch_size):
        return self.bilstm.init_hidden(batch_size)

class Classifier(nn.Module):
    def __init__(self, config):
        super(Classifier, self).__init__()
        self.model = AttentionModel(config)
        self.drop_prob = config['drop_prob']
        self.hidden_size = config['hidden_size']
        self.attention_mechanism = config['attention_mechanism']
        self.do_BN = config['do_BN']
        self.out_actfuc = config['out_actfuc']
        self.class_num = config['class_num']

        # if self.attention_mechanism['Type'] == 'both_concat':
        #     self.out = nn.Sequential(
        #         nn.Linear(in_features=self.hidden_size * 2 * 2 * 2, out_features=self.hidden_size * 2),
        #         self.out_actfuc,
        #         nn.Dropout(self.drop_prob),
        #         nn.Linear(in_features=self.hidden_size * 2, out_features=self.class_num)
        #     )
        # elif self.attention_mechanism['Type'] == 'both_hybrid':
        #     self.out = nn.Sequential(
        #         nn.Linear(in_features=self.hidden_size * 2 * 3 * 2, out_features=self.hidden_size * 2),
        #         self.out_actfuc,
        #         nn.Dropout(self.drop_prob),
        #         nn.Linear(in_features=self.hidden_size * 2, out_features=self.class_num)
        #     )
        # elif self.attention_mechanism['Type'] == 'hybrid':
        #     self.out = nn.Sequential(
        #         nn.Linear(in_features=self.hidden_size * 2 * 3 * 2, out_features=self.hidden_size * 2),
        #         # nn.Linear(in_features=self.hidden_size * 2 * 4 * 2, out_features=self.hidden_size * 2),
        #         # nn.Linear(in_features=self.hidden_size * 2 * 2 * 4, out_features=self.hidden_size * 2), # NLI
        #         self.out_actfuc,
        #         nn.Dropout(self.drop_prob),
        #         nn.Linear(in_features=self.hidden_size * 2, out_features=self.class_num)
        #     )
        # else:
        #     self.out = nn.Sequential(
        #         nn.Linear(in_features=self.hidden_size * 2 * 2, out_features=self.hidden_size * 2),
        #         self.out_actfuc,
        #         nn.Dropout(self.drop_prob),
        #         nn.Linear(in_features=self.hidden_size * 2, out_features=self.class_num)
        #     )

        if self.attention_mechanism['Type'] == 'concat':
            self.out = nn.Sequential(
                nn.Linear(in_features=self.hidden_size * 2 * 2 * 2, out_features=self.hidden_size * 2),
                self.out_actfuc,
                nn.Dropout(self.drop_prob),
                nn.Linear(in_features=self.hidden_size * 2, out_features=self.class_num)
            )
        elif self.attention_mechanism['Type'] == 'hybrid':
            self.out = nn.Sequential(
                nn.Linear(in_features=self.hidden_size * 2 * 3 * 2, out_features=self.hidden_size * 2),
                self.out_actfuc,
                nn.Dropout(self.drop_prob),
                nn.Linear(in_features=self.hidden_size * 2, out_features=self.class_num)
            )
        else:
            self.out = nn.Sequential(
                nn.Linear(in_features=self.hidden_size * 2 * 2, out_features=self.hidden_size * 2),
                self.out_actfuc,
                nn.Dropout(self.drop_prob),
                nn.Linear(in_features=self.hidden_size * 2, out_features=self.class_num)
            )

        if self.do_BN:
            # self.concat_output_BN = nn.BatchNorm1d(num_features=self.hidden_size * 2 * 2)
            if self.attention_mechanism['Type'] != 'both_concat':
                self.concat_output_BN = nn.BatchNorm1d(num_features=self.hidden_size * 2 * 2)
            else:
                self.concat_output_BN = nn.BatchNorm1d(num_features=self.hidden_size * 2 * 2 * 2)
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
        quote_outputs, response_outputs, q_attention, r_attention, self_q_logits, self_r_logits, cross_qr_logits = self.model.forward(X_q_inputs, X_r_inputs, q_init_state, r_init_state)
        concat_outputs = torch.cat((quote_outputs, response_outputs), dim=1)

        # abs_outputs = torch.abs(quote_outputs - response_outputs)
        # multiple_outputs = quote_outputs * response_outputs
        # concat_outputs = torch.cat((quote_outputs, response_outputs, abs_outputs, multiple_outputs), dim=1)

        if self.do_BN:
            concat_outputs = self.concat_output_BN(concat_outputs)

        output = self.out(concat_outputs)
        return output, q_attention, r_attention, self_q_logits, self_r_logits, cross_qr_logits

    def init_hidden(self, batch_size):
        return self.model.init_hidden(batch_size)

# def test_epoch(model, dataset):
#     model.eval()
#     data_size = len(dataset.y)
#     _batch_size = data_size
#     # _batch_size = int(data_size / 2)
#     batch_num = int(data_size / _batch_size)
#     _accs = 0.0
#     _costs = 0.0
#     _pred = []
#     _true = []
#     for i in range(batch_num):
#
#         X_q_batch, X_r_batch, y_batch = dataset.next_batch(_batch_size)
#
#         X_q_batch = Variable(torch.LongTensor(X_q_batch), volatile=True).cuda() if USE_GPU else Variable(torch.LongTensor(X_q_batch), volatile=True)
#         X_r_batch = Variable(torch.LongTensor(X_r_batch), volatile=True).cuda() if USE_GPU else Variable(torch.LongTensor(X_r_batch), volatile=True)
#
#         y_batch = Variable(torch.LongTensor(y_batch), volatile=True).cuda() if USE_GPU else Variable(torch.LongTensor(y_batch), volatile=True)
#
#         q_init_state = model.init_hidden(y_batch.size()[0])
#         r_init_state = model.init_hidden(y_batch.size()[0])
#
#         test_output, q_attention, r_attention = model(X_q_batch, X_r_batch, q_init_state, r_init_state)
#         _cost = loss_func(test_output, y_batch)
#
#         y_pred = torch.max(test_output, 1)[1]
#         _acc = sum(y_pred.data.cpu() == y_batch.data.cpu()) / float(y_batch.size(0)) if USE_GPU else sum(y_pred.data == y_batch.data) / float(y_batch.size(0))
#
#         if i == 0:
#             _pred = y_pred.data.cpu().numpy() if USE_GPU else y_pred.data.numpy()
#             _true = y_batch.data.cpu().numpy() if USE_GPU else y_batch.data.numpy()
#         else:
#             _pred = np.concatenate((_pred, y_pred.data.cpu().numpy())) if USE_GPU else np.concatenate((_pred, y_pred.data.numpy()))
#             _true = np.concatenate((_true, y_batch.data.cpu().numpy())) if USE_GPU else np.concatenate((_true, y_batch.data.numpy()))
#         _accs += _acc
#         _costs += _cost.data[0]
#     mean_acc = _accs / batch_num
#     mean_cost = _costs / batch_num
#     p, r, f1, s = precision_recall_fscore_support(y_true=_true, y_pred=_pred)
#     return mean_acc, mean_cost, f1[0], f1[1]

def test_sample(model, sample):
    model.eval()

    X_q_batch, X_r_batch, y_batch = sample

    X_q_batch = Variable(torch.LongTensor(X_q_batch), volatile=True).cuda() if USE_GPU else Variable(torch.LongTensor(X_q_batch), volatile=True)
    X_r_batch = Variable(torch.LongTensor(X_r_batch), volatile=True).cuda() if USE_GPU else Variable(torch.LongTensor(X_r_batch), volatile=True)

    y_batch = Variable(torch.LongTensor(y_batch), volatile=True).cuda() if USE_GPU else Variable(torch.LongTensor(y_batch), volatile=True)

    q_init_state = model.init_hidden(y_batch.size()[0])
    r_init_state = model.init_hidden(y_batch.size()[0])

    test_output, q_attention, r_attention, self_q_logits, self_r_logits, cross_qr_logits = model(X_q_batch, X_r_batch, q_init_state, r_init_state)

    y_pred = torch.max(test_output, 1)[1]
    acc = sum(y_pred.data.cpu() == y_batch.data.cpu()) / float(y_batch.size(0)) if USE_GPU else sum(y_pred.data == y_batch.data) / float(y_batch.size(0))

    if acc == 1.0:
        return q_attention, r_attention, self_q_logits, self_r_logits, cross_qr_logits
    else:
        return None

if __name__ == '__main__':

    USE_GPU = True
    data_name = 'iac'
    with open('./data/data_disagree_agree_None_64.pkl', 'rb') as fr:
    # data_name = 'debatepedia'
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

    type = 'hybrid'
    type_detail = 'self_dot-cross_dot_v2'
    gpu = '0'
    kf = KFold(n_splits=5)

    attention_mechanism_config = {'Type': type, 'TypeDetail': type_detail, 'ActFunc': F.tanh}
    config = {
        'word2id': word2id, 'pretrain_emb': False, 'class_num': 2,
        'embedding_size': 300, 'hidden_size': 128, 'layer_num': 2,
        'out_actfuc': nn.Tanh(), 'drop_prob': 0.5, 'lstm_drop_prob': 0.5,
        'do_BN': False, 'attention_mechanism': attention_mechanism_config}

    for i, (train_index, test_index) in enumerate(kf.split(X)):
        if i != 1:
            continue
        if USE_GPU:
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu
            print('Using GPU: {}...'.format(os.environ['CUDA_VISIBLE_DEVICES']))

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        data_train = data_helper_2018.BatchGenerator(X_train, y_train, shuffle=False)
        data_test = data_helper_2018.BatchGenerator(X_test, y_test, shuffle=False)

        model = Classifier(config)
        model.load_state_dict(torch.load('./data/model_params.pkl'))

        max_grad_norm = 0.5

        if USE_GPU:
            model.cuda()

        data = []

        # for i in range(len(X_test)):
            # sample = data_test.next_batch(1)
        for i in range(len(X_train)):
            sample = data_train.next_batch(1)

            X_q, X_r, y = sample

            X_q = np.asarray([i for i in X_q[0] if i != 0])
            X_r = np.asarray([i for i in X_r[0] if i != 0])
            if len(X_q) > 16 or len(X_r) > 16:
                continue

            # if len(np.asarray([i for i in X_q[0] if i != 0])) > 12 or len(np.asarray([i for i in X_r[0] if i != 0])) > 12:
            #     continue
            # if len(X_q) == 64 and len(X_r) == 64:
            result = test_sample(model, sample)
            if result != None:

                (self_q_attention, cross_q_attention), (self_r_attention, cross_r_attention), self_q_logits, self_r_logits, cross_qr_logits = result

                self_q_attention = self_q_attention.data.cpu().numpy().reshape(64)[:len(X_q)]
                cross_q_attention = cross_q_attention.data.cpu().numpy().reshape(64)[:len(X_q)]
                q_attention = [self_q_attention, cross_q_attention]

                self_r_attention = self_r_attention.data.cpu().numpy().reshape(64)[:len(X_r)]
                cross_r_attention = cross_r_attention.data.cpu().numpy().reshape(64)[:len(X_r)]
                r_attention = [self_r_attention, cross_r_attention]

                self_q_logits = self_q_logits.data.cpu().numpy()[:len(X_q), :len(X_q)]

                self_r_logits = self_r_logits.data.cpu().numpy()[:len(X_r), :len(X_r)]

                cross_qr_logits = cross_qr_logits.data.cpu().numpy()[:len(X_q), :len(X_r)]

                q_text = list(id2word[X_q])
                r_text = list(id2word[X_r])
                # self_q_attention = self_q_attention.data.cpu().numpy().reshape(64)
                # cross_q_attention = cross_q_attention.data.cpu().numpy().reshape(64)
                # self_r_attention = self_r_attention.data.cpu().numpy().reshape(64)
                # cross_r_attention = cross_r_attention.data.cpu().numpy().reshape(64)
                # self_q_logits = self_q_logits.data.cpu().numpy()
                # self_r_logits = self_r_logits.data.cpu().numpy()
                # cross_qr_logits = cross_qr_logits.data.cpu().numpy()
                # q_text = list(id2word[X_q[0]])
                # r_text = list(id2word[X_r[0]])

                label = 'disagree' if id2label[y[0]] == '0' else 'agree'

                data.append([q_text, r_text, label, q_attention, r_attention, self_q_logits, self_r_logits, cross_qr_logits])

        data_df = pd.DataFrame(data, columns=['q_text', 'r_text', 'label', 'q_attention', 'r_attention', 'self_q_logits', 'self_r_logits', 'cross_qr_logits'])
        with open('./data/iac_attention.pkl', 'wb') as fw:
            pickle.dump(data_df, fw)


