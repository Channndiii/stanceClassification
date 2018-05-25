import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import data_helper_2018
import numpy as np
import time
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
import os
import pickle

torch.manual_seed(12)

class NLIBiLSTM(nn.Module):
    def __init__(self, config):
        super(NLIBiLSTM, self).__init__()
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

        self.embedding.weight.requires_grad = False

        if self.do_BN:
            self.quote_input_BN = nn.BatchNorm1d(num_features=self.embedding_size)
            self.response_input_BN = nn.BatchNorm1d(num_features=self.embedding_size)

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

        if self.do_BN:
            quote_inputs = quote_inputs.transpose(1, 2).contiguous()
            quote_inputs = self.quote_input_BN(quote_inputs)
            quote_inputs = quote_inputs.transpose(1, 2)
            response_inputs = response_inputs.transpose(1, 2).contiguous()
            response_inputs = self.response_input_BN(response_inputs)
            response_inputs = response_inputs.transpose(1, 2)

        quote_outputs, _ = self.bilstm(quote_inputs, q_init_state)
        response_outputs, _ = self.bilstm(response_inputs, r_init_state)

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
        self.bilstm = NLIBiLSTM(config)
        self.drop_prob = config['drop_prob']
        self.hidden_size = config['hidden_size']
        self.do_BN = config['do_BN']

        if self.do_BN:
            if self.attention_mechanism['Type'] != 'both_concat':
                self.quote_output_BN = nn.BatchNorm1d(num_features=self.hidden_size * 2)
                self.response_output_BN = nn.BatchNorm1d(num_features=self.hidden_size * 2)
            else:
                self.quote_output_BN = nn.BatchNorm1d(num_features=self.hidden_size * 2 * 2)
                self.response_output_BN = nn.BatchNorm1d(num_features=self.hidden_size * 2 * 2)

        self.quote_attention_Wy = nn.Linear(in_features=self.hidden_size * 2, out_features=self.hidden_size, bias=False)
        self.quote_attention_Wh = nn.Linear(in_features=self.hidden_size * 2, out_features=self.hidden_size, bias=False)
        self.quote_attention_Wa = nn.Linear(in_features=self.hidden_size, out_features=1, bias=False)

        self.response_attention_Wy = nn.Linear(in_features=self.hidden_size * 2, out_features=self.hidden_size, bias=False)
        self.response_attention_Wh = nn.Linear(in_features=self.hidden_size * 2, out_features=self.hidden_size, bias=False)
        self.response_attention_Wa = nn.Linear(in_features=self.hidden_size, out_features=1, bias=False)

    def forward(self, X_q_inputs, X_r_inputs, q_init_state, r_init_state):
        quote_outputs, response_outputs = self.bilstm.forward(X_q_inputs, X_r_inputs, q_init_state, r_init_state)

        quote_outputs = self.attention_layer(self.quote_attention_Wy, self.quote_attention_Wh, self.quote_attention_Wa, quote_outputs)
        response_outputs = self.attention_layer(self.response_attention_Wy, self.response_attention_Wh, self.response_attention_Wa, response_outputs)

        return quote_outputs, response_outputs

    def attention_layer(self, Wy, Wh, Wa, bilstm_outputs):

        size = bilstm_outputs.size()

        bilstm_outputs_mean = torch.mean(bilstm_outputs, dim=1) # [32, 256]
        attention_inputs = bilstm_outputs.contiguous().view(-1, self.hidden_size * 2)  # [32, 64, 256] -> [32*64, 256]
        M_left = Wy(attention_inputs).view(-1, size[1], self.hidden_size) # [32*64, 256] x [256, 128] -> [32*64, 128] -> [32, 64, 128]

        M_right = Wh(bilstm_outputs_mean) # [32, 256] x [256, 128] -> [32, 128]
        M_right = M_right.repeat(size[1], 1, 1) # [32, 128] -> [64, 32, 128]
        M_right = M_right.transpose(0, 1).contiguous() # [64, 32, 128] -> [32, 64, 128]

        M = F.tanh(M_left + M_right).view(-1, self.hidden_size) # [32, 64, 128] + [32, 64, 128] -> [32, 64, 128] -> [32*64, 128]
        M = Wa(M).view(-1, 1, size[1]) # [32*64, 128] x [128, 1] -> [32*64, 1] -> [32, 1, 64]
        M = F.softmax(M, dim=2) # [32, 1, 64]

        return torch.bmm(M, bilstm_outputs).view(-1, bilstm_outputs.size()[2]) # [32, 1, 64] * [32, 64, 256] -> [32, 1, 256] -> [32, 256]

    def init_hidden(self, batch_size):
        return self.bilstm.init_hidden(batch_size)

class Classifier(nn.Module):
    def __init__(self, config):
        super(Classifier, self).__init__()
        self.model = AttentionModel(config)
        self.drop_prob = config['drop_prob']
        self.hidden_size = config['hidden_size']
        self.do_BN = config['do_BN']
        self.out_actfuc = config['out_actfuc']
        self.class_num = config['class_num']

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
        quote_outputs, response_outputs = self.model.forward(X_q_inputs, X_r_inputs, q_init_state, r_init_state)
        concat_outputs = torch.cat((quote_outputs, response_outputs), dim=1)

        if self.do_BN:
            concat_outputs = self.concat_output_BN(concat_outputs)

        output = self.out(concat_outputs)
        return output

    def init_hidden(self, batch_size):
        return self.model.init_hidden(batch_size)

def train_batch(model, data_train):
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

    optimizer.zero_grad()
    _cost.backward()
    nn.utils.clip_grad_norm(model.parameters(), max_norm=max_grad_norm)
    optimizer.step()

    # valid_acc, valid_cost, _ = test_epoch(model, main_test)
    # print('BATCH {} training {} task acc={:.6f}, cost={:.6f}, test acc={:.6f}, cost={:.6f}'.format(batch, _acc, _cost.data[0], valid_acc, valid_cost))
    # fw.write('BATCH {} training {} task acc={:.6f}, cost={:.6f}, test acc={:.6f}, cost={:.6f}\n'.format(batch, _acc, _cost.data[0], valid_acc, valid_cost))

    valid_acc, valid_cost, disagree_f1, agree_f1 = test_epoch(model, data_test)
    average_f1 = (disagree_f1 + agree_f1) / 2
    print('BATCH {} training acc={:.6f}, cost={:.6f}, test acc={:.6f}, cost={:.6f}, disagree_f1={:.6f}, agree_f1={:.6f}, average_f1={:.6f}'.format(batch, _acc, _cost.data[0], valid_acc, valid_cost, disagree_f1, agree_f1, average_f1))
    fw.write('BATCH {} training acc={:.6f}, cost={:.6f}, test acc={:.6f}, cost={:.6f}, disagree_f1={:.6f}, agree_f1={:.6f}, average_f1={:.6f}\n'.format(batch, _acc, _cost.data[0], valid_acc, valid_cost, disagree_f1, agree_f1, average_f1))
    # return average_f1
    return valid_acc

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
    # cr = classification_report(y_true=_true, y_pred=_pred, target_names=['disagree', 'agree'], digits=4)
    # return mean_acc, mean_cost, cr
    p, r, f1, s = precision_recall_fscore_support(y_true=_true, y_pred=_pred)
    return mean_acc, mean_cost, f1[0], f1[1]


if __name__ == '__main__':

    USE_GPU = True
    # data_name = 'iac'
    # with open('./data/data_disagree_agree_None_64.pkl', 'rb') as fr:
    data_name = 'debatepedia'
    with open('./data/data_debatepedia_None_64.pkl', 'rb') as fr:
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

    settings = ['NLI, 3']

    for setting in settings:

        print(setting)
        type = setting.split(',')[0].strip()
        gpu = setting.split(',')[-1].strip()

        config = {
            'word2id': word2id, 'pretrain_emb': True, 'class_num': 2,
            'embedding_size': 300, 'hidden_size': 128, 'layer_num': 1,
            'out_actfuc': nn.Tanh(), 'drop_prob': 0.5, 'lstm_drop_prob': 0.5,
            'do_BN': False}

        KFold_BEST = []
        for i, (train_index, test_index) in enumerate(kf.split(X)):
            # if i != 1:
            #     continue
            with open('./result_log/%s_BBB_%s_%s.txt' % (data_name, type, i), 'w') as fw:
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

                data_train = data_helper_2018.BatchGenerator(X_train, y_train, shuffle=True)
                data_test = data_helper_2018.BatchGenerator(X_test, y_test, shuffle=False)

                model = Classifier(config)

                max_grad_norm = 0.5

                for name, para in model.named_parameters():
                    if para.requires_grad == True:
                        print(name)
                        fw.write(name + '\n')
                if USE_GPU:
                    model.cuda()

                lr = 1e-3

                trainable_param = list(model.parameters())
                trainable_param = filter(lambda x: x.requires_grad, trainable_param)

                optimizer = torch.optim.Adam(trainable_param, lr=lr, weight_decay=1e-5)
                loss_func = nn.CrossEntropyLoss()

                tr_batch_size = 32
                max_epoch = 30
                max_max_epoch = 100
                display_num = 5

                best_result = None
                best_batch = None
                EARLY_STOP = len(X_train) // tr_batch_size
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

        with open('./result_log/%s_result_summary_NLI.txt' % data_name, 'a') as fw:
            print('{}'.format(type))
            fw.write('{}\n'.format(type))
            for item in KFold_BEST:
                print(item)
                fw.write('{}\n'.format(item))
            print('Average={:.6f}'.format(sum([f1[2] for f1 in KFold_BEST]) / len(KFold_BEST)))
            fw.write('Average={:.6f}\n'.format(sum([f1[2] for f1 in KFold_BEST]) / len(KFold_BEST)))

