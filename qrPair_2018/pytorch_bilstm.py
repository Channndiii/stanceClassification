import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import varied_textLength
import numpy as np
import time
from sklearn.metrics import confusion_matrix
import os

class TextBiLSTM(nn.Module):
    def __init__(self, vocabulary_size, embedding_size, hidden_size, layer_num, class_num, do_BN, attention_mechanism):
        super(TextBiLSTM, self).__init__()

        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.class_num = class_num
        self.do_BN = do_BN
        self.attention_mechanism = attention_mechanism

        self.embedding = nn.Embedding(num_embeddings=self.vocabulary_size, embedding_dim=self.embedding_size)

        if self.do_BN:
            self.quote_input_BN = nn.BatchNorm1d(num_features=self.embedding_size)
            self.response_input_BN = nn.BatchNorm1d(num_features=self.embedding_size)
            self.quote_output_BN = nn.BatchNorm1d(num_features=self.hidden_size * 2)
            self.response_output_BN = nn.BatchNorm1d(num_features=self.hidden_size * 2)
            self.concat_output_BN = nn.BatchNorm1d(num_features=self.hidden_size * 2 * 2)
            
        if attention_mechanism != None:
            self.quote_attention_layer = nn.Linear(in_features=self.hidden_size * 2, out_features=1)
            self.response_attention_layer = nn.Linear(in_features=self.hidden_size * 2, out_features=1)

        self.quote_bilstm = nn.LSTM(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.layer_num,
            batch_first=True,
            bidirectional=True,
            dropout=0.5
        )

        self.response_bilstm = nn.LSTM(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.layer_num,
            batch_first=True,
            bidirectional=True,
            dropout=0.5
        )

        self.out = nn.Linear(in_features=self.hidden_size*2*2, out_features=self.class_num)
        # self.params_initializer()

    def attention_layer(self, layer, bilstm_outputs, ActFunc):
        attention_inputs = bilstm_outputs.contiguous().view(-1, self.hidden_size*2)
        attention_logits = layer(attention_inputs).view(-1, bilstm_outputs.size()[1])
        attention_signals = F.softmax(ActFunc(attention_logits), dim=1).view(-1, bilstm_outputs.size()[1], 1)

        _outputs = attention_signals * bilstm_outputs
        outputs = torch.mean(_outputs, dim=1)
        return outputs
    
    # def params_initializer(self):
    #     for name, param in self.named_parameters():
    #         if name.find('weight') != -1 and name.find('BN') == -1:
    #             nn.init.xavier_normal(param.data)
    #         elif name.find('bias') != -1 and name.find('BN') == -1:
    #             # nn.init.uniform(param.data, -0.1, 0.1)
    #             param.data.uniform_(-0.1, 0.1)
    #         else:
    #             continue

    def forward(self, X_q_inputs, X_r_inputs):
        quote_inputs = self.embedding(X_q_inputs)
        response_inputs = self.embedding(X_r_inputs)

        if self.do_BN:
            quote_inputs = quote_inputs.transpose(1, 2).contiguous()
            quote_inputs = self.quote_input_BN(quote_inputs)
            quote_inputs = quote_inputs.transpose(1, 2)
            response_inputs = response_inputs.transpose(1, 2).contiguous()
            response_inputs = self.response_input_BN(response_inputs)
            response_inputs = response_inputs.transpose(1, 2)

        quote_outputs, _ = self.quote_bilstm(quote_inputs)
        response_outputs, _ = self.response_bilstm(response_inputs)

        if self.attention_mechanism != None:
            ActFunc = self.attention_mechanism['ActFunc']
            if self.attention_mechanism['Type'] == 'self_attention':
                quote_outputs = self.attention_layer(self.quote_attention_layer, quote_outputs, ActFunc)
                response_outputs = self.attention_layer(self.response_attention_layer, response_outputs, ActFunc)
            if self.attention_mechanism['Type'] == 'cross_attention':
                _quote_outputs = self.attention_layer(self.quote_attention_layer, response_outputs, ActFunc)
                _response_outputs = self.attention_layer(self.response_attention_layer, quote_outputs, ActFunc)
                quote_outputs = _quote_outputs
                response_outputs = _response_outputs
        else:
            quote_outputs = torch.mean(quote_outputs, dim=1)
            response_outputs = torch.mean(response_outputs, dim=1)
        if self.do_BN:
            quote_outputs = self.quote_output_BN(quote_outputs)
            response_outputs = self.response_output_BN(response_outputs)

        concat_outputs = torch.cat((quote_outputs, response_outputs), dim=1)
        if self.do_BN:
            concat_outputs = self.concat_output_BN(concat_outputs)
        output = self.out(concat_outputs)
        return output

def test_epoch(model, dataset):
    data_size = dataset.y.shape[0]
    # _batch_size = 1
    _batch_size = data_size
    batch_num = int(data_size / _batch_size)
    _accs = 0.0
    _costs = 0.0
    _pred = []
    _true = []
    for i in range(batch_num):
        
        X_q_batch, X_r_batch, y_batch = dataset.next_batch(_batch_size)
        X_q_batch, X_r_batch = torch.from_numpy(X_q_batch).type(torch.LongTensor), torch.from_numpy(X_r_batch).type(torch.LongTensor)
        if USE_GPU:
            X_q_batch, X_r_batch, y_batch = Variable(X_q_batch, volatile=True).cuda(), Variable(X_r_batch, volatile=True).cuda(), Variable(torch.from_numpy(y_batch).type(torch.LongTensor), volatile=True).cuda()
        else:
            X_q_batch, X_r_batch, y_batch = Variable(X_q_batch, volatile=True), Variable(X_r_batch,  volatile=True), Variable(torch.from_numpy(y_batch).type(torch.LongTensor), volatile=True)

        test_output = model(X_q_batch, X_r_batch)
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

if __name__ == '__main__':


    USE_GPU = True
    if USE_GPU:
        os.environ['CUDA_VISIBLE_DEVICES'] = '2'
        print('Using GPU: {}...'.format(os.environ['CUDA_VISIBLE_DEVICES']))
    
    # attention_mechanism_config = None # gpu: 0
    attention_mechanism_config = {'Type': 'self_attention', 'ActFunc': F.tanh} # gpu: 1 t1 t3
    # attention_mechanism_config = {'Type': 'cross_attention', 'ActFunc': F.tanh} # gpu: 2 t2

    model = TextBiLSTM(
        vocabulary_size=27590, embedding_size=300,
        hidden_size=128, layer_num=2, class_num=2,
        do_BN=True, attention_mechanism=attention_mechanism_config)
    
    max_grad_norm = 5.0
    nn.utils.clip_grad_norm(model.parameters(), max_norm=max_grad_norm)

    # for name, para in model.named_parameters():
    #    if para.requires_grad == True:
    #       print(name)

    if USE_GPU:
        model.cuda()
    # data_train, data_test = varied_textLength.getDataSet()
    import data_helper
    data_train, data_test = data_helper.getDataSet()
    
    lr = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()

    tr_batch_size = 32
    max_epoch = 30
    max_max_epoch = 100
    display_num = 5
    
    tr_batch_num = int(data_train.y.shape[0] / tr_batch_size)
    display_batch = int(tr_batch_num / display_num)

    for epoch in range(1, max_max_epoch+1):
        if epoch > max_epoch:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.97
        lr = optimizer.param_groups[0]['lr']
        print('EPOCH {}, lr={}'.format(epoch, lr))

        start_time = time.time()
        _accs = 0.0
        _costs = 0.0
        show_accs = 0.0
        show_costs = 0.0

        for batch in range(tr_batch_num):
            
            X_q_batch, X_r_batch, y_batch = data_train.next_batch(tr_batch_size)
            X_q_batch, X_r_batch = torch.from_numpy(X_q_batch).type(torch.LongTensor), torch.from_numpy(X_r_batch).type(torch.LongTensor)
            if USE_GPU:
                X_q_batch, X_r_batch, y_batch = Variable(X_q_batch).cuda(), Variable(X_r_batch).cuda(), Variable(torch.from_numpy(y_batch).type(torch.LongTensor)).cuda()
            else:
                X_q_batch, X_r_batch, y_batch = Variable(X_q_batch), Variable(X_r_batch), Variable(torch.from_numpy(y_batch).type(torch.LongTensor))

            output = model(X_q_batch, X_r_batch)
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
            optimizer.step()

            if (batch + 1) % display_batch == 0:
                model.eval()
                valid_acc, valid_cost, valid_cm = test_epoch(model, data_test)
                print('\ttraining acc={:.6f}, cost={:.6f};  valid acc={:.6f}, cost={:.6f}, confusion_matrix=[{}/{}, {}/{}] '.format(show_accs / display_batch, show_costs / display_batch, valid_acc, valid_cost, valid_cm[0][0], valid_cm[0][0] + valid_cm[1][0], valid_cm[1][1], valid_cm[0][1] + valid_cm[1][1]))
                show_accs = 0.0
                show_costs = 0.0
                model.train()

        mean_acc = _accs / tr_batch_num
        mean_cost = _costs / tr_batch_num
        
        model.eval()
        print('Epoch training {}, acc={:.6f}, cost={:.6f}, speed={} s/epoch'.format(data_train.y.shape[0], mean_acc, mean_cost, time.time() - start_time))
        test_acc, test_cost, test_cm = test_epoch(model, data_test)
        print('**Test {}, acc={:.6f}, cost={:.6f}, confusion_matrix=[{}/{}, {}/{}]\n'.format(data_test.y.shape[0], test_acc, test_cost, test_cm[0][0], test_cm[0][0] + test_cm[1][0], test_cm[1][1], test_cm[0][1] + test_cm[1][1]))
        model.train()





