import torch
from torch import nn
from torch.autograd import Variable
import varied_textLength
import numpy as np
import time
from sklearn.metrics import confusion_matrix

class TextBiLSTM(nn.Module):
    def __init__(self, vocabulary_size, embedding_size, hidden_size, layer_num, class_num):
        super(TextBiLSTM, self).__init__()

        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.class_num = class_num

        self.embedding = nn.Embedding(num_embeddings=self.vocabulary_size, embedding_dim=self.embedding_size)

        # self.quote_input_BN = nn.BatchNorm1d(num_features=self.embedding_size, momentum=0.5)
        self.quote_bilstm = nn.LSTM(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.layer_num,
            batch_first=True,
            bidirectional=True
        )
        # self.quote_output_BN = nn.BatchNorm1d(num_features=self.hidden_size*2, momentum=0.5)

        # self.response_input_BN = nn.BatchNorm1d(num_features=self.embedding_size, momentum=0.5)
        self.response_bilstm = nn.LSTM(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.layer_num,
            batch_first=True,
            bidirectional=True
        )
        # self.response_output_BN = nn.BatchNorm1d(num_features=self.hidden_size*2, momentum=0.5)

        # self.concat_output_BN = nn.BatchNorm1d(num_features=self.hidden_size*2*2, momentum=0.5)
        self.out = nn.Linear(in_features=self.hidden_size*2*2, out_features=self.class_num)

    def forward(self, X_q_inputs, X_r_inputs):
        quote_inputs = self.embedding(X_q_inputs)
        # quote_inputs = quote_inputs.transpose(1, 2)
        # quote_inputs = self.quote_input_BN(quote_inputs)
        # quote_inputs = quote_inputs.transpose(1, 2)
        quote_outputs, _ = self.quote_bilstm(quote_inputs, None)
        quote_outputs = torch.mean(quote_outputs, dim=1)
        # quote_outputs = self.quote_output_BN(quote_outputs)

        response_inputs = self.embedding(X_r_inputs)
        # response_inputs = response_inputs.transpose(1, 2)
        # response_inputs = self.response_input_BN(response_inputs)
        # response_inputs = response_inputs.transpose(1, 2)
        response_outputs, _ = self.response_bilstm(response_inputs, None)
        response_outputs = torch.mean(response_outputs, dim=1)
        # response_outputs = self.response_output_BN(response_outputs)

        concat_outputs = torch.cat((quote_outputs, response_outputs), dim=1)
        # concat_outputs = self.concat_output_BN(concat_outputs)
        output = self.out(concat_outputs)
        return output

def test_epoch(model, dataset):
    data_size = dataset.y.shape[0]
    _batch_size = 1
    batch_num = int(data_size / _batch_size)
    _accs = 0.0
    _costs = 0.0
    _pred = []
    _true = []
    for i in xrange(batch_num):
        X_q_batch, X_r_batch, y_batch = dataset.next_batch(_batch_size)
        X_q_batch, X_r_batch = np.asarray(X_q_batch[0]), np.asarray(X_r_batch[0])
        X_q_batch, X_r_batch = torch.from_numpy(X_q_batch).type(torch.LongTensor), torch.from_numpy(X_r_batch).type(torch.LongTensor)
        X_q_batch, X_r_batch = torch.unsqueeze(X_q_batch, dim=0), torch.unsqueeze(X_r_batch, dim=0)
        X_q_batch, X_r_batch, y_batch = Variable(X_q_batch, volatile=True), Variable(X_r_batch, volatile=True), Variable(torch.from_numpy(y_batch).type(torch.LongTensor), volatile=True)

        test_output = model(X_q_batch, X_r_batch)
        _cost = loss_func(test_output, y_batch)

        y_pred = torch.max(test_output, 1)[1]
        _acc = sum(y_pred.data == y_batch.data) / float(y_batch.size(0))

        if i == 0:
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

    model = TextBiLSTM(vocabulary_size=27590, embedding_size=300, hidden_size=128, layer_num=2, class_num=2)
    data_train, data_test = varied_textLength.getDataSet()

    lr = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()

    tr_batch_size = 1
    max_epoch = 100
    display_num = 5
    tr_batch_num = int(data_train.y.shape[0] / tr_batch_size)
    display_batch = int(tr_batch_num / display_num)

    for epoch in range(max_epoch):
        print 'EPOCH {}, lr={}'.format(epoch + 1, lr)

        start_time = time.time()
        _accs = 0.0
        _costs = 0.0
        show_accs = 0.0
        show_costs = 0.0

        for batch in xrange(tr_batch_num):
            X_q_batch, X_r_batch, y_batch = data_train.next_batch(tr_batch_size)
            X_q_batch, X_r_batch = np.asarray(X_q_batch[0]), np.asarray(X_r_batch[0])
            X_q_batch, X_r_batch = torch.from_numpy(X_q_batch).type(torch.LongTensor), torch.from_numpy(X_r_batch).type(torch.LongTensor)
            X_q_batch, X_r_batch = torch.unsqueeze(X_q_batch, dim=0), torch.unsqueeze(X_r_batch, dim=0)
            X_q_batch, X_r_batch, y_batch = Variable(X_q_batch), Variable(X_r_batch), Variable(torch.from_numpy(y_batch).type(torch.LongTensor))

            output = model(X_q_batch, X_r_batch)
            _cost = loss_func(output, y_batch)
            y_pred = torch.max(output, 1)[1]
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
                print '\ttraining acc={}, cost={};  valid acc={}, cost={}, confusion_matrix=[{}/{}, {}/{}] '.format(show_accs / display_batch, show_costs / display_batch, valid_acc, valid_cost, valid_cm[0][0], valid_cm[0][0] + valid_cm[1][0], valid_cm[1][1], valid_cm[0][1] + valid_cm[1][1])
                show_accs = 0.0
                show_costs = 0.0
                model.train()

        mean_acc = _accs / tr_batch_num
        mean_cost = _costs / tr_batch_num

        print 'Epoch training {}, acc={}, cost={}, speed={} s/epoch'.format(data_train.y.shape[0], mean_acc, mean_cost, time.time() - start_time)
        test_acc, test_cost, test_cm = test_epoch(model, data_test)
        print '**Test {}, acc={}, cost={}, confusion_matrix=[{}/{}, {}/{}]\n'.format(data_test.y.shape[0], test_acc, test_cost, test_cm[0][0], test_cm[0][0] + test_cm[1][0], test_cm[1][1], test_cm[0][1] + test_cm[1][1])





