import pandas as pd

def result2csv(taskList, typeList, kFold):
    resultList = []
    for task in taskList:
        for type in typeList:
            for fold in range(kFold):
                with open('./result_log/%s_%s_%s.txt' % (task, type, fold), 'r') as fr:
                    print('Processing %s_%s_%s.txt ...' % (task, type, fold))
                    for line in list(fr)[51:]:
                        line = line[:-1]
                        if 'EPOCH' in line:
                            fold_epoch = str(fold) + '_' + line.split(',')[0].split(' ')[-1]
                            lr = float(line.split(',')[1].split('=')[-1])
                        if 'Epoch training' in line:
                            train_acc = float(line.split(',')[1].split('=')[-1])
                            train_cost = float(line.split(',')[2].split('=')[-1])
                        if '**Test' in line:
                            test_acc = float(line.split(',')[1].split('=')[-1])
                            test_cost = float(line.split(',')[2].split('=')[-1])
                        if 'disagree' in line:
                            disagree_f1 = float(line.split()[-2])
                        elif 'agree' in line:
                            agree_f1 = float(line.split()[-2])
                            resultList.append({'task': task, 'type': type, 'fold_epoch': fold_epoch, 'lr': lr, 'train_acc': train_acc, 'train_cost': train_cost, 'test_acc': test_acc, 'test_cost': test_cost, 'disagree_f1': disagree_f1, 'agree_f1': agree_f1})

    result_df = pd.DataFrame(resultList)
    result_df.to_csv('./data/result.csv', columns=['task', 'type', 'fold_epoch', 'lr', 'train_acc', 'train_cost', 'test_acc', 'test_cost', 'disagree_f1', 'agree_f1'], index=None)

if __name__ == '__main__':
    taskList = ['iac', 'debatepedia']
    typeList = ['None', 'share', 'self_attention', 'cross_attention', 'both_sum', 'both_concat']
    kFold = 5
    result2csv(taskList, typeList, kFold)

