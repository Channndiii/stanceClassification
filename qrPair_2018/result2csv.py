import pandas as pd
import numpy as np

def result2csv(taskList, typeList, kFold):
    resultList = []
    for task in taskList:
        for type in typeList:
            for fold in range(kFold):
                try:
                    with open('./result_log/%s_%s_%s.txt' % (task, type, fold), 'r') as fr:
                        print('Processing %s_%s_%s.txt ...' % (task, type, fold))
                        for line in list(fr)[50:]:
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
                except Exception:
                    pass

    result_df = pd.DataFrame(resultList)
    result_df.to_csv('./data/result_hybrid.csv', columns=['task', 'type', 'fold_epoch', 'lr', 'train_acc', 'train_cost', 'test_acc', 'test_cost', 'disagree_f1', 'agree_f1'], index=None)

def result2csv_bBb(taskList, typeList, kFold):
    resultList = []
    for task in taskList:
        for type in typeList:
            for fold in range(kFold):
                try:
                    with open('./result_log/%s_%s_%s.txt' % (task, type, fold), 'r') as fr:
                        print('Processing %s_%s_%s.txt ...' % (task, type, fold))
                        fr = list(fr)
                        best_batch = fr[-1][:-1].split(',')[0].split('=')[-1]
                        for line in fr[50:]:
                            line = line[:-1]
                            if 'BATCH' in line:
                                batch = line.split(',')[0].split(' ')[1]
                                if batch == best_batch:
                                    train_acc = float(line.split(',')[0].split('=')[-1])
                                    train_cost = float(line.split(',')[1].split('=')[-1])
                                    test_acc = float(line.split(',')[2].split('=')[-1])
                                    test_cost = float(line.split(',')[3].split('=')[-1])
                                    disagree_f1 = float(line.split(',')[4].split('=')[-1])
                                    agree_f1 = float(line.split(',')[5].split('=')[-1])
                                    average_f1 = float(line.split(',')[6].split('=')[-1])
                                    resultList.append({'task': task, 'type': type, 'fold': fold, 'train_acc': train_acc, 'train_cost': train_cost, 'test_acc': test_acc, 'test_cost': test_cost, 'disagree_f1': disagree_f1, 'agree_f1': agree_f1, 'average_f1': average_f1})

                except Exception:
                    pass

    result_df = pd.DataFrame(resultList)
    for task in taskList:
        for type in typeList:
            final_df = result_df[(result_df.task == task) & (result_df.type == type)]
            cv_result = np.mean(final_df.average_f1.values)
            print('{}, {}, cv_result={:.6f}'.format(task, type, cv_result))
    # result_df.to_csv('./data/result.csv', columns=['task', 'type', 'fold', 'train_acc', 'train_cost', 'test_acc', 'test_cost', 'disagree_f1', 'agree_f1', 'average_f1'], index=None)

def result2csv_new(taskList, typeList, kFold):
    resultList = []
    for task in taskList:
        for type in typeList:
            for fold in range(kFold):
                try:
                    with open('./result_log/%s_%s_%s.txt' % (task, type, fold), 'r') as fr:
                        print('Processing %s_%s_%s.txt ...' % (task, type, fold))
                        fr = list(fr)
                        best_epoch = fr[-1][:-1].split(',')[0].split('=')[-1]
                        best_result = fr[-1][:-1].split(',')[1].split('=')[-1]
                        for line in fr[50:]:
                            line = line[:-1]
                            if 'EPOCH' in line:
                                epoch = line.split(',')[0].split(' ')[-1]
                                fold_epoch = str(fold) + '_' + epoch
                                lr = float(line.split(',')[1].split('=')[-1])
                            if 'Epoch training' in line:
                                train_acc = float(line.split(',')[1].split('=')[-1])
                                train_cost = float(line.split(',')[2].split('=')[-1])
                            if '**Test' in line:
                                test_acc = float(line.split(',')[1].split('=')[-1])
                                test_cost = float(line.split(',')[2].split('=')[-1])
                                test_average_f1 = float(line.split(',')[3].split('=')[-1])
                            if 'disagree' in line:
                                disagree_f1 = float(line.split()[-2])
                            elif 'agree' in line:
                                agree_f1 = float(line.split()[-2])
                                if epoch == best_epoch:
                                    resultList.append({'task': task, 'type': type, 'fold_epoch': fold_epoch, 'lr': lr, 'train_acc': train_acc, 'train_cost': train_cost, 'test_acc': test_acc, 'test_cost': test_cost, 'disagree_f1': disagree_f1, 'agree_f1': agree_f1, 'average_f1': test_average_f1})
                                    break
                except Exception:
                    pass

    result_df = pd.DataFrame(resultList)
    result_df.to_csv('./data/result_new.csv',
                     columns=['task', 'type', 'fold_epoch', 'lr', 'train_acc', 'train_cost', 'test_acc', 'test_cost', 'disagree_f1', 'agree_f1', 'average_f1'], index=None)

def result2csv_bBb_acc(taskList, typeList, kFold):
    for task in taskList:
        for type in typeList:
            for fold in range(kFold):
                try:
                    with open('./result_log/%s_%s_%s.txt' % (task, type, fold), 'r') as fr:
                        resultList = []
                        print('Processing %s_%s_%s.txt ...' % (task, type, fold))
                        fr = list(fr)
                        for line in fr[42:-1]:
                            line = line[:-1]
                            if 'BATCH' in line:
                                test_acc = float(line.split(',')[2].split('=')[-1])
                                resultList.append(test_acc)
                        print('Best Acc = {}'.format(max(resultList)))
                except Exception:
                    pass

if __name__ == '__main__':
    # # taskList = ['iac', 'debatepedia', 'create_debate', 'create_debate_preTrain', 'iac_shuffle', 'iac_siamese']
    # # typeList = ['None', 'share', 'self_attention', 'cross_attention', 'both_sum', 'both_concat']
    # taskList = ['create_debate_preTrain']
    # typeList = ['None', 'share', 'self_attention', 'cross_attention', 'both_sum', 'both_concat', 'hybrid']
    # kFold = 5
    # result2csv(taskList, typeList, kFold)
    # # result2csv_bBb(taskList, typeList, kFold)
    # # result2csv_new(taskList, typeList, kFold)

    taskList = ['debatepedia_BBB']
    typeList = ['None_None', 'single_self_dot', 'single_cross_dot_v2', 'sum_self_dot-cross_dot_v2', 'concat_self_dot-cross_dot_v2', 'hybrid_self_dot-cross_dot_v2']
    kFold = 5

    result2csv_bBb_acc(taskList, typeList, kFold)

