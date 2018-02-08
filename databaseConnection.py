import pymysql
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt

def connectDatabase():
    db = pymysql.connect('localhost',
                         'root',
                         '940803',
                         'IAC2.0')
    return db

def queryDatabase(db, tableName):
    cursor = db.cursor()
    sql = 'SELECT * FROM %s' % tableName
    try:
        cursor.execute(sql)
        results = cursor.fetchall()
        return results
    except:
        print 'Error: unable to fetch data'

# def cleanText(text):
#     text = ' '.join([word for word in text.strip().split()])
#     text = re.sub(r'[^\x00-\x7f]', '', text)
#     text = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", text)
#     text = re.sub(r",", " , ", text)
#     text = re.sub(r"!", " ! ", text)
#     text = re.sub(r"\(", " ( ", text)
#     text = re.sub(r"\)", " ) ", text)
#     text = re.sub(r"\?", " ? ", text)
#     text = re.sub(r"\s{2,}", " ", text)
#     return text

def cleanText(text):
    text = ' '.join([word for word in text.strip().split()])
    text = re.sub(r'[^\x00-\x7f]', '', text)
    text = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", text)
    text = re.sub(r"\'s", " \'s", text)
    text = re.sub(r"\'ve", " \'ve", text)
    text = re.sub(r"n\'t", " n\'t", text)
    text = re.sub(r"\'re", " \'re", text)
    text = re.sub(r"\'d", " \'d", text)
    text = re.sub(r"\'ll", " \'ll", text)
    text = re.sub(r",", " , ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\(", " \( ", text)
    text = re.sub(r"\)", " \) ", text)
    text = re.sub(r"\?", " \? ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip().lower()

def dataResolution(results, tableName):
    dataResults = []
    if tableName == 'mturk_2010_qr_entry':
        for row in results:
            pageID = str(row[0])
            tabNum = str(row[1])
            qrID = pageID + '@' +tabNum
            quoteText = cleanText(str(row[6]))
            responseText = cleanText(str(row[7]))
            topic = str(row[9])
            dataResults.append([qrID, quoteText, responseText, topic])
        return dataResults

    if tableName == 'mturk_2010_qr_task1_average_response':
        for row in results:
            pageID = str(row[0])
            tabNum = str(row[1])
            qrID = pageID + '@' + tabNum
            # disagree_agree = 1 if float(row[3]) >= 0 else 0
            # attacking_respectful = 1 if float(row[5]) >= 0 else 0
            # emotion_fact = 1 if float(row[7]) >= 0 else 0
            # nasty_nice = 1 if float(row[9]) >= 0 else 0

            disagree_agree = float(row[3])
            attacking_respectful = float(row[5])
            emotion_fact = float(row[7])
            nasty_nice = float(row[9])

            dataResults.append([qrID, disagree_agree, attacking_respectful, emotion_fact, nasty_nice])
        return dataResults

    # if tableName == 'mturk_2010_qr_task1_average_response':
    #     for row in results:
    #         pageID = str(row[0])
    #         tabNum = str(row[1])
    #         qrID = pageID + '@' + tabNum
    #
    #         # disagree_agree
    #         if float(row[3]) <= -2.0:
    #             disagree_agree = 0
    #             dataResults.append([qrID, disagree_agree])
    #         if float(row[3]) > 0.0:
    #             disagree_agree = 1
    #             dataResults.append([qrID, disagree_agree])
    #         # disagree_agree
    #
    #         # attacking_respectful
    #         # if float(row[5]) < 0.0:
    #         #     attacking_respectful = 0
    #         #     dataResults.append([qrID, attacking_respectful])
    #         # if float(row[5]) > 1.0:
    #         #     attacking_respectful = 1
    #         #     dataResults.append([qrID, attacking_respectful])
    #         # attacking_respectful
    #
    #         # emotion_fact
    #         # if float(row[7]) <= -1.0:
    #         #     emotion_fact = 0
    #         #     dataResults.append([qrID, emotion_fact])
    #         # if float(row[7]) > 1.0:
    #         #     emotion_fact = 1
    #         #     dataResults.append([qrID, emotion_fact])
    #         # emotion_fact
    #
    #         # nasty_nice
    #         # if float(row[9]) < 0.0:
    #         #     nasty_nice = 0
    #         #     dataResults.append([qrID, nasty_nice])
    #         # if float(row[9]) > 1.0:
    #         #     nasty_nice = 1
    #         #     dataResults.append([qrID, nasty_nice])
    #         # nasty_nice
    #
    #     return dataResults

    if tableName == 'post':
        for row in results:
            discussionID = str(row[0])
            postID = str(row[1])
            textID = str(row[6])
            dataResults.append([discussionID+'@'+postID, textID])
        return dataResults

    if tableName == 'quote':
        for row in results:
            quoteTextID = str(row[5])
            responseDPID = str(row[0]) + '@' + str(row[1]) # discussionID + postID
            dataResults.append([quoteTextID, responseDPID])
        return dataResults

    if tableName == 'text':
        for row in results:
            textID = str(row[0])
            text = cleanText(str(row[1]))
            dataResults.append([textID, text])
        return dataResults

def get_qrID2Label(qrPairLabelList):
    qrID2Label = dict()
    for row in qrPairLabelList:
        qrID = row[0]
        qrID2Label[qrID] = row[1:]
    return qrID2Label

def labelDistribution():

    db = connectDatabase()

    qrPairList = queryDatabase(db, 'mturk_2010_qr_entry')
    qrPairList = dataResolution(qrPairList, 'mturk_2010_qr_entry')

    qrPairLabelList = []
    results = queryDatabase(db, 'mturk_2010_qr_task1_average_response')
    for row in results:
        pageID = str(row[0])
        tabNum = str(row[1])
        qrID = pageID + '@' + tabNum
        disagree_agree = float(row[3])
        attacking_respectful = float(row[5])
        emotion_fact = float(row[7])
        nasty_nice = float(row[9])
        qrPairLabelList.append([qrID, disagree_agree, attacking_respectful, emotion_fact, nasty_nice])

    qrID2Label = get_qrID2Label(qrPairLabelList)

    result = []
    for row in qrPairList:
        qrID = row[0]
        try:
            label = qrID2Label[qrID]
            row.extend(label)
            result.append(row)
        except Exception:
            pass

    qrPair_df = pd.DataFrame(result, columns=['qrID', 'quoteText', 'responseText', 'topic', 'disagree_agree', 'attacking_respectful', 'emotion_fact', 'nasty_nice'])


    distributionCount(qrPair_df['disagree_agree'].values, pos=5, neg=3) # final <=-2.0 or >0.0
    # distributionCount(qrPair_df['attacking_respectful'].values, pos=6, neg=5) # final <0.0 or >1.0
    # distributionCount(qrPair_df['emotion_fact'].values, pos=6, neg=4) # final <=-1.0 or >1.0
    # distributionCount(qrPair_df['nasty_nice'].values, pos=6, neg=5) # final <0.0 or >1.0

    # qrPair_df['disagree_agree'].hist(bins=10)
    # # qrPair_df['attacking_respectful'].hist(bins=10)
    # # qrPair_df['emotion_fact'].hist(bins=10)
    # # qrPair_df['nasty_nice'].hist(bins=10)
    # plt.xlim(-5, 5)
    # plt.xlabel('Rate')
    # plt.ylabel('Num')
    # plt.title('Distribution of Rate')
    # plt.show()

def distributionCount(labelList, pos, neg):
    '''
    -5.0 -4.0 -3.0 -2.0 -1.0 0.0 1.0 2.0 3.0 4.0 5.0
        0    1    2    3    4   5   6   7   8   9
    '''
    count = [0] * 10
    for score in labelList:
        if score <= -4.0:
            count[0] += 1
        elif score <= -3.0:
            count[1] += 1
        elif score <= -2.0:
            count[2] += 1
        elif score <= -1.0:
            count[3] += 1
        elif score <= 0.0:
            count[4] += 1
        elif score <= 1.0:
            count[5] += 1
        elif score <= 2.0:
            count[6] += 1
        elif score <= 3.0:
            count[7] += 1
        elif score <= 4.0:
            count[8] += 1
        elif score <= 5.0:
            count[9] += 1
    total = sum(count[:neg]) + sum(count[pos:])
    if sum(count[:neg]) > sum(count[pos:]):
        rate = float(sum(count[:neg])) / total
    else:
        rate = float(sum(count[pos:])) / total
    print count, sum(count), total, rate

    count = [0] * 10
    for score in labelList:
        if score <= -4.0:
            count[0] += 1
        if score > -4.0 and score <= -3.0:
            count[1] += 1
        if score > -3.0 and score <= -2.0:
            count[2] += 1
        if score > -2.0 and score <= -1.0:
            count[3] += 1
        if score > -1.0 and score <= 0.0:
            count[4] += 1
        if score > 0.0 and score <= 1.0:
            count[5] += 1
        if score > 1.0 and score <= 2.0:
            count[6] += 1
        if score > 2.0 and score <= 3.0:
            count[7] += 1
        if score > 3.0 and score <= 4.0:
            count[8] += 1
        if score > 4.0 and score <= 5.0:
            count[9] += 1
    total = sum(count[:neg]) + sum(count[pos:])
    if sum(count[:neg]) > sum(count[pos:]):
        rate = float(sum(count[:neg])) / total
    else:
        rate = float(sum(count[pos:])) / total
    print count, sum(count), total, rate, sum(count[:neg]), sum(count[pos:])

    # disagree_agree
    # count = [0] * 2
    # for score in labelList:
    #     if score <= -2.0:
    #         count[0] += 1
    #     if score > 0.0:
    #         count[1] += 1
    # print count, sum(count)
    # disagree_agree [3903, 1973] 5876 0.664

    # attacking_respectful
    # count = [0] * 2
    # for score in labelList:
    #     if score < 0.0:
    #         count[0] += 1
    #     if score > 1.0:
    #         count[1] += 1
    # print count, sum(count)
    # attacking_respectful [3294, 3618] 6912 0.523

    # emotion_fact
    # count = [0] * 2
    # for score in labelList:
    #     if score <= -1.0:
    #         count[0] += 1
    #     if score > 1.0:
    #         count[1] += 1
    # print count, sum(count)
    # emotion_fact [2383, 3040] 5423 0.561

    # nasty_nice
    # count = [0] * 2
    # for score in labelList:
    #     if score < 0.0:
    #         count[0] += 1
    #     if score > 1.0:
    #         count[1] += 1
    # print count, sum(count)
    # nasty_nice [2455, 4391] 6846 0.641

def get_dpID2TextID():

    db = connectDatabase()
    dpID2TextIDList = queryDatabase(db, 'post')
    dpID2TextIDList = dataResolution(dpID2TextIDList, 'post')
    dpID2TextID = dict()
    for i in dpID2TextIDList:
        dpID = i[0]
        textID = i[1]
        dpID2TextID[dpID] = textID
    return dpID2TextID

def buildUnsupervisedData():

    # step 1
    # db = connectDatabase()
    # qrIDList = queryDatabase(db, 'quote')
    # qrIDList = dataResolution(qrIDList, 'quote')
    #
    # dpID2TextID = get_dpID2TextID()
    # qrTextIDList = []
    # for i in qrIDList:
    #     quoteTextID = i[0]
    #     responseDPID = i[1]
    #     responseTextID = dpID2TextID[responseDPID]
    #     qrTextIDList.append([quoteTextID, responseTextID])
    #
    # qrTextID_df = pd.DataFrame(qrTextIDList, columns=['quoteTextID', 'responseTextID'])
    # qrTextID_df.to_csv('./data/unsupervisedQRTextID.csv', index=None)
    # step 1

    # step 2
    global id2Text
    id2Text = get_id2Text()
    qrTextID_df = pd.read_csv('./data/unsupervisedQRTextID.csv')
    qrTextID_df['quoteText'] = qrTextID_df['quoteTextID'].apply(mapID2Text)
    qrTextID_df['responseText'] = qrTextID_df['responseTextID'].apply(mapID2Text)
    qrTextID_df.to_csv('./data/unsupervisedQRText.csv', index=None)
    # step 2

def get_id2Text():
    db = connectDatabase()
    id2TextList = queryDatabase(db, 'text')
    id2TextList = dataResolution(id2TextList, 'text')
    id2Text = dict()
    for i in id2TextList:
        textID = i[0]
        text = i[1]
        id2Text[textID] = text
    return id2Text

def mapID2Text(textID):
    return id2Text[str(textID)]

def filterText(row):
    maxLen = 150
    quoteText = str(row['quoteText']).split(' ')
    responseText = str(row['responseText']).split(' ')
    if len(quoteText) <= maxLen and len(responseText) <= maxLen:
        filterLabel = False
    else:
        filterLabel = True
    row['filterLabel'] = filterLabel
    return row

def sampleText(row):
    if np.random.random_sample() > 0.5:
        filterLabel = False
    else:
        filterLabel = True
    row['filterLabel'] = filterLabel
    return row

def get_qrTextSample():
    # step 1
    # qrText_df = pd.read_csv('./data/unsupervisedQRText.csv')
    # qrText_df = qrText_df.apply(filterText, axis=1)
    # qrText_df = qrText_df[qrText_df.filterLabel == False]
    # qrText_df.to_csv('./data/unsupervisedQRTextSample.csv', index=None, columns=['quoteTextID', 'responseTextID', 'quoteText', 'responseText'])
    # step 1

    # step 2
    qrText_df = pd.read_csv('./data/unsupervisedQRTextSample.csv')
    qrText_df = qrText_df.apply(sampleText, axis=1)
    qrText_df = qrText_df[qrText_df.filterLabel == False]
    qrText_df.to_csv('./data/unsupervisedQRText_Sample.csv', index=None, columns=['quoteTextID', 'responseTextID', 'quoteText', 'responseText'])
    # step 2

if __name__ == '__main__':

    # task = 'disagree_agree'
    # task = 'attacking_respectful'
    # task = 'emotion_fact'
    # task = 'nasty_nice'

    db = connectDatabase()

    qrPairList = queryDatabase(db, 'mturk_2010_qr_entry')
    qrPairList = dataResolution(qrPairList, 'mturk_2010_qr_entry')

    qrPairLabelList = queryDatabase(db, 'mturk_2010_qr_task1_average_response')
    qrPairLabelList = dataResolution(qrPairLabelList, 'mturk_2010_qr_task1_average_response')

    qrID2Label = get_qrID2Label(qrPairLabelList)

    result = []
    for row in qrPairList:
        qrID = row[0]
        try:
            label = qrID2Label[qrID]
            row.extend(label)
            result.append(row)
        except Exception:
            # print qrID
            pass
    qrPair_df = pd.DataFrame(result, columns=['qrID', 'quoteText', 'responseText', 'topic', 'disagree_agree', 'attacking_respectful', 'emotion_fact', 'nasty_nice'])
    # qrPair_df.to_csv('./data/qrPair.csv', index=None)
    qrPair_df.to_csv('./data/qrPair_2018.csv', index=None)
    # print qrPair_df.head()
    # qrPair_df = pd.DataFrame(result, columns=['qrID', 'quoteText', 'responseText', 'topic', '%s' % task])
    # qrPair_df.to_csv('./data/qrPair_%s.csv' % task, index=None)
    #
    # # labelDistribution()

    # buildUnsupervisedData()

    # get_qrTextSample()
    # qrText_df = pd.read_csv('./data/unsupervisedQRText_Sample.csv')
    # print len(qrText_df)

