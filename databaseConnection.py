import pymysql
import pandas as pd
import re
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

    # if tableName == 'mturk_2010_qr_task1_average_response':
    #     for row in results:
    #         pageID = str(row[0])
    #         tabNum = str(row[1])
    #         qrID = pageID + '@' + tabNum
    #         disagree_agree = 1 if float(row[3]) >= 0 else 0
    #         attacking_respectful = 1 if float(row[5]) >= 0 else 0
    #         emotion_fact = 1 if float(row[7]) >= 0 else 0
    #         nasty_nice = 1 if float(row[9]) >= 0 else 0
    #         dataResults.append([qrID, disagree_agree, attacking_respectful, emotion_fact, nasty_nice])
    #     return dataResults

    if tableName == 'mturk_2010_qr_task1_average_response':
        for row in results:
            pageID = str(row[0])
            tabNum = str(row[1])
            qrID = pageID + '@' + tabNum

            # disagree_agree
            # if float(row[3]) <= -2.0:
            #     disagree_agree = 0
            #     dataResults.append([qrID, disagree_agree])
            # if float(row[3]) > 0.0:
            #     disagree_agree = 1
            #     dataResults.append([qrID, disagree_agree])
            # disagree_agree

            # attacking_respectful
            # if float(row[5]) < 0.0:
            #     attacking_respectful = 0
            #     dataResults.append([qrID, attacking_respectful])
            # if float(row[5]) > 1.0:
            #     attacking_respectful = 1
            #     dataResults.append([qrID, attacking_respectful])
            # attacking_respectful

            # emotion_fact
            # if float(row[7]) <= -1.0:
            #     emotion_fact = 0
            #     dataResults.append([qrID, emotion_fact])
            # if float(row[7]) > 1.0:
            #     emotion_fact = 1
            #     dataResults.append([qrID, emotion_fact])
            # emotion_fact

            # nasty_nice
            if float(row[9]) < 0.0:
                nasty_nice = 0
                dataResults.append([qrID, nasty_nice])
            if float(row[9]) > 1.0:
                nasty_nice = 1
                dataResults.append([qrID, nasty_nice])
            # nasty_nice

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

if __name__ == '__main__':

    # task = 'disagree_agree'
    # task = 'attacking_respectful'
    # task = 'emotion_fact'
    task = 'nasty_nice'

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
    # qrPair_df = pd.DataFrame(result, columns=['qrID', 'quoteText', 'responseText', 'topic', 'disagree_agree', 'attacking_respectful', 'emotion_fact', 'nasty_nice'])
    # qrPair_df.to_csv('./data/qrPair.csv', index=None)
    # print qrPair_df.head()
    qrPair_df = pd.DataFrame(result, columns=['qrID', 'quoteText', 'responseText', 'topic', '%s' % task])
    qrPair_df.to_csv('./data/qrPair_%s.csv' % task, index=None)

    # labelDistribution()