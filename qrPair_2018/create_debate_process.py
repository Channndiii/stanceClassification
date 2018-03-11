import os
from bs4 import BeautifulSoup
import re
import pandas as pd
import numpy as np
import collections

def getFileList(source_dir):
    fileList = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.find('.xml') != -1:
                fileList.append(os.path.join(root, file))
    return fileList

def xmlParsing(filePath):
    soup = BeautifulSoup(open(filePath), 'lxml')
    title = soup.title.get_text()
    dataDict = dict()
    for comment in soup.find_all('comment'):
        quoteID = comment['parent-url']
        responseID = comment['url']
        side = comment['side']
        userName = comment.user.get_text()
        text = comment.find_all('text')[0].get_text()
        dataDict[responseID] = {'title': title, 'quoteID': quoteID, 'responseID': responseID, 'side': side, 'userName': userName, 'text': text}
    return dataDict

def cleanText(text):
    text = str(text)
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

def buildDataset(dataDict, downsample):
    resultList = []
    for responseID in dataDict.keys():
        quoteID = dataDict[responseID]['quoteID']
        user_of_response = dataDict[responseID]['userName']
        if quoteID != '-1':
            user_of_quote = dataDict[quoteID]['userName']
            if user_of_quote != user_of_response:
                title = dataDict[responseID]['title']
                quoteText = cleanText(dataDict[quoteID]['text'])
                responseText = cleanText(dataDict[responseID]['text'])
                side_of_quote = dataDict[quoteID]['side']
                side_of_response = dataDict[responseID]['side']
                relation = 'agreement' if side_of_quote == side_of_response else 'disagreement'
                if relation == 'disagreement':
                    if downsample:
                        if np.random.random_sample() < 0.52:
                            resultList.append({'title': title, 'quoteID': quoteID, 'responseID': responseID, 'quoteText': quoteText, 'responseText': responseText, 'relation': relation})
                    else:
                        resultList.append({'title': title, 'quoteID': quoteID, 'responseID': responseID, 'quoteText': quoteText, 'responseText': responseText, 'relation': relation})
                if relation == 'agreement':
                    resultList.append({'title': title, 'quoteID': quoteID, 'responseID': responseID, 'quoteText': quoteText, 'responseText': responseText, 'relation': relation})
    return resultList

if __name__ == '__main__':

    fileList = getFileList(source_dir='./data/create_debate/training/')
    train_data = []
    for file in fileList:
        dataDict = xmlParsing(file)
        # sideSet = set()
        # for id in dataDict.keys():
        #     sideSet.add(dataDict[id]['side'])
        # if len(sideSet) != 2:
        #     continue
        result = buildDataset(dataDict, downsample=True)
        train_data.extend(result)

    train_df = pd.DataFrame(train_data)
    print(len(train_df))
    print(collections.Counter(train_df.relation.values).most_common())
    train_df.to_csv('./data/create_debate_train.csv', columns=['title', 'quoteID', 'responseID', 'quoteText', 'responseText', 'relation'], index=None)

    fileList = getFileList(source_dir='./data/create_debate/development/')
    dev_data = []
    for file in fileList:
        dataDict = xmlParsing(file)
        # sideSet = set()
        # for id in dataDict.keys():
        #     sideSet.add(dataDict[id]['side'])
        # if len(sideSet) != 2:
        #     continue
        result = buildDataset(dataDict, downsample=False)
        dev_data.extend(result)

    dev_df = pd.DataFrame(dev_data)
    print(len(dev_df))
    print(collections.Counter(dev_df.relation.values).most_common())
    dev_df.to_csv('./data/create_debate_dev.csv', columns=['title', 'quoteID', 'responseID', 'quoteText', 'responseText', 'relation'], index=None)

    fileList = getFileList(source_dir='./data/create_debate/testing/')
    test_data = []
    for file in fileList:
        dataDict = xmlParsing(file)
        # sideSet = set()
        # for id in dataDict.keys():
        #     sideSet.add(dataDict[id]['side'])
        # if len(sideSet) != 2:
        #     continue
        result = buildDataset(dataDict, downsample=False)
        test_data.extend(result)

    test_df = pd.DataFrame(test_data)
    print(len(test_df))
    print(collections.Counter(test_df.relation.values).most_common())
    test_df.to_csv('./data/create_debate_test.csv', columns=['title', 'quoteID', 'responseID', 'quoteText', 'responseText', 'relation'], index=None)