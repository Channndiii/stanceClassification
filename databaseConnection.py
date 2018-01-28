import pymysql
import codecs
import collections
import networkx as nx

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

def dataResolution(results, tableName):

    dataResults = []
    if tableName == 'author':
        for row in results:
            authorID = row[0]
            userName = row[1]
            dataResults.append((authorID, userName))
        return dataResults
    if tableName == 'text':
        for row in results:
            textID = row[0]
            text = row[1]
            dataResults.append((textID, text))
        return dataResults
    if tableName == 'post':
        for row in results:
            discussionID = str(row[0])
            postID = discussionID + '@' + str(row[1])
            authorID = row[2]
            parentPostID = discussionID + '@' + str(row[4]) if str(row[4]) != 'None' else str(row[4])
            textID = str(row[6])
            dataResults.append((discussionID, postID, authorID, parentPostID, textID))
        return dataResults
    if tableName == 'quote':
        for row in results:
            discussionID = str(row[0])
            postID = discussionID + '@' + str(row[1])
            sourceDisscussionID = str(row[6])
            sourcePostID = sourceDisscussionID + '@' + str(row[7])
            textID = str(row[5])
            dataResults.append((discussionID, postID, sourceDisscussionID, sourcePostID, textID))
        return dataResults
    if tableName == 'mturk_author_stance':
        for row in results:
            authorID = str(row[1])
            topicID = authorID + '@' + str(row[2])
            topicStance1 = row[4]
            topicStance2 = row[6]
            topicStance3 = row[7]
            dataResults.append((authorID, topicID, topicStance1, topicStance2, topicStance3))
        return dataResults
    if tableName == 'discussion_topic':
        for row in results:
            discussionID = str(row[0])
            topicID = str(row[1])
            dataResults.append((discussionID, topicID))
        return dataResults
    if tableName == 'discussion':
        for row in results:
            discussionID = str(row[0])
            title = str(row[2])
            dataResults.append((discussionID, title))
        return dataResults

def getPost2Author(db):
    results = queryDatabase(db, 'post')
    results = dataResolution(results, 'post')
    post2Author = dict()
    for index, row in enumerate(results):
        postID = row[1]
        authorID = row[2]
        post2Author[postID] = authorID
    return post2Author

def getUserPostGraph(db, tableName):

    results = queryDatabase(db, tableName)
    results = dataResolution(results, tableName)

    userGraphEdge = []
    post2Author = getPost2Author(db)
    for index, row in enumerate(results):
        authorID = row[2]
        parentPostID = row[3]
        if parentPostID != 'None':
            # userGraphEdge.append((post2Author[parentPostID], authorID, tableName[0]))
            userGraphEdge.append((post2Author[parentPostID], authorID))
            userGraphEdge.append((authorID, post2Author[parentPostID]))
        else:
            continue

    # print 'Finish Processing!'
    # userGraphEdge = collections.Counter(userGraphEdge).most_common()
    # userGraphEdge = sorted(userGraphEdge, key=lambda x: x[0][0])
    # with codecs.open('./userGraph-Post.txt', 'w', 'utf-8') as fw:
    #     for (edge, Count) in userGraphEdge:
    #         fw.write(str(edge[0]) + ' ' + str(edge[1]) + ' ' + edge[2] + ' ' + str(Count) + '\n')

    return userGraphEdge

def getUserQuoteGraph(db, tableName):

    results = queryDatabase(db, tableName)
    results = dataResolution(results, tableName)

    userGraphEdge = []
    post2Author = getPost2Author(db)
    count = 0
    for index, row in enumerate(results):
        try:
            postID = row[1]
            sourcePostID = row[3]
            # userGraphEdge.append((post2Author[sourcePostID], post2Author[postID], tableName[0]))
            userGraphEdge.append((post2Author[sourcePostID], post2Author[postID]))
            userGraphEdge.append((post2Author[postID], post2Author[sourcePostID]))
        except Exception:
            print 'Error row {}-->{}'.format(index + 1, row)
            count += 1
    print 'Error row count={}'.format(count)

    # print 'Finish Processing!'
    # userGraphEdge = collections.Counter(userGraphEdge).most_common()
    # userGraphEdge = sorted(userGraphEdge, key=lambda x: x[0][0])
    # with codecs.open('./userGraph-Quote.txt', 'w', 'utf-8') as fw:
    #     for (edge, Count) in userGraphEdge:
    #         fw.write(str(edge[0]) + ' ' + str(edge[1]) + ' ' + edge[2] + ' ' + str(Count) + '\n')

    return userGraphEdge

def mergeGraph(postGraph, quoteGraph):

    postGraph.extend(quoteGraph)
    userGraphEdge = collections.Counter(postGraph).most_common()
    userGraphEdge = sorted(userGraphEdge, key=lambda x: x[0][0])
    with codecs.open('./userGraph-Merge.txt', 'w', 'utf-8') as fw:
        for (edge, Count) in userGraphEdge:
            fw.write(str(edge[0]) + ' ' + str(edge[1]) + ' ' + str(Count) + '\n')

def getAuthor2Stance(db):

    results = queryDatabase(db, 'mturk_author_stance')
    results = dataResolution(results, 'mturk_author_stance')
    author2Stance = dict()
    for index, row in enumerate(results):
        authorID = row[1]
        if authorID in author2Stance.keys():
            author2Stance[authorID] = (author2Stance[authorID][0] + row[2], author2Stance[authorID][1] + row[3], author2Stance[authorID][2] + row[4])
        else:
            author2Stance[authorID] = (row[2], row[3], row[4])

    for authorID in author2Stance.keys():
        for index, value in enumerate(author2Stance[authorID]):
            if value == max(author2Stance[authorID]):
                author2Stance[authorID] = index
                break
    return author2Stance # author@Topic->Stance
    # author2Stance = sorted(author2Stance.items(), key=lambda x: x[0])
    # with codecs.open('./author2Stance.txt', 'w', 'utf-8') as fw:
    #     for (authorID, stance) in author2Stance: # author@Topic Stance
    #         fw.write(str(authorID) + ' ' + str(stance) + '\n')

def getDiscussion2Topic(db):

    results = queryDatabase(db, 'discussion_topic')
    results = dataResolution(results, 'discussion_topic')
    discussion2Topic = dict()
    for index, row in enumerate(results):
        discussionID = row[0]
        topicID = row[1]
        discussion2Topic[discussionID] = topicID
    return discussion2Topic

def getDiscussionID2Title(db):
    results = queryDatabase(db, 'discussion')
    results = dataResolution(results, 'discussion')
    discussionID2Title = dict()
    for index, row in enumerate(results):
        discussionID = row[0]
        title = row[1]
        discussionID2Title[discussionID] = title
    return discussionID2Title

def getTextWithStance(db):

    tableName = 'post'
    results = queryDatabase(db, tableName)
    results = dataResolution(results, tableName)

    discussion2Topic = getDiscussion2Topic(db)
    author2Stance = getAuthor2Stance(db)
    textWithStance = []
    for index, row in enumerate(results):
        discussionID = row[0]
        if discussionID in discussion2Topic.keys():
            topicID = discussion2Topic[discussionID]
            authorID = str(row[2]) + '@' + topicID
            if authorID in author2Stance.keys():
                postID = row[1]
                textID = row[4]
                stance = author2Stance[authorID]
                textWithStance.append((discussionID, postID, authorID, textID, stance))
    print 'Finish Processing!'
    with codecs.open('./textWithStance.txt', 'w', 'utf-8') as fw:
        for (discussionID, postID, authorID, textID, stance) in textWithStance:
            fw.write(str(discussionID) + ' ' + str(postID) + ' ' + str(authorID) + ' ' + str(textID) + ' ' + str(stance) + '\n')

def getPostSequence(db):
    tableName = 'post'
    results = queryDatabase(db, tableName)
    results = dataResolution(results, tableName)

    parentChild = []
    # count = 0
    for index, row in enumerate(results):
        postID = row[1]
        parentPostID = row[3]
        if parentPostID != 'None':
            parentChild.append((parentPostID, postID))
        else:
            if len(parentChild) != 0:
                DG = nx.DiGraph()
                DG.add_edges_from(parentChild)
                nodeList = [node for node in list(DG.nodes()) if node != 'None']
                pathList = []
                for node in nodeList:
                    for path in nx.all_simple_paths(DG, source='None', target=node):
                        path = path[1:]
                        # sequenceLength = 5
                        # if len(path) < 5:
                        #     tmp = ['null'] * (sequenceLength - len(path))
                        #     tmp.extend(path)
                        #     path = tmp
                        #     count += 1
                        # else:
                        #     path = path[-sequenceLength:]
                        pathList.append(path)
                with codecs.open('./postSequence.txt', 'a', 'utf-8') as fw:
                    for path in pathList:
                        fw.write(' '.join(path) + '\n')
            parentChild = []
            parentChild.append((parentPostID, postID))
    # print count

    # print 'Finish Processing!'
    # with codecs.open('./parent2ChildPost.txt', 'w', 'utf-8') as fw:
    #     for (parent, child) in parentChild:
    #         fw.write(str(parent) + ' ' + str(child) + '\n')


def closeDatabase(db):
    db.close()

if __name__ == '__main__':
    db = connectDatabase()
    # getUserPostGraph(db, 'post')
    # getUserQuoteGraph(db, 'quote')
    # mergeGraph(getUserPostGraph(db, 'post'), getUserQuoteGraph(db, 'quote'))
    # getAuthor2Stance(db)
    # getTextWithStance(db)
    getPostSequence(db)
    closeDatabase(db)
