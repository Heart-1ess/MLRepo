import jieba
import os


def loadDict(fileName, score):
    wordDict = {}
    with open(fileName, encoding='utf-8') as fin:
        for line in fin:
            word = line.strip()
            wordDict[word] = score
    return wordDict


def appendDict(wordDict, fileName, score):
    with open(fileName, encoding='utf-8') as fin:
        for line in fin:
            word = line.strip()
            wordDict[word] = score


def loadExtentDict(fileName, level):
    extentDict = {}
    for i in range(level):
        with open(fileName + ".txt", encoding='utf-8') as fin:
            for line in fin:
                word = line.strip()
                extentDict[word] = i + 1
    return extentDict


def getScore(content, emotional_dict_path):
    postDict = loadDict(os.path.join(emotional_dict_path, "积极情感词语.txt"), 1)  # 积极情感词典
    negDict = loadDict(os.path.join(emotional_dict_path, "消极情感词语.txt"), -1)  # 消极情感词典
    inverseDict = loadDict(os.path.join(emotional_dict_path, "否定词语.txt"), -1)  # 否定词词典
    extentDict = loadExtentDict(os.path.join(emotional_dict_path, "程度级别词语"), 6)
    # punc = loadDict(u"sentimentDict/标点符号.txt", 1)
    # exclamation = {"!": 2, "！": 2}

    # words = jieba.cut(content)
    wordList = list(content)
    # print(wordList)

    totalScore = 0  # 记录最终情感得分
    lastWordPos = 0  # 记录情感词的位置
    # lastPuncPos = 0  # 记录标点符号的位置
    i = 0  # 记录扫描到的词的位置

    for word in wordList:
    #     if word in punc:
    #         lastPuncPos = i

        if word in postDict:
            # if lastWordPos > lastPuncPos:
            start = lastWordPos
            # else:
            #     start = lastPuncPos

            score = 1
            for word_before in wordList[start:i]:
                if word_before in extentDict:
                    score = score * extentDict[word_before]
                if word_before in inverseDict:
                    score = score * -1
            # for word_after in wordList[i + 1:]:
            #     if word_after in punc:
            #         if word_after in exclamation:
            #             score = score + 2
            #         else:
            #             break
            lastWordPos = i
            totalScore += score
        elif word in negDict:
            # if lastWordPos > lastPuncPos:
            start = lastWordPos
            # else:
            #     start = lastPuncPos
            score = -1
            for word_before in wordList[start:i]:
                if word_before in extentDict:
                    score = score * extentDict[word_before]
                if word_before in inverseDict:
                    score = score * -1
            # for word_after in wordList[i + 1:]:
            #     if word_after in punc:
            #         if word_after in exclamation:
            #             score = score - 2
            #         else:
            #             break
            lastWordPos = i
            totalScore += score
        i = i + 1

    return totalScore
