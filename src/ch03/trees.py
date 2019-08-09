"""
Create by 2018-05-18

@author: Shiyipaisizuo
"""
import math
import operator
import pickle


def create_dataset():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


def calc_shannon_ent(dataSet):
    num_entries = len(dataSet)
    label_counts = {}
    # 为所有可能分类创建字典
    for featVec in dataSet:
        current_label = featVec[-1]
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts[current_label] += 1
    shannoent = 0.0

    # 以二为底求对数
    for key in label_counts:
        prob = float(label_counts[key])/num_entries
        shannoent -= prob * math.log(prob, 2)
    return shannoent


def split_dataset(dataSet, axis, value):
    # 创建新的list对象
    retdataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            # 抽取
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retdataSet.append(reducedFeatVec)

    return retdataSet


def choose_best(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calc_shannon_ent(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1

    # 创建唯一分类标签
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueValis = set(featList)
        newEntropy = 0.0

        # 计划每种划分的信息墒
        for value in uniqueValis:
            subDataSet = split_dataset(dataSet, i ,value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calc_shannon_ent(subDataSet)
            infoGain = baseEntropy - newEntropy

            # 计算最好的增益墒
            if infoGain > bestInfoGain:
                bestInfoGain = infoGain
                bestFeature = i

    return bestFeature


def majoritycnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

    return sortedClassCount


def create_tree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):

        # 停止分类直至所有类别相等
        return classList[0]
    if len(dataSet[0]) == 1:

        # 停止分割直至没有更多特征
        return majoritycnt(classList)
    bestfaet = choose_best(dataSet)
    bestfaetlabel = labels[bestfaet]
    mytree = {bestfaetlabel:{}}
    del(labels[bestfaet])

    # 得到包含所有属性的列表
    featvalues = [example[bestfaet] for example in dataSet]
    uniquevalues = set(featvalues)
    for value in uniquevalues:
        sublables = labels[:]
        mytree[bestfaetlabel][value] = create_tree(split_dataset(dataSet, bestfaet, value), sublables)

    return mytree


def classify(inputtree, featlabels, testvec):
    firststr = inputtree.keys()[0]
    seconddict = inputtree[firststr]
    featindex = featlabels.index(firststr)
    key = testvec[featindex]
    valueoffeat = seconddict[key]
    if isinstance(valueoffeat, dict):
        classlabel = classify(valueoffeat, featlabels, testvec)
    else:
        classlabel = valueoffeat
    return classlabel


def store_tree(inputtree, filename):
    fw = open(filename, 'w')
    pickle.dump(inputtree, fw)
    fw.close()


def grab_tree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)
