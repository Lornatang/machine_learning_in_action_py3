import math


def calcshannonent(dataSet):
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
