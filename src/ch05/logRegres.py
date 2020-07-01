"""
Create by 2018-05-21

@author: Shiyipaisizuo
"""
import matplotlib.pyplot as plt
from numpy import *
import math
from numpy.ma import array


# 加载数据
def load_data_set():
    data_matrix = []
    label_matrix = []

    fr = open('testSet.txt')

    for line in fr.readlines():
        line_array = line.strip().split()
        data_matrix.append([1.0, float(line_array[0]), float(line_array[1])])
        label_matrix.append(int(line_array[2]))

    return data_matrix, label_matrix


def sigmoid(X):

    return (1.0 / (1+math.exp(-X))) + 1e-8


# Logistic回归梯度上升优化算法
def grad_ascent(data_matrix, class_label):

    # 将数据转化为numpy矩阵
    data_matrix = mat(data_matrix)
    label_matrix = mat(class_label).transpose()
    m, n = shape(data_matrix)

    alpha = 0.01
    max_cycles = 500
    weights = ones((n, 1))

    # 矩阵之间做乘法
    for k in range(max_cycles):
        h = sigmoid(data_matrix*weights)
        error = (label_matrix - h)
        weights = weights + alpha * data_matrix.transpose() * error

    return weights


# 画图
def plot_best_fit(weights):
    data_matrix, label_matrix = load_data_set()
    data_array = array(data_matrix)
    n = shape(data_array)[0]
    x1 = []
    y1 = []
    x2 = []
    y2 = []

    for i in range(n):
        if int(label_matrix[i]) == 1:
            x1.append(data_array[i, 1])
            x2.append(data_array[i, 2])
        else:
            y1.append(data_array[i, 1])
            y2.append(data_array[i, 2])

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter(x1, x2, s=30, c='red', marker='s')
    ax.scatter(y1, y2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)

    # 最佳似合直线
    y = (-weights[0] - weights[1]*x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


# 随机梯度上升算法
def stroc_grad_ascent0(data_matrix, class_labels):
    m, n = shape(data_matrix)
    alpha = 0.01
    weights = ones(n)

    for i in range(m):
        h = sigmoid(sum(data_matrix[i] * weights))
        error = class_labels[i] - h
        weights = weights + alpha * error * data_matrix[i]

    return weights


# 改进后的随机梯度上升算法
def stoc_grad_ascent1(data_matrix, class_labels, num_iter=150):
    m, n = shape(data_matrix)
    weights = ones(n)

    for j in range(num_iter):
        data_index = list(range(m))
        for i in range(m):

            # 每次来调整alpha的值
            alpha = 4 / (1.0 + j + i) + 0.0001

            # 随机选取更新
            rand_index = int(random.uniform(0, len(data_index)))
            h = sigmoid(sum(data_matrix[rand_index] * weights))
            error = class_labels[rand_index] - h
            weights = weights + alpha * error * data_matrix[rand_index]
            del (data_index[rand_index])

    return weights


# Logistic回归分类函数
def classify_vector(inx, weights):
    prob = sigmoid(sum(inx * weights))

    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def colic_test():
    frtrain = open('horseColicTraining.txt');
    frtest = open('horseColicTest.txt')
    train_set = []
    train_labels = []

    for line in frtrain.readlines():
        curr_line = line.strip().split('\t')
        line_array = []
        for i in range(21):
            line_array.append(float(curr_line[i]))
        train_set.append(line_array)
        train_labels.append(float(curr_line[21]))

    train_weights = stoc_grad_ascent1(array(train_set), train_labels, 1000)
    error_count = 0
    num_test_vector = 0.0

    for line in frtest.readlines():
        num_test_vector += 1.0
        curr_line = line.strip().split('\t')
        line_array = []
        for i in range(21):
            line_array.append(float(curr_line[i]))
        if int(classify_vector(array(line_array), train_weights)) != int(curr_line[21]):
            error_count += 1

    error_rate = (float(error_count) / num_test_vector)
    print(f"the error rate of this test is: {error_rate}")

    return error_rate


# 多重测试
def mul_test():
    num_tests = 10
    error_sum = 0.0

    for k in range(num_tests):
        error_sum += colic_test()

    print(f"after {num_tests} iterations the average error rate is: {error_sum / float(num_tests)}")
