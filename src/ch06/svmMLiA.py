"""
Create by 2018-05-22

@author: 代码来源于网上，书上代码有误
"""
from numpy import *


# 加载数据集
def loadDataSet(fileName):
    # 数据矩阵
    dataMat = []

    # 标签向量
    labelMat = []

    fr = open(fileName)
    for line in fr.readlines():
        # strip()表示删除空白符，split()表示分割
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])  # 1.0表示x0
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat

#功能：在(0, m)的区间范围内随机选择一个除i以外的整数
def selectJrand(i, m):
    j = i
    while (j == i):
        j = int (random.uniform(0, m))
    return j

#功能：保证aj在区间[L, H]里面
#输入：要调整的数aj，区间上界H，区间下界L
#输出：调整好的数aj
def clipAlpha(aj, H, L):
    if aj > H:#aj大于H
        aj = H
    if L > aj:#aj小于L
        aj = L
    return aj

#功能：简化版SMO算法
#输入：数据矩阵dataMatIn，标签向量classLabels，常数C，容错率toler，最大迭代次数maxIter
#输出：超平面位移项b，拉格朗日乘子alpha
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    b = 0
    m, n = shape(dataMatrix)#数据矩阵行数和列数，表示训练样本个数和特征值个数
    alphas = mat(zeros((m, 1)))#m*1阶矩阵
    iter = 0
    while (iter < maxIter):#循环直到超出最大迭代次数
        alphaPairsChanged = 0
        for i in range(m):
            #主窗口输入numpy.info(numpy.multiply)
            #推导见《机器学习》（周志华）公式6.12
            fXi = float(multiply(alphas, labelMat).T * \
                        (dataMatrix * dataMatrix[i, :].T)) + b
            Ei = fXi - float(labelMat[i])#误差
            #误差很大，可以对该数据实例所对应的alpha值进行优化
            if ((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or \
                    ((labelMat[i] * Ei > toler) and (alphas[i] > 0)):
                #在(0, m)的区间范围内随机选择一个除i以外的整数，即随机选择第二个alpha
                j = selectJrand(i, m)
                #求变量alphaJ对应的误差
                fXj = float(multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j, :].T)) + b
                Ej = fXj - float(labelMat[j])
                #不能直接 alphaIold = alphas[i]，否则alphas[i]和alphaIold指向的都是同一内存空间
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                #接下来需要看Plata的论文，待以后看论文后再重读此章
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    print("L == H")
                    continue
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - \
                    dataMatrix[i, :] * dataMatrix[i, :].T - \
                    dataMatrix[j, :] * dataMatrix[j, :].T
                if eta >= 0:
                    print("eta >= 0")
                    continue
                alphas[j] -= labelMat[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)#保证alphas[j]在区间[L, H]里面
                #检查alpha[j]是否有较大改变，如果没有则退出for循环
                if (abs(alphas[j] - alphaJold) < 0.00001):
                    print("j not moving enough")
                    continue
                #labelMat[i]与labelMat[j]绝对值均为1，则alphas[i]与alphas[j]改变大小一样
                #保证alpha[i] * labelMal[i] + alpha[j] * labelMal[j] = c
                #即Delta(alpha[i]) * labelMal[i] + Delta(alpha[j]) * labelMal[j] = 0
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * \
                              dataMatrix[i, :] * dataMatrix[i, :].T - \
                    labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[i, :] * dataMatrix[j, :].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * \
                    dataMatrix[i, :] * dataMatrix[j, :].T - \
                    labelMat[j] * (alphas[j] - alphaJold) * \
                    dataMatrix[j, :] * dataMatrix[j, :].T
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairsChanged += 1
                print("iter: {} i: {}, pairs changed {}".format(iter, i, alphaPairsChanged))
        #不是1，这个迭代思路比较巧妙，是以最后一次迭代没有误差为迭代结束条件
        if (alphaPairsChanged == 0):
            iter += 1
        else:
            iter = 0
        print("iteration number: {}".format(iter))
    return b, alphas

#功能：建立数据结构用于保存所有的重要值
#输入：无
#输出：无
class optStruct:
    def __init__(self, dataMatIn, classLabes, C, toler, kTup):#__init__作用是初始化已实例化后的对象
        self.X = dataMatIn
        self.labelMat = classLabes
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]#dataMatIn行数
        self.alphas = mat(zeros((self.m, 1)))#(self.m, 1)是一个元组，下同
        self.b = 0
        # m*2误差矩阵，第一列为eCache是否有效的标志位，第二列是
        self.eCache = mat(zeros((self.m, 2)))
        self.K = mat(zeros((self.m, self.m)))#建立核矩阵
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)

#功能：计算第k个alpha的误差值
#输入：数据集，alpha数
#输出：误差值
def calcEk(oS, k):
    #fXk = float(multiply(oS.alphas, oS.labelMat).T * (oS.X * oS.X[k, :].T)) + oS.b
    fXk = float(multiply(oS.alphas, oS.labelMat).T * oS.K[:, k] + oS.b)
    Ek = fXk -float(oS.labelMat[k])
    return Ek

#功能：选择有最大步长的alpha值
#输入：第一个alpha值，数据集，第一个alpha对应的误差值
#输出：第二个alpha值和对应的误差值
def selectJ(i, oS, Ei):
    maxK = -1#最大步长对应j值
    maxDeltaE = 0#最大步长
    Ej = 0#最大误差值
    oS.eCache[i] = [1, Ei]#使i值对应的标志位永远有效
    # .A表示将矩阵转化为列表，nonzero()返回值不为零的元素的下标，[0]表示第一列
    #该行表示读取eCache第一列即是否有效标志位的下标
    validEcacheList = nonzero(oS.eCache[:, 0].A)[0]
    if (len(validEcacheList)) > 1:#大于等于2个
        for k in validEcacheList:#在有效标志位中寻找
            if k == i:
                continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):#找到最大步长
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        j = selectJrand(i, oS.m)#随机选择一个j值
        Ej = calcEk(oS, j)#j值对应的误差值Ej
    return j, Ej

#功能：更新第k个alpha的误差值至数据结构中
#输入：数据集，alpha数
#输出：无
def updataEk(oS, k):
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]

#功能：完整版Platt SMO内循环，在数据结构中更新alpha数
#输入：alpha数，数据集
#输出：是否在数据结构中成功更新alpha数，成功返回1，不成功返回0
def innerL(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or \
            ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        j, Ej = selectJ(i, oS, Ei)#选择有最大步长的alpha值
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            print("L == H")
            return 0
        #eta = 2.0 * oS.X[i, :] * oS.X[j, :].T - oS.X[i, :] * oS.X[i, :].T - \
           #oS.X[j, :] * oS.X[j, :].T
        eta = 2.0 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]
        if eta >= 0:
            print("eta >= 0")
            return 0
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updataEk(oS, j)
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            print("j not moving enough")
            return 0
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * \
                        (alphaJold - oS.alphas[j])
        updataEk(oS, i)
        #b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[i, :].T - \
            #oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.X[i, :] * oS.X[j, :].T
        #b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[j, :].T - \
             #oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.X[j, :] * oS.X[j, :].T
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, i] - \
             oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[i, j]
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, j] - \
             oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[j, j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0

#功能：完整版Platt SMO外循环
#输入：数据矩阵dataMatIn，标签向量classLabels，常数C，容错率toler，最大迭代次数maxIter
#输出：超平面位移项b，拉格朗日乘子alpha
def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):
    oS = optStruct(mat(dataMatIn), mat(classLabels).transpose(), C, toler, kTup)#建立数据结构
    iter = 0#一次迭代完成一次循环过程
    entireSet = True
    alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:#判断1
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)
                print("fullSet, iter: {} i: {}, pairs changed {}".format(iter, i, alphaPairsChanged))
            iter += 1
        else:
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print("non-bound, iter: {} i: {}, pairs changed {}".format(iter, i, alphaPairsChanged))
            iter += 1
        # 执行判断1时，如果entireSet = True，表示遍历整个集合，alphaPairsChanged = 0，表示未对任意alpha对进行修改
        if entireSet:
            entireSet = False
        #执行判断1时，第一次迭代遍历整个集合，之后就只遍历非边界值，除非遍历非边界值发现没有任意alpha对进行修改，遍历整个集合
        elif (alphaPairsChanged == 0):
            entireSet = True
        print("iteration number: {}".format(iter))
    return oS.b, oS.alphas


#功能：计算超平面法向量
#输入：拉格朗日乘子alpha,数据矩阵dataArr，标签向量classLabels
#输出：超平面法向量
def calcWs(alphas, dataArr, classLabels):
    X = mat(dataArr)
    labelMat = mat(classLabels).transpose()
    m, n = shape(X)
    w = zeros((n, 1))
    for i in range(m):
        w += multiply(alphas[i] * labelMat[i], X[i, :].T)
    return w

#功能：核转换函数
#输入：数据集，第i行数据集，核函数名称
#输出：对应的核函数
def kernelTrans(X, A, kTup):
    m, n = shape(X)#数据集行数和列数
    K = mat(zeros((m, 1)))
    if kTup[0] == 'lin':#如果核函数是线性核
        K = X * A.T
    elif kTup[0] == 'rbf':#如果核函数是高斯核，即径向基核函数
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T
        K = exp(K / (-1 * kTup[1] ** 2))
    else:#出现不能识别的核函数
        #通过raise显式地引发异常
        raise NameError('Houston We Have a Problem -- \
                        That Kernel is not recognized')
    return K

#功能：利用核函数进行分类的径向基测试函数
#输入：高斯核带宽的平方值
#输出：无
def testRbf(k1 = 1.3):
    dataArr, labelArr = loadDataSet('testSetRBF.txt')
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1))
    datMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A >0)[0]#支持向量的下标
    sVs = datMat[svInd]#支持向量
    labelSV = labelMat[svInd]#支持向量的类别标签
    print("there are {} Support Vectors".format(shape(sVs)[0]))
    m, n = shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], ('rbf', k1))
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b#得预测值
        if sign(predict) != sign(labelArr[i]):
            errorCount += 1
    print("the training error rate is {}".format(float(errorCount) / m))
    dataArr, labelArr = loadDataSet('testSetRBF2.txt')
    errorCount = 0
    datMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    m, n = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], ('rbf', k1))
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b#得验证集预测值
        if sign(predict) != sign(labelArr[i]):
            errorCount += 1
    print("the test error rate is: {}".format(float(errorCount) / m))


#功能：图像矩阵转化为m*1矩阵
#输入：文件名
#输出：m*1矩阵
def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int (lineStr[j])
    return returnVect

#功能：将图像内容导入矩阵
#输入：一级子目录
#输出：图像矩阵，图像标签向量
def loadImages(dirName):
    from os import listdir
    hwLabels = []
    trainingFileList = listdir(dirName)#dirName文件夹下的文件名列表
    m = len(trainingFileList)#dirName文件夹下的文件数目
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]#文件名
        fileStr = fileNameStr.split('.')[0]#去掉.txt的文件名
        classNumStr = int(fileStr.split('_')[0])#要识别的数字
        if classNumStr == 9:#数字9
            hwLabels.append(-1)
        else:#数字1
            hwLabels.append(1)
        trainingMat[i, :] = img2vector('%s/%s' % (dirName, fileNameStr))
    return trainingMat, hwLabels

def testDigits(kTup = ('rbf', 10)):
    dataArr, labelArr = loadImages('trainingDigits')
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, kTup)
    datMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A > 0)[0]  # 支持向量的下标
    sVs = datMat[svInd]  # 支持向量
    labelSV = labelMat[svInd]  # 支持向量的类别标签
    print("there are {} Support Vectors".format(shape(sVs)[0]))
    m, n = shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], kTup)
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b  # 得预测值
        if sign(predict) != sign(labelArr[i]):
            errorCount += 1
    print("the training error rate is {}".format(float(errorCount) / m))
    dataArr, labelArr = loadImages('testDigits')
    errorCount = 0
    datMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    m, n = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], kTup)
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b  # 得验证集预测值
        if sign(predict) != sign(labelArr[i]):
            errorCount += 1
    print("the test error rate is: {}".format(float(errorCount) / m))