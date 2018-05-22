'''
Created by 2018-05-22

@author: Shiyipaisizuo
'''
from numpy import *
from time import sleep
import json
import urllib.request


# 自适应数据加载函数，该函数能够自动检测出特征的数目，假定最后一个特征是类别标签
def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) - 1 # 获得所有标签
    dataMat = []; labelMat = []
    fr = open(fileName)

    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))

    return dataMat,labelMat


# 计算最佳似合曲线
def standRegres(xArr,yArr):
    xMat = mat(xArr); yMat = mat(yArr).T
    xTx = xMat.T*xMat

    if linalg.det(xTx) == 0.0:  # 判断行列式是否为零，为零则计算逆矩阵出现错误
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T*yMat)

    return ws


# 局部加权线性回归函数
def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat = mat(xArr); yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m)))  # 创建对角矩阵

    for j in range(m):
        diffMat = testPoint - xMat[j,:]     # 权重值大小以指数级衰减
        weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx = xMat.T * (weights * xMat)

    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))

    return testPoint * ws


# 为数据集中的每个点调用lwlr()，有助于求解k的大小
def lwlrTest(testArr,xArr,yArr,k=1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)

    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)

    return yHat


# 画出测试(lwlr)图
def lwlrTestPlot(xArr,yArr,k=1.0):
    yHat = zeros(shape(yArr))
    xCopy = mat(xArr)
    xCopy.sort(0)
    for i in range(shape(xArr)[0]):
        yHat[i] = lwlr(xCopy[i],xArr,yArr,k)

    return yHat,xCopy


# 使用平方误差，分析预测误差的大小
def rssError(yArr,yHatArr):

    return ((yArr-yHatArr)**2).sum()

def ridgeRegres(xMat,yMat,lam=0.2):
    xTx = xMat.T*xMat
    denom = xTx + eye(shape(xMat)[1])*lam
    if linalg.det(denom) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = denom.I * (xMat.T*yMat)

    return ws


# 数据的标准化过程，具体过程就是所有特征都减去各自的均值并处理方差
def ridgeTest(xArr,yArr):
    xMat = mat(xArr); yMat=mat(yArr).T
    yMean = mean(yMat,0)
    yMat = yMat - yMean  # 要消去X0，取Y的平均值。
    xMeans = mean(xMat,0)  # calc的意思是减去它。
    xVar = var(xMat,0)  # 然后再除以Xi的calc方差。
    xMat = (xMat - xMeans)/xVar
    numTestPts = 30
    wMat = zeros((numTestPts,shape(xMat)[1]))

    for i in range(numTestPts):
        ws = ridgeRegres(xMat,yMat,exp(i-10))
        wMat[i,:]=ws.T

    return wMat


# 按照均值为0，方差为1，对列进行标准化处理
def regularize(xMat):
    inMat = xMat.copy()
    inMeans = mean(inMat,0)  # calc的意思是减去它。
    inVar = var(inMat,0)  # 然后再除以Xi的calc方差。
    inMat = (inMat - inMeans)/inVar
    return inMat


# 前向逐步线性回归
def stageWise(xArr,yArr,eps=0.01,numIt=100):
    xMat = mat(xArr); yMat=mat(yArr).T
    yMean = mean(yMat,0)
    yMat = yMat - yMean  # 也可以调整ys，但会得到更小的coef
    xMat = regularize(xMat)
    m,n=shape(xMat)
    ws = zeros((n,1)); wsTest = ws.copy(); wsMax = ws.copy()

    for i in range(numIt):
        print(ws.T)
        lowestError = inf; 
        for j in range(n):
            for sign in [-1,1]:
                wsTest = ws.copy()
                wsTest[j] += eps*sign
                yTest = xMat*wsTest
                rssE = rssError(yMat.A,yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()


# 查找数据集
def searchForSet(retX, retY, setNum, yr, numPce, origPrc):
    sleep(10)
    myAPIstr = 'AIzaSyD2cR2KFyx12hXu6PFU-wrWot3NXvko8vY'
    searchURL = 'https://www.googleapis.com/shopping/search/v1/public/products?key=%s&country=US&q=lego+%d&alt=json' % (myAPIstr, setNum)
    pg = urllib.request.urlopen(searchURL)
    retDict = json.loads(pg.read())
    for i in range(len(retDict['items'])):
        try:
            currItem = retDict['items'][i]
            if currItem['product']['condition'] == 'new':
                newFlag = 1
            else: newFlag = 0
            listOfInv = currItem['product']['inventories']
            for item in listOfInv:
                sellingPrice = item['price']
                if  sellingPrice > origPrc * 0.5:
                    print("{}\t{}\t{}\t{}\t{}".format(yr,numPce,newFlag,origPrc, sellingPrice))
                    retX.append([yr, numPce, newFlag, origPrc])
                    retY.append(sellingPrice)
        except:
            print("problem with item {}".format(i))


def setDataCollect(retX, retY):
    searchForSet(retX, retY, 8288, 2006, 800, 49.99)
    searchForSet(retX, retY, 10030, 2002, 3096, 269.99)
    searchForSet(retX, retY, 10179, 2007, 5195, 499.99)
    searchForSet(retX, retY, 10181, 2007, 3428, 199.99)
    searchForSet(retX, retY, 10189, 2008, 5922, 299.99)
    searchForSet(retX, retY, 10196, 2009, 3263, 249.99)


# 交叉验证测试岭回归
def crossValidation(xArr,yArr,numVal=10):
    m = len(yArr)                           
    indexList = range(m)
    errorMat = zeros((numVal,30))  # 创建错误mat 30列numVal行。

    for i in range(numVal):
        # 创建训练集合测试集容器
        trainX=[]; trainY=[]
        testX = []; testY = []
        random.shuffle(indexList)
        for j in range(m):  # 根据索引列表中90%的值创建培训集。
            if j < m*0.9: 
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
        wMat = ridgeTest(trainX,trainY)  # 从测试集上得到30个权向量。
        for k in range(30):  # 对所有的测试集估计
            matTestX = mat(testX); matTrainX=mat(trainX)
            meanTrain = mean(matTrainX,0)
            varTrain = var(matTrainX,0)
            matTestX = (matTestX-meanTrain)/varTrain  # 使用测试参数进行规范化测试。
            yEst = matTestX * mat(wMat[k,:]).T + mean(trainY)  # 测试ridge结果和存储。
            errorMat[i,k]=rssError(yEst.T.A,array(testY))

    meanErrors = mean(errorMat,0)  # 测试不同权向量的calc avg性能。
    minMean = float(min(meanErrors))
    bestWeights = wMat[nonzero(meanErrors==minMean)]
    # 可以不定期的得到模型
    xMat = mat(xArr); yMat=mat(yArr).T
    meanX = mean(xMat,0); varX = var(xMat,0)
    unReg = bestWeights/varX
    print("the best model from Ridge Regression is:\n",unReg)
    print("with constant term: ",-1*sum(multiply(meanX,unReg)) + mean(yMat))