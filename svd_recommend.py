from numpy import *
from numpy import linalg as la


def ecludSim(inA, inB):
    return 1.0/(1.0 + la.norm(inA - inB))


def pearsSim(inA, inB):
    # 检查是否存在3个或更多的点不存在，该函数返回1.0，此时两个向量完全相关。
    if len(inA) < 3:
        return 1.0
    return 0.5 + 0.5 * corrcoef(inA, inB, rowvar=False)[0][1]


def cosSim(inA, inB):
    return 0.5 + 0.5*(float(inA.T*inB) / la.norm(inA)*la.norm(inB))


def loadExData3():
    return[[2, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5],
           [0, 0, 0, 0, 0, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 3, 0, 0, 2, 2, 0, 0],
           [5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0],
           [4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 5],
           [0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0],
           [0, 0, 0, 3, 0, 0, 0, 0, 4, 5, 0],
           [1, 1, 2, 1, 1, 2, 1, 0, 4, 5, 0]]


'''基于物品相似度的推荐引擎
descripte：计算某用户未评分物品中，以对该物品和其他物品评分的用户的物品相似度，然后进行综合评分
dataMat         训练数据集
user            用户编号
simMeas         相似度计算方法
item            未评分的物品编号
Returns: ratSimTotal/simTotal  评分（0～5之间的值）
'''


def standEst(dataMat, user, simMeas, item):
    # 得到数据集中的物品数目
    n = shape(dataMat)[1]
    # 初始化两个评分值
    simTotal = 0.0 ; ratSimTotal = 0.0
    # 遍历行中的每个物品（对用户评过分的物品遍历，并与其他物品进行比较）
    for j in range(n):
        userRating = dataMat[user, j]
        # 如果某个物品的评分值为0，则跳过这个物品
        if userRating == 0: # 终止循环
            continue
        # 寻找两个都评级的物品,变量overLap 给出两个物品中已被评分的元素索引ID
        # logical_and 计算x1和x2元素的真值。
        # print(dataMat[:, item].T.A, ':',dataMat[:, j].T.A )
        overLap = nonzero(logical_and(dataMat[:, item].A > 0, dataMat[:, j].A > 0))[0]
        # 如果相似度为0，则两着没有任何重合元素，终止本次循环
        if len(overLap) == 0:
            similarity = 0
        # 如果存在重合的物品，则基于这些重合物重新计算相似度。
        else:
            similarity = simMeas(dataMat[overLap, item], dataMat[overLap, j])
        # 相似度会不断累加，每次计算时还考虑相似度和当前用户评分的乘积
        # similarity  用户相似度，   userRating 用户评分
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    # 通过除以所有的评分总和，对上述相似度评分的乘积进行归一化，使得最后评分在0~5之间，这些评分用来对预测值进行排序
    else:
        return ratSimTotal/simTotal


'''分析 Sigma 的长度取值
根据自己的业务情况，就行处理，设置对应的 Singma 次数
通常保留矩阵 80% ～ 90% 的能量，就可以得到重要的特征并取出噪声。
'''


def analyse_data(Sigma, loopNum=20):
    # 总方差的集合（总能量值）
    Sig2 = Sigma**2
    SigmaSum = sum(Sig2)
    for i in range(loopNum):
        SigmaI = sum(Sig2[:i+1])
        print('主成分：%s, 方差占比：%s%%' % (format(i+1, '2.0f'), format(SigmaI/SigmaSum*100, '.2f')))


'''基于SVD的评分估计
Args:
    dataMat         训练数据集
    user            用户编号
    simMeas         相似度计算方法
    item            未评分的物品编号
Returns:
    ratSimTotal/simTotal     评分（0～5之间的值）
'''


def svdEst(dataMat, user, simMeas, item):
    # 物品数目
    n = shape(dataMat)[1]
    # 对数据集进行SVD分解
    simTotal = 0.0 ;  ratSimTotal = 0.0
    # 奇异值分解,只利用90%能量值的奇异值，奇异值以NumPy数组形式保存
    U, Sigma, VT = la.svd(dataMat)
    # 分析 Sigma 的长度取值
    # analyse_data(Sigma, 20)

    # 如果要进行矩阵运算，就必须要用这些奇异值构建出一个对角矩阵
    Sig4 = mat(eye(4) * Sigma[: 4]) # eye对角矩阵
    # 利用U矩阵将物品转换到低维空间中，构建转换后的物品(物品+4个主要的特征)
    xformedItems = dataMat.T * U[:, :4] * Sig4.I # I 逆矩阵
    # print('dataMat', shape(dataMat))
    # print('U[:, :4]', shape(U[:, :4]))
    # print('Sig4.I', shape(Sig4.I))
    # print('VT[:4, :]', shape(VT[:4, :]))
    # print('xformedItems', shape(xformedItems))

    # 对于给定的用户，for循环在用户对应行的元素上进行遍历
    # 和standEst()函数的for循环一样，这里相似度计算在低维空间下进行的。
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0 or j == item:
            continue
        # 相似度的计算方法也会作为一个参数传递给该函数
        similarity = simMeas(xformedItems[item, :].T, xformedItems[j, :].T)
        # for 循环中加入了一条print语句，以便了解相似度计算的进展情况。如果觉得累赘，可以去掉
        # print('the %d and %d similarity is: %f' % (item, j, similarity))
        # 对相似度不断累加求和
        simTotal += similarity
        # 对相似度及对应评分值的乘积求和
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        # 计算估计评分
        return ratSimTotal/simTotal


'''recommend函数推荐引擎，默认调用standEst函数，产生最高的N个推荐结果
Args:
    dataMat         训练数据集
    user            用户编号
    simMeas         相似度计算方法
    estMethod       使用的推荐算法
Returns:  返回最终 N 个推荐结果
'''


def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
    # 寻找未评级的物品,对给定的用户建立一个未评分的物品列表
    unratedItems = nonzero(dataMat[user, :].A == 0)[1] # .A: 矩阵转数组
    # 如果不存在未评分物品，那么就退出函数
    if len(unratedItems) == 0:
        return 'you rated everything'
    # 物品的编号和评分值
    itemScores = []
    # 在未评分物品上进行循环
    for item in unratedItems:
        # 获取 item 该物品的评分
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        itemScores.append((item, estimatedScore))
    # 按照评分得分 进行逆排序，获取前N个未评级物品进行推荐
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[: N]


# 计算相似度的方法
myMat = mat(loadExData3())
# 计算相似度的第一种方式
# print(recommend(myMat, 1, estMethod=svdEst))
# 计算相似度的第二种方式
print(recommend(myMat, 1, estMethod=svdEst, simMeas=pearsSim))

# 默认推荐（菜馆菜肴推荐示例）
print(recommend(myMat, 1, simMeas=pearsSim))
