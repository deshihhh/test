#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/8 13:40
# @Author  : GXl
# @File    : 2.2.5.py
# @Software: win10 Tensorflow1.13.1 python3.5.6


import numpy as np
import operator
import tkinter as tk

app = tk.Tk()
app.title('预测结果')





"""
Parameters:
    inX - 用于分类的数据(测试集)
    dataSet - 用于训练的数据(训练集)
    labes - 分类标签
    k - kNN算法参数,选择距离最小的k个点
Returns:
    sortedClassCount[0][0] - 分类结果
"""
# 函数说明:kNN算法,分类器
def classify0(inX, dataSet, labels, k):
    #numpy函数shape[0]返回dataSet的行数
    dataSetSize = dataSet.shape[0]
    #在列向量方向上重复inX共1次(横向),行向量方向上重复inX共dataSetSize次(纵向)
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    #二维特征相减后平方
    sqDiffMat = diffMat**2
    #sum()所有元素相加,sum(0)列相加,sum(1)行相加
    sqDistances = sqDiffMat.sum(axis=1)
    #开方,计算出距离
    distances = sqDistances**0.5
    #返回distances中元素从小到大排序后的索引值
    sortedDistIndices = distances.argsort()
    #定一个记录类别次数的字典
    classCount = {}
    for i in range(k):
        #取出前k个元素的类别
        voteIlabel = labels[sortedDistIndices[i]]
        #dict.get(key,default=None),字典的get()方法,返回指定键的值,如果值不在字典中返回默认值。
        #计算类别次数
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    #python3中用items()替换python2中的iteritems()
    #key=operator.itemgetter(1)根据字典的值进行排序
    #key=operator.itemgetter(0)根据字典的键进行排序
    #reverse降序排序字典
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    #返回次数最多的类别,即所要分类的类别
    return sortedClassCount[0][0]

"""
Parameters:
    filename - 文件名
Returns:
    returnMat - 特征矩阵
    classLabelVector - 分类Label向量
"""
# 函数说明:打开并解析文件，对数据进行分类：1代表low,2代表mid,3代表high
def file2matrix(filename):
    #打开文件
    fr = open(filename)
    #读取文件所有内容
    arrayOLines = fr.readlines()
    #得到文件行数
    numberOfLines = len(arrayOLines)
    #返回的NumPy矩阵,解析完成的数据:numberOfLines行,3列
    returnMat = np.zeros((numberOfLines,3))
    #返回的分类标签向量
    classLabelVector = []
    #行的索引值
    index = 0
    for line in arrayOLines:
        #s.strip(rm)，当rm空时,默认删除空白符(包括'\n','\r','\t',' ')
        line = line.strip()
        #使用s.split(str="",num=string,cout(str))将字符串根据'\t'分隔符进行切片。
        listFromLine = line.split()
        #将数据前三列提取出来,存放到returnMat的NumPy矩阵中,也就是特征矩阵
        returnMat[index,:] = listFromLine[0:3]
        #根据文本中标记的喜欢的程度进行分类,1代表low,2代表mid,3代表high
        if listFromLine[-1] == 'low':
            classLabelVector.append(1)
        elif listFromLine[-1] == 'mid':
            classLabelVector.append(2)
        elif listFromLine[-1] == 'high':
            classLabelVector.append(3)
        index += 1
    return returnMat, classLabelVector

"""
Parameters:
    dataSet - 特征矩阵
Returns:
    normDataSet - 归一化后的特征矩阵
    ranges - 数据范围
    minVals - 数据最小值
"""
# 函数说明:对数据进行归一化
def autoNorm(dataSet):
    #获得数据的最小值
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    #最大值和最小值的范围
    ranges = maxVals - minVals
    #shape(dataSet)返回dataSet的矩阵行列数
    normDataSet = np.zeros(np.shape(dataSet))
    #返回dataSet的行数
    m = dataSet.shape[0]
    #原始值减去最小值
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    #除以最大和最小值的差,得到归一化数据
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    #返回归一化数据结果,数据范围,最小值
    return normDataSet, ranges, minVals

precentTats = 0
ffMiles = 0
iceCream = 0
# 函数说明:通过输入一个人的三维特征,进行分类输出
def classifyPerson():
    #输出结果
    resultList = ['慢','中等','快']
    #三维特征用户输入
    v = tk.StringVar()

    tk.Label(app, text='该地区的湿度').grid(row=0, column=0)
    p = tk.Entry(app)
    p.grid(row=0, column=1)
    tk.Label(app, text='该地区的温度').grid(row=1, column=0)
    q = tk.Entry(app)
    q.grid(row=1, column=1)
    tk.Label(app, text='该地区是否降雨').grid(row=2, column=0)
    r = tk.Entry(app)
    r.grid(row=2, column=1)
    tk.Label(app, text='预测结果：').grid(row=4, column=0)

    def aa():
        global precentTats
        precentTats = float(p.get())
        global ffMiles
        ffMiles = float(q.get())
        global iceCream
        iceCream = float(r.get())
        # 打开的文件名
        filename = "random forest1.txt"
        # 打开并处理数据
        datingDataMat, datingLabels = file2matrix(filename)
        # 训练集归一化
        normMat, ranges, minVals = autoNorm(datingDataMat)
        # 生成NumPy数组,测试集
        inArr = np.array([precentTats, ffMiles, iceCream])
        # 测试集归一化
        norminArr = (inArr - minVals) / ranges
        # 返回分类结果
        classifierResult = classify0(norminArr, normMat, datingLabels, 3)
        # 打印结果
        print("该地区大气腐蚀速率%s" % (resultList[classifierResult - 1]))
        str1 = ("该地区大气腐蚀速率%s" % (resultList[classifierResult - 1]))
        v.set(str1)

    tk.Button(app, text='下一步', command=aa).grid(row=3, column=1, sticky=tk.W)
    s = tk.Entry(app, state='readonly', textvariable=v).grid(row =4,column = 1)

    app.mainloop()



if __name__ == '__main__':
    classifyPerson()