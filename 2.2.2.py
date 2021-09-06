#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/8 13:36
# @Author  : GXl
# @File    : 2.2.2.py
# @Software: win10 Tensorflow1.13.1 python3.5.6

#import matplotlib.pyplot
from matplotlib.font_manager import FontProperties
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

"""
Parameters:
    filename - 文件名
Returns:
    returnMat - 特征矩阵
    classLabelVector - 分类Label向量
"""
# 函数说明:打开并解析文件，对数据进行分类：1代表慢,2代表中,3代表快
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
        #使用s.split(str="",num=string,cout(str))
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
    datingDataMat - 特征矩阵
    datingLabels - 分类Label
Returns:
    无
"""
# 函数说明:可视化数据
def showdatas(datingDataMat, datingLabels):
    #设置汉字格式
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    #将fig画布分隔成1行1列,不共享x轴和y轴,fig画布的大小为(13,8)
    #当nrow=2,nclos=2时,代表fig画布被分为四个区域,axs[0][0]表示第一行第一个区域
    fig, axs = plt.subplots(nrows=2, ncols=2,sharex=False, sharey=False, figsize=(13,13))
    numberOfLabels = len(datingLabels)
    LabelsColors = []
    for i in datingLabels:
        if i == 1:
            LabelsColors.append('black')
        if i == 2:
            LabelsColors.append('orange')
        if i == 3:
            LabelsColors.append('red')
    #画出散点图,以datingDataMat矩阵的第一(相对湿度)、第二列(温度)数据画散点数据,散点大小为15,透明度为0.5
    axs[0][0].scatter(x=datingDataMat[:,0], y=datingDataMat[:,1], color=LabelsColors,s=15, alpha=.5)
    #设置标题,x轴label,y轴label
    axs0_title_text = axs[0][0].set_title(u'相对湿度与温度', FontProperties=font)
    axs0_xlabel_text = axs[0][0].set_xlabel(u'相对湿度', FontProperties=font)
    axs0_ylabel_text = axs[0][0].set_ylabel(u'温度', FontProperties=font)
    plt.setp(axs0_title_text, size=12, weight='bold', color='red')
    plt.setp(axs0_xlabel_text, size=12, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=12, weight='bold', color='black')

    #画出散点图,以datingDataMat矩阵的第一(相对湿度)、第三列(降雨)数据画散点数据,散点大小为15,透明度为0.5
    axs[0][1].scatter(x=datingDataMat[:,0], y=datingDataMat[:,2], color=LabelsColors, s=15, alpha=.5)
    axs[0][1].set_yticks(np.arange(0, 1.1, 1))
    #设置标题,x轴label,y轴label
    axs1_title_text = axs[0][1].set_title(u'相对湿度与降雨', FontProperties=font)
    axs1_xlabel_text = axs[0][1].set_xlabel(u'相对湿度', FontProperties=font)
    axs1_ylabel_text = axs[0][1].set_ylabel(u'降雨', FontProperties=font)

    plt.setp(axs1_title_text, size=12, weight='bold', color='red')
    plt.setp(axs1_xlabel_text, size=12, weight='bold', color='black')
    plt.setp(axs1_ylabel_text, size=12, weight='bold', color='black')

    #画出散点图,以datingDataMat矩阵的第二(温度)、第三列(降雨)数据画散点数据,散点大小为15,透明度为0.5
    axs[1][0].scatter(x=datingDataMat[:, 1], y=datingDataMat[:, 2], color=LabelsColors, s=15, alpha=.5)
    axs[1][0].set_yticks(np.arange(0, 1.1, 1))
    #设置标题,x轴label,y轴label
    axs2_title_text = axs[1][0].set_title(u'温度与降雨', FontProperties=font)
    axs2_xlabel_text = axs[1][0].set_xlabel(u'温度', FontProperties=font)
    axs2_ylabel_text = axs[1][0].set_ylabel(u'降雨', FontProperties=font)
    plt.setp(axs2_title_text, size=12, weight='bold', color='red')
    plt.setp(axs2_xlabel_text, size=12, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=12, weight='bold', color='black')
    #设置图例
    low = mlines.Line2D([], [], color='black', marker='.',
                      markersize=10, label='low')
    mid = mlines.Line2D([], [], color='orange', marker='.',
                      markersize=10, label='mid')
    high = mlines.Line2D([], [], color='red', marker='.',
                      markersize=10, label='high')
    #添加图例
    axs[0][0].legend(handles=[low, mid, high])
    axs[0][1].legend(handles=[low, mid, high])
    axs[1][0].legend(handles=[low, mid, high])
    # fig.tight_layout()
    #显示图片
    plt.show()


if __name__ == '__main__':
    #打开的文件名
    filename = "random forest1.txt"
    #打开并处理数据
    datingDataMat, datingLabels = file2matrix(filename)
    showdatas(datingDataMat, datingLabels)