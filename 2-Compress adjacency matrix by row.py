#!/usr/bin/python3
# -*- coding:utf-8 -*-
import os
import cv2
import scipy.io as scio
import pandas as pd
import numpy as np
import warnings
warnings.simplefilter(action='ignore')
'''
读取测试文件夹下的形式背景set
'''
for path in os.listdir(r'F:\shiyan\new_attempt\CPPDD\2yupu_newAT\xingshibeijing\HC'):
    data = pd.DataFrame(pd.read_excel((r'F:\shiyan\new_attempt\CPPDD\2yupu_newAT\xingshibeijing\HC'+"./"+path),header = 0,index_col=0))#读取数据,设置None可以生成一个字典，字典中的key值即为sheet名字，此时不用使用DataFram，会报错
    print(path)
    nnn =0
    ffff =0
    yihang_matrix = np.zeros((int(9), int(9)))  # #用于邻接矩阵拼接形状为(9，9)的语句  一行中所有邻接矩阵相加得到的邻接矩阵
    diyige_matrix = np.zeros((int(9), int(9)))  # #用于邻接矩阵拼接形状为(9，9)的语句    一行中第1个邻接矩阵
    dierge_matrix = np.zeros((int(9), int(9)))  # #用于邻接矩阵拼接形状为(9，9)的语句    一行中第2个邻接矩阵
    disange_matrix = np.zeros((int(9), int(9)))  # #用于邻接矩阵拼接形状为(9，9)的语句    一行中第3个邻接矩阵
    lianjie_matrix = np.zeros((int(9), int(9)))  # #用于邻接矩阵拼接形状为(9，9)的语句

    for jj in range(1, 193):     #按照语谱图像素选择范围
        huachunag_data = data.iloc[(0 + 64 * (jj - 1)):(64 + 64 * (jj - 1)), 0:10]

        result = huachunag_data.loc[huachunag_data['0°-20°'] == 1]  # 获取列名为0°-20°，内容为1的内容
        result2 = huachunag_data.loc[huachunag_data['20°-40°'] == 1]
        result3 = huachunag_data.loc[huachunag_data['40°-60°'] == 1]
        result4 = huachunag_data.loc[huachunag_data['60°-80°'] == 1]
        result5 = huachunag_data.loc[huachunag_data['80°-100°'] == 1]
        result6 = huachunag_data.loc[huachunag_data['100°-120°'] == 1]
        result7 = huachunag_data.loc[huachunag_data['120°-140°'] == 1]
        result8 = huachunag_data.loc[huachunag_data['140°-160°'] == 1]
        result9 = huachunag_data.loc[huachunag_data['160°-180°'] == 1]
        list = [result, result2, result3, result4, result5, result6, result7, result8, result9]
        print("result----------------")
        print(result)
        print("result2----------------")
        print(result2)
        print("result3----------------")
        print(result3)
        print("result4----------------")
        print(result4)
        print("result5----------------")
        print(result5)
        print("result6----------------")
        print(result6)
        print("result7----------------")
        print(result7)
        print("result8----------------")
        print(result8)
        print("result9----------------")
        print(result9)
        print("list----------------")
        print(list)
        print("---------------")
        adjacency_matrix = np.zeros((int(9), int(9)))
        for i in range(9):
            for j in range(9):
                if i == j:
                    adjacency_matrix[i][j] = len(list[i])  #邻接矩阵的对角元素是单个角度列中为1的个数
                else:
                    adjacency_matrix[i][j] = len(set(list[i].index).intersection(set(list[j].index)))

        print(adjacency_matrix)      #得到一个窗口（形式背景）的对应的邻接矩阵

        kkkk =jj % 3    #jj是第jj个窗口，取余操作
        if kkkk == 1:
            diyige_matrix = adjacency_matrix
        elif kkkk ==2:
            dierge_matrix=adjacency_matrix
        else :
            disange_matrix=adjacency_matrix

        nnn = nnn + 1          #用来记是第几个窗口

        if nnn==3:             #如果是第三个窗口
            for i in range(len(yihang_matrix)):
                for j in range(len(yihang_matrix[0])):
                    yihang_matrix[i][j] = diyige_matrix[i][j]+dierge_matrix[i][j]+disange_matrix[i][j]    #一行的邻接矩阵相加

            nnn = 0        #算完一行给nnn清零，然后用来算下一行的
            ffff =ffff+1

        if nnn == 0:
            if ffff == 1:
                lianjie1_matrix = yihang_matrix.copy()
            elif ffff == 2:
                lianjie_matrix = np.vstack((lianjie1_matrix, yihang_matrix))
            elif ffff > 2:
                # 垂直拼接
                lianjie_matrix = np.vstack((lianjie_matrix, yihang_matrix))


