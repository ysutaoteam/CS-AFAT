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
for path in os.listdir(r'F:\shiyan\new_attempt\SPDD\1yupu_AT\xingshibeijing\HC'):
    data = pd.DataFrame(pd.read_excel((r'F:\shiyan\new_attempt\SPDD\1yupu_AT\xingshibeijing\HC'+"./"+path),header = 0,index_col=0))#读取数据,设置None可以生成一个字典，字典中的key值即为sheet名字，此时不用使用DataFram，会报错
    print(path)
    nnn =0
    ffff =0

    for jj in range(1, 321):
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


        adjacency_matrix = np.zeros((int(9), int(9)))
        for i in range(9):
            for j in range(9):
                if i == j:
                    adjacency_matrix[i][j] = len(list[i])  #邻接矩阵的对角元素是单个角度列中为1的个数
                else:
                    adjacency_matrix[i][j] = len(set(list[i].index).intersection(set(list[j].index)))

        '''
        由每个窗口的形式背景生成每个窗口的邻接矩阵
        '''
        nnn = nnn + 1
        if nnn == 1:
            lianjie1_matrix = adjacency_matrix.copy()
        elif nnn == 2:
            lianjie_matrix = np.vstack((lianjie1_matrix, adjacency_matrix))
        elif nnn>2:
            lianjie_matrix = np.vstack((lianjie_matrix, adjacency_matrix))

    filename = r'F:\shiyan\new_attempt\SPDD\1yupu_AT\lianjiematrix\HC' + "/" + path[0:-8] + '.mat'
    scio.savemat(filename, {'0': lianjie_matrix})  # 以字典格式保存

