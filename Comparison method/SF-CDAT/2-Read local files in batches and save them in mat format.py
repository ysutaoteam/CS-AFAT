#!/usr/bin/python3
# -*- coding:utf-8 -*-

import os
import cv2
import xlrd
import scipy.io as scio
import pandas as pd
import numpy as np
import warnings
warnings.simplefilter(action='ignore')

#读取健康人文件夹下的形式背景set
for path in os.listdir(r'F:\shiyan\Angletian\PCGITA\HOG+AT\classification\normal\xingshibeijing\HC'):
    data = pd.DataFrame(pd.read_excel((r'F:\shiyan\Angletian\PCGITA\HOG+AT\classification\normal\xingshibeijing\HC'+"./"+path),header = 0,index_col=0))#读取数据
    jiaankangpinjie_matrix = np.zeros((int(9), int(9)))
    for jj in range(1, 641):   #  rang的范围是滑动窗口的数量
        huachunag_data = data.iloc[0 + 64 * (jj - 1):64 + 64 * (jj - 1), 0:10]
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
                    adjacency_matrix[i][j] = len(list[i])  # min(len(list[i]), len(list[j]))
                elif set(list[i].index).intersection(set(list[j].index)) == set(list[i].index):
                    adjacency_matrix[i][j] = 0
                else:
                    adjacency_matrix[i][j] = len(set(list[i].index).intersection(set(list[j].index)))

        if jj == 1:
            jiaankangpinjie_matrix = adjacency_matrix
        else:
            jiaankangpinjie_matrix = np.hstack((jiaankangpinjie_matrix, adjacency_matrix))

    filename = r'F:\shiyan\Angletian\PCGITA\HOG+AT\classification\normal\mat-xingshibeijing\HC'+"./"+path+'.mat'  # 保存的文件名
    scio.savemat(filename, {'0': jiaankangpinjie_matrix})  # 以字典格式保存

for path in os.listdir(r'F:\shiyan\Angletian\PCGITA\HOG+AT\classification\normal\xingshibeijing\PD'):       #读取患病人文件夹下的形式背景set
    data = pd.DataFrame(pd.read_excel((r'F:\shiyan\Angletian\PCGITA\HOG+AT\classification\normal\xingshibeijing\PD'+"./"+path),header = 0,index_col=0))#读取数据,设置None可以生成一个字典，字典中的key值即为sheet名字，此时不用使用DataFram，会报错
    youbingpinjie_matrix = np.zeros((int(9), int(9)))
    # 注意改range范围，后边的数字是形式背景数量
    for jj in range(1, 641):
        huachunag_data = data.iloc[0 + 64 * (jj - 1):64 + 64 * (jj - 1), 0:10]
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
                    adjacency_matrix[i][j] = len(list[i])  # min(len(list[i]), len(list[j]))
                elif set(list[i].index).intersection(set(list[j].index)) == set(list[i].index):
                    adjacency_matrix[i][j] = 0
                else:
                    adjacency_matrix[i][j] = len(set(list[i].index).intersection(set(list[j].index)))

        if jj == 1:
            youbingpinjie_matrix = adjacency_matrix
        else:
            youbingpinjie_matrix = np.hstack((youbingpinjie_matrix, adjacency_matrix))

    filename = r'F:\shiyan\Angletian\PCGITA\HOG+AT\classification\normal\mat-xingshibeijing\PD' + "./" + path + '.mat'  # 保存的文件名
    scio.savemat(filename, {'1': youbingpinjie_matrix})  # 注意要以字典格式保存