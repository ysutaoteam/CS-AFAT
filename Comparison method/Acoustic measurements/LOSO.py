import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import csv
from sklearn import preprocessing  # 预处理模块
from sklearn.preprocessing import scale
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from pandas.core.frame import DataFrame
import random
from sklearn.metrics import confusion_matrix
import datetime
from sklearn.naive_bayes import GaussianNB    #导入先验概率为高斯分布的朴素贝叶斯
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score,accuracy_score

print('开始运行!')
starttime = datetime.datetime.now()

path = r'F:\shiyan\new_attempt\SPDD\1yupu_AT\liantongyu\AT_SPDD_LOSO.csv'
data = pd.read_csv(path,header=None , encoding='gbk',low_memory=False)
print(data)
data=data.iloc[1:391,0:322]
print("读取data")
print(data)
from collections import defaultdict
dd = defaultdict(list)
name_list=[]

for i in range(1,len(data[321])+1):
    # print(data[0][i])
    name_list.append(data[321][i])
print("健康加患病的文件名列表")
print(name_list)
print("健康加患病的文件名列表长度：")
print(len(name_list))
hl=name_list[0:142]
print("健康人的文件名列表：")
print(hl)
hl_list=[]
for i in range(len(hl)):
    a=hl[i][0:2]
    hl_list.append(a)
print(hl_list)
print(len(hl_list))

pd=name_list[142:]
print("看看155是啥")
print(name_list[142])
pd_list=[]
for i in range(len(pd)):
    b=pd[i][0:6]
    pd_list.append(b)
print(pd_list)
new_list=hl_list+pd_list
# print(pd_list)
print(len(pd_list))

for k, va in [(v,i) for i, v in enumerate(new_list)]:
    dd[k].append(va)

index_list=[]
for i in dd:
    index=dd[i]
    index_list.append(index)

hh = defaultdict(list)
for k, va in [(v,i) for i, v in enumerate(hl_list)]:
    hh[k].append(va)
# print(hh)
hl_index_list=[]
for i in hh:
    index=hh[i]
    hl_index_list.append(index)

pd_index_list = [item for item in index_list if item not in hl_index_list]
test_list=[]

def read_data(data):
    x_list=[]
    y_list=[]
    for index in data.index:
        one_line=data.loc[index].values[1:321]    #取一行的特征
        one_line=one_line.tolist()
        x_list.append(one_line)
        y=data.loc[index].values[0]   #最后一列为标签，把所有标签保存在一个列表里，存在y_list里边
        y_list.append(y)
    return x_list,y_list


acc_micro_list=[]
sen_micro_list=[]
pre_micro_list=[]
f1_micro_list=[]

y_true_list=[]
y_predict_list=[]
y_test_list=[]

tp_list=[]
tn_list=[]
fp_list=[]
fn_list=[]
acc_list=[]


for i in range(0,len(index_list)):

    test_list=index_list[i]
    X_test = data.iloc[test_list]

    if i == 0:
        train_list = []
        train_list += ((index_list[1:]))
        # print(train_list)
        train_list = [i for item in train_list for i in item]
        # print(train_list)
        #print('train', train_list)
        X_train = data.iloc[train_list]
    if i == 1:
        train_list = []
        train_list.append((index_list[0]))
        # print(train_list)
        train_list += (index_list[i + 1:])
        train_list = [i for item in train_list for i in item]
        #print('train', train_list)
        X_train = data.iloc[train_list]
        # print(X_train)
    if i == len(index_list) - 1:
        train_list = []
        train_list += (index_list[:i])
        train_list = [i for item in train_list for i in item]
        #print('train', train_list)
        X_train = data.iloc[train_list]
    elif i != 0 and i != 1 and i != len(index_list) - 1:
        train_list = []
        train_list += (index_list[:i])
        train_list += (index_list[i + 1:])
        train_list = [i for item in train_list for i in item]
        X_train = data.iloc[train_list]

    x_test, y_test = read_data(X_test)
    x_train, y_train = read_data(X_train)
    y_test = np.array(y_test)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    x_train = np.array(x_train)
    # print(y_test)

    clf_loso = svm.SVC(kernel='rbf',C=10,gamma='auto',cache_size=200, decision_function_shape='ovr') # 构建分类器
    # clf_loso = KNeighborsClassifier(n_neighbors=14)
    # clf_loso = LogisticRegression(max_iter=700)  # 逻辑回归模型
    # clf_loso =  GaussianNB( )                   #高斯朴素贝叶斯
    # clf_loso=RandomForestClassifier(min_samples_leaf=1, n_estimators=300, max_depth=30, max_features=0.02)
    # clf_loso = RandomForestClassifier(n_estimators=100,max_depth=30)  # 随机森林
    # clf_loso = MLPClassifier(solver='adam', hidden_layer_sizes=(20,30),random_state=0,max_iter=700)
    clf_loso.fit(x_train, y_train)  # 进行训练
    score = clf_loso.score(x_test, y_test)
    acc_list.append(score)

    y_predict = clf_loso.predict(x_test)
    y_true = np.array(y_test)  # [0 0 0 ]
    print(y_true)
    print(y_predict)
    C1 = confusion_matrix(y_true, y_predict)
    print(C1)

    if i<=len(hl_index_list):
        if C1.shape==(1,1):
            tn = C1[0][0]
            tn_list.append(tn)
        if C1.shape==(2,2):
            tn = C1[0][0]
            fp = C1[0][1]
            tn_list.append(tn)
            fp_list.append(fp)
    if i>len(hl_index_list):
        if C1.shape==(1,1):
            tp = C1[0][0]
            tp_list.append(tp)
        if C1.shape == (2, 2):
            fn=C1[1][0]
            tp=C1[1][1]
            fn_list.append(fn)
            tp_list.append(tp)
print(tp_list)
print(fn_list)
print(fp_list)
print(tn_list)

acc = (sum(tp_list)+sum(tn_list))/(sum(tp_list)+sum(tn_list)+sum(fp_list)+sum(fn_list))
spe = (sum(tn_list))/(sum(tn_list)+sum(fp_list))
sen = (sum(tp_list))/(sum(tp_list)+sum(fn_list))
pre = (sum(tp_list))/(sum(tp_list)+sum(fp_list))
f1 = 2*(pre*sen)/(pre+sen)
print("留一对象法 Acc     :%0.2f " % (acc*100))
print("留一对象法 Rec     :%0.2f " % (sen*100))
print("留一对象法 Pre     :%0.2f " % (pre*100))
print("留一对象法 f 1     :%0.2f " % (f1*100))
print("留一对象法 spe     :%0.2f " % (spe*100))