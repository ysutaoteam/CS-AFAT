import pandas as pd
import warnings
from sklearn.preprocessing import scale
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate  # 交叉验证所需的函数
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from xgboost.sklearn import XGBClassifier
from sklearn import preprocessing  # 预处理模块
from sklearn.metrics import recall_score  # 模型度量
import pandas as pd
from sklearn.utils import column_or_1d
from sklearn import svm  # SVM算法
from tqdm import tqdm
from sklearn.model_selection import KFold,StratifiedKFold, LeaveOneOut, LeavePOut, ShuffleSplit  # 交叉验证所需的子集划分方法
import csv
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score,accuracy_score
from sklearn.naive_bayes import GaussianNB    #导入先验概率为高斯分布的朴素贝叶斯
from sklearn.naive_bayes import MultinomialNB #导入先验概率为多项式分布的朴素贝叶斯
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
# import lightgbm as lgb

data_all = pd.read_csv('./zhuan/liantongyu/PCGITA_zhuan50.csv', header=None , encoding='gbk',low_memory=False)

df_X=data_all.iloc[0:1346,1:641]

print(df_X)
df_y=data_all.iloc[0:1346,0:1]
print(df_y)
df_y = column_or_1d(df_y, warn=True)
df_y=df_y.astype(int)



def five_fold(model):           # 五折
    acc_list=[]
    rec_list=[]
    pre_list=[]
    f1_list=[]
    spe_list = []
    gkf = StratifiedKFold(n_splits=5,shuffle=False)
    for train_index, test_index in gkf.split(df_X,df_y):
        X_train, X_test = np.array(df_X)[train_index], np.array(df_X)[test_index]
        Y_train, Y_test = np.array(df_y)[train_index], np.array(df_y)[test_index]
        model.fit(X_train, Y_train)
        y_true = Y_test
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_true, y_pred)
        # print(cm)
        tp = cm[1][1]
        fp = cm[0][1]
        fn = cm[1][0]
        tn = cm[0][0]

        acc=(tp+tn)/(tp+tn+fp+fn)
        acc_list.append(acc)     #  macro
        rec=(tp)/(tp+fn)
        rec_list.append(rec)     #  macro
        pre=(tp)/(tp+fp)
        pre_list.append(pre)     #  macro
        f1=(2*pre*rec)/(pre+rec)
        f1_list.append(f1)     #  macro
        spe = (tn)/(tn+fp)
        spe_list.append(spe)

    acc_list = np.array(acc_list)
    rec_list = np.array(rec_list)
    pre_list = np.array(pre_list)
    f1_list = np.array(f1_list)
    spe_list = np.array(spe_list)
    # print(acc_list)
    # print(rec_list)
    # print(pre_list)
    # print(f1_list)
    print("5折 ACC mean±std: %0.2f±%0.2f" % (acc_list.mean()*100, acc_list.std(ddof=1)*100))
    print("5折 REC mean±std: %0.2f±%0.2f" % (rec_list.mean()*100, rec_list.std(ddof=1)*100))
    print("5折 Pre mean±std: %0.2f±%0.2f" % (pre_list.mean()*100, pre_list.std(ddof=1)*100))
    print("5折 F 1 mean±std: %0.2f±%0.2f" % (f1_list.mean()*100, f1_list.std(ddof=1)*100))
    print("5折 Spe mean±std: %0.2f±%0.2f" % (spe_list.mean() * 100, spe_list.std(ddof=1) * 100))

def ten_fold(model):           # 五折
    acc_list=[]
    rec_list=[]
    pre_list=[]
    f1_list=[]
    spe_list = []
    gkf = StratifiedKFold(n_splits=10,shuffle=False)
    for train_index, test_index in gkf.split(df_X,df_y):
        X_train, X_test = np.array(df_X)[train_index], np.array(df_X)[test_index]
        Y_train, Y_test = np.array(df_y)[train_index], np.array(df_y)[test_index]
        model.fit(X_train, Y_train)
        y_true = Y_test
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_true, y_pred)
        # print(cm)
        tp = cm[1][1]
        fp = cm[0][1]
        fn = cm[1][0]
        tn = cm[0][0]

        acc=(tp+tn)/(tp+tn+fp+fn)
        acc_list.append(acc)
        rec=(tp)/(tp+fn)
        rec_list.append(rec)
        pre=(tp)/(tp+fp)
        pre_list.append(pre)
        f1=(2*pre*rec)/(pre+rec)
        f1_list.append(f1)
        spe = tn/(tn+fp)
        spe_list.append(spe)

    acc_list = np.array(acc_list)
    rec_list = np.array(rec_list)
    pre_list = np.array(pre_list)
    f1_list = np.array(f1_list)
    spe_list = np.array(spe_list)

    print("—————————————————————————————————————————————————————")
    print("10折 ACC mean±std: %0.2f±%0.2f" % (acc_list.mean()*100, acc_list.std(ddof=1)*100))
    print("10折 REC mean±std: %0.2f±%0.2f" % (rec_list.mean()*100, rec_list.std(ddof=1)*100))
    print("10折 Pre mean±std: %0.2f±%0.2f" % (pre_list.mean()*100, pre_list.std(ddof=1)*100))
    print("10折 F 1 mean±std: %0.2f±%0.2f" % (f1_list.mean()*100, f1_list.std(ddof=1)*100))
    print("10折 Spe mean±std: %0.2f±%0.2f" % (spe_list.mean() * 100, spe_list.std(ddof=1) * 100))



logistic = LogisticRegression(max_iter=2000)  # 逻辑回归模型
#
svm = svm.SVC(kernel='rbf', C=1,gamma='auto',cache_size=200, decision_function_shape='ovr')   # SVM模型
#
forest = RandomForestClassifier(n_estimators=100,max_depth=30)  # 随机森林70
#
GaussNaiveBayes = GaussianNB( )    #高斯朴素贝叶斯

#
mlp = MLPClassifier(solver='adam', hidden_layer_sizes=(20,20,10), random_state=0,max_iter=1000)  #多层感知机20,30,30,20

knn = KNeighborsClassifier(n_neighbors=12)    # KNN

# model_name=["logistic","svm","forest","GaussNaiveBayes","mlp","knn"]
# model_name=["logistic"]
# model_name=["GaussNaiveBayes"]
# model_name=["svm"]
# model_name=["forest"]
model_name=["mlp"]
# model_name=["knn"]
for name in model_name:
    model=eval(name)
    print("分类器：",name)
    five_fold(model)
    ten_fold(model)


