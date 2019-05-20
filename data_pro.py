#!/usr/bin/env python
# coding=UTF-8
'''
@Description: 
@Author: ZhaoSong
@LastEditors: ZhaoSong
@Date: 2019-05-08 15:37:35
@LastEditTime: 2019-05-10 21:06:03
'''
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def set_missing_ages(df):

    age_df = df[['Age','Fare','Parch','SibSp','Pclass']]

    # 将乘客分为已知年龄和未知年龄两部分
    known_age = age_df[age_df.Age.notnull()].values
    unknown_age = age_df[age_df.Age.isnull()].values

    # y为目标年龄      #冒号：提取两个索引(不包括停止索引)之间的项
    y = known_age[:,0]
    # x为特征属性值
    X = known_age[:,1:]

    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X,y)

    # 用得到的模型进行未知直接给出年龄预测结果。内部还是调用的predict_proba()
    predictedAges = rfr.predict(unknown_age[:,1::])
    # 用得到的预测结果填补缺失数据。.loc按标签访问列
    df.loc[(df.Age.isnull()),'Age'] = predictedAges

    return df,rfr 

def set_Cabin_type(df):
    df.loc[(df.Cabin.notnull()),'Cabin'] = 'Yes'
    df.loc[(df.Cabin.isnull()),'Cabin'] = 'No'
    return df

train = pd.read_csv('F:/titanic/train.csv')

# TODO:缺失值填充
train,rfr = set_missing_ages(train)
train = set_Cabin_type(train)

field_train = train.columns.values[4:].tolist()
# 不用继续
field_train1 = ['Pclass','Sex','SibSp','Parch','Embarked','Survived']
# 年龄处理成三个one-hot representation
train['Age'] = pd.cut(train['Age'], bins=[0, 12, 50, 200], labels=['Child', 'Adult', 'Elder'])
# 处理名字 


'''fields_train_dict = {}
for fields in field_train:
    fields_train_dict[fields] = pd.get_dummies(train[fields],prefix= 'Pclass')
print(fields_train_dict)   ''' 

# 再细分一下，把不需要field的踢出去，换成穷举吧
print(train.info())    
print(train.describe())


sex = set(train['Sex'])
Embarked = set(train['Embarked'])

test = pd.read_csv('F:/titanic/test.csv')
sib_t = set(test['SibSp'])
Pclass_t = set(test['Pclass'])
sex_t = set(test['Sex'])
Embarked_t = set(test['Sex'])
print(test.info())
print(test.describe())