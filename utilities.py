#!/usr/bin/env python
# coding=UTF-8
'''
@Description: 
@Author: ZhaoSong
@LastEditors: ZhaoSong
@Date: 2019-05-01 19:22:38
@LastEditTime: 2019-05-06 13:50:01
'''
import pandas as pd
import numpy as np
import pickle

def one_hot_representation(sample, fields_dict, array_length):
    """
    One hot presentation for every sample data
    :param fields_dict: fields value to array index
    :param sample: sample data, type of pd.series
    :param array_length: length of one-hot representation
    :return: one-hot representation, type of np.array
    """
    array = np.zeros([array_length])
    idx = []
    for field in fields_dict:
        # get index of array index 
        # 效果体现在name pclass sex sibsp parch embarked上
        if field == 'Survived':
            field_value = int(str(sample[field])[-2:])
        else:
            field_value = sample[field]
        ind = fields_dict[field][field_value]
        array[ind] = 1
        idx.append(ind)
    return array,idx[:21]
    
if __name__ == '__main__':
    # setting fields
    fields_train = ['Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked','Survived']

    fields_test = ['Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']

    train = pd.read_csv('F:/titanic/train.csv',chunksize=100)
    test = pd.read_csv('F:/titanic/test.csv',chunksize=100)
    # loading dicts
    fields_train_dict = {}
    for field in fields_train:
        with open('dicts/'+field+'.pkl','rb') as f:
            fields_train_dict[field] = pickle.load(f)

    fields_test_dict = {}
    for field in fields_test:
        with open('dicts/'+field+'.pkl','rb') as f:
            fields_test_dict[field] = pickle.load(f)
    # length of representation
    train_array_length = max(fields_train_dict['click'].values()) + 1
    test_array_length = train_array_length - 2
    # initialize the model

    for data in test:
        # data['click'] = np.zeros(100)
        # data.to_csv('a.csv',mode='a')
        sample = data.iloc[3,:]
        print(one_hot_representation(sample, fields_test_dict, test_array_length))
        break