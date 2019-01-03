import torch.utils.data as data

from PIL import Image
import os
import os.path
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import torch

def fea_normalize(features,dataset):
    if dataset=='flickr25k':
        viewdims = [192, 256, 256, 43, 150, 960, 2000]
        accu_dims = [0,192,448,704,747,897,1857,3857]
        views = 7
    else:
        viewdims = [64, 144, 73, 128, 225, 500]
        accu_dims = [0,64, 208, 281, 409, 634, 1134]
        views = 6

    names=locals()
    for i in range(views):
        names['x_'+ str(i)] = features[0::,accu_dims[i]:accu_dims[i+1]]
        names['max_nums_'+ str(i)] = np.max(names.get('x_' + str(i)), axis=1)
        names['max_nums_' + str(i)] = names.get('max_nums_' + str(i))[:,np.newaxis]
        names['max_nums_' + str(i)] = np.repeat(names.get('max_nums_' + str(i)), viewdims[i], axis = 1)

    max_nums = names.get('max_nums_' + str(0))
    for i in range(1, views):
        max_nums = np.concatenate((max_nums, names.get('max_nums_' + str(i))), axis = 1)
    #max_nums = np.concatenate((max_nums_0,max_nums_1,max_nums_2,max_nums_3,max_nums_4,max_nums_5), axis=1)

    feature =features/max_nums
    return feature

def fea_reader(file_dir, dataset='',is_test= False):


    if dataset == 'flickr25k':
        if is_test:
            file_dir = '/home/wqy/Documents/MvADL-master/nus_train.csv'
        else:
            file_dir = '/home/wqy/Documents/MvADL-master/nus_test.csv'
    else:
        if is_test:
            file_dir = '/home/wqy/Documents/MvADL-master/nus_train.csv'
        else:
            file_dir = '/home/wqy/Documents/MvADL-master/nus_test.csv'

    raw_data = pd.read_csv(file_dir, header=0)
    if dataset == 'flickr25k':
        print('loading dataset {} in path {}'.format(dataset,path_train))
        data = raw_data.values #row feature
        m,n= data.shape
        class_nums = 38
    else:
        data = raw_data.values.T #original stored in column feature,change to row feature
        m,n= data.shape
        class_nums = 31

    imgs = data[0::, 0:n-class_nums:]
    imgs = fea_normalize(imgs , dataset)
    labels = data[0::,n-class_nums::]

    out_features = imgs
    out_labels = labels

    class_id = np.argmax(out_labels, axis=1)
    out_labels = class_id
    out_features.tolist()
    out_labels.tolist()
    out_features = torch.FloatTensor(out_features)
    out_labels = torch.LongTensor(out_labels)
    return out_features, out_labels

class FeaList(data.Dataset):
    def __init__(self, rootsrc=None, dataset='nus-wide-object',roottgt=None, transform=None, is_test = False):
        self.rootsrc   = rootsrc
        self.rootttgt  = roottgt
        self.transform = transform
        self.featList ,self.labelList = fea_reader(file_dir = rootsrc,dataset=dataset, is_test = is_test)

    def __getitem__(self, index):
        src = self.featList[index]
        tgt = self.labelList[index]
        if self.transform is not None:
            src = self.transform(src)
            tgt = self.transform(tgt)
        return src, tgt

    def __len__(self):
        return len(self.featList)