import torch.utils.data as data

from PIL import Image
import os
import os.path
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import torch

def fea_normalize(features):
    max_nums = np.max(features, axis=1)
    max_nums = max_nums[:,np.newaxis]
    bs, dim = features.shape
    np.repeat(max_nums,dim, axis = 1)
    feature =features/max_nums
    return feature

def fea_reader(file_dir, is_test):
    raw_data = pd.read_csv(file_dir, header=0)

    data = raw_data.values.T
    m,n= data.shape
    imgs = data[0::, 0:64] #n-10  #240:316
    imgs = fea_normalize(imgs)
    labels = data[0::,n-31::]
    # # 选取 2/3 数据作为训练集， 1/3 数据作为测试集
    # train_features, test_features, train_labels, test_labels = train_test_split(
    #     imgs, labels, test_size=0.5, random_state=23323)
    # if(is_test):
    #     out_features = test_features
    #     out_labels = test_labels
    # else:
    #     out_features = train_features
    #     out_labels = train_labels
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
    def __init__(self, rootsrc=None, roottgt=None, transform=None, is_test = False):
        self.rootsrc   = rootsrc
        self.rootttgt  = roottgt
        self.transform = transform
        self.featList ,self.labelList = fea_reader(file_dir = rootsrc, is_test = is_test)

    def __getitem__(self, index):
        src = self.featList[index]
        tgt = self.labelList[index]
        if self.transform is not None:
            src = self.transform(src)
            tgt = self.transform(tgt)
        return src, tgt

    def __len__(self):
        return len(self.featList)