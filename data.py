#coding:utf-8

import os
from PIL import Image
import numpy as np
import random
import pandas as pd
import cv2
import time

IMAGE_SIZE = 32
source_root = "./augmentation/"

def load_data():
    img_0 = os.listdir(source_root + '0/')
    img_1 = os.listdir(source_root + '1/')

    num_img = len(img_0) + len(img_1)
    vali_num = int(num_img*0.2)
    datas = np.empty((num_img, 3, IMAGE_SIZE, IMAGE_SIZE), dtype="uint8")
    labels = np.empty((num_img,),dtype="uint8")
    data_train = np.empty((num_img-vali_num,3,IMAGE_SIZE, IMAGE_SIZE),dtype="uint8") # for train
    label_train = np.empty((num_img-vali_num,),dtype="uint8")
    data_test = np.empty((vali_num,3,IMAGE_SIZE, IMAGE_SIZE),dtype="uint8") # for test
    label_test = np.empty((vali_num,),dtype="uint8")

    for i in range(len(img_0)):
        img = Image.open(source_root+ '0/' +img_0[i])
        arr = np.array(img)
        datas[i,:,:,:] = [arr[:,:,0],arr[:,:,1],arr[:,:,2]]
        labels[i] = 0
    for i in range(len(img_1)):
        img = Image.open(source_root+ '1/' +img_1[i])
        arr = np.array(img)
        idx = len(img_0) + i
        datas[idx,:,:,:] = [arr[:,:,0],arr[:,:,1],arr[:,:,2]]
        labels[idx] = 1

    # shuffle
    index = [i for i in range(num_img)]
    random.shuffle(index)

    # train
    for i in range(num_img-vali_num):
        data_train[i] = datas[index[i],:,:,:]
        label_train[i] = labels[index[i]]
    
    # test
    for i in range(vali_num):
        data_test[i] = datas[index[i+num_img-vali_num],:,:,:]
        label_test[i] = labels[index[i+num_img-vali_num]]

    return (data_train,label_train), (data_test,label_test)

def save_image(data_train,label_train, data_test,label_test):
    os.makedirs(source_root+'train')
    os.makedirs(source_root+'test')

    data_train = data_train.transpose(0, 2, 3, 1)
    data_test = data_test.transpose(0, 2, 3, 1)

    for i in range(len(label_train)):
        fileName = source_root+'train/'+str(label_train[i])+'_'+str(i)+'.jpg'
        cv2.imwrite(fileName, data_train[i])
    
    for i in range(len(label_test)):
        fileName = source_root+'test/'+str(label_test[i])+'_'+str(i)+'.jpg'
        cv2.imwrite(fileName, data_test[i])

if __name__ == '__main__':
    (data_train,label_train), (data_test,label_test) = load_data()
    save_image(data_train,label_train,data_test,label_test)