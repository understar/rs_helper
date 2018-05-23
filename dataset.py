# coding:cp936
'''
构造训练及验证数据集合；
步骤：
1、遍历可用遥感影像文件夹，可能是多个；随机抽取n张影像；
2、每一张影像随机生成m张patch；
3、保存patch，手工去除不合适的；
'''
import numpy as np
# import keras
import os
import glob
import random
from skimage.io import imread, imsave
from sklearn.feature_extraction.image import extract_patches_2d

good = [0, 1]
bad = [2, 3, 4, 5, 6, 7, 8, 9, 10]

good_imgs = []
bad_imgs = []

for i in good:
    good_imgs.extend(glob.glob('./thumbfiles/%d/*.jpg' % i))

for i in bad:
    bad_imgs.extend(glob.glob('./thumbfiles/%d/*.jpg' % i))

if not os.path.exists('dataset'):
    os.mkdir('dataset')
    os.mkdir('dataset/good')
    os.mkdir('dataset/bad')

rng = np.random.RandomState()
patch_size = (200, 200)

'''Good patches'''
index = 0
for img in random.sample(good_imgs, 3000):
    temp = imread(img)
    data = extract_patches_2d(temp, patch_size=patch_size, max_patches=10, random_state=rng)
    for i in range(10):
        print('./dataset/good/%d.jpg' % index)
        imsave('./dataset/good/%d.jpg' % index, data[i])
        index += 1

'''Bad patches'''
index = 0
for img in random.sample(bad_imgs, 2000):
    temp = imread(img)
    data = extract_patches_2d(temp, patch_size=patch_size, max_patches=15, random_state=rng)
    for i in range(15):
        print('./dataset/bad/%d.jpg' % index)
        imsave('./dataset/bad/%d.jpg' % index, data[i])
        index += 1

#  print(len(good_imgs))
#  print(len(bad_imgs))
