# Author: Jingxiao Gu
# Baidu Account: Seigato
# Description: Make Data Lists for Lane Segmentation Competition

import os
import pandas as pd
from sklearn.utils import shuffle
import shutil
import numpy as np
import cv2
from utils.process_labels import encode_labels, decode_color_labels

#================================================
# make train & validation lists
#================================================
label_list = []
image_list = []

image_dir = '/home/gujingxiao/projects/PaddlePaddle/Image_Data/'
label_dir = '/home/gujingxiao/projects/PaddlePaddle/Gray_Label/'

for s1 in os.listdir(image_dir):
    image_sub_dir1 = os.path.join(image_dir, s1)
    label_sub_dir1 = os.path.join(label_dir, 'Label_' + str.lower(s1), 'Label')
    # print(image_sub_dir1, label_sub_dir1)

    for s2 in os.listdir(image_sub_dir1):
        image_sub_dir2 = os.path.join(image_sub_dir1, s2)
        label_sub_dir2 = os.path.join(label_sub_dir1, s2)
        # print(image_sub_dir2, label_sub_dir2)

        for s3 in os.listdir(image_sub_dir2):
            image_sub_dir3 = os.path.join(image_sub_dir2, s3)
            label_sub_dir3 = os.path.join(label_sub_dir2, s3)
            # print(image_sub_dir3, label_sub_dir3)

            for s4 in os.listdir(image_sub_dir3):
                s44 = s4.replace('.jpg','_bin.png')
                image_sub_dir4 = os.path.join(image_sub_dir3, s4)
                label_sub_dir4 = os.path.join(label_sub_dir3, s44)
                if not os.path.exists(image_sub_dir4):
                    print(image_sub_dir4)
                if not os.path.exists(label_sub_dir4):
                    print(label_sub_dir4)
                # print(image_sub_dir4, label_sub_dir4)
                image_list.append(image_sub_dir4)
                label_list.append(label_sub_dir4)
print(len(image_list), len(label_list))

save = pd.DataFrame({'image':image_list, 'label':label_list})
save_shuffle = shuffle(save)
save_shuffle.to_csv('../data_list/train.csv', index=False)

#================================================
# Data Augmentation
#================================================
# data_dir = '../data_list/train.csv'
# img_dir = '/home/gujingxiao/projects/PaddlePaddle/aug_data/images/'
# label_dir = '/home/gujingxiao/projects/PaddlePaddle/aug_data/labels/'
# data_list = pd.read_csv(data_dir)
# images = np.array(data_list['image'])
# labels = np.array(data_list['label'])
# new_image = []
# new_label = []
# count = 0
# for index in range(len(labels)):
#     img_name = images[index].split('/')[-1]
#     label_name = labels[index].split('/')[-1]
#
#     ori_mask = cv2.imread(labels[index], cv2.IMREAD_GRAYSCALE)
#     ori_mask = cv2.resize(ori_mask, (768, 384), interpolation=cv2.INTER_NEAREST)
#
#     encode_mask = encode_labels(ori_mask)
    # decode_mask = decode_color_labels(encode_mask)
    # decode_mask = np.transpose(decode_mask, (1, 2, 0))
    # all_zero = 1 if len(encode_mask[encode_mask>0]) == 0 else len(encode_mask[encode_mask>0])
    # ratio = len(encode_mask[encode_mask==7]) / all_zero
    # print(index, label_name, ratio, count)
    # if ratio > 0.04:
    #     count += 1
    #     shutil.copy(images[index], os.path.join(img_dir, img_name))
    #     shutil.copy(labels[index], os.path.join(label_dir, label_name))
        # ori_image = cv2.imread(images[index])
        # ori_image = cv2.resize(ori_image, (1024, 512))
        # cv2.imshow('image', ori_image)
        # cv2.imshow('label', decode_mask)
        # cv2.waitKey(0)

#     new_image.append(img_name)
#     new_label.append(label_name)
#
# save = pd.DataFrame({'image':new_image, 'label':new_label})
# save.to_csv('/home/gujingxiao/projects/validation_set/validation.csv', index=False)