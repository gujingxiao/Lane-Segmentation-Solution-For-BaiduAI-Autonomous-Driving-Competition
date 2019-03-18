# Author: Jingxiao Gu
# Baidu Account: Seigato
# Description: Data Generator for Lane Segmentation Competition

import os
import cv2
import numpy as np
from paddle.fluid import core
from utils.process_labels import encode_labels, verify_labels
from utils.image_process import crop_resize_data, crop_val_resize_data

# Feed Data into Tensor
def get_feeder_data(data, place, for_test=False):
    feed_dict = {}
    image_t = core.LoDTensor()
    image_t.set(data[0], place)
    feed_dict["image"] = image_t

    # if not test, feed label also
    # Otherwise, only feed image
    if not for_test:
        labels_t = core.LoDTensor()
        labels_t.set(data[1], place)
        feed_dict["label"] = labels_t

    return feed_dict

# Train Images Generator
def train_image_gen(train_list, batch_size=4, image_size=[1024, 384], crop_offset=690):
    # Arrange all indexes
    all_batches_index = np.arange(0, len(train_list))
    out_images = []
    out_masks = []
    image_dir = np.array(train_list['image'])
    label_dir = np.array(train_list['label'])
    while True:
        # Random shuffle indexes every epoch
        np.random.shuffle(all_batches_index)
        for index in all_batches_index:
            if os.path.exists(image_dir[index]):
                ori_image = cv2.imread(image_dir[index])
                ori_mask = cv2.imread(label_dir[index], cv2.IMREAD_GRAYSCALE)
                # Crop the top part of the image
                # Resize to train size
                train_img, train_mask = crop_resize_data(ori_image, ori_mask, image_size, crop_offset)
                # Encode
                train_mask = encode_labels(train_mask)

                # verify_labels(train_mask)
                out_images.append(train_img)
                out_masks.append(train_mask)
                if len(out_images) >= batch_size:
                    out_images = np.array(out_images)
                    out_masks = np.array(out_masks)
                    out_images = out_images[:, :, :, ::-1].transpose(0, 3, 1, 2).astype(np.float32) / (255.0 / 2) - 1
                    out_masks = out_masks.astype(np.int64)
                    yield out_images, out_masks
                    out_images, out_masks = [], []
            else:
                print(image_dir, 'does not exist.')

# Validation Images Generator
def val_image_gen(val_list, batch_size=4, image_size=[1024, 384], crop_offset=690):
    all_batches_index = np.arange(0, len(val_list))

    out_images = []
    out_masks = []
    image_dir = np.array(val_list['image'])
    label_dir = np.array(val_list['label'])
    while True:
        np.random.shuffle(all_batches_index)
        for index in all_batches_index:
            if os.path.exists(image_dir[index]):
                ori_image = cv2.imread(image_dir[index])
                ori_mask = cv2.imread(label_dir[index], cv2.IMREAD_GRAYSCALE)
                val_img, val_mask = crop_val_resize_data(ori_image, ori_mask, image_size, crop_offset)
                val_mask = encode_labels(val_mask)
                out_images.append(val_img)
                out_masks.append(val_mask)
                if len(out_images) >= batch_size:
                    out_images = np.array(out_images)
                    out_masks = np.array(out_masks)
                    out_images = out_images[:, :, :, ::-1].transpose(0, 3, 1, 2).astype(np.float32) / (255.0 / 2) - 1
                    out_masks = out_masks.astype(np.int64)
                    yield out_images, out_masks
                    out_images, out_masks = [], []
            else:
                print(image_dir, 'does not exist.')