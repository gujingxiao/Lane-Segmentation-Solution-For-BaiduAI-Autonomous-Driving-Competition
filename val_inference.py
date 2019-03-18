# Author: Jingxiao Gu
# Baidu Account: Seigato
# Description: Val & Inference Code for Lane Segmentation Competition

import cv2
import sys
import os
import time
import pandas as pd
import numpy as np
import paddle.fluid as fluid

from utils.process_labels import decode_color_labels
from utils.image_process import crop_resize_data, expand_resize_data
from utils.data_feeder import get_feeder_data, val_image_gen
from models.unet_base import unet_base
from models.unet_simple import unet_simple
from models.deeplabv3p import deeplabv3p

def mean_iou(pred, label, num_classes):
    pred = fluid.layers.argmax(pred, axis=1)
    pred = fluid.layers.cast(pred, 'int32')
    label = fluid.layers.cast(label, 'int32')
    miou, wrong, correct = fluid.layers.mean_iou(pred, label, num_classes)
    return miou

no_grad_set = []
def create_loss(predict, label, num_classes):
    predict = fluid.layers.transpose(predict, perm=[0, 2, 3, 1])
    predict = fluid.layers.reshape(predict, shape=[-1, num_classes])
    predict = fluid.layers.softmax(predict)
    label = fluid.layers.reshape(label, shape=[-1, 1])
    bce_loss = fluid.layers.cross_entropy(predict, label)
    no_grad_set.append(label.name)
    loss = bce_loss
    miou = mean_iou(predict, label, num_classes)
    return fluid.layers.reduce_mean(loss), miou

def create_network(train_image, train_label, classes, network='unet_simple', image_size=(1024, 384), for_test=False):
    if network == 'unet_base':
        predict = unet_base(train_image, classes, image_size)
    elif network == 'unet_simple':
        predict = unet_simple(train_image, classes, image_size)
    elif network == 'deeplabv3p':
        predict = deeplabv3p(train_image, classes)
    else:
        raise Exception('Not support this model:', network)
    print('The program will run', network)

    if for_test == False:
        loss, miou = create_loss(predict, train_label, classes)
        return loss, miou, predict
    elif for_test == True:
        return predict
    else:
        raise Exception('Wrong Status:', for_test)


# The main method
def main():
    IMG_SIZE =[1536, 512]
    SUBMISSION_SIZE = [3384, 1710]
    save_test_logits = False
    num_classes = 8
    batch_size = 4
    log_iters = 100
    network = 'unet_simple'

    # Define paths for each model
    if network == 'deeplabv3p':
        model_path = "./model_weights/paddle_deeplabv3p_8_end_060223"
        npy_dir = '/npy_save/deeplabv3p/'
    elif network == 'unet_base':
        model_path = "./model_weights/paddle_unet_base_10_end_059909"
        npy_dir = '/npy_save/unet_base/'
    elif network == 'unet_simple':
        model_path = "./model_weights/paddle_unet_simple_12_end_060577"
        npy_dir = '/npy_save/unet_simple/'

    program_choice = 2 # 1 - Validtion; 2 - Test
    show_label = False
    crop_offset = 690
    data_dir = './data_list/val.csv'
    test_dir = '../PaddlePaddle/TestSet_Final/ColorImage/'
    sub_dir = './test_submission/'

    # Get data list and split it into train and validation set.
    val_list = pd.read_csv(data_dir)

    #Initialization
    images = fluid.layers.data(name='image', shape=[3, IMG_SIZE[1], IMG_SIZE[0]], dtype='float32')
    labels = fluid.layers.data(name='label', shape=[1, IMG_SIZE[1], IMG_SIZE[0]], dtype='float32')

    iter_id = 0
    total_loss = 0.0
    total_miou = 0.0
    prev_time = time.time()
    # Validation
    if program_choice == 1:
        val_reader = val_image_gen(val_list, batch_size=batch_size, image_size=IMG_SIZE, crop_offset=crop_offset)
        reduced_loss, miou, pred = create_network(images, labels, num_classes, network=network, image_size=(IMG_SIZE[1], IMG_SIZE[0]), for_test=False)
        place = fluid.CUDAPlace(0)
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        fluid.io.load_params(exe, model_path)
        print("loaded model from: %s" % model_path)

        # Parallel Executor to use multi-GPUs
        exec_strategy = fluid.ExecutionStrategy()
        exec_strategy.allow_op_delay = True
        build_strategy = fluid.BuildStrategy()
        build_strategy.reduce_strategy = fluid.BuildStrategy.ReduceStrategy.Reduce
        train_exe = fluid.ParallelExecutor(use_cuda=True, build_strategy=build_strategy, exec_strategy=exec_strategy)

        print('Start Validation!')
        for iteration in range(int(len(val_list) / batch_size)):
            val_data = next(val_reader)
            results = train_exe.run(
                feed=get_feeder_data(val_data, place),
                fetch_list=[reduced_loss.name, miou.name, pred.name])
            if iter_id % log_iters == 0:
                print('Finished Processing %d Images.' %(iter_id * batch_size))
            iter_id += 1
            total_loss += np.mean(results[0])
            total_miou += np.mean(results[1])
            # label to mask
            if show_label == True:
                label_image = val_data[1][0]
                color_label_mask = decode_color_labels(label_image)
                color_label_mask = np.transpose(color_label_mask, (1, 2, 0))
                cv2.imshow('gt_label', cv2.resize(color_label_mask, (IMG_SIZE[0], IMG_SIZE[1])))

                prediction = np.argmax(results[2][0], axis=0)
                color_pred_mask = decode_color_labels(prediction)
                color_pred_mask = np.transpose(color_pred_mask, (1, 2, 0))
                cv2.imshow('pred_label', cv2.resize(color_pred_mask, (IMG_SIZE[0], IMG_SIZE[1])))
                cv2.waitKey(0)

        end_time = time.time()
        print("validation loss: %.3f, mean iou: %.3f, time cost: %.3f s"
            % (total_loss / iter_id, total_miou / iter_id, end_time - prev_time))
    # Test
    elif program_choice == 2:
        predictions = create_network(images, labels, num_classes, network=network, image_size=(IMG_SIZE[1], IMG_SIZE[0]), for_test=True)
        place = fluid.CUDAPlace(0)
        # place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        fluid.io.load_params(exe, model_path)
        print("loaded model from: %s" % model_path)

        print('Start Making Submissions!')
        test_list = os.listdir(test_dir)
        for test_name in test_list:
            test_ori_image = cv2.imread(os.path.join(test_dir, test_name))
            test_image = crop_resize_data(test_ori_image, label=None, image_size=IMG_SIZE, offset=crop_offset)
            out_image = np.expand_dims(np.array(test_image), axis=0)
            out_image = out_image[:, :, :, ::-1].transpose(0, 3, 1, 2).astype(np.float32) / (255.0 / 2) - 1

            feed_dict = {}
            feed_dict["image"] = out_image
            results_1 = exe.run(
                feed=feed_dict,
                fetch_list=[predictions])

            if iter_id % 20 == 0:
                print('Finished Processing %d Images.' %(iter_id))
            iter_id += 1
            prediction = np.argmax(results_1[0][0], axis=0)

            # Save npy files
            if save_test_logits == True:
                np.save(npy_dir + test_name.replace('.jpg', '.npy'), results_1[0][0])

            # Save Submission PNG
            submission_mask = expand_resize_data(prediction, SUBMISSION_SIZE, crop_offset)
            cv2.imwrite(os.path.join(sub_dir, test_name.replace('.jpg', '.png')), submission_mask)

            # Show Label
            if show_label == True:
                cv2.imshow('test_image', cv2.resize(test_ori_image,(IMG_SIZE[0], IMG_SIZE[1])))
                cv2.imshow('pred_label', cv2.resize(submission_mask,(IMG_SIZE[0], IMG_SIZE[1])))
                cv2.waitKey(0)
        sys.stdout.flush()

# Main
if __name__ == "__main__":
    main()
