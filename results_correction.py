import os
import cv2
import numpy as np

image_dir = '../PaddlePaddle/TestSet_Final/ColorImage/'
label_dir = './test_submission/'
correction_dir = './test_submission_correction/'
index = 0
offset = 4

for item in os.listdir(image_dir):
    print(index, item)
    index += 1
    img = cv2.imread(os.path.join(image_dir, item))
    label = cv2.imread(os.path.join(label_dir, item.replace('.jpg', '.png')), cv2.IMREAD_GRAYSCALE)
    correction_label = np.zeros(label.shape)
    correction_label[:, :3384-offset] = label[:, offset:3384]
    correction_label[:, 3384-offset:] = label[:, 3384-offset:]
    cv2.imwrite(os.path.join(correction_dir, item.replace('.jpg', '.png')), correction_label)
