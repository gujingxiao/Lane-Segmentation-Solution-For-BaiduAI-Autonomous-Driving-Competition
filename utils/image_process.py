import cv2
import numpy as np
from imgaug import augmenters as iaa
from utils.process_labels import decode_labels, decode_color_labels

# crop the image to discard useless parts
def crop_resize_data(image, label=None, image_size=[1024, 384], offset=690):
    roi_image = image[offset:, :]
    if label is not None:
        #roi_image = image_augmentation(roi_image)
        roi_label = label[offset:, :]
        # crop_image, crop_label = random_crop(roi_image, roi_label)
        train_image = cv2.resize(roi_image, (image_size[0], image_size[1]), interpolation=cv2.INTER_LINEAR)
        train_label = cv2.resize(roi_label, (image_size[0], image_size[1]), interpolation=cv2.INTER_NEAREST)
        # cv2.imshow('train_img', train_image)
        # cv2.imshow('train_label', train_label * 100)
        # cv2.waitKey(0)
        return train_image, train_label
    else:
        train_image = cv2.resize(roi_image, (image_size[0], image_size[1]), interpolation=cv2.INTER_LINEAR)
        return train_image

def crop_val_resize_data(image, label=None, image_size=[1024, 384], offset=690):
    roi_image = image[offset:, :]
    roi_label = label[offset:, :]
    val_image = cv2.resize(roi_image, (image_size[0], image_size[1]), interpolation=cv2.INTER_LINEAR)
    val_label = cv2.resize(roi_label, (image_size[0], image_size[1]), interpolation=cv2.INTER_NEAREST)
    return val_image, val_label

def expand_resize_data(prediction=None, submission_size=[3384, 1710], offset=690):
    pred_mask = decode_labels(prediction)
    expand_mask = cv2.resize(pred_mask, (submission_size[0], submission_size[1] - offset), interpolation=cv2.INTER_NEAREST)
    submission_mask = np.zeros((submission_size[1], submission_size[0]), dtype='uint8')
    submission_mask[offset:, :] = expand_mask
    return submission_mask

def expand_resize_color_data(prediction=None, submission_size=[3384, 1710], offset=690):
    color_pred_mask = decode_color_labels(prediction)
    color_pred_mask = np.transpose(color_pred_mask, (1, 2, 0))
    color_expand_mask = cv2.resize(color_pred_mask, (submission_size[0], submission_size[1] - offset), interpolation=cv2.INTER_NEAREST)
    color_submission_mask = np.zeros((submission_size[1], submission_size[0], 3), dtype='uint8')
    color_submission_mask[offset:, :, :] = color_expand_mask
    return color_submission_mask

def random_crop(image, label):
    random_seed = np.random.randint(0, 10)
    if random_seed < 5:
        return image, label
    else:
        width, height = image.shape[1], image.shape[0]
        new_width = int(float(np.random.randint(88, 99)) / 100.0 * width)
        new_height = int(float(np.random.randint(88, 99)) / 100.0 * height)
        offset_w = np.random.randint(0, width - new_width - 1)
        offset_h = np.random.randint(0, height - new_height - 1)
        new_image = image[offset_h : offset_h + new_height, offset_w : offset_w + new_width]
        new_label = label[offset_h: offset_h + new_height, offset_w: offset_w + new_width]
        return new_image, new_label

def image_augmentation(ori_img):
    random_seed = np.random.randint(0, 10)
    if random_seed > 5:
        seq = iaa.Sequential([iaa.OneOf([
                iaa.Sharpen(alpha=(0.1, 0.3), lightness=(0.7, 1.3)),
                iaa.GaussianBlur(sigma=(0, 1.0))])])
        ori_img = seq.augment_image(ori_img)
    return ori_img