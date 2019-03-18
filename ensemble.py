import os
import numpy as np
import cv2
from paddle import fluid
from utils.image_process import expand_resize_data

# Create a bilineraNet to resize predictions to full size
def bilinearNet(predictions, submission_size, crop_offset):
    logit = fluid.layers.resize_bilinear(input=predictions, out_shape=(submission_size[0], submission_size[1] - crop_offset))
    return logit

# Main
if __name__ == "__main__":
    print('Start Making Ensemble Submissions!')
    test_dir = '../PaddlePaddle/TestSet_Final/ColorImage/'
    sub_dir = './test_submission/'
    IMG_SIZE = [1536, 512]
    SUBMISSION_SIZE = [3384, 1710]
    crop_offset = 690
    # Ignore Class 4
    label_num = 8
    test_list = os.listdir(test_dir)

    # Three Folders which save npy files corresponding to all test images
    # ensemble index 1 0.61234
    model_lists = ['/npy_save/deeplabv3p/',
                   '/npy_save/unet_base/',
                   '/npy_save/unet_simple/']

    # Build Model & Initialize Program
    images = fluid.layers.data(name='image', shape=[label_num, IMG_SIZE[1], IMG_SIZE[0]], dtype='float32')
    predictions = bilinearNet(images, SUBMISSION_SIZE, crop_offset)
    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    for index in range(len(test_list)):
        test_name = test_list[index]
        print(index, test_name)

        # Load three diffirent npys and then do average
        model_logits1 = np.load(model_lists[0] + test_name.replace('.jpg', '.npy'))
        model_logits2 = np.load(model_lists[1] + test_name.replace('.jpg', '.npy'))
        model_logits3 = np.load(model_lists[2] + test_name.replace('.jpg', '.npy'))
        avg_model_logits = (model_logits1 + model_logits2 + model_logits3) / 3.0
        logits_input = np.expand_dims(np.array(avg_model_logits), axis=0)

        # Feed data & Run BilinearNet
        feed_dict = {}
        feed_dict["image"] = logits_input
        results = exe.run(
            feed=feed_dict,
            fetch_list=[predictions])

        prediction = np.argmax(results[0][0], axis=0)
        # Convert prediction to submission image
        submission_mask = expand_resize_data(prediction, SUBMISSION_SIZE, crop_offset)
        # Save submission png
        cv2.imwrite(os.path.join(sub_dir, test_name.replace('.jpg', '.png')), submission_mask)