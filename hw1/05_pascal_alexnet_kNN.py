from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import sys
import numpy as np
import tensorflow as tf
import argparse
import os.path as osp
from os import listdir
from os.path import isfile, join
import scipy.misc as sci
from PIL import Image
from functools import partial
import matplotlib.pyplot as plt
from tensorflow.python.tools import inspect_checkpoint as chkp
from tensorflow.python import pywrap_tensorflow
import os 
from sklearn.neighbors import NearestNeighbors

from eval import compute_map
tf.logging.set_verbosity(tf.logging.INFO)

CLASS_NAMES = [
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor',
]

BATCH_SIZE = 10
IMAGE_SIZE = 256
IMAGE_CROP_SIZE = 224
MODEL_PATH = "pascal_model_alexnet"
max_step = 4000
stride = 100
display = 100
# test_num = 10


def cnn_model_fn(features, labels, mode, num_classes=20):
    # Build model
    if mode == tf.estimator.ModeKeys.TRAIN:
        input_layer = tf.reshape(features["x"], [-1, IMAGE_SIZE, IMAGE_SIZE, 3])
    else:
        input_layer = tf.reshape(features["x"], [-1, IMAGE_CROP_SIZE, IMAGE_CROP_SIZE, 3])

    def data_augmentation(inputs):
        for i in xrange(BATCH_SIZE):
            output = tf.image.random_flip_left_right(inputs[i])
            output = tf.image.random_contrast(output, 0.9, 1.1)
            output += tf.random_normal([IMAGE_SIZE, IMAGE_SIZE, 3], 0, 0.1)
            output = tf.random_crop(output, [IMAGE_CROP_SIZE, IMAGE_CROP_SIZE, 3])
            output = tf.expand_dims(output, 0)
            if i == 0:
                outputs = output
            else:
                outputs = tf.concat([outputs, output], 0)
        return outputs

    # def center_crop(inputs, size):
    #     print(size)
    #     ratio = IMAGE_CROP_SIZE / float(IMAGE_SIZE)
    #     for i in xrange(size):
    #         output = tf.image.central_crop(inputs[i], ratio)
    #         output = tf.expand_dims(output, 0)
    #         if i == 0:
    #             outputs = output
    #         else:
    #             outputs = tf.concat([outputs, output], 0)
    #     return outputs

    #data augmentation
    if mode == tf.estimator.ModeKeys.TRAIN:
        input_layer = data_augmentation(input_layer)
    # else:
    #     input_layer = center_crop(input_layer, test_num)
    print(input_layer.shape)

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=96,
        kernel_size=[11, 11],
        strides=[4, 4],
        padding="valid",
        activation=tf.nn.relu,
        kernel_initializer=tf.random_normal_initializer(0, 0.01),
        bias_initializer=tf.zeros_initializer())
    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3, 3], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=256,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.random_normal_initializer(0, 0.01),
        bias_initializer=tf.zeros_initializer())
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[3, 3], strides=2)

    # Convolutional Layer #3
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=384,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.random_normal_initializer(0, 0.01),
        bias_initializer=tf.zeros_initializer())

    # Convolutional Layer #4
    conv4 = tf.layers.conv2d(
        inputs=conv3,
        filters=384,
        kernel_size=[3, 3],
        padding="same",
        kernel_initializer=tf.random_normal_initializer(0, 0.01),
        bias_initializer=tf.zeros_initializer())

    # Convolutional Layer #5 and Max pooling #3
    conv5 = tf.layers.conv2d(
        inputs=conv4,
        filters=256,
        kernel_size=[3, 3],
        padding="same",
        kernel_initializer=tf.random_normal_initializer(0, 0.01),
        bias_initializer=tf.zeros_initializer())
    pool3 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[3, 3], strides=2)

    # print(pool3.shape)
    # Dense Layer #1 and drop out #1
    pool3_flat = tf.reshape(pool3, [-1, 256 * 5 * 5])
    dense1 = tf.layers.dense(inputs=pool3_flat, units=4096,
                            activation=tf.nn.relu,
                            kernel_initializer=tf.random_normal_initializer(0, 0.005),
                            bias_initializer=tf.zeros_initializer())
    dropout1 = tf.layers.dropout(
        inputs=dense1, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    dense2 = tf.layers.dense(inputs=dropout1, units=4096,
                            activation=tf.nn.relu,
                            kernel_initializer=tf.random_normal_initializer(0, 0.005),
                            bias_initializer=tf.zeros_initializer())
    dropout2 = tf.layers.dropout(
        inputs=dense2, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout2, units=20,
                            kernel_initializer=tf.random_normal_initializer(0, 0.01),
                            bias_initializer=tf.zeros_initializer())

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.sigmoid(logits, name="sigmoid_tensor"),
        "pool5": pool3,
        "fc7": dense2
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.identity(tf.losses.sigmoid_cross_entropy(
        multi_class_labels=labels, logits=logits), name='loss')

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        lr = tf.train.exponential_decay(0.001, tf.train.get_global_step(), 10000, 0.5)
        optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])} 
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def load_pascal(data_dir, split='train'):
    """
    Function to read images from PASCAL data folder.
    Args:
        data_dir (str): Path to the VOC2007 directory.
        split (str): train/val/trainval split to use.
    Returns:
        images (np.ndarray): Return a np.float32 array of
            shape (N, H, W, 3), where H, W are 224px each,
            and each image is in RGB format.
        labels (np.ndarray): An array of shape (N, 20) of
            type np.int32, with 0s and 1s; 1s for classes that
            are active in that image.
    """
    # Wrote this function
    img_dir = data_dir + 'JPEGImages/'
    label_dir = data_dir + 'ImageSets/Main/'

    # read images
    label_path = label_dir + split + '.txt'
    file = open(label_path, 'r')
    lines = file.readlines()
    file.close()
    img_num = len(lines)
    first_flag = True
    margin = (IMAGE_SIZE - IMAGE_CROP_SIZE) // 2

    mean_value = [123, 116, 103]
    mean_r = np.tile(np.array(mean_value[0]), (IMAGE_SIZE, IMAGE_SIZE))
    mean_g = np.tile(np.array(mean_value[1]), (IMAGE_SIZE, IMAGE_SIZE))
    mean_b = np.tile(np.array(mean_value[2]), (IMAGE_SIZE, IMAGE_SIZE))
    mean = np.stack((mean_r, mean_g, mean_b), axis=2)
#     print(mean.shape)

    if split != 'test':
        img_list = np.zeros((len(lines), IMAGE_SIZE, IMAGE_SIZE, 3))
    else:
        img_list = np.zeros((len(lines), IMAGE_CROP_SIZE, IMAGE_CROP_SIZE, 3))

    count = 0
    for line in lines:
        line = line[:6]
        img_name = img_dir + line + '.jpg'
        img = sci.imread(img_name)
        img = sci.imresize(img, (IMAGE_SIZE, IMAGE_SIZE, 3))
        img = np.subtract(img, mean)

        if split == 'test':
            img = img[margin:IMAGE_CROP_SIZE+margin, margin:IMAGE_CROP_SIZE+margin, :]
        img_list[count, :, :, :] = img
        count += 1
        if count % 1000 == 1:
            print(count)

    print("finish loading images")
    img_list = img_list.astype(np.float32)
    img_list /= 255.0
    img_list -= 0.5
    img_list *= 2       

    # read labels
    label_list = np.zeros((img_num, 20))
    weight_list = np.zeros((img_num, 20))
    cls_pos = 0
    for class_name in CLASS_NAMES:
        img_pos = 0
        label_path = label_dir + class_name + '_' + split + '.txt'
        # load images
        file = open(label_path, 'r')
        lines = file.readlines()
        file.close()
        for line in lines:
            label = line.split()[1]
            label = int(label)
            if label == 1:
                label_list[img_pos, cls_pos] = 1
                weight_list[img_pos, cls_pos] = 1
            # elif label == 0:
            #     label_list[img_pos, cls_pos] = 1
            else:
                weight_list[img_pos, cls_pos] = 1
            img_pos += 1
        cls_pos += 1
    print("finish loading label")

    img_list = img_list.astype(np.float32)
    label_list = label_list.astype(np.int32)
    weight_list = weight_list.astype(np.int32)
    return img_list, label_list, weight_list
    

def load_test_image(test_data_dir):
    mean_value = [123, 116, 103]
    mean_r = np.tile(np.array(mean_value[0]), (IMAGE_SIZE, IMAGE_SIZE))
    mean_g = np.tile(np.array(mean_value[1]), (IMAGE_SIZE, IMAGE_SIZE))
    mean_b = np.tile(np.array(mean_value[2]), (IMAGE_SIZE, IMAGE_SIZE))
    mean = np.stack((mean_r, mean_g, mean_b), axis=2)
    
    img_path_list = [f for f in listdir(test_data_dir) if isfile(join(test_data_dir, f))]
    img_list = np.zeros((len(img_path_list), IMAGE_CROP_SIZE, IMAGE_CROP_SIZE, 3))
    count = 0
    margin = (IMAGE_SIZE - IMAGE_CROP_SIZE) // 2
    for img_name in img_path_list:
        img_name = test_data_dir + img_name
        img = sci.imread(img_name)
        img = sci.imresize(img, (IMAGE_SIZE, IMAGE_SIZE, 3))
        img = np.subtract(img, mean)
        img = img[margin:IMAGE_CROP_SIZE+margin, margin:IMAGE_CROP_SIZE+margin, :]
        img_list[count, :, :, :] = img
        count += 1

    print("finish loading test images")
    print(img_list.shape)
    img_list = img_list.astype(np.float32)
    img_list /= 255.0
    img_list -= 0.5
    img_list *= 2
    return img_list


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a classifier in tensorflow!')
    parser.add_argument(
        'data_dir', type=str, default='data/VOC2007',
        help='Path to PASCAL data storage')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args


def _get_el(arr, i):
    try:
        return arr[i]
    except IndexError:
        return arr


def findNearestNeighbours(pred_eval, pred_test):
    # Feature extraction
    pool5_eval = np.stack(p['pool5'] for p in pred_eval)
    pool5_eval = np.reshape(pool5_eval, (-1, 5*5*256))
#     print(pool5_eval.shape)
    fc7_eval = np.stack(p['fc7'] for p in pred_eval)
    fc7_eval = np.reshape(fc7_eval, (-1, 4096))
#     print(fc7_eval.shape)

    pool5_test = np.stack(p['pool5'] for p in pred_test)
    pool5_test = np.reshape(pool5_test, (-1, 5*5*256))
#     print(pool5_test.shape)
    fc7_test = np.stack(p['fc7'] for p in pred_test)
    fc7_test = np.reshape(fc7_test, (-1, 4096))
#     print(fc7_test.shape)

    # find kNN
    nb_pool5 = NearestNeighbors(n_neighbors=5, algorithm='ball_tree', n_jobs=-1)
    nb_pool5.fit(pool5_eval)
    dist_pool5, indices_pool5 = nb_pool5.kneighbors(pool5_test)
    print("pool5, kNN")
    print(indices_pool5)
    
    nb_fc7 = NearestNeighbors(n_neighbors=5, algorithm='ball_tree', n_jobs=-1)
    nb_fc7.fit(fc7_eval)
    dist_fc7, indices_fc7 = nb_fc7.kneighbors(fc7_test)
    print("fc5, kNN")
    print(indices_fc7)

    return indices_pool5, indices_fc7


def showNearestNeighbouts(data_dir, eval_data_dir, test_data_dir, indices, mode):
    img_path_list = [f for f in listdir(test_data_dir) if isfile(join(test_data_dir, f))]

    img_file_path = data_dir + 'ImageSets/Main/test.txt'

    # read images
    file = open(img_file_path, 'r')
    lines = file.readlines()
    file.close()
    count = 0
    for img_path in img_path_list:
        img_name = test_data_dir + img_path
        img = sci.imread(img_name)
        img = sci.imresize(img, (IMAGE_SIZE, IMAGE_SIZE, 3))
        for i in xrange(indices.shape[1]):
            idx = indices[count][i]
            line = lines[idx][:6]
            img_name_nn = eval_data_dir + line + '.jpg'
            img_nn = sci.imread(img_name_nn)
            img_nn = sci.imresize(img_nn, (IMAGE_SIZE, IMAGE_SIZE, 3))
            img = np.concatenate((img, img_nn), axis=0)
        if count == 0:
            img_all = img
        else:
            img_all = np.concatenate((img_all, img), axis=1)
        count += 1

    img_save_name = 'task5_kNN_alexnet_' + mode + '.jpg'
    sci.imsave(img_save_name, img_all)


def main():
#     args = parse_args()
    data_dir = 'data/VOC2007/'
    test_data_dir =data_dir + 'testKNN/'

    # Load whole test data
    eval_data, eval_labels, eval_weights = load_pascal(
        data_dir, split='test')
    
    # Load knn test data
    test_data = load_test_image(test_data_dir)

    # Define Estimator
    pascal_classifier = tf.estimator.Estimator(
        model_fn=partial(cnn_model_fn, num_classes=eval_labels.shape[1]),
        model_dir=MODEL_PATH)

    # Build data
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data, "w": eval_weights},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    
    test_weights = np.zeros((test_data.shape[0], eval_labels.shape[1]))
    test_labels = np.zeros((test_data.shape[0], eval_labels.shape[1]))
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": test_data, "w": test_weights},
        y=test_labels,
        num_epochs=1,
        shuffle=False)

    # Extract features
    pred_eval = list(pascal_classifier.predict(input_fn=eval_input_fn))
    pred_prob_eval = np.stack([p['probabilities'] for p in pred_eval])
    AP = compute_map(eval_labels, pred_prob_eval, eval_weights, average=None)
    print('All test data')
    print('Obtained {} mAP'.format(np.mean(AP)))
    print('per class:')
    for cid, cname in enumerate(CLASS_NAMES):
        print('{}: {}'.format(cname, _get_el(AP, cid)))
    
    pred_test = list(pascal_classifier.predict(input_fn=test_input_fn))
    
    indices_pool5, indices_fc7 = findNearestNeighbours(pred_eval, pred_test)
    showNearestNeighbouts(data_dir, data_dir + 'JPEGImages/', test_data_dir, indices_pool5, 'pool5')
    showNearestNeighbouts(data_dir, data_dir + 'JPEGImages/', test_data_dir, indices_fc7, 'fc7')


if __name__ == "__main__":
    main()

