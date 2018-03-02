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
import matplotlib.patches as mpatches
from tensorflow.python.tools import inspect_checkpoint as chkp
from tensorflow.python import pywrap_tensorflow
import os 
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

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

    # mean_value = [123, 116, 103]
    # mean_r = np.tile(np.array(mean_value[0]), (IMAGE_SIZE, IMAGE_SIZE))
    # mean_g = np.tile(np.array(mean_value[1]), (IMAGE_SIZE, IMAGE_SIZE))
    # mean_b = np.tile(np.array(mean_value[2]), (IMAGE_SIZE, IMAGE_SIZE))
    # mean = np.stack((mean_r, mean_g, mean_b), axis=2)
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
        # img = np.subtract(img, mean)

        if split == 'test':
            img = img[margin:IMAGE_CROP_SIZE+margin, margin:IMAGE_CROP_SIZE+margin, :]
        img_list[count, :, :, :] = img
        count += 1
        if count % 1000 == 1:
            print(count)

    print("finish loading images")
    # img_list = img_list.astype(np.float32)
    # img_list /= 255.0
    # img_list -= 0.5
    # img_list *= 2       

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

def showtSNE(pred_eval, eval_labels):
    fc7 = np.stack(p['fc7'] for p in pred_eval)
    fc7 = np.reshape(fc7, (-1, 4096))

    # PCA 
    pca = PCA(n_components=50)
    fc7_pca = pca.fit_transform(fc7)

    # tSNE
    tSNE = TSNE(n_components=2)
    fc7_tSNE = tSNE.fit_transform(fc7_pca)
    print(fc7_tSNE.shape)

    # draw figure
    fc7_min = np.min(fc7_tSNE, 0)
    fc7_max = np.max(fc7_tSNE, 0)
    fc7_normal = (fc7_tSNE - fc7_min) / (fc7_max - fc7_min)

    cls_num = eval_labels.shape[1]

    fig = plt.figure(figsize=(8,8))
    fig.subplots_adjust=0.6
    # fig.subplots_adjust(right=0.2)
    for i in xrange(fc7_normal.shape[0]):
        cls_list = [j for j,x in enumerate(eval_labels[i]) if x == 1]
        clr = sum(cls_list) / len(cls_list)
        clr /= float(cls_num)
        if len(cls_list) == 1:
            cls_name = CLASS_NAMES[sum(cls_list)]
        else:
            cls_name = np.array(CLASS_NAMES)[cls_list]
            cls_name = ' '.join(cls_name)
        plt.plot(fc7_normal[i,0], fc7_normal[i,1], 'o', color=plt.cm.Set1(clr), label=cls_name)


    # patch_list = []
    # for i in xrange(cls_num):
    #     patch = mpatches.Patch(color=plt.cm.Set1(i), label = CLASS_NAMES[i])
    #     patch_list.append(patch)
    # plt.legend(handles=[patch])
    arr = np.concatenate((fc7_normal, eval_labels), axis=1)

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('equal')
    # plt.legend(bbox_to_anchor=(0.5, 1.05), loc=8, ncol=8)
    # plt.legend(bbox_to_anchor=(0.5, -0.05), loc=8, ncol=5)

    # plt.legend()
    # plt.legend(handles=patch_list)
    plt.title('tSNE for fc7 features from AlexNet')
    plt.show()
    fig.savefig("task5_tSNE_alexnet_fc7.jpg")
    np.save('task5_tSNE_alexnet_fc7.npy', arr)


def main():
#     args = parse_args()
    data_dir = 'data/VOC2007/'

    # Load whole test data
    eval_data, eval_labels, eval_weights = load_pascal(
        data_dir, split='test')

    # random select 1000 images
    select_num = 1000
    eval_num = eval_data.shape[0]
    select_list = np.random.randint(eval_num, size=select_num)
    eval_data = eval_data[select_list]
    eval_labels = eval_labels[select_list]
    eval_weights = eval_weights[select_list]

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

    # Extract features
    pred_eval = list(pascal_classifier.predict(input_fn=eval_input_fn))
    
    print("draw tSNE")
    showtSNE(pred_eval, eval_labels)


if __name__ == "__main__":
    main()

