import os
import visdom
import torchvision.transforms as trans
import _init_paths
import torch
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
import pickle
import subprocess
import uuid
from datasets.factory import get_imdb
import cv2

vis = visdom.Visdom(server='http://128.2.176.219', port='8097')

imdb = get_imdb('voc_2007_trainval')
roidb = imdb.selective_search_roidb()
print(len(roidb))

idx = 2018
roi = roidb[idx]

# gt box
print("gt box")
gt_box = roi['boxes'][0]
print(gt_box)

# ss boxes
scores = roi['boxscores']
# print(score)
scores = np.squeeze(scores)
scores = scores[scores.shape[0]//2:]
print(scores.shape)
idx = np.argsort(scores)
idx = idx[::-1]
print(idx.shape)
# print(idx[0:10])
print("ss boxes")
print(roi['boxes'].shape)
ss_box = roi['boxes'][idx[0:10]]
print(ss_box)

img_path = 'data/VOCdevkit2007/VOC2007/JPEGImages/003998.jpg'
img_gt = cv2.imread(img_path)
cv2.rectangle(img_gt,(gt_box[0], gt_box[1]),(gt_box[2], gt_box[3]), (0,255,0),2)
cv2.putText(img_gt, '%s' % (roi['gt_classes'][0]), (gt_box[0], gt_box[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (0, 255, 0), thickness=1)
cv2.imwrite('2008.jpg', img_gt)

img_ss = cv2.imread(img_path)
for i in xrange(10):
    cv2.rectangle(img_ss, (ss_box[i][0],ss_box[i][1]),(ss_box[i][2],ss_box[i][3]),(0,0,255),2)
    cv2.putText(img_ss, '%.3f' % (scores[idx[i]]), (ss_box[i][0], ss_box[i][1] + 15), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (0, 0, 255), thickness=1)
cv2.imwrite('2008_ss.jpg', img_ss)

r, g, b = cv2.split(img_gt)
img_gt = cv2.merge([b, g, r])
img_gt = img_gt.transpose(2,0,1)

r, g, b = cv2.split(img_ss)
img_ss = cv2.merge([b, g, r])
img_ss = img_ss.transpose(2,0,1)
# vis.image(img_gt)
# vis.image(img_ss)
# imgTensor = trans.ToTensor()
# img_gt = imgTensor(img_gt)
vis.image(img_gt, opts=dict(title='Image', caption='Ground True Bounding Box'))
vis.image(img_ss, opts=dict(title='Image', caption='Selective Search Bounding Box'))
