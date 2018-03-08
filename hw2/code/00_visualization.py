import os
import cv2
import visdom
#from faster_rcnn.datasets.imdb import imdb
#import faster_rcnn.datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
import pickle
import subprocess
import uuid
#from faster_rcnn.fast_rcnn.config import cfg
from faster_rcnn.datasets.factory import get_imdb

vis = visdom.Visdom(server='http://address.com', port='8097')

imdb = get_imdb('voc_2007_trainval')
roidb = imdb.selective_search_roidb()
print(len(roidb))

idx = 2018
roi = roidb[idx]
#print(roidb)
print("gt box")
#print(len(roi['boxes']))
gt_box = roi['boxes'][0]
print(gt_box)
print("ss boxes")
ss_box = roi['boxes'][1:11]
print(ss_box)

img_path = 'data/VOCdevkit2007/VOC2007/JPEGImages/003998.jpg'
img_gt = cv2.imread(img_path)
cv2.rectangle(img_gt,(gt_box[0], gt_box[1]),(gt_box[2], gt_box[3]), (0,255,0),2)
cv2.imwrite('2008.jpg', img_gt)

img_ss = cv2.imread(img_path)
for i in xrange(10):
    cv2.rectangle(img_ss, (ss_box[i][0],ss_box[i][1]),(ss_box[i][2],ss_box[i][3]),(0,0,255),2)
cv2.imwrite('2008_ss.jpg', img_ss)

img_gt = img_gt.transpose(2,0,1)
img_ss = img_ss.transpose(2,0,1)
vis.image(img_gt)
vis.image(img_ss)
