import os
import cv2
from faster_rcnn.datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
import pickle
import subprocess
import uuid
from fast_rcnn.config import cfg
from datasets.factory import get_imdb

imdb = get_imdb('voc_2007_trainval')