import numpy as np
import os
from collections import OrderedDict
import pandas as pd
import pickle
import time
import subprocess
import utils.dataloader_utils as dutils
import random
import cv2


def get_data_generators(cf, logger, mode='train'):
    all_data = load_dataset(cf, logger, mode)
    pano_batch_generator = PanoBatchGenerator(all_data, cf, mode)
    return pano_batch_generator


def load_dataset(cf, logger, mode):
    data_path = cf.raw_data_dir + mode
    data = []

    for f in os.listdir(data_path):
        if f[-3:] == 'txt':
            continue

        img_id = f[:-4]
        img_path = os.path.join(data_path, f)
        target_path = os.path.join(data_path, img_id + '.txt')

        if mode == 'train':
            data.append({'img_id': img_id, 'img': img_path, 'target': target_path})
        else:
            data.append({'img_id': img_id, 'img': img_path})

    return data


class PanoBatchGenerator:

    def __init__(self, all_data, cf, mode='train'):
        self.all_data = all_data
        self.cf = cf
        self.batch_size = 1 if mode=='test' else cf.batch_size
        self.mode = mode

    def ready(self, do_shuffle=True):
        self.idx = 0
        if do_shuffle:
            random.shuffle(self.all_data)

    def next_batch(self):
        cur_data_batch = self.all_data[self.idx:self.idx+self.batch_size]
        if len(cur_data_batch) == 0:
            return None

        self.idx += self.batch_size
        cur_id_batch = []
        cur_imgs_batch = []
        cur_bbox_batch = []
        cur_cls_batch = []
        cur_path_batch = []

        for i in range(len(cur_data_batch)):
            data = cur_data_batch[i] #data = {'id': ....}

            img = cv2.imread(data['img'], cv2.IMREAD_GRAYSCALE)
            orig_h, orig_w = img.shape
            img = dutils.crop_center_and_resize(img, size=(512,768))
            img = np.expand_dims(img, axis=0) # (1, y, x)

            if self.mode == 'train':
                bbox, bbox_cls = dutils.get_bbox(data['target'], orig_h, orig_w) # (n_box, (y1, x1, y2, x2))
            else:
                bbox, bbox_cls = [], []

            cur_imgs_batch.append(img)
            cur_id_batch.append(data['img_id'])
            cur_path_batch.append(data['img'])
            cur_bbox_batch.append(bbox)
            cur_cls_batch.append(bbox_cls)

        cur_id_batch = np.array(cur_id_batch)
        cur_path_batch = np.array(cur_path_batch)
        cur_imgs_batch = np.array(cur_imgs_batch)

        if self.mode == 'train':
            cur_bbox_batch = np.array(cur_bbox_batch)
            cur_cls_batch = np.array(cur_cls_batch)
            return {'id': cur_id_batch,
                    'path': cur_path_batch,
                    'img': cur_imgs_batch,
                    'bbox': cur_bbox_batch,
                    'cls': cur_cls_batch}

        return {'id': cur_id_batch,
                'path': cur_path_batch,
                'img': cur_imgs_batch}
