#!/usr/bin/python
'''
<image_classifier.py:
Color-based image classifier. The code uses six different methods to classify the superheroes images.

Copyright (C) <2018>  <Eric Bazan> <eric.bazan@mines-paristech.fr>
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''

import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from skimage.io import imread as read_img
import cv2
import ot
import os
num_cores = multiprocessing.cpu_count()


def ot_hist_comp(hist1, hist2):
    pos1 = np.where(hist1 > 0)
    pos2 = np.where(hist2 > 0)
    w1 = hist1[pos1] / hist1[pos1].sum()
    w2 = hist2[pos2] / hist2[pos2].sum()
    M = ot.dist(np.array(pos1, dtype='int').T, np.array(pos2, dtype='int').T, 'sqeuclidean')
    return ot.emd2(w1, w2, M, processes=multiprocessing.cpu_count())


def bin2bin_hist_comp(hist1, hist2, method):
    hist2 = np.float32(hist2 / hist2.sum())
    return cv2.compareHist(hist1, hist2, method)


class ImageClassifier:

    def __init__(self, color_space='LAB', hist_size=8, p_process='True', database_path='', query_path='', metric='emd'):
        """
        :param color_space:
        :param hist_size:
        :param p_process:
        :param database_path:
        :param query_path:
        :param metric:
        """
        self.color_space = color_space
        self.hist_size = hist_size
        self.p_process = p_process
        self.database_path = database_path
        self.query_path = query_path
        self.metric = metric

    def get_database_histogram(self, database_path):
        imgs_database = sorted(os.listdir(database_path))
        self.classes = [s.replace('.png', '') for s in imgs_database]

        color_pixles = np.array(Parallel(n_jobs=num_cores, require='sharedmem')(delayed(self.img_color_pixels)(imgs_database[ii], self.database_path) for ii in range(len(imgs_database))))
        self.database_hist  = np.array(Parallel(n_jobs=num_cores, require='sharedmem')(delayed(self.histogram_3d)(color_pixles[ii, :, :]) for ii in range(len(imgs_database))))

    def compare_image(self, query_image='ironman'):
        self.get_database_histogram(self.database_path)

        query_image = query_image + '.png'
        query_color_pixels = self.img_color_pixels(query_image, self.query_path)
        query_hist = self.histogram_3d(query_color_pixels)

        if self.metric == 'corr':
            method = 0
        elif self.metric == 'inter':
            method = 2
        elif self.metric == 'bhatt':
            method = 3
        elif self.metric == 'chi2':
            method = 4
        elif self.metric == 'kl':
            method = 5
        elif self.metric == 'emd':
            method = 6
        else:
            raise ValueError('image_classifier_demo.py: the metric ' + self.metric + ' is not supported.')

        if method <= 5:
            hist1 = np.float32(query_hist / query_hist.sum())
            self.distances = np.array(Parallel(n_jobs=num_cores, require='sharedmem')(delayed(bin2bin_hist_comp)(hist1, self.database_hist[ii, :, :, :], method) for ii in range(len(self.classes))))

        if method == 6:
            hist1 = query_hist
            self.distances = np.array(Parallel(n_jobs=num_cores, require='sharedmem')(delayed(ot_hist_comp)(hist1, self.database_hist[ii, :, :, :]) for ii in range(len(self.classes))))

        self.show_result(self.distances, query_image)



    def histogram_3d(self, color_pixels):
        hist, _ = np.histogramdd(color_pixels, bins=self.hist_size)   # range=((0, 255), (0, 255), (0, 255)))
        return hist

    def img_color_pixels(self, img_name, path):

        img = cv2.imread(path + img_name, cv2.CV_8S)

        if self.color_space == 'hls':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        elif self.color_space == 'lab':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        elif self.color_space == 'rgb':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError('image_classifier_demo.py: the color space ' + self.color_space + ' is not supported.')

        m, n, d = img.shape
        color_pixels = np.reshape(img, ((m * n), d))

        return color_pixels

    def show_result(self, distances, query_img):
        order = distances.argsort()
        query_img = read_img(self.query_path + query_img)
        best_math = read_img(self.database_path + self.classes[order[0]] + '.png')

        fig1, ax = plt.subplots(1, 2, figsize=(8, 4))
        fig1.suptitle('COMPARISON RESULT \n Metric: ' + self.metric + ', Hist. size: %i' % self.hist_size + ', Color space: ' + self.color_space)
        ax[0].imshow(query_img)
        ax[1].imshow(best_math)
        ax[0].set_xlabel('Query image')
        ax[1].set_xlabel('Best match, d = %.2f' %distances[order[0]])

        for ii in ax:
            ii.set_xticks([])
            ii.set_yticks([])

        fig2, ax = plt.subplots(5, 5, figsize=(8, 10))
        fig2.suptitle('COMPARISON ARRAY')

        order_array = order[1:].reshape((5, 5))
        for ii in range(5):
            for jj in range(5):
                temp_img = read_img(self.database_path + self.classes[order_array[ii, jj]] + '.png')
                ax[ii, jj].imshow(temp_img)
                ax[ii, jj].set_xticks([])
                ax[ii, jj].set_yticks([])
                ax[ii, jj].set_xlabel('d = %.2f' %distances[order_array[ii, jj]])
        plt.show()

