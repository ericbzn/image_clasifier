#!/usr/bin/python
'''
<image_classifier.py:
Color-based image classifier. This code uses six different methods to classify the superheroes images.

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

    def __init__(self, color_space='LAB', hist_size=8, database_path='', query_path=''):
        """
        :param color_space:
        :param hist_size:
        :param database_path:
        :param query_path:
        :param metric:
        """
        self.color_space = color_space
        self.hist_size = hist_size
        self.database_path = database_path
        self.query_path = query_path

    def get_database_histogram(self, database_path):
        imgs_database = sorted(os.listdir(database_path))
        self.classes = [s.replace('.png', '') for s in imgs_database]

        color_pixles = np.array(Parallel(n_jobs=num_cores, require='sharedmem')(delayed(self.img_color_pixels)(imgs_database[ii], self.database_path) for ii in range(len(imgs_database))))
        self.database_hist  = np.array(Parallel(n_jobs=num_cores, require='sharedmem')(delayed(self.histogram_3d)(color_pixles[ii, :, :]) for ii in range(len(imgs_database))))

    def compare_image(self, query_image='ironman', metric='emd'):
        self.get_database_histogram(self.database_path)

        query_image = query_image + '.png'
        query_color_pixels = self.img_color_pixels(query_image, self.query_path)
        query_hist = self.histogram_3d(query_color_pixels)

        if metric == 'corr':
            method = 0
        elif metric == 'inter':
            method = 2
        elif metric == 'bhatt':
            method = 3
        elif metric == 'chi2':
            method = 4
        elif metric == 'kl':
            method = 5
        elif metric == 'emd':
            method = 6

        if method <= 5:
            hist1 = np.float32(query_hist / query_hist.sum())
            self.distances = np.array(Parallel(n_jobs=num_cores, require='sharedmem')(delayed(bin2bin_hist_comp)(hist1, self.database_hist[ii, :, :, :], method) for ii in range(len(self.classes))))

        if method == 6:
            hist1 = query_hist
            self.distances = np.array(Parallel(n_jobs=num_cores, require='sharedmem')(delayed(ot_hist_comp)(hist1, self.database_hist[ii, :, :, :]) for ii in range(len(self.classes))))

        self.show_result(self.distances, query_image, metric)

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

    def show_result(self, distances, query_img, metric):
        if metric == 'corr':
            order = np.argsort(-distances)
        elif metric == 'inter':
            order = np.argsort(-distances)
        else:
            order = distances.argsort()

        fig2, ax = plt.subplots(4, 9, figsize=(26, 26))
        fig2.suptitle('COMPARISON RESULT (Ordered by Similarity) \n Metric: ' + metric + ', Hist. size: %i' % self.hist_size + ', Color space: ' + self.color_space)
        plt.setp(ax, xticks=[], xticklabels=[], yticks=[], yticklabels=[])

        ax[0, 1].imshow(read_img(self.query_path + query_img))
        ax[0, 1].set_xlabel('Query image')

        ax[0, 4].imshow(read_img(self.database_path + self.classes[order[0]] + '.png'))
        ax[0, 4].set_xlabel('Best match (%s)' % metric)

        if query_img == 'ironman_a.png' or query_img == 'ironman_b.png':
            query_img = 'ironman.png'

        ax[0, 7].imshow(read_img(self.database_path + query_img))
        ax[0, 7].set_xlabel('Ground Truth')

        order_array = np.arange(0, 36, 1) - 9
        order_array = order_array.reshape((4, 9))

        for ii in range(3):
            for jj in range(9):
                if order_array[ii+1, jj] == 26:
                    ax[ii+1, jj].set_xticks([])
                    ax[ii+1, jj].set_yticks([])
                    ax[ii+1, jj].set_xlabel('')
                else:
                    temp_img = read_img(self.database_path + self.classes[order[order_array[ii+1, jj]]] + '.png')
                    name = 'd = %.2f' % distances[order[order_array[ii+1, jj]]]
                    ax[ii+1, jj].imshow(temp_img)
                    ax[ii+1, jj].set_xticks([])
                    ax[ii+1, jj].set_yticks([])
                    ax[ii+1, jj].set_xlabel(name)

        [ax[0, 0].spines[s].set_visible(False) for s in ax[0, 0].spines]
        [ax[0, 2].spines[s].set_visible(False) for s in ax[0, 0].spines]
        [ax[0, 3].spines[s].set_visible(False) for s in ax[0, 0].spines]
        [ax[0, 5].spines[s].set_visible(False) for s in ax[0, 0].spines]
        [ax[0, 6].spines[s].set_visible(False) for s in ax[0, 0].spines]
        [ax[0, 8].spines[s].set_visible(False) for s in ax[0, 0].spines]
        [ax[3, 8].spines[s].set_visible(False) for s in ax[0, 0].spines]
