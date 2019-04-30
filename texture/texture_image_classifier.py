#!/usr/bin/python
'''
<texture_image_classifier.py:
Color-based image classifier. This code uses six different methods to classify the superheroes images.


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
from scipy.signal import fftconvolve as convolve
from skimage import img_as_float32
import cv2
import ot
import os
import pdb
num_cores = multiprocessing.cpu_count()


def ot_hist_comp(hist1, hist2, dist):
    pos1 = np.where(hist1 >= 0)
    pos2 = np.where(hist2 >= 0)
    w1 = hist1[pos1] / hist1[pos1].sum()
    w2 = hist2[pos2] / hist2[pos2].sum()
    return ot.emd2(w1, w2, dist, processes=multiprocessing.cpu_count())


def bin2bin_hist_comp(hist1, hist2, method):
    hist2 = np.float32(hist2 / hist2.sum())
    return cv2.compareHist(hist1, hist2, method)


def genGabor(freq=0.1, theta=0, K=np.pi):
    sigma_x = sigma_y = 10./freq
    sz = [sigma_x, sigma_y]
    radius = (int(sz[0]/2.0), int(sz[1]/2.0))
    [x, y] = np.meshgrid(range(-radius[0], radius[0]+1), range(-radius[1], radius[1]+1))

    x1 = x * np.cos(theta) + y * np.sin(theta)
    y1 = -x * np.sin(theta) + y * np.cos(theta)

    gauss = freq**2 / (4*np.pi * K**2) * np.exp(- freq**2 / (8*K**2) * (4 * x1**2 + y1**2))

    func = np.cos
    cosenoid = func(freq * x1) * np.exp(K**2 / 2)
    gabor_real = gauss * cosenoid
    func = np.sin
    sinusoid = func(freq * x1) * np.exp(K**2 / 2)
    gabor_imag = gauss * sinusoid
    return np.vectorize(complex)(gabor_real, gabor_imag)


def makeGabor_filters(n_freq=6, n_angles=6):

    theta = np.deg2rad(np.linspace(0, 180, n_angles, endpoint=False))
   # the maximum frequency is given by the Shannon theorem
    max_freq = 3.
   # generate an octave-based, geometric sequence of frequences
    freq = [max_freq]
    for ii in range(n_freq - 1):
        freq.append(freq[ii] / 2)

    filters = []
    for f in freq:
        for t in theta:
            filters.append(genGabor(freq=f, theta=t, K=np.pi))
    return np.array(filters)


def applyGabor_filterbank(img, filterbank):
    num_cores = 8
    img = (img - img.mean()) / img.std()
    def convolve_Gabor(img, filter):
        response = np.sqrt(convolve(img, filter.real, mode='same')**2 + convolve(img, filter.imag, mode='same')**2)
        # response = response - response.mean()
        # response = np.log(1 + response)  # Smoothing strategy
        return response

    gabor_responses = np.array(Parallel(n_jobs=num_cores, prefer='threads')(
        delayed(convolve_Gabor)(img, filter) for filter in filterbank))#, require='sharedmem'

    return gabor_responses

class ImageClassifier:

    def __init__(self, texture_descriptor='gabor', n_freq=4, n_angles=4, database_path='', query_path=''):
        """
        :param texture_descriptor:
        :param hist_size:
        :param database_path:
        :param query_path:
        :param metric:
        """
        self.texture_descriptor = texture_descriptor
        self.n_freq = n_freq
        self.n_angles = n_angles
        self.database_path = database_path
        self.query_path = query_path
        self.filterbank = 0

    def get_database_histogram(self, database_path):
        imgs_database = sorted(os.listdir(database_path))
        self.classes = [img.split('.')[0] for img in imgs_database]

        if self.texture_descriptor == 'gabor':
            self.filterbank = makeGabor_filters(n_freq=self.n_freq, n_angles=self.n_angles)
        elif self.texture_descriptor == 'coomatrix':
            pass

        self.database_hist  = np.array(Parallel(n_jobs=num_cores, prefer='processes')(delayed(self.create_texture_hist)(imgs_database[ii], self.filterbank, self.database_path) for ii in range(len(imgs_database))))


    def create_texture_hist(self, img_name, filterbank, path):
        img = img_as_float32(cv2.resize(cv2.imread(path + img_name, 0), (128, 128)))
        if self.texture_descriptor == 'gabor':  # Gabor filterbank
            texture_responses = np.array(applyGabor_filterbank(img, filterbank))
        elif self.texture_descriptor == 'coomatrix':  # Co-ocurrence Matrix filterbank
            pass

        E_fa = np.zeros(texture_responses.shape[0], dtype=np.float)

        for ii in range(texture_responses.shape[0]):
            E_fa[ii] = np.abs(texture_responses[ii].sum()) ** 2

        texture_hist = E_fa.reshape((self.n_freq, self.n_angles))

        return texture_hist

    def compare_image(self, query_image='ironman', metric='emd'):
        # self.get_database_histogram(self.database_path)

        query_image = query_image + '.png'
        query_hist = self.create_texture_hist(query_image, self.filterbank, self.query_path)

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
            self.distances = np.array(Parallel(n_jobs=num_cores, prefer='processes')(
                delayed(bin2bin_hist_comp)(hist1, self.database_hist[ii], method) for ii in range(len(self.classes))))

        if method == 6:
            hist1 = query_hist
            pos1 = np.where(query_hist >= 0)
            pos2 = pos1
            sz = query_hist.shape[0] * query_hist.shape[1]
            dist = np.zeros((sz, sz))
            for ii in range(sz):
                for jj in range(sz):
                    # delta_freq = np.abs(pos1[0][ii] - pos2[0][jj])
                    delta_freq = np.abs(np.logspace(pos1[0][ii], pos1[0][ii], 1, base=2)[0] -
                                        np.logspace(pos2[0][jj], pos2[0][jj], 1, base=2)[0])
                    delta_theta = min(np.abs(pos1[1][ii] - pos2[1][jj]),
                                      hist1.shape[1] - np.abs(pos1[1][ii] - pos2[1][jj]))
                    dist[ii, jj] = delta_freq + delta_theta
            self.distances = np.array(Parallel(n_jobs=num_cores, prefer='processes')(
                delayed(ot_hist_comp)(hist1, self.database_hist[ii], dist) for ii in range(len(self.classes))))

        self.show_result(self.distances, query_image, metric)



    def show_result(self, distances, query_img, metric):
        if metric == 'corr':
            order = np.argsort(-distances)
        elif metric == 'inter':
            order = np.argsort(-distances)
        else:
            order = distances.argsort()

        fig2, ax = plt.subplots(5, 7, figsize=(26, 26))
        fig2.suptitle('COMPARISON RESULT (Ordered by Similarity) \n Metric: ' + metric + ', N. freq: %i' % self.n_freq + ', N. angles: %i' % self.n_angles + ', Texture descriptor: ' + self.texture_descriptor)
        plt.setp(ax, xticks=[], xticklabels=[], yticks=[], yticklabels=[])

        ax[0, 0].imshow(read_img(self.query_path + query_img), cmap='gray')
        ax[0, 3].set_xlabel('%s' % query_img.split('.')[0])
        ax[0, 0].set_ylabel('Query image')

        ax[0, 3].imshow(read_img(self.database_path + self.classes[order[0]] + '.png'), cmap='gray')
        ax[0, 3].set_xlabel('%s' % self.classes[order[0]])
        ax[0, 3].set_ylabel('Best match')

        ax[0, 6].imshow(read_img(self.database_path + query_img.split('.')[0] + '_a.png'), cmap='gray')
        ax[0, 6].set_xlabel(query_img.split('.')[0] + '_a')
        ax[0, 6].set_ylabel('Ground Truth')

        order_array = np.arange(0, 35, 1) - 7
        order_array = order_array.reshape((5, 7))

        for ii in range(4):
            for jj in range(7):
                temp_img = read_img(self.database_path + self.classes[order[order_array[ii+1, jj]]] + '.png')
                name = 'd = %.2f' % distances[order[order_array[ii+1, jj]]]
                ax[ii+1, jj].imshow(temp_img, cmap='gray')
                ax[ii+1, jj].set_xticks([])
                ax[ii+1, jj].set_yticks([])
                ax[ii+1, jj].set_xlabel(name)

        [ax[0, 1].spines[s].set_visible(False) for s in ax[0, 0].spines]
        [ax[0, 2].spines[s].set_visible(False) for s in ax[0, 0].spines]
        [ax[0, 4].spines[s].set_visible(False) for s in ax[0, 0].spines]
        [ax[0, 5].spines[s].set_visible(False) for s in ax[0, 0].spines]

