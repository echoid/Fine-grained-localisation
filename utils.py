import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
import os
import cv2  
from tqdm import tqdm

from sklearn.cluster import KMeans
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76


def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))


def show_image(images,name,row,col,size1=15,size2=15):
    n = len(images)
    plt.subplots(figsize=(15, 15))

    for i in range(n):
        plt.subplot(row,col,i+1)
        plt.title(name[i])
        plt.imshow(images[i], cmap='gray')
        plt.axis('off')   
    plt.show()


def main_color(images,n_color,show = False):

    clf = KMeans(n_clusters = n_color)

    color_total = []
    weight_total = []

    for image in tqdm(images):
        modified_image = cv2.resize(image, (600, 400), interpolation = cv2.INTER_AREA)
        flatten = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1], 3)

        labels = clf.fit_predict(flatten)

        counts = Counter(labels)

        center_colors = clf.cluster_centers_

        hex_colors = [RGB2HEX(center_colors[i]) for i in counts.keys()]

        color_total.append(center_colors)
        weight_total.append( [i/sum(counts.values()) for i in counts.values()])



        if show:
            plt.pie(counts.values(), labels = hex_colors, colors = hex_colors)
            plt.show()

    return color_total,weight_total
