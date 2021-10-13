import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
import os
import cv2  
from tqdm import tqdm
from utils import show_image, main_color

from sklearn.cluster import KMeans
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76

import multiprocessing

train_path = "data/train"
test_path = "data/test"

train = os.listdir(train_path)
test = os.listdir(test_path)

train_image = [cv2.cvtColor(cv2.imread(os.path.join(train_path, image)), cv2.COLOR_BGR2RGB)  for image in train ]

test_image = [cv2.cvtColor(cv2.imread(os.path.join(test_path, image)), cv2.COLOR_BGR2RGB)  for image in test ]


trian_label = pd.read_csv("data/train.csv")
color_total_train,weight_total_train = main_color(train_image,3,False)

trian_label["color"] = pd.Series(color_total_train)
trian_label["weight"] = pd.Series(weight_total_train)
trian_label.to_csv("data/with_color_train.csv")


# test_label = pd.read_csv("data/imagenames.csv")
# color_total_test,weight_total_test = main_color(test_image,3,False)

# test_label["color"] = pd.Series(color_total_test)
# test_label["weight"] = pd.Series(weight_total_test)


# test_label.to_csv("data/with_color_test.csv")

