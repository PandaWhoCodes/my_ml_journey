import os
import timeit
import cv2
from skimage import io as io
import face_recognition as fr
import numpy as np
import pickle
from tqdm import tqdm
from sklearn import datasets, svm, metrics
import random
from matplotlib import pyplot as plt


def extract_data():
    with open('classifier.pkl', 'rb') as f:
        gender_dataset = pickle.load(f)
        # random.shuffle(gender_dataset)
        return gender_dataset


def load_train_data(pkl):
    train_data = list()
    gender_train = list()

    embedding_list_test = list()
    gender_label_list_test = list()

    for emb, label in pkl[1:1200]:
        train_data.append(emb)
        gender_train.append(label)

    for emb, label in pkl[1201:]:
        embedding_list_test.append(emb)
        gender_label_list_test.append(label)

    print('length of embedding train list: {}'.format(len(train_data)))
    print('lenght of label train list: {}'.format(len(gender_train)))
    print('length of embedding test list: {}'.format(len(embedding_list_test)))
    print('lenght of label test list: {}'.format(len(gender_label_list_test)))

load_train_data(extract_data())