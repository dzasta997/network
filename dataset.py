import os
import re

import numpy as np
import cv2
from scipy import stats


def age_regression():
    directory = "data/raw/FGNET/images"
    save_directory = "data/processed/age"
    for filename in os.listdir(directory):
        image = cv2.imread(os.path.join(directory, filename), cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (100, 100))
        cv2.imwrite(os.path.join(save_directory, filename), image)
        # print(numpy.asarray(image))


def load_age_regression():
    directory = "data/processed/age"
    X = np.array([], int)
    Y = np.array([], int)
    X = np.array(
        [np.array(cv2.imread(os.path.join(directory, filename), cv2.IMREAD_GRAYSCALE).flatten()) for filename in
         os.listdir(directory)])
    Y = np.array(
        [int(re.findall(r'\d+', filename.split("A")[1].split(".")[0])[0]) for filename in os.listdir(directory)])
    print(X.shape, Y.shape)
    return X, Y


def heart_classification(filename: str = "processed.hungarian.data"):
    directory = "data/raw/heart"
    matrix = np.genfromtxt(os.path.join(directory, filename), delimiter=",")
    matrix[np.isnan(matrix)] = 0
    inputs = matrix[:, :-1]

    inputs = stats.zscore(inputs, axis=None)
    outputs = matrix[:, -1]

    return inputs, outputs



if __name__ == "__main__":
    heart_classification()
    # age_regression()
