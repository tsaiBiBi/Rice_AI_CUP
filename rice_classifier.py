from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import time
from tensorflow.keras.utils import plot_model
import numpy as np

IMAGE_SIZE = 32

# load model
img_input = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name='img_input')
hidden = layers.Conv2D(filters=12, kernel_size=(3,3), strides=1, activation='relu', name='hidden', padding='same')(img_input)
pool = layers.MaxPooling2D(pool_size=(2, 2), name='pool')(hidden)
hidden_ft = layers.Flatten()(pool)

hidden2 = layers.Dense(512, activation='sigmoid', name='hidden2')(hidden_ft)
dropout2 = layers.Dropout(rate=0.25)(hidden2)

hidden3 = layers.Dense(512, activation='relu', name='hidden3')(dropout2)
dropout3 = layers.Dropout(rate=0.25)(hidden3)

hidden4 = layers.Dense(512, activation='relu', name='hidden4')(dropout3)
dropout4 = layers.Dropout(rate=0.25)(hidden4)

outputs = layers.Dense(2, activation='softmax', name='Output')(dropout4)
model = keras.Model(inputs=img_input, outputs=outputs)
model.load_weights("./riceClassifier_model.h5")

def classify(img):
    img_normalize = img.astype('float32') / 255.0
    prediction = model.predict(img_normalize)
    # prediction=np.argmax(prediction,axis=1)
    prediction = prediction[:,1]
    print(prediction)
    prediction = np.where(prediction> 0.5, 1, 0)
    print(prediction)
    return prediction

import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

def cluster(data_set):
    # parameter setting
    eps = IMAGE_SIZE//2
    MinPts = 1

    # DBSCAN method
    model = DBSCAN(eps, MinPts)
    cluster = model.fit(data_set)
    unique = np.unique(cluster.labels_)
    labels = cluster.labels_

    # results visualization
    # plt.figure()
    # plt.scatter(data_set[:,0], data_set[:,1], c = labels)
    # plt.axis('equal')
    # plt.title('Prediction')
    # plt.show()
    
    return labels

def find_center(data_set, labels):
    group = np.unique(labels)
    center_xy = np.empty((group.shape[0],2))
    for i in group:
        idx = np.where(labels==i)
        center_xy[i] = (np.sum(data_set[idx], axis=0)/data_set[idx].shape).astype('int')
    return center_xy