import numpy as np
import os
import pandas as pd
import tensorflow as tf

from typing import Any
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
from tensorflow.keras.models import Model


def generate_csv(src_path: str, dataset: str):
    """
    generate csv from datasets (ps: customize your class2index)
    :param src_path:
    :param dataset:
    :return: random shuffled csv contains path and class
    """
    print('Generating csv...')
    class2index = {
        'sipakmed': {
            'im_Dyskeratotic': 0,
            'im_Koilocytotic': 1,
            'im_Metaplastic': 2,
            'im_Parabasal': 3,
            'im_Superficial-Intermediate': 4
        },
        'herlev': {
            'normal': 0,
            'abnormal': 1,
        },
        'mendeley': {
            'High squamous intra-epithelial lesion': 0,
            'Low squamous intra-epithelial lesion': 1,
            'Negative for Intraepithelial malignancy': 2,
            'Squamous cell carcinoma': 3
        }
    }
    data = []
    for cls in os.listdir(src_path):
        for img in os.listdir(os.path.join(src_path, cls)):
            data.append((os.path.join(src_path, cls, img),
                        class2index[dataset][cls]))
    print(f'The total number of images: {len(data)}')
    df = pd.DataFrame(data, columns=['path', 'class'])
    df = shuffle(df)  # random shuffle
    df.to_csv('data.csv', index=False)
    print('Generating csv successfully!')
    return df


def k_fold_splits(x: np.ndarray, y: np.ndarray, n_splits: int = 5):
    """
    k-fold cross-validation splits
    :param x: samples
    :param y: labels
    :param n_splits: number of folds
    :return: train splits and val splits
    """
    func = StratifiedKFold(n_splits=n_splits, random_state=0, shuffle=True)
    x_train_splits, x_val_splits, y_train_splits, y_val_splits = [], [], [], []
    for train_index, val_index in func.split(x, y):
        x_train, x_val = x[train_index], x[val_index]
        y_train, y_val = y[train_index], y[val_index]

        x_train_splits.append(x_train)
        x_val_splits.append(x_val)

        y_train_splits.append(y_train)
        y_val_splits.append(y_val)
    return x_train_splits, x_val_splits, y_train_splits, y_val_splits


def create_model(model_name: str, output: int):
    """
    create model as base learner
    :param model_name:
    :param output: classes of datasets
    :return: keras model
    """
    if model_name == 'InceptionResNetV2':
        model = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(
            include_top=False,
            weights='imagenet',
            input_shape=(256,256,3)
        )
    elif model_name == 'InceptionV3':
        model = tf.keras.applications.inception_v3.InceptionV3(
            include_top=False,
            weights='imagenet',
            input_shape=(256,256,3)
        )
    elif model_name == 'Xception':
        model = tf.keras.applications.Xception(
            include_top=False,
            weights='imagenet',
            input_shape=(256,256,3)
        )
    else:
        return None
    x = tf.keras.layers.GlobalAveragePooling2D()(model.output)
    x = tf.keras.layers.Dense(output, activation='softmax')(x)
    model = Model(inputs=model.input, outputs=x)
    return model
