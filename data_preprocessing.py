import numpy as np
from keras_preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau


def process(x: list):
    """
    load images
    :param x: paths of images
    :return: ndarray of images
    """
    return np.array([
        img_to_array(
            load_img(img_path, target_size=(256, 256))
        ) for img_path in x
    ])


def data_preprocessing(x_train: list, y_train: list, x_val: list, y_val: list, x_test: list, y_test: list, batch_size: int, fold_number: int):
    """
    stage 1: data preprocessing
    :return training (split 20% for validation), validation set and testing set for k-fold cross validation, labels
    """
    train_generator = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
    )
    val_generator = ImageDataGenerator()
    test_generator = ImageDataGenerator()

    x_train, y_train, x_val, y_val, x_test, y_test = process(x_train), np.array(y_train), process(x_val), np.array(
        y_val), process(x_test), np.array(y_test)

    x_train, x_val_2, y_train, y_val_2 = train_test_split(x_train, y_train, test_size=0.2, stratify=y_train)

    print(f'fold no -- {fold_number}')
    print(f'size of k-fold training set: {x_train.shape}, {y_train.shape}')
    print(f'size of k-fold validation set: {x_val_2.shape}, {y_val_2.shape}')
    print(f'size of k-fold testing set: {x_val.shape}, {y_val.shape}')
    print(f'size of testing set: {x_test.shape}, {y_test.shape}')

    k_fold_train = train_generator.flow(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
    )
    k_fold_val = val_generator.flow(
        x=x_val_2,
        y=y_val_2,
        batch_size=batch_size,
    )
    k_fold_test = test_generator.flow(
        x=x_val,
        y=y_val,
        batch_size=batch_size,
        shuffle=False
    )
    test = test_generator.flow(
        x=x_test,
        y=y_test,
        batch_size=batch_size,
        shuffle=False
    )
    return k_fold_train, k_fold_val, k_fold_test, test, y_val, y_test 
