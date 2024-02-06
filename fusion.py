import numpy as np
import tensorflow as tf

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from utils import create_model


def fusion(k_fold_train, k_fold_val, k_fold_test, test, y_test, models: list, training_set_len, testing_set_len, classes, batch_size, epoch_number, learning_rate, fold_number):
    """
    stage 2 fusion: training of base learners
    :return: new training set split and testing set split for one fold
    """
    new_train_split = np.empty(shape=(training_set_len, 0))  # training set split
    new_test_split = np.empty(shape=(testing_set_len, 0))  # testing set split
    model_acc, model_pre, model_rec, model_f1 = [], [], [], []
    for model_name in models:
        base_learner = create_model(model_name, classes)
        filepath = f'models/{model_name}_fold_{fold_number}.hdf5'

        # experimental configuration
        base_learner.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=tf.keras.optimizers.experimental.Adam(
                learning_rate = learning_rate
            ),
            metrics=['accuracy']
        )
        lr_scheduler = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=5
        )
        checkpoint = ModelCheckpoint(
            filepath=filepath, monitor="val_loss", save_best_only=True, mode="min", save_weights_only=True
        )
        base_learner.fit(
            x=k_fold_train,
            callbacks=[checkpoint, lr_scheduler],
            validation_data=k_fold_val,
            epochs=epoch_number
        )
        # load best model
        base_learner.load_weights(filepath)

        # generate training set for meta learner
        preds1 = base_learner.predict(
            x=k_fold_test,
            batch_size=batch_size
        )
        # concatenate the outputs of each base learner
        new_train_split = np.concatenate((new_train_split, preds1), axis=1)

        # generate testing set for meta learner
        preds2 = base_learner.predict(
            x=test,
            batch_size=batch_size
        )
        new_test_split = np.concatenate((new_test_split, preds2), axis=1)
        # save metrics of each base learner
        y_preds = []
        for pred in preds2:
            y_preds.append(np.argmax(pred))
        accuracy = accuracy_score(y_test, y_preds)
        precison = np.average(precision_score(y_test, y_preds, average=None))
        recall = np.average(recall_score(y_test, y_preds, average=None))
        f1 = np.average(f1_score(y_test, y_preds, average=None))
        model_acc.append(accuracy)
        model_pre.append(precison)
        model_rec.append(recall)
        model_f1.append(f1)

    return new_train_split, new_test_split, model_acc, model_pre, model_rec, model_f1
