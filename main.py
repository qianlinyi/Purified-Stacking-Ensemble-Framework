import numpy as np
from fusion import fusion
import argparse
from utils import generate_csv, k_fold_splits
from sklearn.model_selection import train_test_split
from data_preprocessing import data_preprocessing
from purification import purification
from relearning import relearning
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # filter meaningless log


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=16, help='batch size for mini-batch training')
parser.add_argument('--classes', type=int, default=5, help='classes of datasets')
parser.add_argument('--classifier_num', type=int, default=3, help='number of base_learners')
parser.add_argument('--dataset', type=str, default='sipakmed', help='name of dataset')
parser.add_argument('--epoch_number', type=int, default=60, help='number of epochs for k-fold training')
parser.add_argument('--fold_number', type=int, default=5, help='number of folds')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate for k-fold training')
parser.add_argument('--path', type=str, default='./', help='path of datasets')
args = parser.parse_args()
models = ['InceptionV3', 'InceptionResNetV2', 'Xception']

df = generate_csv(src_path=args.path, dataset=args.dataset)
x, y = np.array(df['path']), np.array(df['class'])

# split datasets into training set and testing set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0, stratify=y)

# k-fold splits
x_train_splits, x_val_splits, y_train_splits, y_val_splits = k_fold_splits(x_train, y_train, n_splits=args.fold_number)

new_x_train = np.empty(shape=(0, args.classes * args.classifier_num))
new_y_train = np.empty(shape=(0,))
new_x_test = np.zeros(shape=(len(y_test), args.classes * args.classifier_num))
new_y_test = np.zeros(shape=(len(y_test),))


# metrics of base_learner
base_acc = [[] for _ in range(args.classifier_num)]
base_pre = [[] for _ in range(args.classifier_num)]
base_rec = [[] for _ in range(args.classifier_num)]
base_f1 = [[] for _ in range(args.classifier_num)]


for i in range(args.fold_number):
    # stage 1
    k_fold_train, k_fold_val, k_fold_test, test, y_val, y_test = data_preprocessing(
        x_train_splits[i], y_train_splits[i], x_val_splits[i], y_val_splits[i], x_test, y_test, args.batch_size, i+1)
    models = ['InceptionV3', 'InceptionResNetV2', 'Xception']
    # stage 2
    new_train_split, new_test_split, model_acc, model_pre, model_rec, model_f1 = fusion(
        k_fold_train, k_fold_val, k_fold_test, test, y_test, models, len(y_val), len(y_test), args.classes, args.batch_size, args.epoch_number, args.lr, i+1)
    new_x_train = np.concatenate((new_x_train, new_train_split), axis=0)
    new_y_train = np.concatenate((new_y_train, y_val), axis=0)
    new_x_test = np.add(new_x_test, new_test_split)
    new_y_test = np.add(new_y_test, y_test)
    for j in range(args.classifier_num):
        base_acc[j].append(model_acc[j])
        base_pre[j].append(model_pre[j])
        base_rec[j].append(model_rec[j])
        base_f1[j].append(model_f1[j])

print(f'size of training set for meta-learner: {new_x_train.shape}, {new_y_train.shape}')
print(f'size of testing set for meta-learner: {new_x_test.shape}, {new_y_test.shape}')
np.savetxt('results/base_acc.csv', base_acc, delimiter=',')
np.savetxt('results/base_pre.csv', base_pre, delimiter=',')
np.savetxt('results/base_rec.csv', base_rec, delimiter=',')
np.savetxt('results/base_f1.csv', base_f1, delimiter=',')
meta_learner_x_test_data = np.divide(new_x_test, args.fold_number)
meta_learner_y_test_data = np.divide(new_y_test, args.fold_number)
np.savetxt('results/new_x_train.csv', new_x_train, delimiter=',')
np.savetxt('results/new_y_train.csv', new_y_train, delimiter=',')
np.savetxt('results/new_x_test.csv', meta_learner_x_test_data, delimiter=',')
np.savetxt('results/new_y_test.csv', meta_learner_y_test_data, delimiter=',')

# base_acc = np.loadtxt('results/base_acc.csv', delimiter=',')
# base_pre = np.loadtxt('results/base_pre.csv', delimiter=',')
# base_rec = np.loadtxt('results/base_rec.csv', delimiter=',')
# base_f1 = np.loadtxt('results/base_f1.csv', delimiter=',')

for i in range(args.classifier_num):
    print(f'Model Name: {models[i]}')
    print(f'Accuracy - avg:{np.average(base_acc[i])} std:{np.std(base_acc[i])}')
    print(f'Precision - avg:{np.average(base_pre[i])} std:{np.std(base_pre[i])}')
    print(f'Recall - avg:{np.average(base_rec[i])} std:{np.std(base_rec[i])}')
    print(f'F1 - avg:{np.average(base_f1[i])} std:{np.std(base_f1[i])}')
    print('')

# new_x_train = np.loadtxt('results/new_x_train.csv', delimiter=',')
# new_y_train = np.loadtxt('results/new_y_train.csv', delimiter=',')
# new_x_test = np.loadtxt('results/new_x_test.csv', delimiter=',')
# new_y_test = np.loadtxt('results/new_y_test.csv', delimiter=',')

# stage 3
meta_x_train, meta_y_train, meta_x_test, meta_y_test, correct_preds, correct_labels = purification(
    new_x_train, new_y_train, new_x_test, new_y_test, args.classes, args.classifier_num)

# stage 4
relearning(meta_x_train, meta_y_train, meta_x_test, meta_y_test, correct_preds, correct_labels, args.classes)
