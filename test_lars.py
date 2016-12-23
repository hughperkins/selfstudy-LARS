import numpy as np
import os
import urllib
import shutil
import hashlib
from os import path
from os.path import join
import argparse
import sklearn.datasets
import csv


unprocessed_data_url = 'https://web.stanford.edu/~hastie/Papers/LARS/diabetes.data'
diabetes_md5 = 'af0c583c28547d76cd2db5f5c67de7e8'
diabetes_train_splitsize = 0.8

lars_datadir = join(os.environ['HOME'], 'lars_data')


def get_md5(filepath):
    with open(filepath, 'rb') as f:
        dat = f.read()
    return hashlib.md5(dat).hexdigest()


def download_dataset(url, dirpath, filename, expected_md5):
    if not path.isdir(dirpath):
        os.makedirs(dirpath)
    file_exists = False
    datafilepath = join(dirpath, filename)
    if path.isfile(datafilepath):
        md5sum = get_md5(datafilepath)
        if md5sum == expected_md5:
            file_exists = True
    if not file_exists:
        print('downloading %s...' % filename)
        with urllib.request.urlopen(url) as response, open(datafilepath, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
            print(get_md5(datafilepath))
            print('... downloaded %s' % filename)


def fetch_diabetes(subset='train'):
    download_dataset(unprocessed_data_url, lars_datadir, 'diabetes-data', diabetes_md5)
    filepath = join(lars_datadir, 'diabetes-data')
    rows = []
    # x_rows = []
    # y_rows = []
    with open(filepath, 'r') as f:
        csvreader = csv.reader(f, delimiter='\t')
        for i, row in enumerate(csvreader):
            if i == 0:
                continue
            row = [float(v) for v in row]
            data = row[:-1]
            target = row[-1]
            rows.append({'x': data, 'y': target})
    train_rows = []
    test_rows = []
    total_N = len(rows)
    train_N = int(total_N * diabetes_train_splitsize)
    rand = np.random.mtrand.RandomState(seed=123)
    train_idx = set(rand.choice(total_N, size=(train_N,), replace=False))
    for n, row in enumerate(rows):
        if n in train_idx:
            train_rows.append(row)
        else:
            test_rows.append(row)

    def rows_to_np(rows):
        x_rows = []
        y_rows = []
        for row in rows:
            x_rows.append(row['x'])
            y_rows.append(row['y'])
        X = np.array(x_rows)
        y = np.array(y_rows)
        return X, y

    train_X, train_y = rows_to_np(train_rows)
    test_X, test_y = rows_to_np(test_rows)

    def get_add_mul(X):
        add = - np.average(X, 0)
        X1 = X + add
        mul = 1 / np.sqrt((X1 * X1).sum(0))
        return add, mul

    X_add, X_mul = get_add_mul(train_X)
    y_add = - np.average(train_y)

    train_X = (train_X + X_add) * X_mul
    train_y = train_y + y_add
    test_X = (test_X + X_add) * X_mul
    test_y = test_y + y_add

    if subset == 'train':
        return sklearn.datasets.base.Bunch(data=train_X, target=train_y)
    elif subset == 'test':
        return sklearn.datasets.base.Bunch(data=test_X, target=test_y)
    else:
        raise Exception('unknown subset %s' % subset)


def run_lars(train):
    X = train.data
    y = train.target
    m = len(X[0])
    print('m', m)
    n = len(X)
    print('n', n)


def run():
    train = fetch_diabetes(subset='train')
    test = fetch_diabetes(subset='test')
    print((train.data * train.data).sum(0))
    print((test.data * test.data).sum(0))
    print(np.average(train.target, 0))
    print(np.average(test.target, 0))
    run_lars(train)


if __name__ == '__main__':
    run()
