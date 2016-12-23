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
diabetes_train_splitsize = 1.0

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
    if len(test_X) > 0:
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

    active_set = set()
    cur_pred = np.zeros((n,), dtype=np.float32)
    residual = y - cur_pred
    cur_corr = X.transpose().dot(residual)
    j = np.argmax(np.abs(cur_corr), 0)
    print('j', j)
    active_set.add(j)

    while len(active_set) < m:
        # print('cur_pred', cur_pred[:5])
        residual = y - cur_pred
        # print('residual', residual[:5])
        print('avg abs residual', np.average(np.abs(residual)))
        print('avg square residual', np.average(residual * residual))
        cur_corr = X.transpose().dot(residual)
        # print('cur_corr', cur_corr)

        X_a = X[:, list(active_set)]
        # print('X_a', X_a)
        G_a = X_a.transpose().dot(X_a)
        # print('G_a', G_a)
        G_a_inv = np.linalg.inv(G_a)
        G_a_inv_red_cols = np.sum(G_a_inv, 1)
        # print(G_a_inv_red_cols)
        A_a = 1 / np.sqrt(np.sum(G_a_inv_red_cols))
        # print('A_a', A_a)
        omega = A_a * G_a_inv_red_cols
        equiangular = X_a.dot(omega)  # .reshape(n)
        # print('equiangular.shape', equiangular.shape)
        # assert equiangular.shape[1] == 1
        # equiangular = equiangular[:, 0]
        # print('equiangular[:5]', equiangular[:5])
        # print('equiangular.shape', equiangular.shape)
        a = X.transpose().dot(equiangular)
        # print('a.shape', a.shape)
        # print('a[:5, :5]', a[:5, :5])
        gamma = None
        largest_abs_correlation = cur_corr.max()
        next_j = None
        for j in range(m):
            if j in active_set:
                continue
            # print('j', j)
            # print(cur_corr.shape, a.shape)
            v0 = (largest_abs_correlation - cur_corr[j]) / (A_a - a[j]).item()
            v1 = (largest_abs_correlation + cur_corr[j]) / (A_a + a[j]).item()
            # print('v0', v0, 'v1', v1)
            if v0 > 0 and (gamma is None or v0 < gamma):
                next_j = j
                gamma = v0
            if v1 > 0 and (gamma is None or v1 < gamma):
                gamma = v1
                next_j = j
        print('next j', next_j)
        active_set.add(next_j)
        # print('gamma', gamma)
        cur_pred += gamma * equiangular
        # print('cur_pred[:5]', cur_pred[:5])

    # residual = y - cur_pred
    # print('resid[:5]', residual[:5])
    # print('avg abs residual', np.average(np.abs(residual)))


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
