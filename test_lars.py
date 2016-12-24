import numpy as np
import os
import urllib
import shutil
import math
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


def vector_len(vector):
    return np.sqrt(np.sum(vector * vector))


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
    beta = np.zeros((m,), dtype=np.float32)
    sign = np.zeros((m,), dtype=np.int32)
    sign[j] = 1

    while len(active_set) < m:
        # print('cur_pred', cur_pred[:5])
        residual = y - cur_pred
        print('len residual', vector_len(residual), 'len y', vector_len(y), 'len(cur_pred)', vector_len(cur_pred))
        mse = np.sqrt(np.sum(residual * residual))
        print('mse', mse)

        pred_from_beta = X.dot(beta)
        print(np.sum(np.abs(pred_from_beta - cur_pred)))

        # print('residual', residual[:5])
        # print('avg abs residual', np.average(np.abs(residual)))
        # print('avg square residual', np.average(residual * residual))
        cur_corr = X.transpose().dot(residual)
        # print('cur_corr', cur_corr)

        X_a = X[:, list(active_set)]
        X_a *= sign[list(active_set)]
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

        # print('length equiangular', np.sqrt(np.sum(equiangular * equiangular)))

        # print('equiangular.shape', equiangular.shape)
        # assert equiangular.shape[1] == 1
        # equiangular = equiangular[:, 0]
        # print('equiangular[:5]', equiangular[:5])
        # print('equiangular.shape', equiangular.shape)
        cos_angle = X.transpose().dot(equiangular)
        # print('a.shape', a.shape)
        # print('cos_angle', cos_angle)
        print('np.arccos(cos_angle) * 180/3.1416', np.arccos(cos_angle) * 180/3.1416)
        # print('a[:5, :5]', a[:5, :5])
        gamma = None
        largest_abs_correlation = np.abs(cur_corr).max()
        print('largest_abs_correlation', largest_abs_correlation)
        next_j = None
        next_sign = 0
        for j in range(m):
            if j in active_set:
                continue
            # print('j', j)
            # print(cur_corr.shape, a.shape)
            v0 = (largest_abs_correlation - cur_corr[j]) / (A_a - cos_angle[j]).item()
            v1 = (largest_abs_correlation + cur_corr[j]) / (A_a + cos_angle[j]).item()
            print(j, 'v0', v0, 'v1', v1)
            if v0 > 0 and (gamma is None or v0 < gamma):
                next_j = j
                gamma = v0
                next_sign = 1
            if v1 > 0 and (gamma is None or v1 < gamma):
                gamma = v1
                next_j = j
                next_sign = -1
        # print('gamma', gamma)

        # a . b = |a| |b| cos theta
        # theta = acos(a . b / |a| / |b|)
        # equiangular and x_j are length 1, so
        # angle = acos(equiangular . x_j)
        # then coeff = gamma / len(active_set) * cos(theta)
        #            = gamma / len(active_set) * x_j . equiangular
        # for j in range(m):
            # angle = math.acos(cos_angle[j]) * 180 / 3.1416
            # coeff = gamma / len(active_set) / cos_angle[j]
            # print('j %s angle %s' % (j, angle), 'coeff', coeff)
            # new_pred = cur_pred + coeff * X[:, j]
            # print()
            # print('coeff', coeff)
        # test_pred = cur_pred
        # test_pred = np.zeros((n,), dtype=np.float32)
        # print('active set')

        # coeffs, eg for 3 vectors
        # c_0 * x_0 + c_1 * x_1 + c_2 * x_2 = equiangular
        # we know: x_0, x_1, x_2, equiangular
        # they're all vectors, so equation must hold for each dimension
        # ie, we have equations:
        # c_0 * x_0_0 + c_1 * x_1_0 + c_2 * x_2_0 = equi_0
        # c_0 * x_0_1 + c_1 * x_1_1 + c_2 * x_2_1 = equi_1
        # c_0 * x_0_2 + c_1 * x_1_2 + c_2 * x_2_2 = equi_2
        # ...
        # comparing with doc at
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.solve.html#numpy.linalg.solve
        # their x_0, x_1, x_2 is our c_0, c_1, c_2
        # their constants are our x_0_0, ...
        # edit since not square, trying
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.lstsq.html#numpy.linalg.lstsq
        sa = X_a
        sb = equiangular * gamma
        print(sa.shape)
        print(sb.shape)
        sx = np.linalg.lstsq(sa, sb)
        print('sx', sx)
        for i, j in enumerate(active_set):
            beta[j] += sx[0][i] * sign[j]
        # for j in active_set:
            # angle = math.acos(cos_angle[j]) * 180 / 3.1416
            # coeff = gamma / len(active_set) / cos_angle[j]  # * sign[j]
            # coeff = gamma / cos_angle[j]
            # print('j %s angle %s' % (j, angle), 'coeff', coeff)
            # beta[j] += coeff
            # proj = (coeff * X[:, j]).dot(equiangular)
            # print('j', j, 'proj', proj)
            # test_pred += coeff * X[:, j]
        # print('len test_pred', vector_len(test_pred))
        # print('new_pred - test_pred', vector_len(cur_pred + gamma * equiangular - test_pred))
        # print('gamma * equiangular - test_pred', vector_len(gamma * equiangular - test_pred))

        print('next j', next_j, 'next sign', next_sign, 'gamma', gamma, 'new max correlation: %s' % (
            largest_abs_correlation - gamma * A_a))
        # angle = 
        # for j in active_set:
        #     beta[j] += gamma * 
        # cur_pred += gamma * equiangular
        cur_pred = X.dot(beta)
        active_set.add(next_j)
        sign[next_j] = next_sign
        # print('cur_pred[:5]', cur_pred[:5])
        print('beta', beta)

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
