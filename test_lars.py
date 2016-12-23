import numpy as np
import requests
import os
import urllib
import shutil
import hashlib
from os import path
from os.path import join
import argparse
import sklearn.datasets
import csv
import pandas


unprocessed_data_url = 'https://web.stanford.edu/~hastie/Papers/LARS/diabetes.data'
diabetes_md5 = 'af0c583c28547d76cd2db5f5c67de7e8'

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


def fetch_diabetes():
    download_dataset(unprocessed_data_url, lars_datadir, 'diabetes-data', diabetes_md5)
    filepath = join(lars_datadir, 'diabetes-data')
    x_rows = []
    y_rows = []
    with open(filepath, 'r') as f:
        csvreader = csv.reader(f, delimiter='\t')
        for i, row in enumerate(csvreader):
            if i == 0:
                continue
            row = [float(v) for v in row]
            data = row[:9]
            target = row[10]
            x_rows.append(data)
            y_rows.append(target)
            # print(row)
    X = np.array(x_rows)
    y = np.array(y_rows)
    print(np.mean(X, 0))
    X = X - np.average(X, 0)
    X = X / np.sqrt(X * X).sum(0)
    print('X', X)
    y -= np.mean(y)
    print('y', y)
    # print(df)


def run():
    fetch_diabetes()


if __name__ == '__main__':
    run()
