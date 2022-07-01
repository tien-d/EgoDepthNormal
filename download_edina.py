#!/usr/bin/env python

"""
Downloads EDINA public data release
Run with python download_edina.py
Modified from ScanNet's download script
"""

import argparse
import os
import urllib.request
import urllib
import tempfile
from tqdm import tqdm
import shutil


BASE_URL = 'https://edina.s3.amazonaws.com'
TRAIN_URL_PREFIX = 'scenes_train'
TEST_URL_PREFIX = 'scenes_test'


# Progress bar util. Adopted from https://github.com/tqdm/tqdm/blob/master/examples/tqdm_wget.py
def my_hook(t):
    """Wraps tqdm instance.
    Don't forget to close() or __exit__()
    the tqdm instance once you're done with it (easiest using `with` syntax).
    Example
    -------
    # >>> with tqdm(...) as t:
    # ...     reporthook = my_hook(t)
    # ...     urllib.urlretrieve(..., reporthook=reporthook)
    """
    last_b = [0]

    def update_to(b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] or -1,
            remains unchanged.
        """
        if tsize not in (None, -1):
            t.total = tsize
        displayed = t.update((b - last_b[0]) * bsize)
        last_b[0] = b
        return displayed

    return update_to


def get_release_scans(release_file):
    scan_lines = urllib.request.urlopen(release_file)
    scans = []
    for scan_line in scan_lines:
        scan_id = scan_line.decode('utf8').rstrip('\n')
        scans.append(scan_id)
    return scans


def download_file(url, out_file):
    out_dir = os.path.dirname(out_file)
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isfile(out_file):
        print('\t' + url + ' > ' + out_file)
        fh, out_file_tmp = tempfile.mkstemp(dir=out_dir)
        f = os.fdopen(fh, 'w')
        f.close()

        # wrap a progress bar hook with tqdm
        with tqdm(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:  # all optional kwargs
            reporthook = my_hook(t)
            urllib.request.urlretrieve(url, out_file_tmp, reporthook=reporthook, data=None)

        os.rename(out_file_tmp, out_file)
    else:
        print('WARNING: skipping download of existing file ' + out_file)


def download_release(release_scans, out_dir, bucket, unzip=False):
    if len(release_scans) == 0:
        return
    print('Downloading EDINA release to ' + out_dir + '...')
    for scan_id in release_scans:
        scan_out_dir = out_dir  # os.path.join(out_dir, scan_id)
        download_scan(scan_id, scan_out_dir, bucket, unzip=unzip)
    print('Downloaded EDINA release.')


def download_scan(scan_id, out_dir, bucket, unzip=False):
    print('Downloading EDINA ' + scan_id + ' ...')
    os.makedirs(out_dir, exist_ok=True)
    url = BASE_URL + '/' + bucket + '/' + scan_id + '.zip'
    out_file = out_dir + '/' + scan_id + '.zip'
    download_file(url, out_file)

    if unzip:
        out_extract_dir = os.path.dirname(out_file)
        print('Unzipping to ', os.path.join(out_extract_dir, scan_id))
        shutil.unpack_archive(out_file, out_extract_dir)

    print('Downloaded scene ' + scan_id)


def main():
    parser = argparse.ArgumentParser(description='Downloads EDINA public data release.')
    parser.add_argument('-o', '--out_dir', required=True, help='directory in which to download')
    parser.add_argument('--id', help='specific scan id to download')
    parser.add_argument('--unzip', action='store_true', help='unzip the downloaded zip file or not')
    args = parser.parse_args()

    # TODO: Training data will be released soon. Will be empty for now!
    release_train_scans = []

    release_test_file = BASE_URL + '/' + 'scenes_test.txt'
    release_test_scans = get_release_scans(release_test_file)

    out_dir_train_scans = os.path.join(args.out_dir, 'scenes_train')
    out_dir_test_scans = os.path.join(args.out_dir, 'scenes_test')

    if args.id:
        scan_id = args.id
        is_test_scan = scan_id in release_test_scans

        # TODO: Training data will be released soon! Only accepting scenes in the test set for now
        if scan_id not in release_train_scans and (not is_test_scan):
            raise Exception(f'ERROR: Invalid scan id: {scan_id}. Please double check the ID at {release_test_file}')
        else:
            print('Downloading scene {}'.format(scan_id))
            out_dir = out_dir_train_scans if not is_test_scan else out_dir_test_scans
            download_scan(scan_id, out_dir, bucket=TEST_URL_PREFIX, unzip=args.unzip)
    else:
        # Download training scenes
        # TODO: Training data will be released soon! Will not download for now

        # Download testing scenes
        print('Downloading the full test set, with {} scenes: {}'.format(len(release_test_scans), release_test_scans))
        download_release(release_test_scans, out_dir_test_scans, bucket=TEST_URL_PREFIX, unzip=args.unzip)


if __name__ == '__main__':
    main()
