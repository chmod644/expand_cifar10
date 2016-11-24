#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import argparse
import numpy as np
import cv2

LABEL_BYTES = 1
HEIGHT = 32
WIDTH = 32
DEPTH = 3
IMAGE_BYTES = HEIGHT * WIDTH * DEPTH


def read_data(fobj):
    record = fobj.read(LABEL_BYTES + IMAGE_BYTES)

    # Detect EOF
    if len(record) == 0:
        return None, None

    record = np.fromstring(record, dtype=np.uint8)

    label = record[0]
    image = record[1:]

    image = np.reshape(image, [DEPTH, HEIGHT, WIDTH])
    image = np.transpose(image, [1, 2, 0])

    return label, image


def expand_files(files, in_dir, out_dir):
    cnt = 0

    image_dir = os.path.join(out_dir, "image")
    label_dir = os.path.join(out_dir, "label")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

    for f in files:
        ifpath = os.path.join(in_dir, f)

        with open(ifpath, 'rb') as fr:
            while True:
                label, image = read_data(fr)

                if label is None:
                    break

                # Write image
                ofpath = os.path.join(image_dir, "{:05}.png".format(cnt))
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(ofpath, image)

                # Write label
                ofpath = os.path.join(label_dir, "{:05}.txt".format(cnt))
                with open(ofpath, 'wt') as fw:
                    fw.write(str(label))

                cnt += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--in', dest='in_dir', required=True,
            help='input directory.')
    parser.add_argument(
            '--out', dest='out_dir', required=True,
            help='output directory.')

    args = parser.parse_args()

    train_files = ["data_batch_{}.bin".format(i) for i in range(1, 5)]
    test_files = ["test_batch.bin"]

    train_out_dir = os.path.join(args.out_dir, "train")
    test_out_dir = os.path.join(args.out_dir, "test")

    expand_files(train_files, args.in_dir, train_out_dir)
    expand_files(test_files, args.in_dir, test_out_dir)
