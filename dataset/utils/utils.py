import os
import cv2
import glob
import numpy as np
from csv import reader, writer


def make_directory(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def read_csv(paths, trainset_mask=False):
    data = list()
    paths = paths if isinstance(paths, list) else [paths]
    for path in paths:
        read = list()
        with open(path, "r") as csvfile:
            # For trainset mask we want to have "wireframe/02%-03%" instead of the "wireframe/0203"
            if trainset_mask:
                path_split = os.path.basename(os.path.splitext(path)[0]).split("-") # ["wireframe", "0203"]
                path_split[1] = f"{path_split[1][:2]}%-{path_split[1][2:]}%"
                path = "/".join(path_split)
            else:
                path = "/".join(os.path.basename(os.path.splitext(path)[0]).split("-"))

            lines = reader(csvfile)
            for x in lines:
                read.append(x + [path])
        data += read
    return data


def write_csv(path, out):
    assert isinstance(out, list)
    with open(path, "a") as csvfile:
        writer(csvfile).writerow(out)


def read_txt(path):
    out = list()
    with open(path, "r") as txt:
        for x in txt:
            out.append(x[:-1])
    return out


def write_txt(path, out):
    out = out if isinstance(out, list) else [out]
    with open(path, "a") as txt:
        txt.write("\n".join(out)+"\n")


def count_file(path, ext=[]):
    num = 0
    for x in ext:
        num += len(glob.glob(os.path.join(path, "**", "*"+x), recursive=True))
    return num
