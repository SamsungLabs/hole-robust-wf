import glob
import json
import math
import os
import random
import cv2

import numpy as np
import numpy.linalg as LA
import torch
from skimage import io
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from lcnn.config import C, M


class WireframeDataset(Dataset):
    def __init__(self, rootdir, split, valid_mask_dir=None):
        rootdir = rootdir if not C.io.is_pseudo_labeling or split == "valid" else C.io.pseudodir
        rootdir = rootdir if not rootdir.endswith("/") else rootdir[:-1]
        self.rootdir = rootdir
        filelist = [os.path.join(root, filename)
                for root, dirnames, filenames in os.walk(f"{rootdir}/{split}/")
                for filename in filenames if filename.endswith("_label.npz")]
        filelist.sort()

        print(f"n{split}:", len(filelist))
        self.split = split
        self.filelist = filelist

        self.epoch = 0
        self.intervals = list(map(lambda x: os.path.basename(x), sorted(glob.glob(os.path.join(C.io.datadir, C.io.dir_conditional_data, "*")))))
        if valid_mask_dir == None or valid_mask_dir == "None":
            self.valid_mask = [None] * len(self.filelist)
        else:
            self.valid_mask = sorted(glob.glob(os.path.join(valid_mask_dir, "*")))

        if self.split == "train" and C.io.is_pseudo_labeling:
            self.train_mask = list()
            for interval in self.intervals:
                self.train_mask.append([os.path.basename(x) for x in sorted(glob.glob(os.path.join(C.io.datadir, C.io.dir_conditional_data, interval, "*")))])
        else:
            self.train_mask = None

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        iname = self.filelist[idx][:-10].replace("_a0", "").replace("_a1", "") + ".png"
        image = io.imread(iname).astype(float)[:, :, :3]
        if not image.shape[:2] == (512,512):
            image = np.clip(cv2.resize(image, (512,512), interpolation=cv2.INTER_CUBIC), 0, 255)
        if "a1" in self.filelist[idx]:
            image = image[:, ::-1, :]
        image_gt = np.copy(image)
        image_gt = (image_gt - M.image.mean) / M.image.stddev
        image_gt = np.rollaxis(image_gt, 2).copy()
        image = (image - M.image.mean) / M.image.stddev

        if C.io.is_conditional_training:
            if self.split == "valid":
                if self.valid_mask[idx] == None:
                    mask = np.zeros(image.shape)[:,:,0:1]
                else:
                    mask = io.imread(self.valid_mask[idx]).astype(float)[:, :, np.newaxis] / 255.0
            elif self.split == "train":
                interval = self.intervals[int((len(self.intervals)/C.optim.max_epoch)*self.epoch)]
                if C.io.is_pseudo_labeling:
                    mname = self.train_mask[int((len(self.intervals)/C.optim.max_epoch)*self.epoch)][random.randint(0, len(self.train_mask[0]) - 1)]
                else:
                    mname = ("_%02d" % random.randint(0, len(self.intervals) - 1)).join(os.path.splitext(os.path.basename(iname)))
                mname = os.path.join(C.io.datadir, C.io.dir_conditional_data, interval, mname)
                mask = io.imread(mname).astype(float)[:, :, np.newaxis] / 255.0
            else:
                raise ValueError("Undefined self.split: ", self.split)
            image = np.concatenate((image * (1 - mask), mask), axis=2)

        image = np.rollaxis(image, 2).copy()

        # -------------------------------------- pseudo labeled dataset --------------------------------------
        # npz["lines_pseudo"]: [Np, 2, 2]   Pseudo-positive lines represented with junction coordinates
        # 
        # To use pseudo labeled dataset on L-CNN framework,
        # npz["lines_pseudo"] must be converted to npz["jmap"], npz["joff"], ..., npz["lneg"].
        # ----------------------------------------------------------------------------------------------------

        # npz["jmap"]: [J, H, W]    Junction heat map
        # npz["joff"]: [J, 2, H, W] Junction offset within each pixel
        # npz["lmap"]: [H, W]       Line heat map with anti-aliasing
        # npz["junc"]: [Na, 3]      Junction coordinates
        # npz["Lpos"]: [M, 2]       Positive lines represented with junction indices
        # npz["Lneg"]: [M, 2]       Negative lines represented with junction indices
        # npz["lpos"]: [Np, 2, 3]   Positive lines represented with junction coordinates
        # npz["lneg"]: [Nn, 2, 3]   Negative lines represented with junction coordinates
        #
        # For junc, lpos, and lneg that stores the junction coordinates, the last
        # dimension is (y, x, t), where t represents the type of that junction.
        with np.load(self.filelist[idx]) as npz:
            target = {
                name: torch.from_numpy(npz[name]).float()
                for name in ["jmap", "joff", "lmap"]
            }
            lpos = np.random.permutation(npz["lpos"])[: M.n_stc_posl]
            lneg = np.random.permutation(npz["lneg"])[: M.n_stc_negl]
            npos, nneg = len(lpos), len(lneg)
            lpre = np.concatenate([lpos, lneg], 0)
            for i in range(len(lpre)):
                if random.random() > 0.5:
                    lpre[i] = lpre[i, ::-1]
            ldir = lpre[:, 0, :2] - lpre[:, 1, :2]
            ldir /= np.clip(LA.norm(ldir, axis=1, keepdims=True), 1e-6, None)
            feat = [
                lpre[:, :, :2].reshape(-1, 4) / 128 * M.use_cood,
                ldir * M.use_slop,
                lpre[:, :, 2],
            ]
            feat = np.concatenate(feat, 1)
            meta = {
                "junc": torch.from_numpy(npz["junc"][:, :2]),
                "jtyp": torch.from_numpy(npz["junc"][:, 2]).byte(),
                "Lpos": self.adjacency_matrix(len(npz["junc"]), npz["Lpos"]),
                "Lneg": self.adjacency_matrix(len(npz["junc"]), npz["Lneg"]),
                "lpre": torch.from_numpy(lpre[:, :, :2]),
                "lpre_label": torch.cat([torch.ones(npos), torch.zeros(nneg)]),
                "lpre_feat": torch.from_numpy(feat),
            }

        return torch.from_numpy(image_gt).float(), torch.from_numpy(image).float(), meta, target

    def adjacency_matrix(self, n, link):
        mat = torch.zeros(n + 1, n + 1, dtype=torch.uint8)
        link = torch.from_numpy(link)
        if len(link) > 0:
            mat[link[:, 0], link[:, 1]] = 1
            mat[link[:, 1], link[:, 0]] = 1
        return mat


def collate(batch):
    return (
        default_collate([b[0] for b in batch]),
        default_collate([b[1] for b in batch]),
        [b[2] for b in batch],
        default_collate([b[3] for b in batch]),
    )
