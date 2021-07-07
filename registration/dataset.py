import h5py
import numpy as np
import os
import open3d as o3d
import torch
from torch.utils.data import Dataset


def jitter_pcd(pcd, sigma=0.01, clip=0.05):
    pcd += np.clip(sigma * np.random.randn(*pcd.shape), -1 * clip, clip)
    return pcd


def random_pose(max_angle, max_trans):
    R = random_rotation(max_angle)
    t = random_translation(max_trans)
    return np.concatenate([np.concatenate([R, t], 1), [[0, 0, 0, 1]]], 0)


def random_rotation(max_angle):
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    angle = np.random.rand() * max_angle
    A = np.array([[0, -axis[2], axis[1]],
                    [axis[2], 0, -axis[0]],
                    [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle) * A + (1 - np.cos(angle)) * np.dot(A, A)
    return R


def random_translation(max_dist):
    t = np.random.randn(3)
    t /= np.linalg.norm(t)
    t *= np.random.rand() * max_dist
    return np.expand_dims(t, 1)


class MVP_RG(Dataset):
    """docstring for MVP_RG"""
    def __init__(self, prefix, args):
        self.prefix = prefix

        if self.prefix == "train":
            f = h5py.File('./data/MVP_Train_RG.h5', 'r')
        elif self.prefix == "val":
            f = h5py.File('./data/MVP_Test_RG.h5', 'r')
        elif self.prefix == "test":
            f = h5py.File('./data/MVP_ExtraTest_RG.h5', 'r')
        
        self.max_angle = args.max_angle / 180 * np.pi
        self.max_trans = args.max_trans

        self.label = f['cat_labels'][:].astype('int32')
        if self.prefix == "test":
            self.src = np.array(f['rotated_src'][:].astype('float32'))
            self.tgt = np.array(f['rotated_tgt'][:].astype('float32'))
        else:
            self.match_level = np.array(f['match_level'][:].astype('int32'))

            match_id = []
            for i in range(len(f['match_id'].keys())):
                ds_data = f['match_id'][str(i)][:]
                match_id.append(ds_data)
            self.match_id = np.array(match_id, dtype=object)

            if self.prefix == "train":
                self.src = np.array(f['src'][:].astype('float32'))
                self.tgt = np.array(f['tgt'][:].astype('float32'))
                if args.max_angle > 45:
                    self.rot_level = int(1)
                else:
                    self.rot_level = int(0)
            elif self.prefix == "val":
                self.src = np.array(f['rotated_src'][:].astype('float32'))
                self.tgt = np.array(f['rotated_tgt'][:].astype('float32'))
                self.transforms = np.array(f['transforms'][:].astype('float32'))
                self.rot_level = np.array(f['rot_level'][:].astype('int32'))

        f.close()
        
        if args.category:
            self.src = self.src[self.label==args.category]
            self.tgt = self.tgt[self.label==args.category]

            if self.prefix is not "test":
                self.match_id = self.match_id[self.label==args.category]
                self.match_level = self.match_level[self.label==args.category]
                if self.prefix == False:
                    self.transforms = self.transforms[self.label==args.category]
                    self.rot_level = self.rot_level[self.label==args.category]
            self.label = self.label[self.label==args.category]

        # print(self.src.shape, self.tgt.shape, self.match_id.shape, self.match_level.shape, self.label.shape)

    def __len__(self):
        return self.src.shape[0]

    def __getitem__(self, index):
        src = self.src[index]
        tgt = self.tgt[index]

        if self.prefix == "train":
            transform = random_pose(self.max_angle, self.max_trans / 2)
            pose1 = random_pose(np.pi, self.max_trans)
            pose2 = transform @ pose1
            src = src @ pose1[:3, :3].T + pose1[:3, 3]
            tgt = tgt @ pose2[:3, :3].T + pose2[:3, 3]
            rot_level = self.rot_level
            match_level = self.match_level[index]

        elif self.prefix == "val":
            transform = self.transforms[index]
            rot_level = self.rot_level[index]
            match_level = self.match_level[index]

        # src = np.random.permutation(src)
        # tgt = np.random.permutation(tgt)

        src = torch.from_numpy(src)
        tgt = torch.from_numpy(tgt)

        if self.prefix is not "test":
            transform = torch.from_numpy(transform)
            match_level = match_level
            rot_level = rot_level
            return src, tgt, transform, match_level, rot_level
        else:
            return src, tgt
