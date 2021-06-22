import os
import sys
import glob
import h5py
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# from util import knn, batch_choice
import open3d as o3d
# from open3d.open3d.geometry import estimate_normals
from train_utils import rotation_error, translation_error, rmse_loss, rt_to_transformation
from model_utils import rotation_geodesic_error


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def batch_choice(data, k, p=None, replace=False):
    # data is [B, N]
    out = []
    for i in range(len(data)):
        out.append(np.random.choice(data[i], size=k, p=p[i], replace=replace))
    out = np.stack(out, 0)
    return out


class FPFH(nn.Module):
    def __init__(self, radius_normal=0.1, radius_feature=0.2):
        super(FPFH, self).__init__()
        self.radius_normal = radius_normal
        self.radius_feature = radius_feature

    def forward(self, xyz):
        xyz = xyz.transpose(1, 2).cpu().numpy()
        res = np.zeros((xyz.shape[0], 33, xyz.shape[1]))

        for i in range(len(xyz)):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz[i])
            # estimate_normals(pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=self.radius_normal, max_nn=30))
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=self.radius_normal, max_nn=30))
            pcd_fpfh = o3d.registration.compute_fpfh_feature(pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=self.radius_feature, max_nn=100))
            res[i] = pcd_fpfh.data

        res = torch.from_numpy(res).float().cuda()
        return res


class Conv1DBNReLU(nn.Module):
    def __init__(self, in_channel, out_channel, ksize):
        super(Conv1DBNReLU, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, ksize, bias=False)
        self.bn = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv1DBlock(nn.Module):
    def __init__(self, channels, ksize):
        super(Conv1DBlock, self).__init__()
        self.conv = nn.ModuleList()
        for i in range(len(channels)-2):
            self.conv.append(Conv1DBNReLU(channels[i], channels[i+1], ksize))
        self.conv.append(nn.Conv1d(channels[-2], channels[-1], ksize))

    def forward(self, x):
        for conv in self.conv:
            x = conv(x)
        return x


class Conv2DBNReLU(nn.Module):
    def __init__(self, in_channel, out_channel, ksize):
        super(Conv2DBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, ksize, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv2DBlock(nn.Module):
    def __init__(self, channels, ksize):
        super(Conv2DBlock, self).__init__()
        self.conv = nn.ModuleList()
        for i in range(len(channels)-2):
            self.conv.append(Conv2DBNReLU(channels[i], channels[i+1], ksize))
        self.conv.append(nn.Conv2d(channels[-2], channels[-1], ksize))

    def forward(self, x):
        for conv in self.conv:
            x = conv(x)
        return x


class Propagate(nn.Module):
    def __init__(self, in_channel, emb_dims):
        super(Propagate, self).__init__()
        self.conv2d = Conv2DBlock((in_channel, emb_dims, emb_dims), 1)
        self.conv1d = Conv1DBlock((emb_dims, emb_dims), 1)

    def forward(self, x, idx):
        batch_idx = np.arange(x.size(0)).reshape(x.size(0), 1, 1)
        nn_feat = x[batch_idx, :, idx].permute(0, 3, 1, 2)
        x = nn_feat - x.unsqueeze(-1)
        x = self.conv2d(x)
        x = x.max(-1)[0]
        x = self.conv1d(x)
        return x


class GNN(nn.Module):
    def __init__(self, emb_dims=64):
        super(GNN, self).__init__()
        self.propogate1 = Propagate(3, 64)
        self.propogate2 = Propagate(64, 64)
        self.propogate3 = Propagate(64, 64)
        self.propogate4 = Propagate(64, 64)
        self.propogate5 = Propagate(64, emb_dims)

    def forward(self, x):
        nn_idx = knn(x, k=12)

        x = self.propogate1(x, nn_idx)
        x = self.propogate2(x, nn_idx)
        x = self.propogate3(x, nn_idx)
        x = self.propogate4(x, nn_idx)
        x = self.propogate5(x, nn_idx)

        return x


class SVDHead(nn.Module):
    def __init__(self, args):
        super(SVDHead, self).__init__()
        self.emb_dims = 33 if args.use_fpfh else args.descriptor_size 
        self.reflect = nn.Parameter(torch.eye(3), requires_grad=False)
        self.reflect[2, 2] = -1

    def forward(self, src, src_corr, weights):
        src_centered = src - src.mean(dim=2, keepdim=True)
        src_corr_centered = src_corr - src_corr.mean(dim=2, keepdim=True)

        H = torch.matmul(src_centered * weights.unsqueeze(1), src_corr_centered.transpose(2, 1).contiguous())

        U, S, V = [], [], []
        R = []

        for i in range(src.size(0)):
            u, s, v = torch.svd(H[i])
            r = torch.matmul(v, u.transpose(1, 0).contiguous())
            r_det = torch.det(r)
            if r_det < 0:
                u, s, v = torch.svd(H[i])
                v = torch.matmul(v, self.reflect)
                r = torch.matmul(v, u.transpose(1, 0).contiguous())
            R.append(r)

            U.append(u)
            S.append(s)
            V.append(v)

        U = torch.stack(U, dim=0)
        V = torch.stack(V, dim=0)
        S = torch.stack(S, dim=0)
        R = torch.stack(R, dim=0)

        t = torch.matmul(-R, (weights.unsqueeze(1) * src).sum(dim=2, keepdim=True)) + (weights.unsqueeze(1) * src_corr).sum(dim=2, keepdim=True)
        return R, t.view(src.size(0), 3)
        # return R, t.view(batch_size, 3)


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.emb_dims = 33 if args.use_fpfh else args.descriptor_size 
        self.num_iter = args.num_iters
        self.emb_nn = FPFH() if args.use_fpfh else GNN(self.emb_dims)
        self.significance_fc = Conv1DBlock((self.emb_dims, 64, 32, 1), 1)
        self.sim_mat_conv1 = nn.ModuleList([Conv2DBlock((self.emb_dims*2+4, 32, 32), 1) for _ in range(self.num_iter)])
        self.sim_mat_conv2 = nn.ModuleList([Conv2DBlock((32, 32, 1), 1) for _ in range(self.num_iter)])
        self.weight_fc = nn.ModuleList([Conv1DBlock((32, 32, 1), 1) for _ in range(self.num_iter)])
        self.head = SVDHead(args=args)


    def forward(self, src, tgt, T_gt):

        # ##### only pass ground truth while training #####
        # if not (self.training or (R_gt is None and t_gt is None)):
        #     raise Exception('Passing ground truth while testing')
        # ##### only pass ground truth while training #####
        self.pts = src
        R_gt = T_gt[:, :3, :3]
        t_gt = T_gt[:, :3, 3]

        src = src.transpose(1,2).contiguous()
        tgt = tgt.transpose(1,2).contiguous()

        ##### getting ground truth correspondences #####
        if self.training:
            src_gt = torch.matmul(R_gt, src) + t_gt.unsqueeze(-1)
            dist = src_gt.unsqueeze(-1) - tgt.unsqueeze(-2)
            min_dist, min_idx = (dist ** 2).sum(1).min(-1) # [B, npoint], [B, npoint]
            min_dist = torch.sqrt(min_dist)
            min_idx = min_idx.cpu().numpy() # drop to cpu for numpy
            match_labels = (min_dist < 0.05).float()
            indicator = match_labels.cpu().numpy()
            indicator += 1e-5
            pos_probs = indicator / indicator.sum(-1, keepdims=True)
            indicator = 1 + 1e-5 * 2 - indicator
            neg_probs = indicator / indicator.sum(-1, keepdims=True)
        ##### getting ground truth correspondences #####

        ##### get embedding and significance score #####
        tgt_embedding = self.emb_nn(tgt)
        src_embedding = self.emb_nn(src)
        src_sig_score = self.significance_fc(src_embedding).squeeze(1)
        tgt_sig_score = self.significance_fc(tgt_embedding).squeeze(1)
        ##### get embedding and significance score #####

        ##### hard point elimination #####
        num_point_preserved = src.size(-1) // 6
        if self.training:
            candidates = np.tile(np.arange(src.size(-1)), (src.size(0), 1))
            pos_idx = batch_choice(candidates, num_point_preserved//2, p=pos_probs)
            neg_idx = batch_choice(candidates, num_point_preserved-num_point_preserved//2, p=neg_probs)
            src_idx = np.concatenate([pos_idx, neg_idx], 1)
            tgt_idx = min_idx[np.arange(len(src))[:, np.newaxis], src_idx]
        else:
            src_idx = src_sig_score.topk(k=num_point_preserved, dim=-1)[1]
            src_idx = src_idx.cpu().numpy()
            tgt_idx = tgt_sig_score.topk(k=num_point_preserved, dim=-1)[1]
            tgt_idx = tgt_idx.cpu().numpy()
        batch_idx = np.arange(src.size(0))[:, np.newaxis]
        if self.training:
            match_labels = match_labels[batch_idx, src_idx]
        src = src[batch_idx, :, src_idx].transpose(1, 2)
        src_embedding = src_embedding[batch_idx, :, src_idx].transpose(1, 2)
        src_sig_score = src_sig_score[batch_idx, src_idx]
        tgt = tgt[batch_idx, :, tgt_idx].transpose(1, 2)
        tgt_embedding = tgt_embedding[batch_idx, :, tgt_idx].transpose(1, 2)
        tgt_sig_score = tgt_sig_score[batch_idx, tgt_idx]
        ##### hard point elimination #####

        ##### initialize #####
        similarity_matrix_list = []
        R = torch.eye(3).unsqueeze(0).expand(src.size(0), -1, -1).cuda().float()
        t = torch.zeros(src.size(0), 3).cuda().float()
        loss = 0.
        ##### initialize #####

        for i in range(self.num_iter):

            ##### stack features #####
            batch_size, num_dims, num_points = src_embedding.size()
            _src_emb = src_embedding.unsqueeze(-1).repeat(1, 1, 1, num_points)
            _tgt_emb = tgt_embedding.unsqueeze(-2).repeat(1, 1, num_points, 1)
            similarity_matrix = torch.cat([_src_emb, _tgt_emb], 1)
            ##### stack features #####

            ##### compute distances #####
            diff = src.unsqueeze(-1) - tgt.unsqueeze(-2)
            dist = (diff ** 2).sum(1, keepdim=True)
            dist = torch.sqrt(dist)
            diff = diff / (dist + 1e-8)
            ##### compute distances #####

            ##### similarity matrix convolution to get features #####
            similarity_matrix = torch.cat([similarity_matrix, dist, diff], 1)
            similarity_matrix = self.sim_mat_conv1[i](similarity_matrix)
            ##### similarity matrix convolution to get features #####

            ##### soft point elimination #####
            weights = similarity_matrix.max(-1)[0]
            weights = self.weight_fc[i](weights).squeeze(1)
            ##### soft point elimination #####

            ##### similarity matrix convolution to get similarities #####
            similarity_matrix = self.sim_mat_conv2[i](similarity_matrix)
            similarity_matrix = similarity_matrix.squeeze(1)
            similarity_matrix = similarity_matrix.clamp(min=-20, max=20)
            ##### similarity matrix convolution to get similarities #####

            ##### negative entropy loss #####
            if self.training and i == 0:
                src_neg_ent = torch.softmax(similarity_matrix, dim=-1)
                src_neg_ent = (src_neg_ent * torch.log(src_neg_ent)).sum(-1)
                tgt_neg_ent = torch.softmax(similarity_matrix, dim=-2)
                tgt_neg_ent = (tgt_neg_ent * torch.log(tgt_neg_ent)).sum(-2)
                loss = loss + F.mse_loss(src_sig_score, src_neg_ent.detach()) + F.mse_loss(tgt_sig_score, tgt_neg_ent.detach())
            ##### negative entropy loss #####

            ##### matching loss #####
            if self.training:
                temp = torch.softmax(similarity_matrix, dim=-1)
                temp = temp[:, np.arange(temp.size(-2)), np.arange(temp.size(-1))]
                temp = - torch.log(temp)
                match_loss = (temp * match_labels).sum() / match_labels.sum()
                loss = loss + match_loss
            ##### matching loss #####

            ##### finding correspondences #####
            corr_idx = similarity_matrix.max(-1)[1]
            src_corr = tgt[np.arange(tgt.size(0))[:, np.newaxis], :, corr_idx].transpose(1, 2)
            ##### finding correspondences #####

            ##### soft point elimination loss #####
            if self.training:
                weight_labels = (corr_idx == torch.arange(corr_idx.size(1)).cuda().unsqueeze(0)).float()
                weight_loss = F.binary_cross_entropy_with_logits(weights, weight_labels)
                loss = loss + weight_loss
            ##### soft point elimination loss #####

            ##### hybrid point elimination #####
            weights = torch.sigmoid(weights)
            weights = weights * (weights >= weights.median(-1, keepdim=True)[0]).float()
            weights = weights / (weights.sum(-1, keepdim=True) + 1e-8)
            ##### normalize weights #####

            ##### get R and t #####
            rotation_ab, translation_ab = self.head(src, src_corr, weights)
            rotation_ab = rotation_ab.detach() # prevent backprop through svd
            translation_ab = translation_ab.detach() # prevent backprop through svd
            src = torch.matmul(rotation_ab, src) + translation_ab.unsqueeze(-1)
            R = torch.matmul(rotation_ab, R)
            t = torch.matmul(rotation_ab, t.unsqueeze(-1)).squeeze() + translation_ab
            ##### get R and t #####

        mse = (rotation_geodesic_error(R, R_gt) + translation_error(t, t_gt)) #.mean()
        r_err = rotation_error(R, R_gt)
        t_err = translation_error(t, t_gt)
        self.T = rt_to_transformation(R, t.unsqueeze(-1))
        rmse = rmse_loss(self.pts, self.T, T_gt)

        # return R, t, loss
        return loss, r_err, t_err, rmse, mse

    def get_transform(self):
        return self.T #, self.scores12