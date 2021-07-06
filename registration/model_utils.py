import sys
import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import open3d as o3d

from copy import deepcopy


_EPS = 1e-5  # To prevent division by zero


def clones(module, N):
	return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
	d_k = query.size(-1)
	# scores = torch.matmul(query, key.transpose(-2, -1).contiguous()) / math.sqrt(d_k)
	scores = (torch.matmul(query.cpu(), key.transpose(-2, -1).contiguous().cpu()) / math.sqrt(d_k)).cuda()
	# B x 4 x points x points
	if mask is not None:
		scores = scores.masked_fill(mask == 0, -1e9)
	p_attn = F.softmax(scores, dim=-1)
	return torch.matmul(p_attn, value), p_attn
# attention is re-weighted to [0~1] with a sum=1
# B x 4 x points x 128; B x 4 x points x points; 


def nearest_neighbor(src, dst):
	inner = -2 * torch.matmul(src.transpose(1, 0).contiguous(), dst)  # src, dst (num_dims, num_points)
	distances = -torch.sum(src ** 2, dim=0, keepdim=True).transpose(1, 0).contiguous() - inner - torch.sum(dst ** 2, dim=0, keepdim=True)
	distances, indices = distances.topk(k=1, dim=-1)
	return distances, indices


def get_edge_features(x, idx):
	batch_size, num_points, k = idx.size()
	device = torch.device('cuda')

	idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
	idx = idx + idx_base
	idx = idx.view(-1)

	_, num_dims, _ = x.size()
	x = x.transpose(2, 1).contiguous() 
    # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
	feature = x.view(batch_size * num_points, -1)[idx, :]
	feature = feature.view(batch_size, num_points, k, num_dims)
	return feature


def get_graph_feature(x, k=20):
	# x = x.squeeze()
	idx = knn(x, k)  # (batch_size, num_points, k)
	batch_size, num_points, _ = idx.size()
	_, num_dims, _ = x.size()
	feature = get_edge_features(x, idx)
	x = x.transpose(2, 1).contiguous()
	x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
	feature = torch.cat((feature, x), dim=3).permute(0, 3, 1, 2).contiguous()
	return feature


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()
    idx = pairwise_distance.topk(k=k, dim=-1)[1].detach()  # (batch_size, num_points, k)
    return idx


def get_rri_cluster(cluster_pts, k):
	'''
	Input:
		cluster_pts: B 3 S M; k: int;
	Output:
		cluster_feats: B 4k S M;
	'''
	batch_size = cluster_pts.size()[0]
	num_samples = cluster_pts.size()[2]
	num_clusters = cluster_pts.size()[3]
	
	cluster_pts_ = cluster_pts.permute(0, 3, 1, 2).contiguous().view(batch_size*num_clusters, 3, num_samples) # BM 3 S
	idx = knn(cluster_pts_, k+1)[:,:,1:] # BM S k
	cluster_npts_ = (get_edge_features(cluster_pts_, idx)).permute(0, 3, 2, 1).contiguous() # BM 3 k S
	
	p = cluster_pts_.transpose(1, 2).contiguous().unsqueeze(2).repeat(1,1,k,1) # BM S k 3
	q = cluster_npts_.transpose(1, 3).contiguous() # BM S k 3

	rp = torch.norm(p, None, dim=-1, keepdim=True) # BM S k 1
	rq = torch.norm(q, None, dim=-1, keepdim=True) # BM S k 1
	pn = p / rp
	qn = q / rq
	dot = torch.sum(pn * qn, dim=-1, keepdim=True) # BM S k 1

	theta = torch.acos(torch.clamp(dot, -1, 1)) # BM S k 1

	T_q = (q - dot * p)
	T_q = T_q.cpu().numpy()
	pn = pn.cpu().numpy()
	sin_psi = np.sum(np.cross(T_q[:, :, None], T_q[:, :, :, None]) * pn[:, :, None], -1)
	cos_psi = np.sum(T_q[:, :, None] * T_q[:, :, :, None], -1)
	
	psi = np.arctan2(sin_psi, cos_psi) % (2*np.pi)

	# psi = np.where(psi < 0, psi+2*np.pi, psi)

	idx = np.argpartition(psi, 1)[:, :, :, 1:2]
	# phi: BM x S x k x 1, projection angles
	phi = torch.from_numpy(np.take_along_axis(psi, idx, axis=-1)).to(theta.device)

	feat = torch.cat([rp, rq, theta, phi], axis=-1).view(batch_size, num_clusters, num_samples, 4*k).transpose(1,3).contiguous() # B 4k S M
	return feat


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


class FCBNReLU(nn.Module):
	def __init__(self, in_planes, out_planes):
		super(FCBNReLU, self).__init__()
		self.linear = nn.Linear(in_planes, out_planes, bias=False)
		self.bn = nn.BatchNorm1d(out_planes)
		self.relu = nn.ReLU(inplace=True)
	def forward(self, x):
		return self.relu(self.bn(self.linear(x)))


class Conv1DBNReLU(nn.Module):
    def __init__(self, in_channel, out_channel, ksize=1):
        super(Conv1DBNReLU, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, ksize, bias=False)
        self.bn = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv1DBlock(nn.Module):
    def __init__(self, channels, ksize=1):
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
    def __init__(self, in_channel, out_channel, ksize=1):
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
    def __init__(self, channels, ksize=1):
        super(Conv2DBlock, self).__init__()
        self.conv = nn.ModuleList()
        for i in range(len(channels)-2):
            self.conv.append(Conv2DBNReLU(channels[i], channels[i+1], ksize))
        self.conv.append(nn.Conv2d(channels[-2], channels[-1], ksize))

    def forward(self, x):
        for conv in self.conv:
            x = conv(x)
        return x


class SVDHead(nn.Module):
    def __init__(self, args):
        super(SVDHead, self).__init__()
        self.emb_dims = 33 if args.use_fpfh else args.descriptor_size 
        self.reflect = nn.Parameter(torch.eye(3), requires_grad=False)
        self.reflect[2, 2] = -1

    def forward(self, src, src_corr, weights=None):
        src_centered = src - src.mean(dim=2, keepdim=True)
        src_corr_centered = src_corr - src_corr.mean(dim=2, keepdim=True)

        if weights == None:
            H = torch.matmul(src_centered, src_corr_centered.transpose(2, 1).contiguous())
        else:
            H = torch.matmul(src_centered * weights, src_corr_centered.transpose(2, 1).contiguous())

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
        
        if weights == None:
            t = torch.matmul(-R, src.mean(dim=2, keepdim=True)) + src_corr.mean(dim=2, keepdim=True)
        else:
            t = torch.matmul(-R, (weights * src).sum(dim=2, keepdim=True)) + (weights * src_corr).sum(dim=2, keepdim=True)
        return R, t.view(src.size(0), 3)