import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from copy import deepcopy
from visu_utils import visualize, plot_pcd, plot_matches
from train_utils import *


# import pn2_utils.mm3d_pn2.ops as pn2
sys.path.append("../utils")
from mm3d_pn2 import three_interpolate, furthest_point_sample, gather_points, grouping_operation, three_nn

_EPS = 1e-5  # To prevent division by zero


# def get_ds_ppf(pts, normals, k = 5, ds_ratio = 4):
# 	num_points = pts.size()[1]
# 	num_clusters = num_points // ds_ratio
	
# 	ds_pts, _, _, _, ds_feats = get_ppf(pts, normals, num_clusters, k)
# 	pn_idx = knn_point(1, ds_pts, pts).detach().int()
# 	out_feats = pn2.grouping_operation(ds_feats, pn_idx).squeeze(-1)
# 	return ds_pts, out_feats


def gmm_feats_params(gamma, pts, feats):
	'''
	Inputs:
		gamma: B x N x J (B, 1024, 16)
		pts: B x N x 3
		feats: B x C x N
	'''
	# pi: B x J
	pi = gamma.mean(dim=1) # (B, 16)
	Npi = pi * gamma.shape[1] 
	# mu: B x J x 3; feats: B x C x J
	mu = gamma.transpose(1, 2) @ pts / Npi.unsqueeze(2) # (B x J x N) (B x N x 3) maintain permutation-invariance (B, 16, 3)
	feats = feats @ gamma / Npi.unsqueeze(1)
	# diff: B x N x J x 3
	diff = pts.unsqueeze(2) - mu.unsqueeze(1) # (B x N x 1 x 3) (B x 1 x j x 3) 
	# sigma: B x J x 3 x 3
	eye = torch.eye(3).unsqueeze(0).unsqueeze(1).to(gamma.device)
	sigma = (
		((diff.unsqueeze(3) @ diff.unsqueeze(4)).squeeze() * gamma).sum(dim=1) / Npi
	).unsqueeze(2).unsqueeze(3) * eye # elementwise times, ((BxNxJx1x3, BxNxJx3x1) => BxNxJ => BxJ => BxJx3x3 (diagonal))
	return pi, mu, sigma, feats


def robust_gmm_register(pi_s, mu_s, mu_t, sigma_t, feat_s, feat_t, alpha=0.2, num_iters=5):
	'''
	Inputs:
		pi: B x J
		mu: B x J x 3
		sigma: B x J x 3 x 3
		feat: B x C x J
	'''

	d_k = feat_s.size(1) # 128

	scores = alpha - square_distance(feat_s.transpose(2, 1).contiguous(), feat_t.transpose(1, 2).contiguous()) / math.sqrt(d_k)
	scores = sinkhorn(scores, num_iters=num_iters)
	scores = torch.exp(scores[:, :-1, :-1])

	mu_s_corr = torch.matmul(scores, mu_t) / (torch.sum(scores, dim=2, keepdim=True) + _EPS) # B x J x 3

	conf = torch.sum(scores, dim=2, keepdim=True) / (torch.sum(torch.sum(scores, 2, keepdim=True), 1, keepdim=True) + _EPS) # B x J x 1

	c_s =  pi_s.unsqueeze(1) @ (mu_s * conf) # B x 1 x 3 (center of src J)
	c_s_corr =  pi_s.unsqueeze(1) @ (mu_s_corr * conf) # B x 1 x 3 (center of tgt J)

	mu_s_centered = mu_s - c_s
	mu_s_corr_centered = mu_s_corr - c_s_corr

	Ms = torch.sum((conf * pi_s.unsqueeze(2) * mu_s_centered).unsqueeze(3) @ mu_s_corr_centered.unsqueeze(2) @ sigma_t.inverse(), dim=1)

	U, _, V = torch.svd(Ms.cpu()) # Bx3x3, _, Bx3x3
	U = U.cuda()
	V = V.cuda()
	S = torch.eye(3).unsqueeze(0).repeat(U.shape[0], 1, 1).to(U.device) # Bx3x3
	S[:, 2, 2] = torch.det(V @ U.transpose(1, 2)) # Bx3x3; (3, 3) = det (B, 1, 1)
	R = V @ S @ U.transpose(1, 2)

	u, s, v = torch.svd(Ms, some=False, compute_uv=True)
	rot_mat_pos = v @ u.transpose(-1, -2)
	v_neg = v.clone()
	v_neg[:, :, 2] *= -1
	rot_mat_neg = v_neg @ u.transpose(-1, -2)
	R = torch.where(torch.det(rot_mat_pos)[:, None, None] > 0, rot_mat_pos, rot_mat_neg)
	assert torch.all(torch.det(R) > 0)

	t = c_s_corr.transpose(1, 2) - R @ c_s.transpose(1, 2)
	bot_row = torch.Tensor([[[0, 0, 0, 1]]]).repeat(R.shape[0], 1, 1).to(R.device)
	T = torch.cat([torch.cat([R, t], dim=2), bot_row], dim=1)
	return T, scores


class KeyPointNet(nn.Module):
	def __init__(self, num_keypoints):
		super(KeyPointNet, self).__init__()
		self.num_keypoints = num_keypoints # 256
		self.k_ppf = 5
		# self.k_rri = 5
		# self.ds_ratio = 4

	def forward(self, *input):
		src = input[0]
		tgt = input[1]
		src_n = input[2]
		tgt_n = input[3]
		src_eb = input[4]
		tgt_eb = input[5]

		# _, src_eb1 = get_ds_ppf(src, src_n, k = self.k_ppf, ds_ratio=4)
		# _, src_eb2 = get_ds_ppf(src, src_n, k = self.k_ppf, ds_ratio=8)

		# _, tgt_eb1 = get_ds_ppf(tgt, tgt_n, k = self.k_ppf, ds_ratio=4)
		# _, tgt_eb2 = get_ds_ppf(tgt, tgt_n, k = self.k_ppf, ds_ratio=8)

		# src_eb = torch.cat((src_eb, src_eb1, src_eb2), dim=1)
		# tgt_eb = torch.cat((tgt_eb, tgt_eb1, tgt_eb2), dim=1)

		batch_size, num_dims, num_points = src_eb.size()
		# # b x features x points

		score = torch.matmul(F.normalize(src_eb.transpose(1, 2).contiguous(), dim=2), F.normalize(tgt_eb, dim=1))
		# # score1 = torch.matmul(F.normalize(src_eb1.transpose(1, 2).contiguous(), dim=2), F.normalize(tgt_eb1, dim=1))
		# # score2 = torch.matmul(F.normalize(src_eb2.transpose(1, 2).contiguous(), dim=2), F.normalize(tgt_eb2, dim=1))

		# # score = score + score1 + score2
		
		score_src, _ = torch.max(score, dim=2)
		score_tgt, _ = torch.max(score, dim=1)
		score_src = score_src.unsqueeze(2) # B N 1
		score_tgt = score_tgt.unsqueeze(2)

		src_topk_idx = torch.topk(score_src, k=self.num_keypoints, dim=1, sorted=False)[1]
		tgt_topk_idx = torch.topk(score_tgt, k=self.num_keypoints, dim=1, sorted=False)[1]
		src_keypoints_idx = src_topk_idx.repeat(1, 1, 3)
		tgt_keypoints_idx = tgt_topk_idx.repeat(1, 1, 3)
		src_embedding_idx = src_topk_idx.repeat(1, 1, num_dims).transpose(1,2).contiguous()
		tgt_embedding_idx = tgt_topk_idx.repeat(1, 1, num_dims).transpose(1,2).contiguous()

		src_keypoints = torch.gather(src, dim=1, index=src_keypoints_idx)
		tgt_keypoints = torch.gather(tgt, dim=1, index=tgt_keypoints_idx)
		src_eb = torch.gather(src_eb, dim=2, index=src_embedding_idx)
		tgt_eb = torch.gather(tgt_eb, dim=2, index=tgt_embedding_idx)
		# # gather value according to point index

		# pc_src_pts, pc_src_feats = get_cluster_rri(src, src_eb, self.num_keypoints, 16) # B 3 M S; B C M S
		# pc_tgt_pts, pc_tgt_feats = get_cluster_rri(tgt, tgt_eb, self.num_keypoints, 16)

		# score = torch.matmul(F.normalize(pc_src_feats.permute(0, 2, 3, 1).contiguous(), dim=-1), F.normalize(pc_tgt_feats.permute(0, 2, 1, 3).contiguous(), dim=-2)) # B M S S
		# score_src, _ = torch.max(score, dim=3) # B M S
		# score_src = torch.sum(score_src, dim=-1) # B M
		# score_tgt, _ = torch.max(score, dim=2)
		# score_tgt = torch.sum(score_tgt, dim=-1)

		# # _, src_index = torch.max(score_src, dim=-1) # B
		# # _, tgt_index = torch.max(score_tgt, dim=-1)
		# src_index = torch.topk(score_src, k=1, dim=-1, sorted=False)[1] # B k
		# tgt_index = torch.topk(score_tgt, k=1, dim=-1, sorted=False)[1]
		# src_index = src_index.unsqueeze(1).unsqueeze(-1) # B 1 k 1
		# tgt_index = tgt_index.unsqueeze(1).unsqueeze(-1)
		# src_pts_index = src_index.repeat(1, 3, 1, self.num_keypoints)
		# src_feats_index = src_index.repeat(1, num_dims, 1, self.num_keypoints)
		# tgt_pts_index = tgt_index.repeat(1, 3, 1, self.num_keypoints)
		# tgt_feats_index = tgt_index.repeat(1, num_dims, 1, self.num_keypoints)

		# src_keypoints = torch.gather(pc_src_pts, dim=2, index=src_pts_index).view(batch_size, 3, -1).transpose(1,2).contiguous() # B S 3
		# tgt_keypoints = torch.gather(pc_tgt_pts, dim=2, index=tgt_pts_index).view(batch_size, 3, -1).transpose(1,2).contiguous()
		# src_eb = torch.gather(pc_src_feats, dim=2, index=src_feats_index).view(batch_size, num_dims, -1) # B C S
		# tgt_eb = torch.gather(pc_tgt_feats, dim=2, index=tgt_feats_index).view(batch_size, num_dims, -1)

		return src_keypoints, tgt_keypoints, src_eb, tgt_eb


def robust_SVD(attention, pts1, pts2, weights):
	'''
	Inputs:
		attention: B x M x 1
		pts: B x M x 3
		weights: B x M
	'''
	weights_normalized = weights[..., None] / (torch.sum(weights[..., None], dim=1, keepdim=True) + _EPS) # B M 1

	c_s = torch.sum(attention * pts1 * weights_normalized, dim=1, keepdim=True) # B 1 3
	c_t = torch.sum(attention * pts2 * weights_normalized, dim=1, keepdim=True)
	
	s_centered = pts1 - c_s # B M 3
	t_centered = pts2 - c_t
	
	cov = torch.sum( (attention * s_centered).unsqueeze(3) @ (t_centered * weights_normalized).unsqueeze(2), dim=1 )
	# B J 3 1; B J 1 3; B J 3 3
	
	u, s, v = torch.svd(cov, some=False, compute_uv=True)
	rot_mat_pos = v @ u.transpose(-1, -2)
	v_neg = v.clone()
	v_neg[:, :, 2] *= -1
	rot_mat_neg = v_neg @ u.transpose(-1, -2)
	R = torch.where(torch.det(rot_mat_pos)[:, None, None] > 0, rot_mat_pos, rot_mat_neg)
	assert torch.all(torch.det(R) > 0)
	
	t = c_t.transpose(1, 2) - R @ c_s.transpose(1, 2)
	bot_row = torch.Tensor([[[0, 0, 0, 1]]]).repeat(R.shape[0], 1, 1).to(R.device)
	T = torch.cat([torch.cat([R, t], dim=2), bot_row], dim=1)

	return T


def DCP_SVDHead(src_embedding, tgt_embedding, src, tgt):
	batch_size = src.size(0)

	d_k = src_embedding.size(1) # 512
	scores = torch.matmul(src_embedding.transpose(2, 1).contiguous(), tgt_embedding) / math.sqrt(d_k)
	scores = torch.softmax(scores, dim=2)
	# b x points x points

	src_corr = torch.matmul(tgt, scores.transpose(2, 1).contiguous())

	src_centered = src - src.mean(dim=2, keepdim=True)

	src_corr_centered = src_corr - src_corr.mean(dim=2, keepdim=True)

	H = torch.matmul(src_centered, src_corr_centered.transpose(2, 1).contiguous())
	# b x 3 x 3

	u, s, v = torch.svd(H, some=False, compute_uv=True)
	rot_mat_pos = v @ u.transpose(-1, -2)
	v_neg = v.clone()
	v_neg[:, :, 2] *= -1
	rot_mat_neg = v_neg @ u.transpose(-1, -2)
	R = torch.where(torch.det(rot_mat_pos)[:, None, None] > 0, rot_mat_pos, rot_mat_neg)
	assert torch.all(torch.det(R) > 0)

	t = torch.matmul(-R, src.mean(dim=2, keepdim=True)) + src_corr.mean(dim=2, keepdim=True) # B 3 1
	bot_row = torch.Tensor([[[0, 0, 0, 1]]]).repeat(R.shape[0], 1, 1).to(R.device) # B 1 4
	T = torch.cat([torch.cat([R, t], dim=2), bot_row], dim=1)
	return T


def SVDHead(pts1, pts2, conf=None, std_noise=0):
	# pts1 B 3 N, pts2 B 3 N, conf B N
	num_points = pts1.shape[-1]
	if conf is None:
		conf = 1 / num_points
	else:
		conf = conf.unsqueeze(1) # B 1 N

	if std_noise > 0:
		pts1 = pts1 + torch.normal(0, std_noise, size=pts1.shape).to(pts1.device)
		pts2 = pts2 + torch.normal(0, std_noise, size=pts2.shape).to(pts2.device)
	
	center_pts1 = (pts1 * conf).sum(dim=2, keepdim=True) # B 3 1
	center_pts2 = (pts2 * conf).sum(dim=2, keepdim=True)
	
	pts1_centered = pts1 - center_pts1
	pts2_centered = pts2 - center_pts2

	cov = torch.matmul(pts1_centered * conf, pts2_centered.transpose(1, 2).contiguous())

	u, s, v = torch.svd(cov, some=False, compute_uv=True)
	rot_mat_pos = v @ u.transpose(-1, -2)
	v_neg = v.clone()
	v_neg[:, :, 2] *= -1
	rot_mat_neg = v_neg @ u.transpose(-1, -2)
	R = torch.where(torch.det(rot_mat_pos)[:, None, None] > 0, rot_mat_pos, rot_mat_neg)
	assert torch.all(torch.det(R) > 0)
	
	t = center_pts2 - torch.matmul(R, center_pts1) # R * pts0 + t = pts1
	bot_row = torch.Tensor([[[0, 0, 0, 1]]]).repeat(R.shape[0], 1, 1).to(R.device)
	T = torch.cat([torch.cat([R, t], dim=2), bot_row], dim=1)
	return T


#matrices batch*3*3
#both matrix are orthogonal rotation matrices
#out theta between 0 to 180 degree batch
def rotation_geodesic_error(m1, m2):
	batch=m1.shape[0]
	m = torch.bmm(m1, m2.transpose(1,2)) #batch*3*3

	cos = (  m[:,0,0] + m[:,1,1] + m[:,2,2] - 1 )/2
	cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch).cuda()) )
	cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch).cuda())*-1 )

	theta = torch.acos(cos)

	#theta = torch.min(theta, 2*np.pi - theta)

	return theta


class ChamferLossNd(nn.Module):

	def __init__(self):
		super(ChamferLossNd, self).__init__()
		self.use_cuda = torch.cuda.is_available()

	# B N D; B M D
	def forward(self, gts, preds):
		P = self.batch_pairwise_dist(gts, preds)
		mins, _ = torch.min(P, 1)
		loss_1 = torch.mean(mins)
		mins, _ = torch.min(P, 2)
		loss_2 = torch.mean(mins)

		return (loss_1 + loss_2)/2.0

	def batch_pairwise_dist(self, x, y):
		x = x.float()
		y = y.float()
		bs, num_points_x, points_dim = x.size()
		_, num_points_y, _ = y.size()
		xx = torch.bmm(x, x.transpose(2, 1))
		yy = torch.bmm(y, y.transpose(2, 1))
		zz = torch.bmm(x, y.transpose(2, 1))
		if self.use_cuda:
			dtype = torch.cuda.LongTensor
		else:
			dtype = torch.LongTensor
		diag_ind_x = torch.arange(0, num_points_x).type(dtype)
		diag_ind_y = torch.arange(0, num_points_y).type(dtype)

		rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz.transpose(2, 1))
		ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
		P = (rx.transpose(2, 1) + ry - 2 * zz)
		return P


def gmm_params(gamma, pts):
	'''
	Inputs:
		gamma: B x N x J (B, 1024, 16)
		pts: B x N x 3
	'''
	# pi: B x J
	pi = gamma.mean(dim=1) # (B, 16)
	Npi = pi * gamma.shape[1] 
	# mu: B x J x 3
	mu = gamma.transpose(1, 2) @ pts / Npi.unsqueeze(2) # (B x J x N) (B x N x 3) maintain permutation-invariance (B, 16, 3)
	# diff: B x N x J x 3
	diff = pts.unsqueeze(2) - mu.unsqueeze(1) # (B x N x 1 x 3) (B x 1 x j x 3) 
	# sigma: B x J x 3 x 3
	eye = torch.eye(3).unsqueeze(0).unsqueeze(1).to(gamma.device)
	sigma = (
		((diff.unsqueeze(3) @ diff.unsqueeze(4)).squeeze() * gamma).sum(dim=1) / Npi
	).unsqueeze(2).unsqueeze(3) * eye # elementwise times, ((BxNxJx1x3, BxNxJx3x1) => BxNxJ => BxJ => BxJx3x3 (diagonal))
	return pi, mu, sigma


# pi, sigma pose-free, mu is not pose-free
def gmm_register(pi_s, mu_s, mu_t, sigma_t):
	'''
	Inputs:
		pi: B x J
		mu: B x J x 3
		sigma: B x J x 3 x 3
	'''
	c_s = pi_s.unsqueeze(1) @ mu_s # B x 1 x 3 (center of src J)
	c_t = pi_s.unsqueeze(1) @ mu_t # B x 1 x 3 (center of tgt J)
	Ms = torch.sum((pi_s.unsqueeze(2) * (mu_s - c_s)).unsqueeze(3) @
				   (mu_t - c_t).unsqueeze(2) @ sigma_t.inverse(), dim=1) # ((BxJx1, BxJx3)=> BxJx3x1, BxJx1x3, BxJx3x3) => Bx3x3
	U, _, V = torch.svd(Ms.cpu()) # Bx3x3, _, Bx3x3
	U = U.cuda()
	V = V.cuda()
	S = torch.eye(3).unsqueeze(0).repeat(U.shape[0], 1, 1).to(U.device) # Bx3x3
	S[:, 2, 2] = torch.det(V @ U.transpose(1, 2)) # Bx3x3; (3, 3) = det (B, 1, 1)
	R = V @ S @ U.transpose(1, 2)
	t = c_t.transpose(1, 2) - R @ c_s.transpose(1, 2)
	bot_row = torch.Tensor([[[0, 0, 0, 1]]]).repeat(R.shape[0], 1, 1).to(R.device)
	T = torch.cat([torch.cat([R, t], dim=2), bot_row], dim=1)
	return T


class Conv1dBNReLU(nn.Sequential):
	def __init__(self, in_planes, out_planes):
		super(Conv1dBNReLU, self).__init__(
			nn.Conv1d(in_planes, out_planes, kernel_size=1, bias=False),
			nn.BatchNorm1d(out_planes),
			nn.ReLU(inplace=True))


class Conv2dBNReLU(nn.Sequential):
	def __init__(self, in_planes, out_planes):
		super(Conv2dBNReLU, self).__init__(
			nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
			nn.BatchNorm2d(out_planes),
			nn.ReLU(inplace=True))


class Conv1dGNReLU(nn.Sequential):
	def __init__(self, in_planes, out_planes):
		super(Conv1dGNReLU, self).__init__(
			nn.Conv1d(in_planes, out_planes, kernel_size=1, bias=False),
			nn.GroupNorm(8, out_planes),
			nn.ReLU(inplace=True))


class Conv2dGNReLU(nn.Sequential):
	def __init__(self, in_planes, out_planes):
		super(Conv2dGNReLU, self).__init__(
			nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
			nn.GroupNorm(8, out_planes),
			nn.ReLU(inplace=True))


class FCBNReLU(nn.Sequential):
	def __init__(self, in_planes, out_planes):
		super(FCBNReLU, self).__init__(
			nn.Linear(in_planes, out_planes, bias=False),
			nn.BatchNorm1d(out_planes),
			nn.ReLU(inplace=True))


class TNet(nn.Module):
	def __init__(self):
		super(TNet, self).__init__()
		self.encoder = nn.Sequential(
			Conv1dBNReLU(3, 64),
			Conv1dBNReLU(64, 128),
			Conv1dBNReLU(128, 256))
		self.decoder = nn.Sequential(
			FCBNReLU(256, 128),
			FCBNReLU(128, 64),
			nn.Linear(64, 6))

	@staticmethod
	def f2R(f):
		r1 = F.normalize(f[:, :3])
		proj = (r1.unsqueeze(1) @ f[:, 3:].unsqueeze(2)).squeeze(2)
		r2 = F.normalize(f[:, 3:] - proj * r1)
		r3 = r1.cross(r2)
		return torch.stack([r1, r2, r3], dim=2)

	def forward(self, pts):
		f = self.encoder(pts)
		f, _ = f.max(dim=2)
		f = self.decoder(f)
		R = self.f2R(f)
		return R @ pts

	def __init__(self, args):
		super(PointNet, self).__init__()
		self.use_tnet = args.use_tnet #False
		self.tnet = TNet() if self.use_tnet else None #False
		# d_input = args.k * 4 if args.use_rri else 3 # 80
		d_input = args.k_rri * 4
		self.encoder = nn.Sequential(
			Conv1dBNReLU(d_input, 64),
			Conv1dBNReLU(64, 128),
			Conv1dBNReLU(128, 256),
			Conv1dBNReLU(256, 1024)) # 1024
		self.decoder = nn.Sequential(
			Conv1dBNReLU(1024 * 2, 512),
			Conv1dBNReLU(512, 256),
			Conv1dBNReLU(256, 128),
			nn.Conv1d(128, args.num_groups, kernel_size=1)) # 16

	def forward(self, pts):
		pts = self.tnet(pts) if self.use_tnet else pts
		f_loc = self.encoder(pts)
		f_glob, _ = f_loc.max(dim=2)
		f_glob = f_glob.unsqueeze(2).expand_as(f_loc)
		y = self.decoder(torch.cat([f_loc, f_glob], dim=1))
		return y.transpose(1, 2)


def angle(v1, v2):
	"""Compute angle between 2 vectors
	For robustness, we use the same formulation as in PPFNet, i.e.
		angle(v1, v2) = atan2(cross(v1, v2), dot(v1, v2)).
	This handles the case where one of the vectors is 0.0, since torch.atan2(0.0, 0.0)=0.0
	Input:
		v1: (B, *, 3)
		v2: (B, *, 3)
	Output:
		
	"""

	cross_prod = torch.stack([v1[..., 1] * v2[..., 2] - v1[..., 2] * v2[..., 1],
								v1[..., 2] * v2[..., 0] - v1[..., 0] * v2[..., 2],
								v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]], dim=-1)
	cross_prod_norm = torch.norm(cross_prod, dim=-1)
	dot_prod = torch.sum(v1 * v2, dim=-1)

	return torch.atan2(cross_prod_norm, dot_prod)


def get_cluster_ppf(points, normals, ppf=None, num_clusters=64, num_samples=128, num_neighbors=5):
	"""
	Input:
		points: B N 3
		normal: B N 3
		ppf: B 4K N
		M; S; K
	Output:
		center_pts, center_n: B M 3
		cluster_pts, cluster_n: B M S 3
		cluster_ppf: B 4K S M
	"""
	batch_size = points.size()[0]
	p_idx = furthest_point_sample(points, num_clusters) # pts => B N 3, M; B M
	center_pts = gather_points(points.transpose(1,2).contiguous(), p_idx).transpose(1,2).contiguous() # B M 3
	center_n = gather_points(normals.transpose(1,2).contiguous(), p_idx).transpose(1,2).contiguous()

	pn_idx = knn_point(num_samples+1, points, center_pts).detach().int()[:, :, 1:].contiguous() # B M S
	# pn_idx = knn_point(num_samples, points, center_pts).detach().int() # B M S
	cluster_pts = grouping_operation(points.transpose(1,2).contiguous(), pn_idx).permute(0, 2, 3, 1).contiguous() # B M S 3
	cluster_n = grouping_operation(normals.transpose(1,2).contiguous(), pn_idx).permute(0, 2, 3, 1).contiguous()

	if ppf is not None:
		cluster_ppf = grouping_operation(ppf, pn_idx).transpose(2,3).contiguous() # B 4K S M

	else:
		pts0 = cluster_pts.view(batch_size*num_centers, num_samples, 3) # BM S 3
		normal0 = cluster_n.view(batch_size*num_centers, num_samples, 3) # BM S 3

		p_idx0 = knn_feature(pts0.transpose(1,2).contiguous(), num_neighbors) # BM S K
		cluster_pts0 = get_edge_features(pts0.transpose(1,2).contiguous(), p_idx0).permute(0, 2, 3, 1).contiguous() # BM S K 3
		cluster_n0 = get_edge_features(normal0.transpose(1,2).contiguous(), p_idx0).permute(0, 2, 3, 1).contiguous()

		center_pts0 = pts0.unsqueeze(2) # BM S 1 3
		center_n0 = normal0.unsqueeze(2)

		d = cluster_pts0 - center_pts0 # BM S K 1
		nr_d = angle(center_n0, d)
		ni_d = angle(cluster_n0, d)
		nr_ni = angle(center_n0, cluster_n0)
		d_norm = torch.norm(d, dim=-1)

		cluster_ppf = torch.stack([nr_d, ni_d, nr_ni, d_norm], dim=-1).view(batch_size, num_clusters, 
							num_samples, 4*num_neighbors).transpose(1,3).contiguous() # B 4K S M
	
	return center_pts, center_n, cluster_pts, cluster_n, cluster_ppf


def get_ppf(points, normals, num_clusters=0, num_samples=5):
	"""
	Input:
		points: B N 3
		normal: B N 3
		M; S
	Output:
		center_pts, center_n: B N 3
		cluster_pts, cluster_n: B N S 3
		ppf: B 4S M
	"""
	batch_size = points.size()[0]
	if num_clusters > 0:
		p_idx = furthest_point_sample(points, num_clusters) # pts => B N 3, M; B M
		center_pts = gather_points(points.transpose(1,2).contiguous(), p_idx).transpose(1,2).contiguous() # B M 3
		center_n = gather_points(normals.transpose(1,2).contiguous(), p_idx).transpose(1,2).contiguous()

		pn_idx = knn_point(num_samples+1, points, center_pts).detach().int()[:, :, 1:].contiguous() # B M S
		# pn_idx = knn_point(num_samples, points, center_pts).detach().int() # B M S
		cluster_pts = grouping_operation(points.transpose(1,2).contiguous(), pn_idx).permute(0, 2, 3, 1).contiguous() # B M S 3
		cluster_n = grouping_operation(normals.transpose(1,2).contiguous(), pn_idx).permute(0, 2, 3, 1).contiguous()

	else:
		p_idx = knn_feature(points.transpose(1,2).contiguous(), num_samples+1)[:, :, 1:].contiguous() # B N S
		# p_idx = knn_feature(points.transpose(1,2).contiguous(), num_samples) # B N S
		cluster_pts = get_edge_features(points.transpose(1,2).contiguous(), p_idx).permute(0, 3, 2, 1).contiguous() # B N S 3
		cluster_n = get_edge_features(normals.transpose(1,2).contiguous(), p_idx).permute(0, 3, 2, 1).contiguous()
		
		center_pts = points # B N 3
		center_n = normals
	
	d = cluster_pts - center_pts.unsqueeze(2) # B M S 3
	
	nr_d = angle(center_n.unsqueeze(2), d)
	ni_d = angle(cluster_n, d)
	nr_ni = angle(center_n.unsqueeze(2), cluster_n)
	d_norm = torch.norm(d, dim=-1)

	ppf = torch.stack([nr_d, ni_d, nr_ni, d_norm], dim=-1).view(batch_size, -1, 4*num_samples).transpose(1,2).contiguous() # B 4S M
	return center_pts, center_n, cluster_pts, cluster_n, ppf


def knn_feature(x, k):
	"""
		x: B C N
		k: int
	"""
	inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
	xx = torch.sum(x ** 2, dim=1, keepdim=True)
	pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()

	idx = pairwise_distance.topk(k=k, dim=-1)[1].detach()  # (batch_size, num_points, k)
	return idx


def knn_point(k, point_input, point_output=None):
	"""
		k: int
		point_input: B n C
		point_output: B m C
	"""
	if point_output == None:
		point_output = point_input

	m = point_output.size()[1]
	n = point_input.size()[1]

	inner = -2 * torch.matmul(point_output, point_input.transpose(2, 1).contiguous())
	xx = torch.sum(point_output ** 2, dim=2, keepdim=True).repeat(1, 1, n)
	yy = torch.sum(point_input ** 2, dim=2, keepdim=False).unsqueeze(1).repeat(1, m, 1)
	pairwise_distance = -xx - inner - yy
	idx = pairwise_distance.topk(k=k, dim=-1)[1].detach() # (batch_size, m, k)

	return idx


def get_edge_features(x, idx):
	"""
	Input:
		x: B C N
		idx: B N k
	Output:
		feature: B C k N
	"""
	batch_size, num_points, k = idx.size()
	device = torch.device('cuda')

	idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

	idx = idx + idx_base

	idx = idx.view(-1)

	x = x.squeeze(2) #B, C, 1, N
	_, num_dims, _ = x.size()

	x = x.transpose(2,
					1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
	feature = x.view(batch_size * num_points, -1)[idx, :]
	feature = feature.view(batch_size, num_points, k, num_dims).permute(0, 3, 2, 1) #B, C, K, N

	return feature


def edge_preserve_sampling(feature_input, point_input, num_samples, k=10):
	# batch_size = feature_input.size()[0]
	# feature_size = feature_input.size()[1]
	num_points = feature_input.size()[2]

	p_idx = furthest_point_sample(point_input, num_samples) # B M
	# point_output = gather_operation(point_input.transpose(1,2).contiguous(), p_idx).transpose(1,2).contiguous() # B M 3
	point_output = gather_points(point_input.transpose(1,2).contiguous(), p_idx).transpose(1,2).contiguous() # B M 3

	pk = int(min(k, num_points))
	pn_idx = knn_point(pk, point_input, point_output).detach().int() # B M pk
	# print(pn_idx.size())

	# neighbor_feature = grouping_operation(feature_input, pn_idx)
	# neighbor_feature = index_points(feature_input.transpose(1,2).contiguous(), pn_idx).permute(0, 3, 1, 2)
	# neighbor_feature = gather_operation(feature_input, pn_idx.view(batch_size, num_samples*pk)).view(batch_size, feature_size, num_samples, pk)
	neighbor_feature = grouping_operation(feature_input, pn_idx) # B C M k
	neighbor_feature, _ = torch.max(neighbor_feature, 3)

	# center_feature = grouping_operation(feature_input, p_idx.unsqueeze(2)).view(batch_size, -1, num_samples)
	center_feature = gather_points(feature_input, p_idx)

	net = torch.cat((center_feature, neighbor_feature), 1)

	return net, p_idx, pn_idx, point_output


def three_nn_upsampling(target_points, source_points):
	dist, idx = three_nn(target_points, source_points)
	dist = torch.max(dist, torch.ones(1).cuda()*1e-10)
	norm = torch.sum((1.0/dist), 2, keepdim=True)
	norm = norm.repeat(1,1,3)
	weight = (1.0/dist) / norm

	return idx, weight


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
	idx = knn_feature(cluster_pts_, k+1)[:,:,1:] # BM S k
	cluster_npts_ = get_edge_features(cluster_pts_, idx) # BM 3 k S
	
	p = cluster_pts_.transpose(1, 2).contiguous().unsqueeze(2).repeat(1,1,k,1) # BM S k 3
	q = cluster_npts_.transpose(1, 3).contiguous() # BM S k 3

	rp = torch.norm(p, None, dim=-1, keepdim=True) # BM S k 1
	rq = torch.norm(q, None, dim=-1, keepdim=True) # BM S k 1
	pn = p / rp
	qn = q / rq
	dot = torch.sum(pn * qn, dim=-1, keepdim=True) # BM S k 1

	theta = torch.acos(torch.clamp(dot, -1, 1)) # BM S k 1
	# T_q = q - dot * p # BM S k 3
	# BM S k k
	# sin_psi = torch.sum(torch.cross(T_q[:,:,None].repeat(1,1,k,1,1), T_q[:,:,:,None].repeat(1,1,1,k,1), dim=-1) * pn[:,:,None], -1)
	# cos_psi = torch.sum(T_q[:,:,None] * T_q[:,:,:,None], -1)
	# psi = torch.atan2(sin_psi, cos_psi) % (2*np.pi)
	# # psi = torch.from_numpy(np.arctan2(sin_psi.cpu().numpy(), cos_psi.cpu().numpy()) % (2*np.pi)).cuda()
	# # BM S k 1
	# _, idx = psi.topk(k=2, dim=-1, largest=False)
	# idx = idx[:, :, :, 1:2]
	# phi = torch.gather(psi, dim=-1, index=idx)

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


def attention(query, key, value):
	dim = query.shape[1]
	scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
	prob = torch.nn.functional.softmax(scores, dim=-1)
	return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob


def log_sinkhorn_iterations(Z, log_mu, log_nu, iters: int):
	""" Perform Sinkhorn Normalization in Log-space for stability"""
	u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu) # b (m+1); b (n+1)
	for _ in range(iters):
		u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2) # b m+1
		v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1) # b n+1
	return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores, alpha, iters: int):
	""" Perform Differentiable Optimal Transport in Log-space for stability"""
	b, m, n = scores.shape
	one = scores.new_tensor(1)
	ms, ns = (m*one).to(scores), (n*one).to(scores) # return a Tensor with same torch.dtype and torch.device as "scores"

	bins0 = alpha.expand(b, m, 1) # value 1
	bins1 = alpha.expand(b, 1, n)
	alpha = alpha.expand(b, 1, 1)

	couplings = torch.cat([torch.cat([scores, bins0], -1),
							torch.cat([bins1, alpha], -1)], 1) # b m+1 n+1

	norm = - (ms + ns).log()
	log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm]) # m+1
	log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm]) # n+1
	log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1) # b (m+1); b (n+1)

	Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
	Z = Z - norm  # multiply probabilities by M+N
	return Z


def arange_like(x, dim: int):
	return x.new_ones(x.shape[dim]).cumsum(0) - 1 


def sample_group(num_centers, num_samples, pts, feats=None):
	'''
	Input:
		num_centers: int M; 
		num_samples: int S;
		pts: B N 3;	feats: B C N;
	Output:
		center_pts: B M 3; cluster_pts: B 3 S M; cluster_feat: B C S M; 
	'''
	p_idx = furthest_point_sample(pts, num_centers) # pts => B N 3, M
	center_pts = gather_points(pts.transpose(1,2).contiguous(), p_idx).transpose(1,2).contiguous() # B M 3
	pn_idx = knn_point(num_samples, pts, center_pts).detach().int() # B M S

	if feats is not None:
		cluster_feats = grouping_operation(feats, pn_idx).transpose(2, 3).contiguous() # B C S M
	else:
		cluster_feats = None
	cluster_pts = grouping_operation(pts.transpose(1,2).contiguous(), pn_idx).transpose(2, 3).contiguous() # B 3 S M
	return center_pts, cluster_pts, cluster_feats


class SA_module(nn.Module):
	"""docstring for SA_module"""
	def __init__(self, in_planes, rel_planes, mid_planes, out_planes, share_planes=8, k=16):
		super(SA_module, self).__init__()
		self.share_planes = share_planes
		self.k = k
		self.conv1 = nn.Conv2d(in_planes, rel_planes, kernel_size=1)
		self.conv2 = nn.Conv2d(in_planes, rel_planes, kernel_size=1)
		self.conv3 = nn.Conv2d(in_planes, mid_planes, kernel_size=1)

		self.conv_w = nn.Sequential(nn.ReLU(inplace=False), nn.Conv2d(rel_planes*(k+1), mid_planes//share_planes, kernel_size=1, bias=False),
									nn.ReLU(inplace=False), nn.Conv2d(mid_planes//share_planes, k*mid_planes//share_planes, kernel_size=1))
		self.activation_fn = nn.ReLU(inplace=False)
		
		self.conv_out = nn.Conv2d(mid_planes, out_planes, kernel_size=1)

	def forward(self, input):
		x, idx = input
		batch_size, _, num_points = x.size()
		x = x.unsqueeze(2)
		identity = x # B C 1 N
		x = self.activation_fn(x)
		xn = get_edge_features(x, idx) # B C K N
		x1, x2, x3 = self.conv1(x), self.conv2(xn), self.conv3(xn)

		x2 = x2.view(batch_size, -1, 1, num_points).contiguous() # B kC 1 N
		w = self.conv_w(torch.cat([x1, x2], 1)).view(batch_size, -1, self.k, num_points)
		w = w.repeat(1, self.share_planes, 1, 1)
		out = w*x3
		out = torch.sum(out, dim=2, keepdim=True)

		out = self.activation_fn(out)
		out = self.conv_out(out) # B C 1 N
		out += identity
		out = out.squeeze(2) # B C N
		return [out, idx]


class SK_SA_module(nn.Module):
	"""docstring for SK_SA_module"""
	def __init__(self, in_planes, rel_planes, mid_planes, out_planes, share_planes=8, k=[10, 20], r=2, L=32):
		super(SK_SA_module, self).__init__()
		
		self.num_kernels = len(k)
		d = max(int(out_planes/r), L)

		self.sams = nn.ModuleList([])

		for i in range(len(k)):
			self.sams.append(
				SA_module(in_planes, rel_planes, mid_planes, out_planes, share_planes, k[i])
				)
		
		self.fc = nn.Linear(out_planes, d)
		self.fcs = nn.ModuleList([])

		for i in range(len(k)):
			self.fcs.append(
				nn.Linear(d, out_planes)
				)

		self.softmax = nn.Softmax(dim=1)
		self.af = nn.ReLU(inplace=False)

	def forward(self, input):
		x, idxs = input
		assert(self.num_kernels == len(idxs))
		for i, sam in enumerate(self.sams):
			fea, _ = sam([x, idxs[i]]) # B C N
			fea = self.af(fea)
			fea = fea.unsqueeze(dim=1)
			if i == 0:
				feas = fea
			else:
				feas = torch.cat([feas, fea], dim=1) # B #K C N

		fea_U = torch.sum(feas, dim=1) # B C N

		fea_s = fea_U.mean(-1)
		fea_z = self.fc(fea_s)

		for i, fc in enumerate(self.fcs):
			vector = fc(fea_z).unsqueeze_(dim=1) 
			if i == 0:
				attention_vectors = vector
			else:
				attention_vectors = torch.cat([attention_vectors, vector], dim=1)

		attention_vectors = self.softmax(attention_vectors) # B #K C
		attention_vectors = attention_vectors.unsqueeze(-1) # B #K C 1

		fea_v = (feas * attention_vectors).sum(dim=1) # B C N
		return [fea_v, idxs]


class SKN_Res_module(nn.Module):
	"""docstring for SKN_Res_module"""
	def __init__(self, input_size, output_size, k=[10, 20], layers=1):
		super(SKN_Res_module, self).__init__()
		self.conv1 = nn.Conv1d(input_size, output_size, 1, bias=False)
		self.net = self._make_layer(output_size, output_size//16, output_size//4, output_size, int(layers), 8, k=k)
		self.conv2 = nn.Conv1d(output_size, output_size, 1, bias=False)
		self.conv_res = nn.Conv1d(input_size, output_size, 1, bias=False)
		self.af = nn.ReLU(inplace=False)

	def _make_layer(self, in_planes, rel_planes, mid_planes, out_planes, blocks, share_planes=8, k=16):
		layers = []
		for _ in range(0, blocks):
			layers.append(SK_SA_module(in_planes, rel_planes, mid_planes, out_planes, share_planes, k))
		return nn.Sequential(*layers)
	def forward(self, feat, idx):
		x, _ = self.net([self.conv1(feat), idx])
		x = self.conv2(self.af(x))
		return x + self.conv_res(feat)


class SA_Res_encoder1(nn.Module):
	"""docstring for SA_Res_encoder1"""
	def __init__(self, input_size=3, k=[10], pk=10, output_size=128, ds_ration=2):
		super(SA_Res_encoder1, self).__init__()
		
		self.k = k
		self.pk = pk
		self.rate = ds_ration

		self.dropout = nn.Dropout()
		self.af = nn.ReLU(inplace=False)

		c1 = 32
		self.sam_res1 = SKN_Res_module(input_size, c1, k)
		c2 = c1 * 2
		self.sam_res2 = SKN_Res_module(c1*2, c2, k)
		c3 = c2 * 2
		self.sam_res3 = SKN_Res_module(c2*2, c3, k)
		c4 = c3 * 2
		self.sam_res4 = SKN_Res_module(c3*2, c4, k)
		c5 = c4 * 2
		self.sam_res5 = SKN_Res_module(c4*2, c5, k)

		self.conv_g = nn.Conv1d(c5, 1024, 1)

		self.fc1 = nn.Linear(1024, 512)
		self.fc2 = nn.Linear(512, 1024)

		self.conv1 = nn.Conv1d(c5+1024, c5, 1)
		self.conv2 = nn.Conv1d(c4+c5, c4, 1)
		self.conv3 = nn.Conv1d(c3+c4, c3, 1)
		self.conv4 = nn.Conv1d(c2+c3, c2, 1)
		self.conv5 = nn.Conv1d(c1+c2, output_size, 1)

		self.conv_out = nn.Conv1d(output_size, output_size, 1)
	

	def _edge_pooling(self, features, points):
		input_points_num = int(features.size()[2])
		sample_num = int(input_points_num // self.rate)
		ds_features, p_idx, pn_idx, ds_points = edge_preserve_sampling(features, points, sample_num, self.pk)
		return ds_features, p_idx, pn_idx, ds_points


	def _edge_unpooling(self, features, src_pts, tgt_pts):
		idx, weight = three_nn_upsampling(tgt_pts, src_pts)
		features = three_interpolate(features, idx, weight)
		return features


	def forward(self, features, pts):
		batch_size, _, _ = features.size()
		pt1 = pts.contiguous() # B N 3

		idx1 = []
		for i in range(len(self.k)):
			idx = knn_feature(pt1.transpose(1,2).contiguous(), self.k[i])
			idx1.append(idx)
		
		x = self.sam_res1(features, idx1)
		x1 = self.af(x)

		x, _, _, pt2 = self._edge_pooling(x1, pt1)
		
		idx2 = []
		for i in range(len(self.k)):
			idx = knn_feature(pt2.transpose(1,2).contiguous(), self.k[i])
			idx2.append(idx)

		x = self.sam_res2(x, idx2)
		x2 = self.af(x)

		x, _, _, pt3 = self._edge_pooling(x2, pt2)

		idx3 = []
		for i in range(len(self.k)):
			idx = knn_feature(pt3.transpose(1,2).contiguous(), self.k[i])
			idx3.append(idx)

		x = self.sam_res3(x, idx3)
		x3 = self.af(x)

		x, _, _, pt4 = self._edge_pooling(x3, pt3)

		idx4 = []
		for i in range(len(self.k)):
			idx = knn_feature(pt4.transpose(1,2).contiguous(), self.k[i])
			idx4.append(idx)
		
		x = self.sam_res4(x, idx4)
		x4 = self.af(x)

		x, _, _, pt5 = self._edge_pooling(x4, pt4)

		idx5 = []
		for i in range(len(self.k)):
			idx = knn_feature(pt5.transpose(1,2).contiguous(), self.k[i])
			idx5.append(idx)
		
		x = self.sam_res5(x, idx5)
		x5 = self.af(x)

		x = self.conv_g(x5)

		x, _ = torch.max(x, -1)
		x = x.view(batch_size, -1)
		x = self.dropout(self.af(self.fc2(self.dropout(self.af(self.fc1(x))))))

		x = x.unsqueeze(2).repeat(1, 1, int(pt5.shape[1]))
		x = self.af(self.conv1(torch.cat([x, x5], 1)))
		x = self._edge_unpooling(x, pt5, pt4)
		x = self.af(self.conv2(torch.cat([x, x4], 1)))
		x = self._edge_unpooling(x, pt4, pt3)
		x = self.af(self.conv3(torch.cat([x, x3], 1)))
		x = self._edge_unpooling(x, pt3, pt2)
		x = self.af(self.conv4(torch.cat([x, x2], 1)))
		x = self._edge_unpooling(x, pt2, pt1)
		x = self.af(self.conv5(torch.cat([x, x1], 1)))

		x = self.conv_out(x)

		return x


class SA_Res_encoder(nn.Module):
	"""docstring for SA_Res_encoder"""
	def __init__(self, input_size=3, k=[10], pk=10, output_size=128, ds_ratio=4):
		super(SA_Res_encoder, self).__init__()
		
		self.k = k
		self.pk = pk
		self.rate = ds_ratio

		self.dropout = nn.Dropout()
		# self.af = nn.ReLU(inplace=False)

		self.conv_res = nn.Conv1d(input_size, output_size, 1)

		c1 = 64
		self.conv0 = nn.Conv2d(4, c1, 1)
		self.sam_res1 = SKN_Res_module(c1, c1, k)
		c2 = c1 * 2
		self.sam_res2 = SKN_Res_module(c1*2, c2, k)
		c3 = c2 * 2
		self.sam_res3 = SKN_Res_module(c2*2, c3, k)

		self.conv_g = nn.Conv1d(c3, 1024, 1)

		self.fc1 = nn.Linear(1024, 512)
		self.fc2 = nn.Linear(512, 1024)

		self.conv3 = nn.Conv1d(c3+1024, c3, 1)
		self.conv2 = nn.Conv1d(c2+c3, c2, 1)
		self.conv1 = nn.Conv1d(c1+c2, output_size, 1)

		self.conv_out = nn.Conv1d(c1+output_size, output_size, 1)
	

	def _edge_pooling(self, features, points):
		input_points_num = int(features.size()[2])
		sample_num = int(input_points_num // self.rate)
		ds_features, p_idx, pn_idx, ds_points = edge_preserve_sampling(features, points, sample_num, self.pk)
		return ds_features, p_idx, pn_idx, ds_points


	def _edge_unpooling(self, features, src_pts, tgt_pts):
		idx, weight = three_nn_upsampling(tgt_pts, src_pts)
		features = three_interpolate(features, idx, weight)
		return features


	def forward(self, features, pts):
		batch_size, _, num_points = features.size()
		pt1 = pts.contiguous() # B N 3

		x_res = self.conv_res(features)

		features = features.transpose(1, 2).reshape(batch_size, num_points, -1, 4).transpose(3, 1).contiguous()
		
		x = self.conv0(features)
		x0, _ = torch.max(x, 2)

		idx1 = []
		for i in range(len(self.k)):
			idx = knn_feature(pt1.transpose(1,2).contiguous(), self.k[i])
			idx1.append(idx)
		
		x = self.sam_res1(x0, idx1)
		x1 = F.relu(x, inplace=True)

		x, _, _, pt2 = self._edge_pooling(x1, pt1)
		
		idx2 = []
		for i in range(len(self.k)):
			idx = knn_feature(pt2.transpose(1,2).contiguous(), self.k[i])
			idx2.append(idx)

		x = self.sam_res2(x, idx2)
		x2 = F.relu(x, inplace=True)

		x, _, _, pt3 = self._edge_pooling(x2, pt2)

		idx3 = []
		for i in range(len(self.k)):
			idx = knn_feature(pt3.transpose(1,2).contiguous(), self.k[i])
			idx3.append(idx)

		x = self.sam_res3(x, idx3)
		x3 = F.relu(x, inplace=True)

		x = self.conv_g(x3)

		x, _ = torch.max(x, -1)
		x = x.view(batch_size, -1)
		x = self.dropout(F.relu(self.fc2(self.dropout(F.relu(self.fc1(x), inplace=True))), inplace=True))

		x = x.unsqueeze(2).repeat(1, 1, int(pt3.shape[1]))
		x = F.relu(self.conv3(torch.cat([x, x3], 1)), inplace=True)
		x = self._edge_unpooling(x, pt3, pt2)
		x = F.relu(self.conv2(torch.cat([x, x2], 1)), inplace=True)
		x = self._edge_unpooling(x, pt2, pt1)
		x = F.relu(self.conv1(torch.cat([x, x1], 1)), inplace=True)

		x = self.conv_out(torch.cat([x, x0], 1)) + x_res

		return x


class PT_block(nn.Module):
	def __init__(self, in_planes, mid_planes, out_planes, share_planes=8, k=16):
		super(PT_block, self).__init__()
		self.share_planes = share_planes
		self.k = k
		
		self.conv0 = nn.Conv1d(in_planes, in_planes, kernel_size=1)

		self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=1)
		self.conv2 = nn.Conv2d(in_planes, mid_planes, kernel_size=1)
		self.conv3 = nn.Conv2d(in_planes, mid_planes, kernel_size=1)

		self.conv_w = nn.Sequential(
							nn.Conv2d(mid_planes*k, mid_planes//share_planes, kernel_size=1, bias=False),
							nn.ReLU(inplace=False), 
							nn.Conv2d(mid_planes//share_planes, k*mid_planes//share_planes, kernel_size=1))
		self.activation_fn = nn.ReLU(inplace=False)
		
		self.conv_out = nn.Conv2d(mid_planes, out_planes, kernel_size=1)

		self.pt_conv = nn.Sequential(
							nn.Conv2d(3, mid_planes, kernel_size=1, bias=False), 
							nn.ReLU(inplace=False), 
							nn.Conv2d(mid_planes, mid_planes, kernel_size=1, bias=False))
	
	def forward(self, input):
		x, pts, idx = input

		batch_size, _, num_points = x.size()

		x = F.relu(self.conv0(x), inplace=True) # B C N

		x = x.unsqueeze(2)
		identity = x # B C 1 N
		x = self.activation_fn(x)
		xn = get_edge_features(x, idx) # B C K N
		x1, x2, x3 = self.conv1(x), self.conv2(xn), self.conv3(xn)

		pts1 = pts.transpose(1, 2).contiguous().unsqueeze(2) # B 3 1 N
		ptsn = get_edge_features(pts1, idx) # B 3 K N
		ptsn = pts1 - ptsn # B 3 K N
		ptf = self.pt_conv(ptsn) # B C K N

		x3 += ptf # B C K N

		xfs = x1 - x2 + ptf # B C K N

		xfs = xfs.view(batch_size, -1, 1, num_points).contiguous() # B KC 1 N
		w = F.softmax(self.conv_w(xfs).view(batch_size, -1, self.k, num_points), dim=-2) # B C K N
		w = w.repeat(1, self.share_planes, 1, 1) # B C K N
		out = w*x3 # B C K N
		out = torch.sum(out, dim=2, keepdim=True)

		out = self.activation_fn(out)
		out = self.conv_out(out) # B C 1 N
		out += identity
		out = out.squeeze(2) # B C N
		return [out, pts, idx]



class PointTransformer(nn.Module):
	"""docstring for PT"""
	def __init__(self, input_size=3, k=10, pk=10, output_size=128, ds_ratio=4):
		super(PointTransformer, self).__init__()
		self.k = k
		self.pk = pk
		self.rate = ds_ratio
		self.dropout = nn.Dropout()

		self.input_size = input_size
		if input_size is not 3:
			self.input_size = 4

		c1 = 64
		self.conv0 = nn.Conv2d(self.input_size, c1, 1)
		self.PT1 = PT_block(c1, c1//4, c1, k=k)
		c2 = c1 * 2
		self.PT2 = PT_block(c2, c2//4, c2, k=k)
		c3 = c2 * 2
		self.PT3 = PT_block(c3, c3//4, c3, k=k)

		self.conv_g = nn.Conv1d(c3, 1024, 1)

		self.fc1 = nn.Linear(1024, 512)
		self.fc2 = nn.Linear(512, 1024)

		self.conv3 = nn.Conv1d(c3+1024, c3, 1)
		self.conv2 = nn.Conv1d(c2+c3, c2, 1)
		self.conv1 = nn.Conv1d(c1+c2, output_size, 1)

		self.conv_out = nn.Conv1d(c1+output_size, output_size, 1)
	

	def _edge_pooling(self, features, points):
		input_points_num = int(features.size()[2])
		sample_num = int(input_points_num // self.rate)
		ds_features, p_idx, pn_idx, ds_points = edge_preserve_sampling(features, points, sample_num, self.pk)
		return ds_features, p_idx, pn_idx, ds_points


	def _edge_unpooling(self, features, src_pts, tgt_pts):
		idx, weight = three_nn_upsampling(tgt_pts, src_pts)
		features = three_interpolate(features, idx, weight)
		return features


	def forward(self, features, pts):
		batch_size, _, num_points = features.size()
		pt1 = pts.contiguous() # B N 3

		if self.input_size == 4:
			features = features.transpose(1, 2).reshape(batch_size, num_points, -1, 4).transpose(3, 1).contiguous()
			x = self.conv0(features)
			x0, _ = torch.max(x, 2)
		else:
			features = features.unsqueeze(-1)
			x = self.conv0(features)
			x0 = x.squeeze(-1)

		idx1 = knn_feature(pt1.transpose(1,2).contiguous(), self.k)		
		x, _, _ = self.PT1([x0, pt1, idx1])
		x1 = F.relu(x, inplace=True)


		x, _, _, pt2 = self._edge_pooling(x1, pt1)
		idx2 = knn_feature(pt2.transpose(1,2).contiguous(), self.k)
		x, _, _ = self.PT2([x, pt2, idx2])
		x2 = F.relu(x, inplace=True)


		x, _, _, pt3 = self._edge_pooling(x2, pt2)
		idx3 = knn_feature(pt3.transpose(1,2).contiguous(), self.k)
		x, _, _ = self.PT3([x, pt3, idx3])
		x3 = F.relu(x, inplace=True)

		x = self.conv_g(x3)

		x, _ = torch.max(x, -1)
		x = x.view(batch_size, -1)
		x = self.dropout(F.relu(self.fc2(self.dropout(F.relu(self.fc1(x), inplace=True))), inplace=True))

		x = x.unsqueeze(2).repeat(1, 1, int(pt3.shape[1]))
		x = F.relu(self.conv3(torch.cat([x, x3], 1)), inplace=True)
		x = self._edge_unpooling(x, pt3, pt2)
		x = F.relu(self.conv2(torch.cat([x, x2], 1)), inplace=True)
		x = self._edge_unpooling(x, pt2, pt1)
		x = F.relu(self.conv1(torch.cat([x, x1], 1)), inplace=True)

		x = self.conv_out(torch.cat([x, x0], 1))

		return x