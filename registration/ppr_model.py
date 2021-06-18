import math
import open3d as o3d
import torch
import torch.nn as nn
import torch.nn.functional as F
from visu_util import visualize
from utils import *
from model_utils import *

_EPS = 1e-5  # To prevent division by zero


class Point_Transformer(nn.Module):
	def __init__(self, in_planes, mid_planes, out_planes, share_planes=8, k=10):
		super(Point_Transformer, self).__init__()
		self.share_planes = share_planes
		self.k = k
		
		self.conv0 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
		self.conv1 = nn.Conv2d(out_planes, mid_planes, kernel_size=1)
		self.conv2 = nn.Conv2d(out_planes, mid_planes, kernel_size=1)
		self.conv3 = nn.Conv2d(out_planes, mid_planes, kernel_size=1)

		self.conv_w = nn.Sequential(
							nn.Conv2d(mid_planes*k, mid_planes//share_planes, kernel_size=1, bias=False),
							nn.ReLU(inplace=False), 
							nn.Conv2d(mid_planes//share_planes, k*mid_planes//share_planes, kernel_size=1))
		# self.activation_fn = nn.ReLU(inplace=False)
		
		self.conv_out = nn.Conv2d(mid_planes, out_planes, kernel_size=1)

		self.pt_conv = nn.Sequential(
							nn.Conv2d(1, mid_planes, kernel_size=1, bias=False), 
							nn.ReLU(inplace=False), 
							nn.Conv2d(mid_planes, mid_planes, kernel_size=1, bias=False))
	
	def forward(self, input):
		# B C M S; B M 3; B 3 M S
		sm_feats, cent_pts, sm_pts = input
		batch_size, _, num_points, num_samples = sm_feats.size()
		
		xn = F.relu(self.conv0(sm_feats)) # B C M S

		x = xn[:, :, :, 0, None] # B C M 1
		identity = x
		x1, x2, x3 = self.conv1(x), self.conv2(xn), self.conv3(xn)

		pts = cent_pts.transpose(1,2).contiguous().unsqueeze(-1) # B 3 M 1
		ptsn = torch.sum(torch.square(pts - sm_pts), dim=1, keepdim=True) # B 1 M S

		ptf = self.pt_conv(ptsn) # B C M S

		x3 += ptf
		xfs = x1 - x2 + ptf # B C M S

		xfs = xfs.transpose(2, 3).contiguous().view(batch_size, -1, 1, num_points) # B CS 1 M
		w = F.softmax(self.conv_w(xfs).view(batch_size, -1, num_samples, num_points), dim=2).transpose(2, 3).contiguous() # B C M S
		w = w.repeat(1, self.share_planes, 1, 1) # B C M S

		out = w * x3 # B C M S
		out = F.relu(torch.sum(out, dim=3, keepdim=True)) # B C M 1
		out = (self.conv_out(out) + identity).squeeze(3) # B C M
		
		return out


def cal_descriptor_loss(feats1, feats2, pts1, pts2, T, pos_pts_dist_th=0.01, neg_pts_dist_th=0.2, pos_feat_th=0.1, neg_feat_th=1.4):
	pts1_trans = pts1 @ T[:, :3, :3].transpose(1, 2) + T[:, :3, 3].unsqueeze(1)
	pts_dist = torch.sqrt(square_distance(pts1_trans, pts2))

	pos_mask = pts_dist < pos_pts_dist_th
	neg_mask = pts_dist > neg_pts_dist_th

	feats_dist = torch.sqrt(square_distance(feats1.transpose(2, 1).contiguous(), feats2.transpose(1, 2).contiguous()))
	# num_pos = torch.sum(torch.where(pts_dist < pts_dist_th, torch.ones_like(feat_dist), torch.zeros_like(feat_dist)))
	# pos_feat_dist = torch.where(pts_dist < pts_dist_th, feat_dist - pos_feat_th, torch.zeros_like(feat_dist))
	# neg_feat_dist = torch.where(pts_dist >= pts_dist_th, neg_feat_th - feat_dist, torch.zeros_like(feat_dist))

	## get anchors that have both positive and negative pairs
	row_sel = ((pos_mask.sum(-1)>0) * (neg_mask.sum(-1)>0)).detach()
	col_sel = ((pos_mask.sum(-2)>0) * (neg_mask.sum(-2)>0)).detach()

	# get alpha for both positive and negative pairs
	pos_weight = feats_dist - 1e5 * (~pos_mask).float() # mask the non-positive 
	pos_weight = (pos_weight - pos_feat_th) # mask the uninformative positive
	pos_weight = torch.max(torch.zeros_like(pos_weight), pos_weight).detach() 

	neg_weight = feats_dist + 1e5 * (~neg_mask).float() # mask the non-negative
	neg_weight = (neg_feat_th - neg_weight) # mask the uninformative negative
	neg_weight = torch.max(torch.zeros_like(neg_weight),neg_weight).detach()

	lse_pos_row = torch.logsumexp(16 * (feats_dist - pos_feat_th) * pos_weight,dim=-1)
	lse_pos_col = torch.logsumexp(16 * (feats_dist - pos_feat_th) * pos_weight,dim=-2)

	lse_neg_row = torch.logsumexp(16 * (neg_feat_th - feats_dist) * neg_weight,dim=-1)
	lse_neg_col = torch.logsumexp(16 * (neg_feat_th - feats_dist) * neg_weight,dim=-2)

	loss_row = F.softplus(lse_pos_row + lse_neg_row)/16
	loss_col = F.softplus(lse_pos_col + lse_neg_col)/16

	circle_loss = (loss_row[row_sel].mean() + loss_col[col_sel].mean()) / 2
	return circle_loss


def get_cluster_feats(feats, pts, k=10, return_cluster_pts=False):
	"""
		feats: B C N; pts: B N 3;
	"""
	idx = knn_feature(pts.transpose(1,2).contiguous(), k) # B N k
	batch_size, num_dims, num_points = feats.size()

	device = torch.device('cuda')
	idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
	idx = idx + idx_base
	idx = idx.view(-1)
	
	feats = feats.transpose(2, 1).contiguous() # B, N, C
	out_feats = feats.view(batch_size * num_points, -1)[idx, :]
	out_feats = out_feats.view(batch_size, num_points, k, num_dims).permute(0, 3, 1, 2).contiguous() # B C N k

	if return_cluster_pts:
		out_pts = pts.view(batch_size * num_points, -1)[idx, :]
		out_pts = out_pts.view(batch_size, num_points, k, 3).permute(0, 3, 1, 2).contiguous() # B 3 N k
	else:
		out_pts = None

	return out_feats, out_pts


def sinkhorn(log_alpha, num_iters: int = 5):
	zero_pad = nn.ZeroPad2d((0, 1, 0, 1))
	log_alpha_padded = zero_pad(log_alpha[:, None, :, :])

	log_alpha_padded = torch.squeeze(log_alpha_padded, dim=1)

	for i in range(num_iters):
		# Row normalization
		log_alpha_padded = torch.cat((
				log_alpha_padded[:, :-1, :] - (torch.logsumexp(log_alpha_padded[:, :-1, :], dim=2, keepdim=True)),
				log_alpha_padded[:, -1, None, :]),  # Don't normalize last row
			dim=1)

		# Column normalization
		log_alpha_padded = torch.cat((
				log_alpha_padded[:, :, :-1] - (torch.logsumexp(log_alpha_padded[:, :, :-1], dim=1, keepdim=True)),
				log_alpha_padded[:, :, -1, None]),  # Don't normalize last column
			dim=2)
	return log_alpha_padded


class MultiHeadedAttention(nn.Module):
	""" Multi-head attention to increase model expressivitiy """
	def __init__(self, num_heads: int, d_model: int):
		super(MultiHeadedAttention, self).__init__()
		assert d_model % num_heads == 0
		self.dim = d_model // num_heads
		self.num_heads = num_heads
		self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
		self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

	def forward(self, query, key, value):
		batch_dim = query.size(0)
		query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
								for l, x in zip(self.proj, (query, key, value))]
		x, _ = attention(query, key, value)
		return self.merge(x.contiguous().view(batch_dim, self.dim*self.num_heads, -1))


class AttentionalPropagation(nn.Module):
	def __init__(self, feature_dim: int, num_heads: int):
		super(AttentionalPropagation, self).__init__()
		self.attn = MultiHeadedAttention(num_heads, feature_dim)
		self.conv_in = nn.Sequential(
			Conv1dBNReLU(feature_dim*2, feature_dim*2)
		)
		self.conv_out = nn.Conv1d(feature_dim*2, feature_dim, 1)

	def forward(self, x, source):
		message = self.attn(x, source, source)
		return self.conv_out(self.conv_in(torch.cat([x, message], dim=1)))


class AttentionalGNN(nn.Module):
	def __init__(self, feature_size, layer_names):
		super(AttentionalGNN, self).__init__()
		self.layers = nn.ModuleList([
			AttentionalPropagation(feature_size, 4) for _ in range(len(layer_names))])
		self.names = layer_names

	def forward(self, desc0, desc1):
		for layer, name in zip(self.layers, self.names):
			if name == 'cross':
				src0, src1 = desc1, desc0
			else:  # if name == 'self':
				src0, src1 = desc0, desc1
			delta0, delta1 = layer(desc0, src0), layer(desc1, src1)
			desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
		return desc0, desc1


class Robust_DCP_SVDHead(nn.Module):
	def __init__(self, alpha=0.1, num_niters=5):
		super(Robust_DCP_SVDHead, self).__init__()
		self.alpha = 0.20
		self.num_iters = num_niters
		bin_score = nn.Parameter(torch.tensor(1.))
		self.register_parameter('bin_score', bin_score)
	

	def regular_scores(self, scores):
		scores = torch.where(torch.isnan(scores), torch.zeros_like(scores), scores)
		scores = torch.where(torch.isinf(scores), torch.ones_like(scores), scores)
		return scores

	def forward(self, src_embedding, tgt_embedding, src, tgt):
		# B C N; ; B 3 N;
		batch_size, _, num_points = src.size()

		d_k = src_embedding.size(1) # 512
		# scores = torch.matmul(F.normalize(src_embedding.transpose(2, 1).contiguous(), dim=2), F.normalize(tgt_embedding, dim=1))
		# scores = torch.matmul(src_embedding.transpose(2, 1).contiguous(), tgt_embedding) / math.sqrt(d_k)
		scores = self.alpha - square_distance(src_embedding.transpose(2, 1).contiguous(), 
									tgt_embedding.transpose(1, 2).contiguous()) / math.sqrt(d_k)
		# scores = -square_distance(F.normalize(src_embedding.transpose(2, 1).contiguous(), dim=2), 
		# 	F.normalize(tgt_embedding.transpose(1, 2).contiguous(), dim=2))
		# scores = torch.softmax(scores, dim=2)
		# b x points x points

		# scores = log_optimal_transport(
		# 	scores, self.bin_score,
		# 	iters=self.num_iters)
		
		scores = sinkhorn(scores, num_iters=self.num_iters)
		
		scores = torch.exp(scores[:, :-1, :-1])

		scores = self.regular_scores(scores)

		# src_corr = torch.matmul(tgt, scores.transpose(2, 1).contiguous())
		src_corr = torch.matmul(tgt, scores.transpose(2, 1).contiguous()) / (torch.sum(scores, dim=2).unsqueeze(1) + _EPS)

		# src_centered = src - src.mean(dim=2, keepdim=True)
		# src_corr_centered = src_corr - src_corr.mean(dim=2, keepdim=True)

		conf = torch.sum(scores, 2).unsqueeze(1) / (torch.sum(torch.sum(scores, 2, keepdim=True), 1, keepdim=True) + _EPS) # B 1 N
		
		cent_src = (src * conf).sum(dim=2, keepdim=True) # B 3 1
		cent_src_corr = (src_corr * conf).sum(dim=2, keepdim=True)

		src_centered = src - cent_src # B 3 N
		src_corr_centered = src_corr - cent_src_corr

		# H = torch.matmul(src_centered, src_corr_centered.transpose(2, 1).contiguous())
		H = torch.matmul(src_centered * conf, src_corr_centered.transpose(2, 1).contiguous())
		# b x 3 x 3

		# u, s, v = torch.svd(H, some=False, compute_uv=True)
		# rot_mat_pos = v @ u.transpose(-1, -2)
		# v_neg = v.clone()
		# v_neg[:, :, 2] *= -1
		# rot_mat_neg = v_neg @ u.transpose(-1, -2)
		# R = torch.where(torch.det(rot_mat_pos)[:, None, None] > 0, rot_mat_pos, rot_mat_neg)
		# assert torch.all(torch.det(R) > 0)

		U, _, V = torch.svd(H.cpu()) # Bx3x3, _, Bx3x3
		U = U.to(H.device)
		V = V.to(H.device)
		S = torch.eye(3).unsqueeze(0).repeat(U.shape[0], 1, 1).to(U.device)
		S[:, 2, 2] = torch.det(V @ U.transpose(1, 2)) # Bx3x3; (3, 3) = det
		R = V @ S @ U.transpose(1, 2)

		# t = torch.matmul(-R, src.mean(dim=2, keepdim=True)) + src_corr.mean(dim=2, keepdim=True) # B 3 1
		t = torch.matmul(-R, cent_src) + cent_src_corr # B 3 1

		bot_row = torch.Tensor([[[0, 0, 0, 1]]]).repeat(R.shape[0], 1, 1).to(R.device) # B 1 4
		T = torch.cat([torch.cat([R, t], dim=2), bot_row], dim=1)
		
		return T, scores


def square_distance(src, dst):
    """Calculate Euclid distance between each two points.
        src^T * dst = xn * xm + yn * ym + zn * zm；
        sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
        sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
        dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
             = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Args:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Returns:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, dim=-1)[:, :, None]
    dist += torch.sum(dst ** 2, dim=-1)[:, None, :]
    return dist


class FeatNetEncoder(nn.Module):
	def __init__(self, input_size, output_size):
		super(FeatNetEncoder, self).__init__()
		self.encoder = nn.Sequential(
			Conv2dGNReLU(4, output_size),
			Conv2dGNReLU(output_size, output_size),
			Conv2dGNReLU(output_size, output_size*2)
		)
		self.decoder = nn.Sequential(
			Conv1dGNReLU(output_size*2, output_size*2),
			Conv1dGNReLU(output_size*2, output_size),
			Conv1dGNReLU(output_size, output_size)
		)

	def forward(self, x):
		# batch_size, _, num_samples, num_points = x.size()
		x = self.encoder(x) # B 2C S N
		x = torch.max(x, dim=2)[0] # B 2C N
		x = self.decoder(x) # B C N
		return x

class PPR(nn.Module):
	def __init__(self, args):
		super(PPR, self).__init__()

		# self.conv1 = nn.Sequential(
		# 	Conv1dBNReLU(4*args.k_ppf, 64),
		# 	Conv1dBNReLU(64, 128)
		# 	)
		# self.hie1 = FeatNetEncoder(input_size=4, output_size=128)
		# self.hie2 = Conv2dBNReLU(128, 16)
		# self.hie3 = Conv2dBNReLU(16, 2)
		
		self.knn = 5 # 5
		self.use_ppf = args.use_ppf
		self.use_fpfh = args.use_fpfh
		self.k_ppf = args.k_ppf

		if self.use_ppf:
			self.hie1 = nn.Sequential(
				Conv1dBNReLU(4*args.k_ppf, 64),
			# 	Conv1dBNReLU(64, 64),
				# Conv1dBNReLU(64, 64),
				Conv1dBNReLU(64, 128)
				)
		elif self.use_fpfh:
			# self.emb_fpfh = FPFH()
			self.hie1 = nn.Sequential(
				Conv1dBNReLU(33, 64),
				Conv1dBNReLU(64, 128)
				)

		self.hie2 = nn.Sequential(
			Point_Transformer(128, 64, 128, k=self.knn),
			Conv1dBNReLU(128, 16)
			# Conv1dBNReLU(128, 32)
			# Conv1dBNReLU(128, 64)
			)
		self.hie3 = nn.Sequential(
			Point_Transformer(16, 8, 16, k=self.knn),
			Conv1dBNReLU(16, 2)
			# Point_Transformer(32, 16, 32, k=self.knn),
			# Conv1dBNReLU(32, 8)
			# Point_Transformer(64, 32, 64, k=self.knn),
			# Conv1dBNReLU(64, 32)
			)

		self.fuse_proj = nn.Conv1d(128+16+2, 128, 1)
		# self.fuse_proj = nn.Conv1d(128+32+8, 128, 1)
		# self.fuse_proj = nn.Conv1d(128+64+32, 128, 1)
		# self.final_proj = nn.Conv1d(128, 128, 1)
		# self.attention_proj = nn.Conv1d(128+16+2, 32, 1)

		# self.key_selection = KeyPointNet(args.num_keypoints)
		# self.backbone2 = PointNet(args)
		# self.use_rri = args.use_data_rri
		# self.k_rri = args.k_rri

		# self.pt_block1 = Point_Transformer(4, 32, 64, k=10)
		# self.pt_block2 = Point_Transformer(64, 32, 64, k=10)
		# self.pt_block3 = Point_Transformer(64, 32, 64, k=10)
		# self.pt_block4 = Point_Transformer(64, 64, 128, k=10)

		# self.gnn = AttentionalGNN(feature_size=128, layer_names=['self', 'cross'] * 2)

		self.robust_pointer = Robust_DCP_SVDHead()

	
	@staticmethod
	def get_ds_cent_feats(feats, points, num_centers):
		p_idx = pn2.furthest_point_sample(points, num_centers) # B M
		cent_pts = pn2.gather_points(points.transpose(1,2).contiguous(), p_idx).transpose(1,2).contiguous() # B M 3
		cent_feats = pn2.gather_points(feats, p_idx) # B C M
		return p_idx, cent_pts, cent_feats

	
	@staticmethod
	def get_ds_feats(feats, points, cent_pts, num_samples):
		pn_idx = knn_point(num_samples, points, cent_pts).detach().int() # B M S
		cluster_feats = pn2.grouping_operation(feats, pn_idx) # B C M S
		cluster_pts = pn2.grouping_operation(points.transpose(1,2).contiguous(), pn_idx) # B 3 M S
		return pn_idx, cluster_pts, cluster_feats
	
	@staticmethod
	def get_us_feats(feats, src_pts, tgt_pts):
		idx, weight = three_nn_upsampling(tgt_pts, src_pts)
		feats = pn2.three_interpolate(feats, idx, weight)
		return feats

	def regis_err(self, T_gt, reverse=False):
		if reverse:
			self.r_err_21 = rotation_error(self.T_21[:, :3, :3], T_gt[:, :3, :3])
			self.t_err_21 = translation_error(self.T_21[:, :3, 3], T_gt[:, :3, 3])
			return self.r_err_21.mean().item(), self.t_err_21.mean().item()
		else:
			self.r_err_12 = rotation_error(self.T_12[:, :3, :3], T_gt[:, :3, :3])
			self.t_err_12 = translation_error(self.T_12[:, :3, 3], T_gt[:, :3, 3])
			return self.r_err_12.mean().item(), self.t_err_12.mean().item()

	def forward(self, pts1, pts2, nm1, nm2, T_gt):
		batch_size, num_points1, _ = pts1.size()
		_, num_points2, _ = pts2.size()
		if self.use_ppf:
			self.pts1 = pts1
			self.pts2 = pts2
			_, _, _, _, feats1 = get_ppf(pts1, nm1, num_samples=self.k_ppf)
			_, _, _, _, feats2 = get_ppf(pts2, nm2, num_samples=self.k_ppf)
			# _, _, _, _, feats1 = get_ppf(pts1, nm1, num_samples=1)
			# _, _, _, _, feats2 = get_ppf(pts2, nm2, num_samples=1)
		
		elif self.use_fpfh:
			self.pts1 = pts1
			self.pts2 = pts2
			feats1 = nm1 #.transpose(1,2).contiguous()
			feats2 = nm2 #.transpose(1,2).contiguous()

		else:
			# self.pts1 = pts1
			# self.pts2 = pts2
			# feats1 = (pts1 - pts1.mean(dim=1, keepdim=True)).transpose(1, 2).contiguous()
			# feats2 = (pts2 - pts2.mean(dim=1, keepdim=True)).transpose(1, 2).contiguous()
			raise Exception('Either ppf or fpfh')

		# feats1 = feats1.transpose(1,2).contiguous().view(batch_size, num_points, self.k_ppf, 4).transpose(1,3).contiguous()
		# feats2 = feats2.transpose(1,2).contiguous().view(batch_size, num_points, self.k_ppf, 4).transpose(1,3).contiguous()

		feats11 = self.hie1(feats1)
		feats21 = self.hie1(feats2)

		_, cent_pts12, cent_feats12 = self.get_ds_cent_feats(feats11, pts1, num_points1//2)
		_, cent_pts22, cent_feats22 = self.get_ds_cent_feats(feats21, pts2, num_points2//2)

		_, sm_pts12, sm_feats12 = self.get_ds_feats(feats11, pts1, cent_pts12, self.knn) # B C M S
		_, sm_pts22, sm_feats22 = self.get_ds_feats(feats21, pts2, cent_pts22, self.knn)

		# # feats12, _ = torch.max(self.hie2(sm_feats12), dim=-1)
		# # feats22, _ = torch.max(self.hie2(sm_feats22), dim=-1)
		feats12 = self.hie2([sm_feats12, cent_pts12, sm_pts12])
		feats22 = self.hie2([sm_feats22, cent_pts22, sm_pts22])

		_, cent_pts13, cent_feats13 = self.get_ds_cent_feats(feats12, cent_pts12, num_points1//8)
		_, cent_pts23, cent_feats23 = self.get_ds_cent_feats(feats22, cent_pts22, num_points2//8)

		_, sm_pts13, sm_feats13 = self.get_ds_feats(feats12, cent_pts12, cent_pts13, self.knn) # B C M S
		_, sm_pts23, sm_feats23 = self.get_ds_feats(feats22, cent_pts22, cent_pts23, self.knn)

		# # feats13, _ = torch.max(self.hie3(sm_feats13), dim=-1)
		# # feats23, _ = torch.max(self.hie3(sm_feats23), dim=-1)
		feats13 = self.hie3([sm_feats13, cent_pts13, sm_pts13])
		feats23 = self.hie3([sm_feats23, cent_pts23, sm_pts23])

		feats12 = self.get_us_feats(feats12, cent_pts12, pts1)
		feats22 = self.get_us_feats(feats22, cent_pts22, pts2)

		feats13 = self.get_us_feats(feats13, cent_pts13, pts1)
		feats23 = self.get_us_feats(feats23, cent_pts23, pts2)

		feats1 = torch.cat((feats11, feats12, feats13), dim=1)
		feats2 = torch.cat((feats21, feats22, feats23), dim=1)

		# _, _, cent_feats12 = self.get_ds_cent_feats(cent_feats12, cent_pts12, num_points//8)
		# _, _, cent_feats22 = self.get_ds_cent_feats(cent_feats22, cent_pts22, num_points//8)

		# feats1 = torch.cat((cent_feats12, cent_feats13, feats13), dim=1)
		# feats2 = torch.cat((cent_feats22, cent_feats23, feats23), dim=1)

		# # self.Spts1, self.Spts2, Sfeats1, Sfeats2 = self.key_selection(self.pts1, self.pts2, feats1, feats2)
		# # self.pts1, self.pts2, feats1, feats2 = self.key_selection(self.pts1, self.pts2, nm1, nm2, feats1, feats2)

		# # gamma1 = self.attention_proj(feats1) # B x 32 x N
		# # gamma2 = self.attention_proj(feats2)

		feats1 = self.fuse_proj(feats1) # B x C x N
		feats2 = self.fuse_proj(feats2)

		# _, sample_pts1, sample_feats1 = self.get_ds_cent_feats(feats1, pts1, num_points//4)
		# _, sample_pts2, sample_feats2 = self.get_ds_cent_feats(feats2, pts2, num_points//4)

		# self.loss_feat = cal_descriptor_loss(feats1, feats2, pts1, pts2, T_gt) * 0.1
		# self.loss_feat = cal_descriptor_loss(sample_feats1, sample_feats2, sample_pts1, sample_pts2, T_gt) * 0.1

		# feats1, feats2 = self.gnn(feats1, feats2)

		# sm_feats1, sm_pts1 = get_cluster_feats(feats1, pts1, 10, True)
		# sm_feats2, sm_pts2 = get_cluster_feats(feats2, pts2, 10, True)

		# feats1 = self.pt_block1([sm_feats1, pts1, sm_pts1])
		# feats2 = self.pt_block1([sm_feats2, pts2, sm_pts2])

		# sm_feats1, _ = get_cluster_feats(feats1, pts1, 10)
		# sm_feats2, _ = get_cluster_feats(feats2, pts2, 10)

		# feats1 = self.pt_block2([sm_feats1, pts1, sm_pts1])
		# feats2 = self.pt_block2([sm_feats2, pts2, sm_pts2])

		# sm_feats1, _ = get_cluster_feats(feats1, pts1, 10)
		# sm_feats2, _ = get_cluster_feats(feats2, pts2, 10)

		# feats1 = self.pt_block3([sm_feats1, pts1, sm_pts1])
		# feats2 = self.pt_block3([sm_feats2, pts2, sm_pts2])

		# sm_feats1, _ = get_cluster_feats(feats1, pts1, 10)
		# sm_feats2, _ = get_cluster_feats(feats2, pts2, 10)

		# feats1 = self.pt_block4([sm_feats1, pts1, sm_pts1])
		# feats2 = self.pt_block4([sm_feats2, pts2, sm_pts2])

		# feats1 = self.final_proj(feats1) # B x C x N
		# feats2 = self.final_proj(feats2)

		# feats1 = feats1 / torch.norm(feats1, dim=1, keepdim=True)
		# feats2 = feats2 / torch.norm(feats2, dim=1, keepdim=True)

		# self.gamma1 = F.softmax(gamma1.transpose(1, 2).contiguous(), dim=2) # B x N x 32
		# self.gamma2 = F.softmax(gamma2.transpose(1, 2).contiguous(), dim=2)
		
		# self.pi1, self.mu1, self.sigma1, feats1 = gmm_feats_params(self.gamma1, self.pts1, feats1)
		# self.pi2, self.mu2, self.sigma2, feats2 = gmm_feats_params(self.gamma2, self.pts2, feats2)

		# self.pi1, self.mu1, self.sigma1 = gmm_params(self.gamma1, self.Spts1)
		# self.pi2, self.mu2, self.sigma2 = gmm_params(self.gamma2, self.Spts2)

		# self.gamma1 = F.softmax(self.backbone2(feats1), dim=2) # B x N x 16
		# self.pi1, self.mu1, self.sigma1 = gmm_params(self.gamma1, self.pts1)
		# self.gamma2 = F.softmax(self.backbone2(feats2), dim=2)
		# self.pi2, self.mu2, self.sigma2 = gmm_params(self.gamma2, self.pts2)

		# self.T_12 = gmm_register(self.pi1, self.mu1, self.mu2, self.sigma2)
		# self.T_21 = gmm_register(self.pi2, self.mu2, self.mu1, self.sigma1)

		# self.T_12 = DCP_SVDHead(feats1, feats2, self.pts1.transpose(1,2), self.pts2.transpose(1,2))
		# self.T_21 = DCP_SVDHead(feats2, feats1, self.pts2.transpose(1,2), self.pts1.transpose(1,2))
		
		self.T_12, self.scores12 = self.robust_pointer(feats1, feats2, self.pts1.transpose(1,2), self.pts2.transpose(1,2))
		self.T_21, self.scores21 = self.robust_pointer(feats2, feats1, self.pts2.transpose(1,2), self.pts1.transpose(1,2))

		# self.T_12, scores12 = self.robust_pointer(feats1, feats2, cent_pts13.transpose(1,2), cent_pts23.transpose(1,2))
		# self.T_21, scores21 = self.robust_pointer(feats2, feats1, cent_pts23.transpose(1,2), cent_pts13.transpose(1,2))
		
		# self.T_12, scores12 = robust_gmm_register(self.pi1, self.mu1, self.mu2, self.sigma2, feats1, feats2)
		# self.T_21, scores21 = robust_gmm_register(self.pi2, self.mu2, self.mu1, self.sigma1, feats2, feats1)
		self.T_gt = T_gt

		self.outlier_loss1 = ((1.0 - torch.sum(self.scores12, dim=2)).mean() + (1.0 - torch.sum(self.scores12, dim=1)).mean()) * 0.1
		self.outlier_loss2 = ((1.0 - torch.sum(self.scores21, dim=2)).mean() + (1.0 - torch.sum(self.scores21, dim=1)).mean()) * 0.1

		# eye = torch.eye(4).expand_as(self.T_gt).to(self.T_gt.device)
		# self.mse1 = F.mse_loss(self.T_12 @ torch.inverse(T_gt), eye)
		# self.mse2 = F.mse_loss(self.T_21 @ T_gt, eye)

		T_gt_inv = torch.inverse(T_gt)
		self.mse1 = (rotation_geodesic_error(self.T_12[:, :3, :3], T_gt[:, :3, :3]) 
						+ translation_error(self.T_12[:, :3, 3], T_gt[:, :3, 3])).mean()
		self.mse2 = (rotation_geodesic_error(self.T_21[:, :3, :3], T_gt_inv[:, :3, :3]) 
						+ translation_error(self.T_21[:, :3, 3], T_gt_inv[:, :3, 3])).mean()

		loss = self.mse1 + self.mse2 + self.outlier_loss1 + self.outlier_loss2 #+ self.loss_feat

		self.r_err = rotation_error(self.T_12[:, :3, :3], T_gt[:, :3, :3])
		self.t_err = translation_error(self.T_12[:, :3, 3], T_gt[:, :3, 3])
		# self.rmse = rmse_loss(self.pts1[:, :100], self.T_12, T_gt)
		self.rmse = rmse_loss(self.pts1, self.T_12, T_gt)

		return loss, self.r_err, self.t_err, self.rmse, self.mse1
		# return feats1, my_feats1

	def visualize(self, i):
		init_r_err = torch.acos((self.T_gt[i, :3, :3].trace() - 1) / 2) * 180 / math.pi
		init_t_err = torch.norm(self.T_gt[i, :3, 3])
		eye = torch.eye(4).unsqueeze(0).to(self.T_gt.device)
		init_rmse = rmse_loss(self.pts1[i:i+1], eye, self.T_gt[i:i+1])[0]
		pts1_trans = self.pts1[i] @ self.T_12[i, :3, :3].T + self.T_12[i, :3, 3]
		fig = visualize([self.pts1[i], self.gamma1[i], self.pi1[i], self.mu1[i], self.sigma1[i],
						 self.pts2[i], self.gamma2[i], self.pi2[i], self.mu2[i], self.sigma2[i],
						 pts1_trans, init_r_err, init_t_err, init_rmse,
						 self.r_err[i], self.t_err[i], self.rmse[i]])
		# pts1_trans = self.Spts1[i] @ self.T_12[i, :3, :3].T + self.T_12[i, :3, 3]
		# fig = visualize([self.Spts1[i], self.gamma1[i], self.pi1[i], self.mu1[i], self.sigma1[i],
		# 				 self.Spts2[i], self.gamma2[i], self.pi2[i], self.mu2[i], self.sigma2[i],
		# 				 pts1_trans, init_r_err, init_t_err, init_rmse,
		# 				 self.r_err[i], self.t_err[i], self.rmse[i]])
		return fig


	def get_transform(self):
		return self.T_12, self.scores12