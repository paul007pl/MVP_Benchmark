import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from visu_utils import visualize
from train_utils import *
# from model_utils import get_rri_cluster, Conv1DBNReLU, FCBNReLU


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()
    idx = pairwise_distance.topk(k=k, dim=-1)[1].detach()  # (batch_size, num_points, k)
    return idx


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


def gmm_params(gamma, pts):
	'''
	Inputs:
		gamma: B x N x J
		pts: B x N x 3
	Outputs:
		pi: B x J
		mu: B x J x 3
		sigma: B x J x 3 x 3
	'''
	# pi: B x J
	pi = gamma.mean(dim=1)
	Npi = pi * gamma.shape[1] 
	# mu: B x J x 3
	mu = gamma.transpose(1, 2) @ pts / Npi.unsqueeze(2) # (B x J x N) (B x N x 3) maintain permutation-invariance
	# diff: B x N x J x 3
	diff = pts.unsqueeze(2) - mu.unsqueeze(1) # (B x N x 1 x 3) (B x 1 x j x 3) 
	# sigma: B x J x 3 x 3
	eye = torch.eye(3).unsqueeze(0).unsqueeze(1).to(gamma.device)
	sigma = (
		((diff.unsqueeze(3) @ diff.unsqueeze(4)).squeeze() * gamma).sum(dim=1) / Npi
	).unsqueeze(2).unsqueeze(3) * eye
	return pi, mu, sigma


def gmm_register(pi_s, mu_s, mu_t, sigma_t):
	'''
	Inputs:
		pi: B x J
		mu: B x J x 3
		sigma: B x J x 3 x 3
	'''
	c_s = pi_s.unsqueeze(1) @ mu_s # B x 1 x 3
	c_t = pi_s.unsqueeze(1) @ mu_t # B x 1 x 3
	Ms = torch.sum((pi_s.unsqueeze(2) * (mu_s - c_s)).unsqueeze(3) @
				   (mu_t - c_t).unsqueeze(2) @ sigma_t.inverse(), dim=1) # 3x3
	U, _, V = torch.svd(Ms.cpu()) # Bx3x3, _, Bx3x3
	U = U.cuda()
	V = V.cuda()
	S = torch.eye(3).unsqueeze(0).repeat(U.shape[0], 1, 1).to(U.device)
	S[:, 2, 2] = torch.det(V @ U.transpose(1, 2)) # Bx3x3; (3, 3) = det
	R = V @ S @ U.transpose(1, 2)
	t = c_t.transpose(1, 2) - R @ c_s.transpose(1, 2)
	bot_row = torch.Tensor([[[0, 0, 0, 1]]]).repeat(R.shape[0], 1, 1).to(R.device)
	T = torch.cat([torch.cat([R, t], dim=2), bot_row], dim=1)
	return T


class TNet(nn.Module):
	def __init__(self):
		super(TNet, self).__init__()
		self.encoder = nn.Sequential(
			Conv1DBNReLU(3, 64),
			Conv1DBNReLU(64, 128),
			Conv1DBNReLU(128, 256))
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


class PointNet(nn.Module):
	def __init__(self, args):
		super(PointNet, self).__init__()
		self.use_tnet = args.use_tnet
		self.tnet = TNet() if self.use_tnet else None
		d_input = args.rri_size * 4 if args.use_rri else 3
		self.encoder = nn.Sequential(
			Conv1DBNReLU(d_input, 64),
			Conv1DBNReLU(64, 128),
			Conv1DBNReLU(128, 256),
			Conv1DBNReLU(256, 1024))
		self.decoder = nn.Sequential(
			Conv1DBNReLU(1024 * 2, 512),
			Conv1DBNReLU(512, 256),
			Conv1DBNReLU(256, 128),
			nn.Conv1d(128, args.num_groups, kernel_size=1))

	def forward(self, pts):
		pts = self.tnet(pts) if self.use_tnet else pts
		f_loc = self.encoder(pts)
		f_glob, _ = f_loc.max(dim=2)
		f_glob = f_glob.unsqueeze(2).expand_as(f_loc)
		y = self.decoder(torch.cat([f_loc, f_glob], dim=1))
		return y.transpose(1, 2)


class Model(nn.Module):
	def __init__(self, args):
		super(Model, self).__init__()
		self.backbone = PointNet(args)
		self.use_rri = args.use_rri
		self.k = args.rri_size

	def regis_err(self, T_gt, reverse=False):
		if reverse:
			self.r_err_21 = rotation_error(self.T_21[:, :3, :3], T_gt[:, :3, :3])
			self.t_err_21 = translation_error(self.T_21[:, :3, 3], T_gt[:, :3, 3])
			return self.r_err_21.mean().item(), self.t_err_21.mean().item()
		else:
			self.r_err_12 = rotation_error(self.T_12[:, :3, :3], T_gt[:, :3, :3])
			self.t_err_12 = translation_error(self.T_12[:, :3, 3], T_gt[:, :3, 3])
			return self.r_err_12.mean().item(), self.t_err_12.mean().item()

	def forward(self, pts1, pts2, T_gt=None, prefix="train"):
		if self.use_rri:
			self.pts1 = pts1
			self.pts2 = pts2
			feats1 = get_rri_cluster(pts1.transpose(1,2).unsqueeze(-1), self.k)
			feats2 = get_rri_cluster(pts2.transpose(1,2).unsqueeze(-1), self.k)
			feats1 = feats1.squeeze(-1)
			feats2 = feats2.squeeze(-1)
		else:
			self.pts1 = pts1
			self.pts2 = pts2
			feats1 = (pts1 - pts1.mean(dim=1, keepdim=True)).transpose(1, 2)
			feats2 = (pts2 - pts2.mean(dim=1, keepdim=True)).transpose(1, 2)

		self.gamma1 = F.softmax(self.backbone(feats1), dim=2) # B x N x 16
		self.pi1, self.mu1, self.sigma1 = gmm_params(self.gamma1, self.pts1)
		self.gamma2 = F.softmax(self.backbone(feats2), dim=2)
		self.pi2, self.mu2, self.sigma2 = gmm_params(self.gamma2, self.pts2)

		self.T_12 = gmm_register(self.pi1, self.mu1, self.mu2, self.sigma2)
		if prefix=="test":
			return self.T_12
		else:
			self.T_21 = gmm_register(self.pi2, self.mu2, self.mu1, self.sigma1)
			self.T_gt = T_gt

			eye = torch.eye(4).expand_as(self.T_gt).to(self.T_gt.device)
			self.mse1 = F.mse_loss(self.T_12 @ torch.inverse(T_gt), eye)
			self.mse2 = F.mse_loss(self.T_21 @ T_gt, eye)
			loss = self.mse1 + self.mse2

			self.r_err = rotation_error(self.T_12[:, :3, :3], T_gt[:, :3, :3])
			self.t_err = translation_error(self.T_12[:, :3, 3], T_gt[:, :3, 3])
			# self.rmse = rmse_loss(self.pts1[:, :100], self.T_12, T_gt)
			self.rmse = rmse_loss(self.pts1, self.T_12, T_gt)

			self.mse = (rotation_geodesic_error(self.T_12[:, :3, :3], T_gt[:, :3, :3]) + translation_error(self.T_12[:, :3, 3], T_gt[:, :3, 3]))

			return loss, self.r_err, self.t_err, self.rmse, self.mse

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
		return fig
