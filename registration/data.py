import h5py
import numpy as np
import os
import open3d as o3d
import torch
from pycuda import gpuarray
from pycuda.compiler import SourceModule
from scipy.spatial import cKDTree
from torch.utils.data import Dataset

from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import minkowski

def knn_idx(pts, k):
	kdt = cKDTree(pts) 
	_, idx = kdt.query(pts, k=k+1)
	return idx[:, 1:]


def get_rri(pts, k):
	# pts: N x 3, original points
	# q: N x K x 3, nearest neighbors
	q = pts[knn_idx(pts, k)]
	p = np.repeat(pts[:, None], k, axis=1)
	# rp, rq: N x K x 1, norms
	rp = np.linalg.norm(p, axis=-1, keepdims=True)
	rq = np.linalg.norm(q, axis=-1, keepdims=True)
	pn = p / rp
	qn = q / rq
	dot = np.sum(pn * qn, -1, keepdims=True)
	# theta: N x K x 1, angles
	theta = np.arccos(np.clip(dot, -1, 1))
	T_q = q - dot * p
	sin_psi = np.sum(np.cross(T_q[:, None], T_q[:, :, None]) * pn[:, None], -1)
	cos_psi = np.sum(T_q[:, None] * T_q[:, :, None], -1)
	psi = np.arctan2(sin_psi, cos_psi) % (2*np.pi)
	idx = np.argpartition(psi, 1)[:, :, 1:2]
	# phi: N x K x 1, projection angles
	phi = np.take_along_axis(psi, idx, axis=-1)
	feat = np.concatenate([rp, rq, theta, phi], axis=-1)
	return feat.reshape(-1, k * 4)


def get_rri_cuda(pts, k, npts_per_block=1):
	import pycuda.autoinit
	mod_rri = SourceModule(open('rri.cu').read() % (k, npts_per_block))
	rri_cuda = mod_rri.get_function('get_rri_feature')

	N = len(pts)
	pts_gpu = gpuarray.to_gpu(pts.astype(np.float32).ravel())
	k_idx = knn_idx(pts, k)
	k_idx_gpu = gpuarray.to_gpu(k_idx.astype(np.int32).ravel())
	feat_gpu = gpuarray.GPUArray((N * k * 4,), np.float32)

	rri_cuda(pts_gpu, np.int32(N), k_idx_gpu, feat_gpu,
			 grid=(((N-1) // npts_per_block)+1, 1),
			 block=(npts_per_block, k, 1))
	
	feat = feat_gpu.get().reshape(N, k * 4).astype(np.float32)
	return feat


def normal_estimation(points, radius=0.1, max_nn=10):
	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(points)
	pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
	pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))
	normal = np.array(pcd.normals)
	return normal


def fpfh_estimation(points, radius=0.1, max_nn=10):
	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(points)
	pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
	pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))
	pcd_fpfh = o3d.registration.compute_fpfh_feature(pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=100))
	return np.array(pcd_fpfh.data)


def farthest_subsample_points(pointcloud1, pointcloud2, num_subsampled_points=768):
	pointcloud1 = pointcloud1.T
	pointcloud2 = pointcloud2.T
	num_points = pointcloud1.shape[0]
	nbrs1 = NearestNeighbors(n_neighbors=num_subsampled_points, algorithm='auto',
							 metric=lambda x, y: minkowski(x, y)).fit(pointcloud1)
	random_p1 = np.random.random(size=(1, 3)) + np.array([[500, 500, 500]]) * np.random.choice([1, -1, 1, -1])
	idx1 = nbrs1.kneighbors(random_p1, return_distance=False).reshape((num_subsampled_points,))
	nbrs2 = NearestNeighbors(n_neighbors=num_subsampled_points, algorithm='auto',
							 metric=lambda x, y: minkowski(x, y)).fit(pointcloud2)
	random_p2 = random_p1 #np.random.random(size=(1, 3)) + np.array([[500, 500, 500]]) * np.random.choice([1, -1, 2, -2])
	idx2 = nbrs2.kneighbors(random_p2, return_distance=False).reshape((num_subsampled_points,))
	return pointcloud1[idx1, :].T, pointcloud2[idx2, :].T


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


class TestData(Dataset):
	def __init__(self, path, args):
		super(TestData, self).__init__()
		with h5py.File(path, 'r') as f:
			self.source = f['source'][...]
			self.target = f['target'][...]
			self.transform = f['transform'][...]
		self.n_cpoints = args.n_cpoints
		self.use_rri = args.use_data_rri
		self.k = args.k if self.use_rri else None
		self.get_rri = get_rri_cuda if torch.cuda.is_available() else get_rri

	def __getitem__(self, index):
		pcd1 = self.source[index][:self.n_cpoints]
		pcd2 = self.target[index][:self.n_cpoints]
		if self.use_rri:
			pcd1 = np.concatenate([pcd1, self.get_rri(pcd1 - pcd1.mean(axis=0), self.k)], axis=1)
			pcd2 = np.concatenate([pcd2, self.get_rri(pcd2 - pcd2.mean(axis=0), self.k)], axis=1)
		transform = self.transform[index]
		return pcd1.astype('float32'), pcd2.astype('float32'), transform.astype('float32')

	def __len__(self):
		return self.transform.shape[0]


class TrainData(Dataset):
	def __init__(self, path, args):
		super(TrainData, self).__init__()
		with h5py.File(path, 'r') as f:
			self.points = f['points'][...]
		self.n_cpoints = args.n_cpoints
		self.max_angle = args.max_angle / 180 * np.pi
		self.max_trans = args.max_trans
		self.noisy = not args.clean
		self.use_rri = args.use_data_rri
		self.k = args.k if self.use_rri else None
		self.get_rri = get_rri_cuda if torch.cuda.is_available() else get_rri

	def __getitem__(self, index):
		pcd1 = self.points[index][:self.n_cpoints]
		pcd2 = self.points[index][:self.n_cpoints]
		transform = random_pose(self.max_angle, self.max_trans / 2)
		pose1 = random_pose(np.pi, self.max_trans)
		pose2 = transform @ pose1
		pcd1 = pcd1 @ pose1[:3, :3].T + pose1[:3, 3]
		pcd2 = pcd2 @ pose2[:3, :3].T + pose2[:3, 3]
		if self.noisy:
			pcd1 = jitter_pcd(pcd1)
			pcd2 = jitter_pcd(pcd2)
		if self.use_rri:
			pcd1 = np.concatenate([pcd1, self.get_rri(pcd1 - pcd1.mean(axis=0), self.k)], axis=1)
			pcd2 = np.concatenate([pcd2, self.get_rri(pcd2 - pcd2.mean(axis=0), self.k)], axis=1)
		return pcd1.astype('float32'), pcd2.astype('float32'), transform.astype('float32')

	def __len__(self):
		return self.points.shape[0]


class ModelNet40_rot(Dataset):
	"""docstring for ModelNet40_rot"""
	def __init__(self, path, partition, args, category=None):
		super(ModelNet40_rot, self).__init__()
		self.partition = partition
		self.gaussian_noise = args.gaussian_noise
		self.unseen = args.unseen
		self.partial = args.partial
		
		self.n_cpoints = 1024
		self.n_ppoints = 768
		self.max_angle = args.max_angle / 180 * np.pi
		self.max_trans = args.max_trans

		self.use_rri = args.use_data_rri
		self.k = args.k_rri if self.use_rri else None
		self.get_rri = get_rri_cuda if torch.cuda.is_available() else get_rri
		
		if args.test_rot is not None:
			self.rot_angle = args.test_rot / 180 * np.pi
			sinz = np.sin(self.rot_angle)
			cosz = np.cos(self.rot_angle)

			self.pose = np.array([[cosz, -sinz, 0],
									[sinz, cosz, 0],
									[0, 0, 1]])
		else:
			self.rot_angle = None

		self.use_ppf = args.use_ppf
		self.use_fpfh = args.use_fpfh
		
		f = h5py.File(path, 'r')
		self.data = np.array(f['data'][:].astype('float32'))
		self.label = np.squeeze(np.array(f['label'][:].astype('int64')))

		print(self.data.shape, self.label.shape)

		if partition == "test":
			self.src_cc = np.array(f['complete_src_clean'][:].astype('float32'))
			self.tgt_cc = np.array(f['complete_tgt_clean'][:].astype('float32'))
			self.src_cn = np.array(f['complete_src_noise'][:].astype('float32'))
			self.tgt_cn = np.array(f['complete_tgt_noise'][:].astype('float32'))

			self.src_pc = np.array(f['partial_src_clean'][:].astype('float32'))
			self.tgt_pc = np.array(f['partial_tgt_clean'][:].astype('float32'))
			self.src_pn = np.array(f['partial_src_noise'][:].astype('float32'))
			self.tgt_pn = np.array(f['partial_tgt_noise'][:].astype('float32'))

			self.transforms = np.array(f['transforms'][:].astype('float32'))

		f.close()

		if category is not None:
			self.data = self.data[self.label==category]

			if partition == "test":
				self.src_cc = self.src_cc[self.label==category]
				self.tgt_cc = self.tgt_cc[self.label==category]
				self.src_cn = self.src_cn[self.label==category]
				self.tgt_cn = self.tgt_cn[self.label==category]

				self.src_pc = self.src_pc[self.label==category]
				self.tgt_pc = self.tgt_pc[self.label==category]
				self.src_pn = self.src_pn[self.label==category]
				self.tgt_pn = self.tgt_pn[self.label==category]

				self.transforms = self.transforms[self.label==category]

			self.label = self.label[self.label==category]

		if self.unseen:
			if self.partition == "test":
				self.data = self.data[self.label>=20]

				self.src_cc = self.src_cc[self.label>=20]
				self.tgt_cc = self.tgt_cc[self.label>=20]
				self.src_cn = self.src_cn[self.label>=20]
				self.tgt_cn = self.tgt_cn[self.label>=20]

				self.src_pc = self.src_pc[self.label>=20]
				self.tgt_pc = self.tgt_pc[self.label>=20]
				self.src_pn = self.src_pn[self.label>=20]
				self.tgt_pn = self.tgt_pn[self.label>=20]

				self.transforms = self.transforms[self.label>=20]

				self.label = self.label[self.label>=20]
			elif self.partition == "train":
				self.data = self.data[self.label<20]
				self.label = self.label[self.label<20]

		print(self.data.shape, self.label.shape)

	def __len__(self):
		return self.data.shape[0]

	def __getitem__(self, index):
		if self.partition == "train":
			src = self.data[index][:self.n_cpoints]
			tgt = src
			
			transform = random_pose(self.max_angle, self.max_trans / 2)
			pose1 = random_pose(np.pi, self.max_trans)
			pose2 = transform @ pose1

			src = src @ pose1[:3, :3].T + pose1[:3, 3]
			tgt = tgt @ pose2[:3, :3].T + pose2[:3, 3]

			# src = np.random.permutation(src)
			# tgt = np.random.permutation(tgt)

			if self.partial:
				src, tgt = farthest_subsample_points(src.T, tgt.T, num_subsampled_points=self.n_ppoints)
				src = src.T
				tgt = tgt.T

			if self.gaussian_noise:
				src = jitter_pcd(src)
				tgt = jitter_pcd(tgt)

		elif self.partition == "test":
			if self.partial:
				if self.gaussian_noise:
					src = self.src_pn[index][:self.n_ppoints]
					tgt = self.tgt_pn[index][:self.n_ppoints]
				else:
					src = self.src_pc[index][:self.n_ppoints]
					tgt = self.tgt_pc[index][:self.n_ppoints]
			else:
				if self.gaussian_noise:
					src = self.src_cn[index][:self.n_cpoints]
					tgt = self.tgt_cn[index][:self.n_cpoints]
				else:
					src = self.src_cc[index][:self.n_cpoints]
					tgt = self.tgt_cc[index][:self.n_cpoints]
			transform = self.transforms[index]

			# src = np.random.permutation(src)
			# tgt = np.random.permutation(tgt)
		
		if self.rot_angle is not None:
			src = src @ transform[:3, :3].T
			tgt = tgt @ self.pose.T
			transform[:3, :3] = self.pose
		
		if self.use_ppf:
			src_n = normal_estimation(src)
			tgt_n = normal_estimation(tgt)
		
		if self.use_fpfh:
			src_fpfh = fpfh_estimation(src)
			tgt_fpfh = fpfh_estimation(tgt)

		if self.use_rri:
			src = np.concatenate([src, self.get_rri(src - src.mean(axis=0), self.k)], axis=1)
			tgt = np.concatenate([tgt, self.get_rri(tgt - tgt.mean(axis=0), self.k)], axis=1)

		if self.use_ppf:
			return src.astype('float32'), tgt.astype('float32'), src_n.astype('float32'), tgt_n.astype('float32'), transform.astype('float32')
		if self.use_fpfh:
			return src.astype('float32'), tgt.astype('float32'), src_fpfh.astype('float32'), tgt_fpfh.astype('float32'), transform.astype('float32')
		else:
			return src.astype('float32'), tgt.astype('float32'), transform.astype('float32')


class ModelNet40(Dataset):
	"""docstring for ModelNet40"""
	def __init__(self, path, partition, args, category=None):
		super(ModelNet40, self).__init__()
		self.partition = partition
		self.gaussian_noise = args.gaussian_noise
		self.unseen = args.unseen
		self.partial = args.partial
		
		self.n_cpoints = 1024
		self.n_ppoints = 768
		self.max_angle = args.max_angle / 180 * np.pi
		self.max_trans = args.max_trans

		self.use_rri = args.use_data_rri
		self.k = args.k_rri if self.use_rri else None
		self.get_rri = get_rri_cuda if torch.cuda.is_available() else get_rri
		# self.get_rri = get_rri
		# self.use_normal = args.use_normal

		self.use_ppf = args.use_ppf
		self.use_fpfh = args.use_fpfh
		
		f = h5py.File(path, 'r')
		self.data = np.array(f['data'][:].astype('float32'))
		self.label = np.squeeze(np.array(f['label'][:].astype('int64')))
		# self.normal = np.array(f['normal'][:].astype('float32'))
		# self.faceId = f['faceId']

		print(self.data.shape, self.label.shape)

		if partition == "test":
			self.src_cc = np.array(f['complete_src_clean'][:].astype('float32'))
			self.tgt_cc = np.array(f['complete_tgt_clean'][:].astype('float32'))
			self.src_cn = np.array(f['complete_src_noise'][:].astype('float32'))
			self.tgt_cn = np.array(f['complete_tgt_noise'][:].astype('float32'))

			self.src_pc = np.array(f['partial_src_clean'][:].astype('float32'))
			self.tgt_pc = np.array(f['partial_tgt_clean'][:].astype('float32'))
			self.src_pn = np.array(f['partial_src_noise'][:].astype('float32'))
			self.tgt_pn = np.array(f['partial_tgt_noise'][:].astype('float32'))

			self.transforms = np.array(f['transforms'][:].astype('float32'))

		f.close()

		if category is not None:
			self.data = self.data[self.label==category]

			if partition == "test":
				self.src_cc = self.src_cc[self.label==category]
				self.tgt_cc = self.tgt_cc[self.label==category]
				self.src_cn = self.src_cn[self.label==category]
				self.tgt_cn = self.tgt_cn[self.label==category]

				self.src_pc = self.src_pc[self.label==category]
				self.tgt_pc = self.tgt_pc[self.label==category]
				self.src_pn = self.src_pn[self.label==category]
				self.tgt_pn = self.tgt_pn[self.label==category]

				self.transforms = self.transforms[self.label==category]

			self.label = self.label[self.label==category]

		if self.unseen:
			if self.partition == "test":
				self.data = self.data[self.label>=20]

				self.src_cc = self.src_cc[self.label>=20]
				self.tgt_cc = self.tgt_cc[self.label>=20]
				self.src_cn = self.src_cn[self.label>=20]
				self.tgt_cn = self.tgt_cn[self.label>=20]

				self.src_pc = self.src_pc[self.label>=20]
				self.tgt_pc = self.tgt_pc[self.label>=20]
				self.src_pn = self.src_pn[self.label>=20]
				self.tgt_pn = self.tgt_pn[self.label>=20]

				self.transforms = self.transforms[self.label>=20]

				self.label = self.label[self.label>=20]
			elif self.partition == "train":
				self.data = self.data[self.label<20]
				self.label = self.label[self.label<20]

		print(self.data.shape, self.label.shape)

	def __len__(self):
		return self.data.shape[0]

	def __getitem__(self, index):
		if self.partition == "train":
			src = self.data[index][:self.n_cpoints]
			tgt = src
			
			transform = random_pose(self.max_angle, self.max_trans / 2)
			pose1 = random_pose(np.pi, self.max_trans)
			pose2 = transform @ pose1

			src = src @ pose1[:3, :3].T + pose1[:3, 3]
			tgt = tgt @ pose2[:3, :3].T + pose2[:3, 3]

			# src = np.random.permutation(src)
			# tgt = np.random.permutation(tgt)

			if self.partial:
				src, tgt = farthest_subsample_points(src.T, tgt.T, num_subsampled_points=self.n_ppoints)
				src = src.T
				tgt = tgt.T

			if self.gaussian_noise:
				src = jitter_pcd(src)
				tgt = jitter_pcd(tgt)

		elif self.partition == "test":
			if self.partial:
				if self.gaussian_noise:
					src = self.src_pn[index][:self.n_ppoints]
					tgt = self.tgt_pn[index][:self.n_ppoints]
				else:
					src = self.src_pc[index][:self.n_ppoints]
					tgt = self.tgt_pc[index][:self.n_ppoints]
			else:
				if self.gaussian_noise:
					src = self.src_cn[index][:self.n_cpoints]
					tgt = self.tgt_cn[index][:self.n_cpoints]
				else:
					src = self.src_cc[index][:self.n_cpoints]
					tgt = self.tgt_cc[index][:self.n_cpoints]
			transform = self.transforms[index]

			# src = np.random.permutation(src)
			# tgt = np.random.permutation(tgt)
		
		if self.use_ppf:
			src_n = normal_estimation(src)
			tgt_n = normal_estimation(tgt)
		
		if self.use_fpfh:
			src_fpfh = fpfh_estimation(src)
			tgt_fpfh = fpfh_estimation(tgt)

		if self.use_rri:
			# # center = (src.mean(axis=0) + tgt.mean(axis=0)) / 2.0
			# src_center = src.mean(axis=0)
			# tgt_center = tgt.mean(axis=0)
			# c_src = src - src_center
			# c_tgt = tgt - tgt_center

			# src = np.concatenate([src, self.get_rri(src - center, self.k)], axis=1)
			# tgt = np.concatenate([tgt, self.get_rri(tgt - center, self.k)], axis=1)

			src = np.concatenate([src, self.get_rri(src - src.mean(axis=0), self.k)], axis=1)
			tgt = np.concatenate([tgt, self.get_rri(tgt - tgt.mean(axis=0), self.k)], axis=1)

		if self.use_ppf:
			return src.astype('float32'), tgt.astype('float32'), src_n.astype('float32'), tgt_n.astype('float32'), transform.astype('float32')
		if self.use_fpfh:
			return src.astype('float32'), tgt.astype('float32'), src_fpfh.astype('float32'), tgt_fpfh.astype('float32'), transform.astype('float32')
		else:
			return src.astype('float32'), tgt.astype('float32'), transform.astype('float32')


class MVP(Dataset):
	"""docstring for MVP"""
	def __init__(self, path, partition, args, category=None):
		self.partition = partition
		
		self.max_angle = args.max_angle / 180 * np.pi
		self.max_trans = args.max_trans

		self.use_rri = args.use_data_rri
		self.k = args.k_rri if self.use_rri else None
		self.get_rri = get_rri_cuda if torch.cuda.is_available() else get_rri

		self.use_ppf = args.use_ppf
		self.use_fpfh = args.use_fpfh

		f = h5py.File(path, 'r')

		self.match_level = f['match_level'][:].astype('int32')
		self.label = f['cat_labels'][:].astype('int32')

		match_id = []
		for i in range(len(f['match_id'].keys())):
			ds_data = f['match_id'][str(i)][:]
			match_id.append(ds_data)
		self.match_id = np.array(match_id, dtype=object)

		if self.partition == "train":
			self.src = np.array(f['src'][:].astype('float32'))
			self.tgt = np.array(f['tgt'][:].astype('float32'))
		elif self.partition == "test":
			self.src = np.array(f['rotated_src'][:].astype('float32'))
			self.tgt = np.array(f['rotated_tgt'][:].astype('float32'))
			self.transforms = np.array(f['transforms'][:].astype('float32'))
		
		f.close()
		
		if category is not None:
			self.src = self.src[self.label==category]
			self.tgt = self.tgt[self.label==category]
			self.match_id = self.match_id[self.label==category]
			self.match_level = self.match_level[self.label==category]
			if self.partition == "test":
				self.transforms = self.transforms[self.label==category]
			self.label = self.label[self.label==category]

		print(self.src.shape, self.tgt.shape, self.match_id.shape, self.match_level.shape, self.label.shape)
	
	def __len__(self):
		return self.src.shape[0]
	
	def __getitem__(self, index):
		src = self.src[index]
		tgt = self.tgt[index]
		level = self.match_level[index]

		if self.partition == "train":
			transform = random_pose(self.max_angle, self.max_trans / 2)
			pose1 = random_pose(np.pi, self.max_trans)
			pose2 = transform @ pose1

			src = src @ pose1[:3, :3].T + pose1[:3, 3]
			tgt = tgt @ pose2[:3, :3].T + pose2[:3, 3]
		
		elif self.partition == "test":
			transform = self.transforms[index]

		# src = np.random.permutation(src)
		# tgt = np.random.permutation(tgt)

		if self.use_ppf:
			src_n = normal_estimation(src)
			tgt_n = normal_estimation(tgt)
		
		if self.use_fpfh:
			src_fpfh = fpfh_estimation(src)
			tgt_fpfh = fpfh_estimation(tgt)

		if self.use_rri:
			src = np.concatenate([src, self.get_rri(src - src.mean(axis=0), self.k)], axis=1)
			tgt = np.concatenate([tgt, self.get_rri(tgt - tgt.mean(axis=0), self.k)], axis=1)
		
		if self.use_ppf:
			return src.astype('float32'), tgt.astype('float32'), src_n.astype('float32'), tgt_n.astype('float32'), transform.astype('float32'), level.astype('int32')
		if self.use_fpfh:
			return src.astype('float32'), tgt.astype('float32'), src_fpfh.astype('float32'), tgt_fpfh.astype('float32'), transform.astype('float32'), level.astype('int32')
		else:
			return src.astype('float32'), tgt.astype('float32'), transform.astype('float32'), level.astype('int32')


class ThreeDMatchLow(Dataset):
	"""docstring for ThreeDMatchLow"""
	def __init__(self, path, partition, args, category=None):
		self.partition = partition
		
		self.max_angle = args.max_angle / 180 * np.pi
		self.max_trans = args.max_trans

		self.use_rri = args.use_data_rri
		self.k = args.k_rri if self.use_rri else None
		self.get_rri = get_rri_cuda if torch.cuda.is_available() else get_rri

		self.use_ppf = args.use_ppf
		self.use_fpfh = args.use_fpfh

		f = h5py.File(path, 'r')
		if self.partition == "train":
			self.match_level = f['match_level'][:].astype('int32')

			match_id = []
			for i in range(len(f['match_id'].keys())):
				ds_data = f['match_id'][str(i)][:]
				match_id.append(ds_data)
			self.match_id = np.array(match_id, dtype=object)
		
			self.src = np.array(f['src'][:].astype('float32'))
			self.tgt = np.array(f['tgt'][:].astype('float32'))

			self.src = self.src[self.match_level > 0]
			self.tgt = self.tgt[self.match_level > 0]
			self.match_id = self.match_id[self.match_level > 0]
			self.match_level = self.match_level[self.match_level > 0]

		elif self.partition == "test":
			match_level1 = f['match_level1'][:].astype('int32')
			match_level2 = f['match_level2'][:].astype('int32')

			src1 = np.array(f['src1'][:].astype('float32'))
			tgt1 = np.array(f['tgt1'][:].astype('float32'))
			transforms1 = np.array(f['transforms1'][:].astype('float32'))

			src2 = np.array(f['src2'][:].astype('float32'))
			tgt2 = np.array(f['tgt2'][:].astype('float32'))
			transforms2 = np.array(f['transforms2'][:].astype('float32'))

			match_id1 = []
			for i in range(len(f['match_id1'].keys())):
				ds_data = f['match_id1'][str(i)][:]
				match_id1.append(ds_data)
			match_id1 = np.array(match_id1, dtype=object)

			match_id2 = []
			for i in range(len(f['match_id2'].keys())):
				ds_data = f['match_id2'][str(i)][:]
				match_id2.append(ds_data)
			match_id2 = np.array(match_id2, dtype=object)

			self.src = np.concatenate((src1, src2), axis=0)
			self.tgt = np.concatenate((tgt1, tgt2), axis=0)
			self.transforms = np.concatenate((transforms1, transforms2), axis=0)
			self.match_level = np.concatenate((match_level1, match_level2), axis=0)
			self.match_id = np.concatenate((match_id1, match_id2), axis=0)

			self.src = self.src[self.match_level >= 0]
			self.tgt = self.tgt[self.match_level >= 0]
			self.transforms = self.transforms[self.match_level >= 0]
			self.match_id = self.match_id[self.match_level >= 0]
			self.match_level = self.match_level[self.match_level >= 0]

		f.close()

		print(self.src.shape, self.tgt.shape, self.match_id.shape, self.match_level.shape)

	def __len__(self):
		return self.src.shape[0]
	
	def __getitem__(self, index):
		src = self.src[index]
		tgt = self.tgt[index]
		level = self.match_level[index]

		if self.partition == "train":
			transform = random_pose(self.max_angle, self.max_trans / 2)
			pose1 = random_pose(np.pi, self.max_trans)
			pose2 = transform @ pose1

			src = src @ pose1[:3, :3].T + pose1[:3, 3]
			tgt = tgt @ pose2[:3, :3].T + pose2[:3, 3]
		
		elif self.partition == "test":
			transform = self.transforms[index]

		# src = np.random.permutation(src)
		# tgt = np.random.permutation(tgt)
		
		if self.use_ppf:
			src_n = normal_estimation(src)
			tgt_n = normal_estimation(tgt)
		
		if self.use_fpfh:
			src_fpfh = fpfh_estimation(src)
			tgt_fpfh = fpfh_estimation(tgt)

		if self.use_rri:
			src = np.concatenate([src, self.get_rri(src - src.mean(axis=0), self.k)], axis=1)
			tgt = np.concatenate([tgt, self.get_rri(tgt - tgt.mean(axis=0), self.k)], axis=1)
		
		if self.use_ppf:
			return src.astype('float32'), tgt.astype('float32'), src_n.astype('float32'), tgt_n.astype('float32'), transform.astype('float32'), level.astype('int32')
		if self.use_fpfh:
			return src.astype('float32'), tgt.astype('float32'), src_fpfh.astype('float32'), tgt_fpfh.astype('float32'), transform.astype('float32'), level.astype('int32')
		else:
			return src.astype('float32'), tgt.astype('float32'), transform.astype('float32'), level.astype('int32')


class ThreeDMatch(Dataset):
	"""docstring for ThreeDMatch"""
	def __init__(self, path, partition, args, category=None):
		self.partition = partition

		self.max_angle = args.max_angle / 180 * np.pi
		self.max_trans = args.max_trans

		self.use_rri = args.use_data_rri
		self.k = args.k_rri if self.use_rri else None
		self.get_rri = get_rri_cuda if torch.cuda.is_available() else get_rri

		self.use_ppf = args.use_ppf
		self.use_fpfh = args.use_fpfh

		f = h5py.File(path, 'r')
		self.match_level = f['match_level'][:].astype('int32')

		self.src = np.array(f['src'][:].astype('float32'))
		self.tgt = np.array(f['tgt'][:].astype('float32'))

		if self.partition == "test":
			self.transforms = np.array(f['transforms'][:].astype('float32'))

		f.close()
		print(self.src.shape, self.tgt.shape, self.match_level.shape)

	def __len__(self):
		return self.src.shape[0]
	
	def __getitem__(self, index):
		src = self.src[index]
		tgt = self.tgt[index]
		level = self.match_level[index]
		
		if self.partition == "train":
			transform = random_pose(self.max_angle, self.max_trans / 2)
			pose1 = random_pose(np.pi, self.max_trans)
			pose2 = transform @ pose1

			src = src @ pose1[:3, :3].T + pose1[:3, 3]
			tgt = tgt @ pose2[:3, :3].T + pose2[:3, 3]
		
		elif self.partition == "test":
			transform = self.transforms[index]

		# src = np.random.permutation(src)
		# tgt = np.random.permutation(tgt)
		
		if self.use_ppf:
			src_n = normal_estimation(src)
			tgt_n = normal_estimation(tgt)
		
		if self.use_fpfh:
			src_fpfh = fpfh_estimation(src)
			tgt_fpfh = fpfh_estimation(tgt)

		if self.use_rri:
			src = np.concatenate([src, self.get_rri(src - src.mean(axis=0), self.k)], axis=1)
			tgt = np.concatenate([tgt, self.get_rri(tgt - tgt.mean(axis=0), self.k)], axis=1)
		
		if self.use_ppf:
			return src.astype('float32'), tgt.astype('float32'), src_n.astype('float32'), tgt_n.astype('float32'), transform.astype('float32'), level.astype('int32')
		if self.use_fpfh:
			return src.astype('float32'), tgt.astype('float32'), src_fpfh.astype('float32'), tgt_fpfh.astype('float32'), transform.astype('float32'), level.astype('int32')
		else:
			return src.astype('float32'), tgt.astype('float32'), transform.astype('float32'), level.astype('int32')


class ThreeDMatchVoxel(Dataset):
	"""docstring for ThreeDMatchVoxel"""
	def __init__(self, path, partition, args, category=None):
		self.partition = partition
		
		self.max_angle = args.max_angle / 180 * np.pi
		self.max_trans = args.max_trans

		self.use_rri = args.use_data_rri
		self.k = args.k_rri if self.use_rri else None
		self.get_rri = get_rri_cuda if torch.cuda.is_available() else get_rri

		self.use_ppf = args.use_ppf
		self.use_fpfh = args.use_fpfh

		f = h5py.File(path, 'r')
		
		self.match_level = f['match_level'][:].astype('int32')

		match_id = []
		for i in range(len(f['match_id'].keys())):
			ds_data = f['match_id'][str(i)][:]
			match_id.append(ds_data)
		self.match_id = np.array(match_id, dtype=object)

		src, tgt = [], []
		for i in range(len(f['src'].keys())):
			src_data = f['src'][str(i)][:]
			tgt_data = f['tgt'][str(i)][:]
			src.append(src_data)
			tgt.append(tgt_data)

		self.src = np.array(src, dtype=object)
		self.tgt = np.array(tgt, dtype=object)

		if self.partition == "test":
			self.transforms = np.array(f['transforms'][:].astype('float32'))

			self.src = self.src[self.match_level >= 0]
			self.tgt = self.tgt[self.match_level >= 0]
			self.transforms = self.transforms[self.match_level >= 0]
			self.match_id = self.match_id[self.match_level >= 0]
			self.match_level = self.match_level[self.match_level >= 0]
		else:
			self.src = self.src[self.match_level > 0]
			self.tgt = self.tgt[self.match_level > 0]
			self.match_id = self.match_id[self.match_level > 0]
			self.match_level = self.match_level[self.match_level > 0]

		f.close()

		print(self.src.shape, self.tgt.shape, self.match_id.shape, self.match_level.shape)

	def __len__(self):
		return self.src.shape[0]
	
	def __getitem__(self, index):
		src = self.src[index]
		tgt = self.tgt[index]
		level = self.match_level[index]

		if self.partition == "train":
			transform = random_pose(self.max_angle, self.max_trans / 2)
			pose1 = random_pose(np.pi, self.max_trans)
			pose2 = transform @ pose1

			src = src @ pose1[:3, :3].T + pose1[:3, 3]
			tgt = tgt @ pose2[:3, :3].T + pose2[:3, 3]
		
		elif self.partition == "test":
			transform = self.transforms[index]
			# pose = random_pose(self.max_angle, self.max_trans)
			# tgt = tgt @ pose[:3, :3].T + pose[:3, 3]
			# transform = pose @ transform

		# src = np.random.permutation(src)
		# tgt = np.random.permutation(tgt)
		
		if self.use_ppf:
			src_n = normal_estimation(src)
			tgt_n = normal_estimation(tgt)
		
		if self.use_fpfh:
			src_fpfh = fpfh_estimation(src)
			tgt_fpfh = fpfh_estimation(tgt)

		if self.use_rri:
			src = np.concatenate([src, self.get_rri(src - src.mean(axis=0), self.k)], axis=1)
			tgt = np.concatenate([tgt, self.get_rri(tgt - tgt.mean(axis=0), self.k)], axis=1)
		
		if self.use_ppf:
			return src.astype('float32'), tgt.astype('float32'), src_n.astype('float32'), tgt_n.astype('float32'), transform.astype('float32'), level.astype('int32')
		if self.use_fpfh:
			return src.astype('float32'), tgt.astype('float32'), src_fpfh.astype('float32'), tgt_fpfh.astype('float32'), transform.astype('float32'), level.astype('int32')
		else:
			return src.astype('float32'), tgt.astype('float32'), transform.astype('float32'), level.astype('int32')