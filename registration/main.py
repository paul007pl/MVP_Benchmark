import argparse
import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from time import time
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import os
import random
import sys

from data import TrainData, ModelNet40, MVP, ThreeDMatch, ThreeDMatchVoxel
from DeepGMR_model import DeepGMR
from DCP_model import DCP
from PRNet_model import PRNet
from ppr_model import PPR
from idam_model import IDAM, FPFH, GNN

def _init_(args):
	if not os.path.exists('checkpoints'):
		os.makedirs('checkpoints')
	if not os.path.exists('checkpoints/' + args.log_dir):
		os.makedirs('checkpoints/' + args.log_dir)
	if not os.path.exists('checkpoints/' + args.log_dir + '/' + 'models'):
		os.makedirs('checkpoints/' + args.log_dir + '/' + 'models')
	os.system('cp main.py checkpoints' + '/' + args.log_dir + '/' + 'main.py.backup')
	if args.method == 'deepgmr':
		os.system('cp DeepGMR_model.py checkpoints' + '/' + args.log_dir + '/' + 'DeepGMR_model.py.backup')
	elif args.method == 'dcp':
		os.system('cp DCP_model.py checkpoints' + '/' + args.log_dir + '/' + 'DCP_model.py.backup')
	elif args.method == 'prnet':
		os.system('cp PRNet_model.py checkpoints' + '/' + args.log_dir + '/' + 'PRNet_model.py.backup')
	elif args.method == 'ppr':
		os.system('cp ppr_model.py checkpoints' + '/' + args.log_dir + '/' + 'ppr_model.py.backup')
	os.system('cp data.py checkpoints' + '/' + args.log_dir + '/' + 'data.py.backup')


parser = argparse.ArgumentParser()
# general
parser.add_argument('--benchmark', type=str, default='ModelNet40', choices=['ModelNet40', 'MVP', '3DMatch'])
parser.add_argument('--log_dir', type=str, default='exp', metavar='N',
					help='Name of the experiment')
# dataset
parser.add_argument('--max_angle', type=float, default=180)
parser.add_argument('--max_trans', type=float, default=0.5)
# parser.add_argument('--n_cpoints', type=int, default=1024)
# parser.add_argument('--n_ppoints', type=int, default=768)
parser.add_argument('--gaussian_noise', action='store_true')
parser.add_argument('--unseen', action='store_true')
parser.add_argument('--partial', action='store_true')

# model
parser.add_argument('--method', type=str, default='ppr', help='which model to use', 
	choices=['deepgmr', 'dcp', 'ppr', 'idam_gnn', 'idam_fpfh'])
# parser.add_argument('--d_model', type=int, default=1024)
parser.add_argument('--descriptor_size', type=int, default=64) # 512
parser.add_argument('--num_groups', type=int, default=8) # 16, 8, 4
parser.add_argument('--num_samples', type=int, default=128)
parser.add_argument('--ds_ratio', type=int, default=16) # 64
parser.add_argument('--num_keypoints', type=int, default=256) # 16, 12
# parser.add_argument('--n_selected_clusters', type=int, default=16)
# parser.add_argument('--n_keypoints', type=int, default=512)

parser.add_argument('--use_rri', action='store_true')
parser.add_argument('--use_data_rri', action='store_true')
# parser.add_argument('--use_normal', action='store_true')
parser.add_argument('--use_ppf', action='store_true')
parser.add_argument('--use_fpfh', action='store_true')
parser.add_argument('--use_tnet', action='store_true')
parser.add_argument('--k_rri', type=int, default=5) # 20
parser.add_argument('--k_ppf', type=int, default=5) # 20
# train setting
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--log_freq', type=int, default=30)
parser.add_argument('--plot_freq', type=int, default=250)
parser.add_argument('--save_freq', type=int, default=10)
parser.add_argument('--manual_seed', action='store_true')
parser.add_argument('--seed', type=int, default=1234, metavar='S',
					help='random seed (default: 1)')
# eval setting
# parser.add_argument('--val_fraction', type=float, default=0.1)
parser.add_argument('--eval_batch_size', type=int, default=32)
parser.add_argument('--eval_plot_freq', type=int, default=10)

parser.add_argument('--eval_only', action='store_true')
parser.add_argument('--model_path', type=str, default='', help='the path for the trained model')
parser.add_argument('--visu', action='store_true')
parser.add_argument('--split_level', action='store_true')
args = parser.parse_args()

_init_(args)

LOG_FOUT = open(os.path.join('checkpoints', args.log_dir, 'log_train.txt'), 'a')

def log_string(out_str):
	LOG_FOUT.write(out_str+'\n')
	LOG_FOUT.flush()
	print(out_str)

log_string(str(args))


def train_one_epoch(epoch, model, loader, writer, log_freq, plot_freq, benchmark='ModelNet40'):
	model.train()

	log_fmt = 'Epoch {:03d} Step {:03d}/{:03d} Train: ' + \
			  'batch time {:.2f}, data time {:.2f}, loss {:.4f}, ' + \
			  'rotation error {:.2f}, translation error {:.4f}, RMSE {:.4f}, MSE {:.4f}'
	batch_time = []
	data_time = []
	losses = []
	r_errs = []
	t_errs = []
	rmses = []
	mses = []
	# losses = 0
	# r_errs = 0
	# t_errs = 0
	# rmses = 0
	# # n_correct = 0
	# mses = 0
	# N = 0
	total_steps = len(loader)

	start = time()
	for step, data in enumerate(loader):
		if benchmark == 'ModelNet40':
			if args.use_ppf or args.use_fpfh:
				pts1, pts2, nm1, nm2, T_gt = data
			else:
				pts1, pts2, T_gt = data
		elif benchmark == 'MVP' or benchmark == '3DMatch':
			if args.use_ppf or args.use_fpfh:
				pts1, pts2, nm1, nm2, T_gt, _ = data
			else:
				pts1, pts2, T_gt, _ = data

		if torch.cuda.is_available():
			pts1 = pts1.cuda()
			pts2 = pts2.cuda()
			T_gt = T_gt.cuda()
			if args.use_ppf or args.use_fpfh:
				nm1 = nm1.cuda()
				nm2 = nm2.cuda()
		data_time.append(time() - start)

		# rri, my_rri = model(pts1, pts2, T_gt)
		# print(rri.size(), my_rri.size())
		# print(rri, '\n')
		# print(my_rri, '\n')
		# for i in range(rri.size()[2]):
		# 	print(rri[:,:,i])
		# 	print(my_rri[:,:,i])
		# sys.exit()

		# N += pts1.shape[0]
		optimizer.zero_grad()
		if args.use_ppf or args.use_fpfh:
			loss, r_err, t_err, rmse, mse = model(pts1, pts2, nm1, nm2, T_gt)
		else:
			loss, r_err, t_err, rmse, mse = model(pts1, pts2, T_gt)
		# loss.backward()

		loss_net = loss.mean()
		loss_net.backward()
		optimizer.step()
		batch_time.append(time() - start)

		# losses.append(loss.item())
		losses.append(loss_net.item())
		r_errs.append(r_err.mean().item())
		t_errs.append(t_err.mean().item())
		rmses.append(rmse.mean().item())
		mses.append(mse.mean().item())

		# losses += loss_net.mean().item()
		# r_errs += r_err.sum().item()
		# t_errs += t_err.sum().item()
		# rmses += rmse.sum().item()
		# # n_correct += (rmse < rmse_thresh).sum().item()
		# mses += mse.sum().item()

		global_step = epoch * len(loader) + step + 1

		if global_step % log_freq == 0:
			log_str = log_fmt.format(
				epoch+1, step+1, total_steps,
				np.mean(batch_time), np.mean(data_time), np.mean(losses),
				np.mean(r_errs), np.mean(t_errs), np.mean(rmses), np.mean(mses)
			)
			# log_str = log_fmt.format(
			# 	epoch+1, step+1, total_steps,
			# 	np.mean(batch_time), np.mean(data_time), losses / N,
			# 	r_errs / N, t_errs / N, rmses / N, mses / N
			# )
			log_string(log_str)
			writer.add_scalar('train/loss', np.mean(losses), global_step)
			writer.add_scalar('train/rotation_error', np.mean(r_errs), global_step)
			writer.add_scalar('train/translation_error', np.mean(t_errs), global_step)
			writer.add_scalar('train/RMSE', np.mean(rmses), global_step)
			writer.add_scalar('train/MSE', np.mean(mses), global_step)
			# writer.add_scalar('test/loss', losses / N, global_step)
			# writer.add_scalar('test/rotation_error', r_errs / N, global_step)
			# writer.add_scalar('test/translation_error', t_errs / N, global_step)
			# writer.add_scalar('test/RMSE', rmses / N, global_step)
			# writer.add_scalar('test/MSE', mses / N, global_step)

			batch_time.clear()
			data_time.clear()
			losses.clear()
			r_errs.clear()
			t_errs.clear()
			rmses.clear()
			mses.clear()
			# losses = 0
			# r_errs = 0
			# t_errs = 0
			# rmses = 0
			# mses = 0
			# N = 0

		if args.method == 'deepgmr' and (global_step % plot_freq == 0) and args.visu:
			fig = model.module.visualize(0)
			writer.add_figure('train', fig, global_step)

		if args.method == 'ppr' and (global_step % plot_freq == 0) and args.visu:
			fig = model.module.visualize(0)
			writer.add_figure('train', fig, global_step)

		start = time()


def eval_one_epoch(epoch, model, loader, writer, global_step, plot_freq, rmse_thresh=0.1, benchmark='ModelNet40'):
	model.eval()

	log_fmt = 'Test: epoch {:03d}, inference time {:.3f}, preprocessing time {:.3f}, ' + \
			  'loss {:.4f}, rotation error {:.2f}, translation error {:.4f}, RMSE {:.4f}, recall {:.3f}, MSE {:.4f}'
	inference_time = 0
	preprocess_time = 0
	losses = 0
	r_errs = 0
	t_errs = 0
	rmses = 0
	n_correct = 0
	mses = 0
	N = 1e-6

	# if (benchmark == 'MVP' or benchmark == '3DMatch') and args.split_level:
	if (benchmark == 'MVP' or benchmark == '3DMatch'):
		log_fmt_part = 'Level: {:02d}, rotation error {:.2f}, translation error {:.4f}, RMSE {:.4f}, recall {:.3f}'
		rmse_thresh = 0.4

		r_errs1, r_errs2, r_errs3 = 0, 0, 0
		t_errs1, t_errs2, t_errs3 = 0, 0, 0
		rmses1, rmses2, rmses3 = 0, 0, 0
		n_correct1, n_correct2, n_correct3 = 0, 0, 0
		# mses1, mses2 = 0, 0
		N1, N2, N3 = 1e-6, 1e-6, 1e-6

	start = time()
	for step, data in enumerate(tqdm(loader, leave=False)):
		if benchmark == 'ModelNet40':
			if args.use_ppf or args.use_fpfh:
				pts1, pts2, nm1, nm2, T_gt = data
			else:
				pts1, pts2, T_gt = data
		elif benchmark == 'MVP' or benchmark == '3DMatch':
			if args.use_ppf or args.use_fpfh:
				pts1, pts2, nm1, nm2, T_gt, level = data
			else:
				pts1, pts2, T_gt, level = data

		if torch.cuda.is_available():
			pts1 = pts1.cuda()
			pts2 = pts2.cuda()
			T_gt = T_gt.cuda()
			if args.use_ppf or args.use_fpfh:
				nm1 = nm1.cuda()
				nm2 = nm2.cuda()
		preprocess_time += time() - start
		N += pts1.shape[0]

		with torch.no_grad():
			if args.use_ppf or args.use_fpfh:
				loss, r_err, t_err, rmse, mse = model(pts1, pts2, nm1, nm2, T_gt)
			else:
				loss, r_err, t_err, rmse, mse = model(pts1, pts2, T_gt)
			inference_time += time() - start

		if not (args.method is 'idam_gnn' or 'idam_fpfh'):
			losses += loss.mean().item()
		r_errs += r_err.sum().item()
		t_errs += t_err.sum().item()
		rmses += rmse.sum().item()
		n_correct += (rmse < rmse_thresh).sum().item()
		mses += mse.sum().item()

		# if (benchmark == 'MVP' or benchmark == '3DMatch') and args.split_level:
		if (benchmark == 'MVP' or benchmark == '3DMatch'):
			curr_bs = pts1.shape[0]
			for i in range(curr_bs):
				if level[i] == 0:
					r_errs1 += r_err[i].item()
					t_errs1 += t_err[i].item()
					rmses1 += rmse[i].item()
					if rmse[i].item() < rmse_thresh:
						n_correct1 += 1
					# mses1 += mse[i].item()
					N1 += 1
				elif level[i] == 1:
					r_errs2 += r_err[i].item()
					t_errs2 += t_err[i].item()
					rmses2 += rmse[i].item()
					if rmse[i].item() < rmse_thresh:
						n_correct2 += 1
					# mses2 += mse[i].item()
					N2 += 1
				elif level[i] == 2:
					r_errs3 += r_err[i].item()
					t_errs3 += t_err[i].item()
					rmses3 += rmse[i].item()
					if rmse[i].item() < rmse_thresh:
						n_correct3 += 1
					# mses2 += mse[i].item()
					N3 += 1

		if writer is not None:
			if args.method == 'deepgmr' and (step+1) % plot_freq == 0 and args.visu:
				fig = model.module.visualize(0)
				writer.add_figure('test/{:02d}'.format(step+1), fig, global_step)
			
			if args.method == 'ppr' and (step+1) % plot_freq == 0 and args.visu:
				fig = model.module.visualize(0)
				writer.add_figure('test/{:02d}'.format(step+1), fig, global_step)

		start = time()

	log_str = log_fmt.format(
		epoch+1, inference_time / N, preprocess_time / N, losses / len(loader),
		r_errs / N, t_errs / N, rmses / N, n_correct / N, mses / N
	)
	log_string(log_str)
	writer.add_scalar('test/loss', losses / len(loader), global_step)
	writer.add_scalar('test/rotation_error', r_errs / N, global_step)
	writer.add_scalar('test/translation_error', t_errs / N, global_step)
	writer.add_scalar('test/RMSE', rmses / N, global_step)
	writer.add_scalar('test/recall', n_correct / N, global_step)
	writer.add_scalar('test/MSE', mses / N, global_step)

	info_test = {
		'epoch': epoch+1,
		'loss': losses / len(loader),
		'rotation_error': r_errs / N,
		'translation_error': t_errs / N,
		'RMSE': rmses / N,
		'recall': n_correct / N,
		'MSE': mses / N
	}

	# if (benchmark == 'MVP' or benchmark == '3DMatch') and args.split_level:
	if (benchmark == 'MVP' or benchmark == '3DMatch'):
		log_str = log_fmt_part.format(
			0, r_errs1 / N1, t_errs1 / N1, rmses1 / N1, n_correct1 / N1
		)
		log_string(log_str)

		log_str = log_fmt_part.format(
			1, r_errs2 / N2, t_errs2 / N2, rmses2 / N2, n_correct2 / N2
		)
		log_string(log_str)

		log_str = log_fmt_part.format(
			2, r_errs3 / N3, t_errs3 / N3, rmses3 / N3, n_correct3 / N3
		)
		log_string(log_str)


	return info_test


if __name__ == "__main__":

	torch.backends.cudnn.deterministic = True

	if not args.manual_seed:
		args.seed = random.randint(1, 10000) 
	
	log_string("Random Seed: %d" % args.seed)

	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)
	np.random.seed(args.seed)

	# data = TrainData(args.data_file, args)
	# ids = np.random.permutation(len(data))
	# n_val = int(args.val_fraction * len(data))
	# train_data = Subset(data, ids[n_val:])
	# valid_data = Subset(data, ids[:n_val])

	if args.benchmark == 'ModelNet40':
		if args.partial:
			args.num_points = 768
		else:
			args.num_points = 1024
		train_data = ModelNet40('./data/ModelNet40_train.h5', 'train', args)
		test_data = ModelNet40('./data/ModelNet40_test.h5', 'test', args)
		# test_data = ModelNet40('./data/ModelNet40_train.h5', 'train', args)
		# test_data = ModelNet40('./data/ModelNet40_test.h5', 'train', args)

	elif args.benchmark == 'MVP':
		args.num_points = 2048
		train_data = MVP('./data/MVP_train_2048pts.h5', 'train', args)
		test_data = MVP('./data/MVP_test_2048pts.h5', 'test', args)
		
	elif args.benchmark == '3DMatch':
		# args.num_points = 4096
		# train_data = ThreeDMatch('./data/3DMatch_train_4096pts.h5', 'train', args)
		# test_data = ThreeDMatch('./data/3DMatch_all_test_4096pts.h5', 'test', args)
		
		# args.num_points = 3072
		# train_data = ThreeDMatch('./data/3DMatch_train_3072pts.h5', 'train', args)
		# test_data = ThreeDMatch('./data/3DMatch_test_3072pts.h5', 'test', args)
		
		args.num_points = None
		train_data = ThreeDMatchVoxel('./data/3DMatch_train_voxel_pts.h5', 'train', args)
		test_data = ThreeDMatchVoxel('./data/3DMatch_all_test_voxel_pts.h5', 'test', args)
		

	train_loader = DataLoader(train_data, args.batch_size, drop_last=True, shuffle=True)
	test_loader = DataLoader(test_data, args.eval_batch_size, drop_last=False, shuffle=False)

	# _init_(args)
	if args.method == 'deepgmr':
		model = DeepGMR(args)
	elif args.method == 'dcp':
		model = DCP(args)
	elif args.method == 'prnet':
		model = PRNet(args)
	elif args.method == 'ppr':
		model = PPR(args)
	elif args.method == 'idam_gnn':
		args.emb_dims = 64
		model = IDAM(GNN(args.emb_dims), args)
	elif args.method == 'idam_fpfh':
		args.emb_dims = 33
		model = IDAM(FPFH(), args)

	if args.model_path is not "":
		model.load_state_dict(torch.load(args.model_path))

	if torch.cuda.is_available():
		model.cuda()
		model = nn.DataParallel(model)


	optimizer = torch.optim.Adam(model.parameters(), args.lr)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, min_lr=1e-6)

	writer_path = 'checkpoints/%s/' % args.log_dir
	print(writer_path)
	writer = SummaryWriter(writer_path)

	if args.eval_only:
		info_test_best = eval_one_epoch(0, model, test_loader, writer, 0, args.eval_plot_freq, benchmark=args.benchmark)
		test_loss = info_test_best['loss']

		log_fmt = 'BestTestEpoch {:03d} loss {:.4f}, rotation error {:.2f}, translation error {:.4f}, RMSE {:.4f}, recall {:.4f}, MSE {:.4f}'
		log_str = log_fmt.format(
		info_test_best['epoch'], info_test_best['loss'], info_test_best['rotation_error'],
		info_test_best['translation_error'], info_test_best['RMSE'], info_test_best['recall'], info_test_best['MSE']
		)
		log_string(log_str)

	else:
		info_test_best = None

		for epoch in range(args.num_epochs):
			train_one_epoch(epoch, model, train_loader, writer, args.log_freq, args.plot_freq, benchmark=args.benchmark)

			global_step = (epoch+1) * len(train_loader)
			info_test = eval_one_epoch(epoch, model, test_loader, writer, global_step, args.eval_plot_freq, benchmark=args.benchmark)

			# info_test['epoch'] = epoch
			test_loss = info_test['loss']

			# if (info_test_best is None) or (info_test_best['RMSE'] > info_test['RMSE']):
			if (info_test_best is None) or (info_test_best['MSE'] > info_test['MSE']):
				info_test_best = info_test
				# info_test_best['stage'] = 'best_test'
				print('save best model')
				torch.save(model.state_dict(), ('checkpoints/%s/models/model_best.pth' % args.log_dir))

			log_fmt = 'BestTestEpoch {:03d} loss {:.4f}, rotation error {:.2f}, translation error {:.4f}, RMSE {:.4f}, recall {:.4f}, MSE {:.4f}'
			log_str = log_fmt.format(
			info_test_best['epoch'], info_test_best['loss'], info_test_best['rotation_error'],
			info_test_best['translation_error'], info_test_best['RMSE'], info_test_best['recall'], info_test_best['MSE']
			)
			log_string(log_str)

			scheduler.step(test_loss)
			for param_group in optimizer.param_groups:
				lr = float(param_group['lr'])
				break
			writer.add_scalar('train/learning_rate', lr, global_step)

			if (epoch+1) % args.save_freq == 0:
				filename = ('checkpoints/%s/models/checkpoint_epoch-%d.pth' % (args.log_dir, epoch+1))
				torch.save(model.state_dict(), filename)


LOG_FOUT.close()
