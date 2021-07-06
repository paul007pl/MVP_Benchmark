#!/usr/bin/env python
# -*- coding: utf-8 -*-


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
from train_utils import *
from model_utils import clones, attention, get_graph_feature, SVDHead

# Part of the code is referred from: http://nlp.seas.harvard.edu/2018/04/03/attention.html#positional-encoding


class EncoderDecoder(nn.Module):
	"""
	A standard Encoder-Decoder architecture. Base for this and many
	other models.
	"""

	def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
		super(EncoderDecoder, self).__init__()
		self.encoder = encoder
		self.decoder = decoder
		self.src_embed = src_embed
		self.tgt_embed = tgt_embed
		self.generator = generator

	def forward(self, src, tgt, src_mask, tgt_mask):
		"Take in and process masked src and target sequences."
		return self.decode(self.encode(src, src_mask), src_mask,
						   tgt, tgt_mask)

	def encode(self, src, src_mask):
		return self.encoder(self.src_embed(src), src_mask)

	def decode(self, memory, src_mask, tgt, tgt_mask):
		return self.generator(self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask))


class Generator(nn.Module):
	def __init__(self, emb_dims):
		super(Generator, self).__init__()
		self.nn = nn.Sequential(nn.Linear(emb_dims, emb_dims // 2),
								nn.BatchNorm1d(emb_dims // 2),
								nn.ReLU(),
								nn.Linear(emb_dims // 2, emb_dims // 4),
								nn.BatchNorm1d(emb_dims // 4),
								nn.ReLU(),
								nn.Linear(emb_dims // 4, emb_dims // 8),
								nn.BatchNorm1d(emb_dims // 8),
								nn.ReLU())
		self.proj_rot = nn.Linear(emb_dims // 8, 4)
		self.proj_trans = nn.Linear(emb_dims // 8, 3)

	def forward(self, x):
		x = self.nn(x.max(dim=1)[0])
		rotation = self.proj_rot(x)
		translation = self.proj_trans(x)
		rotation = rotation / torch.norm(rotation, p=2, dim=1, keepdim=True)
		return rotation, translation


class Encoder(nn.Module):
	def __init__(self, layer, N):
		super(Encoder, self).__init__()
		self.layers = clones(layer, N)
		self.norm = LayerNorm(layer.size)
		# layer.size = 512

	def forward(self, x, mask):
		for layer in self.layers:
			x = layer(x, mask)
		return self.norm(x)


class Decoder(nn.Module):
	"Generic N layer decoder with masking."

	def __init__(self, layer, N):
		super(Decoder, self).__init__()
		self.layers = clones(layer, N)
		self.norm = LayerNorm(layer.size)

	def forward(self, x, memory, src_mask, tgt_mask):
		for layer in self.layers:
			x = layer(x, memory, src_mask, tgt_mask)
		return self.norm(x)


class LayerNorm(nn.Module):
	def __init__(self, features, eps=1e-6):
		super(LayerNorm, self).__init__()
		self.a_2 = nn.Parameter(torch.ones(features))
		self.b_2 = nn.Parameter(torch.zeros(features))
		self.eps = eps

	def forward(self, x):
		mean = x.mean(-1, keepdim=True)
		std = x.std(-1, keepdim=True)
		return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
	def __init__(self, size, dropout=None):
		super(SublayerConnection, self).__init__()
		self.norm = LayerNorm(size)

	def forward(self, x, sublayer):
		return x + sublayer(self.norm(x))


class EncoderLayer(nn.Module):
	def __init__(self, size, self_attn, feed_forward, dropout):
		super(EncoderLayer, self).__init__()
		self.self_attn = self_attn
		# attn => multihead attention
		self.feed_forward = feed_forward
		# ff => 2 sequential linear layers
		self.sublayer = clones(SublayerConnection(size, dropout), 2) # 2 sublayers
		# sublayer => x + sub_l(N(x))
		self.size = size # 512

	def forward(self, x, mask):
		x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
		return self.sublayer[1](x, self.feed_forward)


class DecoderLayer(nn.Module):
	"Decoder is made of self-attn, src-attn, and feed forward (defined below)"

	def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
		super(DecoderLayer, self).__init__()
		self.size = size
		self.self_attn = self_attn
		self.src_attn = src_attn
		self.feed_forward = feed_forward
		self.sublayer = clones(SublayerConnection(size, dropout), 3)

	def forward(self, x, memory, src_mask, tgt_mask):
		"Follow Figure 1 (right) for connections."
		m = memory
		x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
		x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
		# m is the output of encoder; x x m => m attention
		return self.sublayer[2](x, self.feed_forward)


class MultiHeadedAttention(nn.Module):
	def __init__(self, h, d_model, dropout=0.1):
		"Take in model size and number of heads."
		super(MultiHeadedAttention, self).__init__()
		assert d_model % h == 0
		# We assume d_v always equals d_k
		self.d_k = d_model // h # 512 // 4
		self.h = h # 4
		self.linears = clones(nn.Linear(d_model, d_model), 4) # deep clone will be another independent item
		self.attn = None
		self.dropout = None

	def forward(self, query, key, value, mask=None):
		"Implements Figure 2"
		if mask is not None:
			# Same mask applied to all h heads.
			mask = mask.unsqueeze(1)
		nbatches = query.size(0)

		# 1) Do all the linear projections in batch from d_model => h x d_k
		query, key, value = \
			[l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2).contiguous()
			 for l, x in zip(self.linears, (query, key, value))]
		# only 3 linears are used; b x 4 x points x 128

		# 2) Apply attention on all the projected vectors in batch.
		x, self.attn = attention(query, key, value, mask=mask,
								 dropout=self.dropout)
		# b x 4 x points x 128

		# 3) "Concat" using a view and apply a final linear.
		x = x.transpose(1, 2).contiguous() \
			.view(nbatches, -1, self.h * self.d_k)
		return self.linears[-1](x)
		# the last linear is used here; b x points x 512


class PositionwiseFeedForward(nn.Module):
	"Implements FFN equation."

	def __init__(self, d_model, d_ff, dropout=0.1):
		super(PositionwiseFeedForward, self).__init__()
		self.w_1 = nn.Linear(d_model, d_ff)
		self.norm = nn.Sequential()  # nn.BatchNorm1d(d_ff)
		self.w_2 = nn.Linear(d_ff, d_model)
		self.dropout = None
		# d_model 512; d_ff 1024; dp 0

	def forward(self, x):
		return self.w_2(self.norm(F.relu(self.w_1(x)).transpose(2, 1).contiguous()).transpose(2, 1).contiguous())


class DGCNN(nn.Module):
	def __init__(self, emb_dims=512):
		super(DGCNN, self).__init__()
		self.conv1 = nn.Conv2d(6, 64, kernel_size=1, bias=False)
		self.conv2 = nn.Conv2d(64, 64, kernel_size=1, bias=False)
		self.conv3 = nn.Conv2d(64, 128, kernel_size=1, bias=False)
		self.conv4 = nn.Conv2d(128, 256, kernel_size=1, bias=False)
		self.conv5 = nn.Conv2d(512, emb_dims, kernel_size=1, bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.bn2 = nn.BatchNorm2d(64)
		self.bn3 = nn.BatchNorm2d(128)
		self.bn4 = nn.BatchNorm2d(256)
		self.bn5 = nn.BatchNorm2d(emb_dims)

	def forward(self, x):
		batch_size, num_dims, num_points = x.size()
		x = get_graph_feature(x)
		x = F.relu(self.bn1(self.conv1(x)))
		x1 = x.max(dim=-1, keepdim=True)[0]
		x = F.relu(self.bn2(self.conv2(x)))
		x2 = x.max(dim=-1, keepdim=True)[0]
		x = F.relu(self.bn3(self.conv3(x)))
		x3 = x.max(dim=-1, keepdim=True)[0]
		x = F.relu(self.bn4(self.conv4(x)))
		x4 = x.max(dim=-1, keepdim=True)[0]
		x = torch.cat((x1, x2, x3, x4), dim=1)
		x = F.relu(self.bn5(self.conv5(x))).view(batch_size, -1, num_points)
		return x


class Transformer(nn.Module):
	def __init__(self, args):
		super(Transformer, self).__init__()
		self.emb_dims = 512 # 512
		self.N = 1 # 1
		self.dropout = 0.0 # 0.0
		self.ff_dims = 1024 # 1024
		self.n_heads = 4 # 4
		c = copy.deepcopy
		attn = MultiHeadedAttention(self.n_heads, self.emb_dims)
		ff = PositionwiseFeedForward(self.emb_dims, self.ff_dims, self.dropout)
		self.model = EncoderDecoder(Encoder(EncoderLayer(self.emb_dims, c(attn), c(ff), self.dropout), self.N),
						Decoder(DecoderLayer(self.emb_dims, c(attn), c(attn), c(ff), self.dropout), self.N), nn.Sequential(), nn.Sequential(), nn.Sequential())

	def forward(self, *input):
		src = input[0]
		tgt = input[1]
		src = src.transpose(2, 1).contiguous() # B x N x C
		tgt = tgt.transpose(2, 1).contiguous()
		tgt_embedding = self.model(src, tgt, None, None).transpose(2, 1).contiguous() # B x C x N
		src_embedding = self.model(tgt, src, None, None).transpose(2, 1).contiguous()
		return src_embedding, tgt_embedding


class Model(nn.Module):
	def __init__(self, args):
		super(Model, self).__init__()
		self.emb_dims = 512
		self.emb_nn = DGCNN(emb_dims=512)
		self.pointer = Transformer(args=args)
		self.head = SVDHead(args=args)

	def forward(self, src, tgt, T_gt=None, prefix="train"):
		feat1 = src[..., :3].transpose(1, 2)
		feat2 = tgt[..., :3].transpose(1, 2)
		src = src[..., :3]
		tgt = tgt[..., :3]

		src_embedding = self.emb_nn(feat1)
		tgt_embedding = self.emb_nn(feat2)
		src_embedding_p, tgt_embedding_p = self.pointer(src_embedding, tgt_embedding)
		src_embedding = src_embedding + src_embedding_p
		tgt_embedding = tgt_embedding + tgt_embedding_p

		scores = torch.matmul(src_embedding.transpose(2, 1).contiguous(), tgt_embedding) / math.sqrt(self.emb_dims)
		scores = torch.softmax(scores, dim=2)
		# b x points x points
		feat1_corr = torch.matmul(feat2, scores.transpose(2, 1).contiguous())
		rotation_ab, translation_ab = self.head(feat1, feat1_corr)

		T_12 = rt_to_transformation(rotation_ab, translation_ab.unsqueeze(2))
		
		if T_gt == None:
			return T_12
		else:
			r_err = rotation_error(T_12[:, :3, :3], T_gt[:, :3, :3])
			t_err = translation_error(T_12[:, :3, 3], T_gt[:, :3, 3])
			rmse = rmse_loss(src, T_12, T_gt)
			eye = torch.eye(4).expand_as(T_gt).to(T_gt.device)
			mse = F.mse_loss(T_12 @ torch.inverse(T_gt), eye)
			loss = mse
			rt_mse = (rotation_geodesic_error(T_12[:, :3, :3], T_gt[:, :3, :3]) + translation_error(T_12[:, :3, 3], T_gt[:, :3, 3]))
			return loss, r_err, t_err, rmse, rt_mse
