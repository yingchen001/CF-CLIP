import os
from argparse import Namespace
from tkinter import W

import torchvision
import numpy as np
import torch
from torch.utils.data import DataLoader
import sys
import time
import clip
from tqdm import tqdm



sys.path.append(".")
sys.path.append("..")
import criteria.clip_loss as clip_loss
from mapper.datasets.latents_dataset import LatentsDataset
from mapper.options.test_options import TestOptions
from mapper.styleclip_mapper import StyleCLIPMapper


def run(test_opts):
	out_path_results = os.path.join(test_opts.exp_dir)
	os.makedirs(out_path_results, exist_ok=True)

	# update test options with options used during training
	ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
	opts = ckpt['opts']
	opts.update(vars(test_opts))
	opts = Namespace(**opts)
	clip_feat = clip_loss.CLIPText(opts)
	net = StyleCLIPMapper(opts)
	net.eval()
	net.cuda()

	test_latents = torch.load(opts.latents_test_path)
	dataset = LatentsDataset(latents=test_latents.cpu(), opts=opts)
	dataloader = DataLoader(dataset,
	                        batch_size=opts.test_batch_size,
	                        shuffle=False,
	                        num_workers=int(opts.test_workers),
	                        drop_last=True)
	if opts.n_images is None:
		opts.n_images = len(dataset)
	
	global_i = 0
	global_time = []
	for input_batch in tqdm(dataloader):
		if global_i >= opts.n_images:
			break
		with torch.no_grad():
			input_cuda = input_batch
			if opts.w_space:
				input_cuda = input_cuda.unsqueeze(1).repeat(1,net.decoder.n_latent,1).float()
			input_cuda = input_cuda.cuda()
			text_inputs = torch.cat([clip.tokenize(opts.description)]).cuda()
			tic = time.time()
			result_batch = run_on_batch(input_cuda, text_inputs, net, clip_feat, opts.couple_outputs, opts.work_in_stylespace)
			toc = time.time()
			global_time.append(toc - tic)

		for i in range(opts.test_batch_size):
			im_path = str(global_i).zfill(5)
			if test_opts.couple_outputs:
				gt_path_results = os.path.join(test_opts.exp_dir, 'gt')
				os.makedirs(gt_path_results, exist_ok=True)
				torchvision.utils.save_image(result_batch[0][i], os.path.join(out_path_results, f"{im_path}.jpg"), normalize=True, range=(-1, 1))
				torchvision.utils.save_image(result_batch[2][i], os.path.join(gt_path_results, f"{im_path}.jpg"), normalize=True, range=(-1, 1))
			else:
				torchvision.utils.save_image(result_batch[0][i], os.path.join(out_path_results, f"{im_path}.jpg"), normalize=True, range=(-1, 1))

			global_i += 1

	stats_path = os.path.join(opts.exp_dir, 'stats.txt')
	result_str = 'Runtime {:.4f}+-{:.4f}'.format(np.mean(global_time), np.std(global_time))
	print(result_str)

	with open(stats_path, 'w') as f:
		f.write(result_str)


def run_on_batch(inputs, text_inputs, net, clip_feat, couple_outputs=False, stylespace=False):
	w = inputs
	t_feat = clip_feat(text_inputs).cuda().detach()
	t_feat = t_feat.unsqueeze(1).repeat(w.shape[0],18,1).float()
	t_w = net.text_mapper(t_feat)
	with torch.no_grad():
		w_hat = w + 0.1 * net.mapper(torch.cat([w, t_w], dim=-1))
		x_hat, w_hat, _ = net.decoder([w_hat], input_is_latent=True, return_latents=True,
			                                   randomize_noise=False, truncation=1)
		result_batch = (x_hat, w_hat)
		if couple_outputs:
			x, _ = net.decoder([w], input_is_latent=True, randomize_noise=False, truncation=1, input_is_stylespace=stylespace)
			result_batch = (x_hat, w_hat, x)
	return result_batch


if __name__ == '__main__':
	test_opts = TestOptions().parse()
	run(test_opts)