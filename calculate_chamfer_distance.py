# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

'''Chamfer distance calculation.
'''

import torch
import numpy as np 
from scipy.io import loadmat
import os
from pytorch3d.loss import chamfer_distance
import trimesh

def normalize_pts(pts):
	# pts [N,3]
	center = np.mean(pts,0)
	pts -= center
	dist = np.linalg.norm(pts,axis=1)
	pts /= np.max(dist*2) # align in a sphere with diameter equal to 1

	return pts

def compute_chamfer(recon_pts,gt_pts,num_pts=10000):
	np.random.seed(0)

	recon_pts = normalize_pts(recon_pts)
	idx = np.random.choice(len(recon_pts),size=(num_pts),replace=True)
	recon_pts = recon_pts[idx,:]

	gt_pts = normalize_pts(gt_pts)
	idx = np.random.choice(len(gt_pts),size=(num_pts),replace=True)
	gt_pts = gt_pts[idx,:]

	with torch.no_grad():
		recon_pts = torch.from_numpy(recon_pts).float().cuda()[None,...]
		gt_pts = torch.from_numpy(gt_pts).float().cuda()[None,...]
		dist,_ = chamfer_distance(recon_pts,gt_pts,batch_reduction=None)
		dist = dist.cpu().squeeze().numpy()
	return dist

def compute_recon_error(recon_path,gt_path):
	recon_mesh = trimesh.load(recon_path)
	if isinstance(recon_mesh,trimesh.Scene):
		recon_mesh = recon_mesh.dump().sum()
	recon_pts = recon_mesh.vertices

	gt_pts = loadmat(gt_path)['p']
	gt_pts = gt_pts[:,:3]

	return compute_chamfer(recon_pts,gt_pts)