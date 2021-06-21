# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

'''Dataset for DIF-Net.
'''

import os

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat

class PointCloud_with_FreePoints(Dataset):
    def __init__(self, pointcloud_path, on_surface_points, instance_idx=None, expand=-1, max_points=-1):
        super().__init__()

        self.instance_idx = instance_idx
        self.expand = expand

        print("Loading point cloud of subject%04d"%self.instance_idx)
        # surface points
        point_cloud = loadmat(pointcloud_path)
        point_cloud = point_cloud['p']

        # free space points
        free_points = loadmat(pointcloud_path.replace('surface_pts_n_normal','free_space_pts'))
        free_points = free_points['p_sdf']
        print("Finished loading point cloud")

        free_points_coords = free_points[:,:3]
        free_points_sdf = free_points[:,3:]

        # surface points with normals
        self.coords = point_cloud[:, :3]
        self.normals = point_cloud[:, 3:]

        self.free_points_coords = free_points_coords
        self.free_points_sdf = free_points_sdf
        
        self.on_surface_points = on_surface_points
        self.max_points = max_points

    def __len__(self):
        if self.max_points != -1:
            return self.max_points // self.on_surface_points
        return self.coords.shape[0] // self.on_surface_points

    def __getitem__(self, idx):
        point_cloud_size = self.coords.shape[0]
        free_point_size = self.free_points_coords.shape[0]

        off_surface_samples = self.on_surface_points 
        total_samples = self.on_surface_points + off_surface_samples

        # Random coords
        rand_idcs = np.random.choice(point_cloud_size, size=self.on_surface_points)

        on_surface_coords = self.coords[rand_idcs, :]
        on_surface_normals = self.normals[rand_idcs, :]

        if self.expand != -1:
            on_surface_coords += on_surface_normals*self.expand  # expand the shape surface if its structure is too thin

        off_surface_coords = np.random.uniform(-1, 1, size=(off_surface_samples//2, 3))
        free_rand_idcs = np.random.choice(free_point_size, size=off_surface_samples//2)
        free_points_coords = self.free_points_coords[free_rand_idcs,:]

        off_surface_normals = np.ones((off_surface_samples, 3)) * -1

        sdf = np.zeros((total_samples, 1))  # on-surface = 0
        sdf[self.on_surface_points:, :] = -1  # off-surface = -1

        # if a free space point has gt SDF value, replace -1 with it.
        if self.expand != -1:
            sdf[self.on_surface_points+off_surface_samples//2:,:] = (self.free_points_sdf[free_rand_idcs] - self.expand)
        else:
            sdf[self.on_surface_points+off_surface_samples//2:,:] = self.free_points_sdf[free_rand_idcs]

        coords = np.concatenate((on_surface_coords, off_surface_coords,free_points_coords), axis=0)
        normals = np.concatenate((on_surface_normals, off_surface_normals), axis=0)

        return {'coords': torch.from_numpy(coords).float(),
        'sdf': torch.from_numpy(sdf).float(),
        'normals': torch.from_numpy(normals).float(),
        'instance_idx':torch.Tensor([self.instance_idx]).squeeze().long()}

class PointCloudMulti(Dataset):
    def __init__(self,root_dir,on_surface_points,max_num_instances=-1,expand=-1,max_points=-1,**kwargs):
        #This class adapted from SIREN https://vsitzmann.github.io/siren/

        super().__init__()
        self.root_dir = root_dir
        print(root_dir)
        if isinstance(root_dir,list):
            self.instance_dirs = root_dir
        else:
            self.instance_dirs = []
            for file in sorted(os.listdir(root_dir)):
                if file.endswith('mat'):
                    if os.path.isfile(os.path.join(root_dir,file).replace('surface_pts_n_normal','free_space_pts')):
                        self.instance_dirs.append(os.path.join(root_dir,file))

        assert (len(self.instance_dirs) != 0), "No objects in the data directory"

        if max_num_instances != -1:
            self.instance_dirs = self.instance_dirs[:max_num_instances]

        self.all_instances = [PointCloud_with_FreePoints(instance_idx=idx,
            pointcloud_path=dir,
            on_surface_points=on_surface_points,expand=expand,max_points=max_points) for idx, dir in enumerate(self.instance_dirs)]

        self.num_instances = len(self.all_instances)
        self.num_per_instance_observations = [len(obj) for obj in self.all_instances]

    def __len__(self):
        return np.sum(self.num_per_instance_observations)

    def get_instance_idx(self, idx):
        """Maps an index into all tuples of all objects to the idx of the tuple relative to the other tuples of that
        object
        """
        obj_idx = 0
        while idx >= 0:
            idx -= self.num_per_instance_observations[obj_idx]
            obj_idx += 1
        return obj_idx - 1, int(idx + self.num_per_instance_observations[obj_idx - 1])

    def collate_fn(self, batch_list):
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            # make them all into a new dict
            ret = {}
            for k in entry[0][0].keys():
                ret[k] = []
            # flatten the list of list
            for b in entry:
                for k in entry[0][0].keys():
                    ret[k].extend( [bi[k] for bi in b])
            for k in ret.keys():
                if type(ret[k][0]) == torch.Tensor:
                   ret[k] = torch.stack(ret[k])
            all_parsed.append(ret)

        return tuple(all_parsed)

    def __getitem__(self, idx):
        """Each __getitem__ call yields a list of self.samples_per_instance observations of a single scene (each a dict),
        as well as a list of ground-truths for each observation (also a dict)."""
        obj_idx, rel_idx = self.get_instance_idx(idx)

        observations = []
        observations.append(self.all_instances[obj_idx][rel_idx])

        ground_truth = [{'sdf':obj['sdf'] ,
        'normals': obj['normals']} for obj in observations]

        return observations, ground_truth