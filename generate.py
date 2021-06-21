# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

'''Generation script for DIF-Net.
'''

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import configargparse
import re
import numpy as np

import torch
import modules, utils
from dif_net import DeformedImplicitField
import sdf_meshing


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def parse_idx_range(s):
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

p = configargparse.ArgumentParser()

p.add_argument('--logging_root', type=str, default='./recon', help='root for logging')
p.add_argument('--config', required=True, help='generation configuration')
p.add_argument('--subject_idx',type=parse_idx_range, help='index of subject to generate')
p.add_argument('--level',type=float ,default=0, help='level of iso-surface for marching cube')

# load configs
opt = p.parse_args()
with open(os.path.join(opt.test_configs),'r') as stream:
    meta_params = yaml.safe_load(stream)

# define DIF-Net
model = DeformedImplicitField(**meta_params)
model.load_state_dict(torch.load(meta_params['checkpoint_path']))
model.cuda()

# create save path
root_path = os.path.join(opt.logging_root, meta_params['experiment_name'])
utils.cond_mkdir(root_path)

# generate meshes with color-coded coordinates
for idx in opt.subject_idx:
    print('generate_instance:', idx)
    sdf_meshing.create_mesh(model, os.path.join(root_path,'test%04d'%idx), subject_idx=idx, N=128,level=opt.level)