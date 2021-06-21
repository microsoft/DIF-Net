# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

'''Training losses for DIF-Net.
'''

import torch
import torch.nn.functional as F

def deform_implicit_loss(model_output, gt,loss_grad_deform=5,loss_grad_temp=1e2,loss_correct=1e2):

    gt_sdf = gt['sdf']
    gt_normals = gt['normals']

    coords = model_output['model_in']
    pred_sdf = model_output['model_out']

    embeddings = model_output['latent_vec']

    gradient_sdf = model_output['grad_sdf']
    gradient_deform = model_output['grad_deform']
    gradient_temp = model_output['grad_temp']
    sdf_correct = model_output['sdf_correct']

    # sdf regression loss from Sitzmannn et al. 2020
    sdf_constraint = torch.where(gt_sdf != -1, torch.clamp(pred_sdf,-0.5,0.5)-torch.clamp(gt_sdf,-0.5,0.5), torch.zeros_like(pred_sdf))
    inter_constraint = torch.where(gt_sdf != -1, torch.zeros_like(pred_sdf), torch.exp(-1e2 * torch.abs(pred_sdf)))
    normal_constraint = torch.where(gt_sdf == 0, 1 - F.cosine_similarity(gradient_sdf, gt_normals, dim=-1)[..., None],
                                    torch.zeros_like(gradient_sdf[..., :1]))
    grad_constraint = torch.abs(gradient_sdf.norm(dim=-1) - 1)

    # deformation smoothness prior
    grad_deform_constraint = gradient_deform.norm(dim=-1)

    # normal consistency prior
    grad_temp_constraint = torch.where(gt_sdf == 0, 1 - F.cosine_similarity(gradient_temp, gt_normals, dim=-1)[..., None],
                                    torch.zeros_like(gradient_temp[..., :1]))

    # minimal correction prior
    sdf_correct_constraint = torch.abs(sdf_correct)

    # latent code prior
    embeddings_constraint = torch.mean(embeddings ** 2)

    # -----------------
    return {'sdf': torch.abs(sdf_constraint).mean() * 3e3, 
            'inter': inter_constraint.mean() * 5e2,
            'normal_constraint': normal_constraint.mean() * 1e2,
            'grad_constraint': grad_constraint.mean() * 5e1,
            'embeddings_constraint': embeddings_constraint.mean() * 1e6,
            'grad_temp_constraint': grad_temp_constraint.mean() * loss_grad_temp,
            'grad_deform_constraint':grad_deform_constraint.mean()* loss_grad_deform,
            'sdf_correct_constraint':sdf_correct_constraint.mean()* loss_correct}

def embedding_loss(model_output, gt):

    gt_sdf = gt['sdf']
    gt_normals = gt['normals']

    coords = model_output['model_in']
    pred_sdf = model_output['model_out']

    embeddings = model_output['latent_vec']
    gradient_sdf = model_output['grad_sdf']

    # sdf regression loss from Sitzmannn et al. 2020
    sdf_constraint = torch.where(gt_sdf != -1, torch.clamp(pred_sdf,-0.5,0.5)-torch.clamp(gt_sdf,-0.5,0.5), torch.zeros_like(pred_sdf))
    inter_constraint = torch.where(gt_sdf != -1, torch.zeros_like(pred_sdf), torch.exp(-1e2 * torch.abs(pred_sdf)))
    normal_constraint = torch.where(gt_sdf == 0, 1 - F.cosine_similarity(gradient_sdf, gt_normals, dim=-1)[..., None],
                                    torch.zeros_like(gradient_sdf[..., :1]))
    grad_constraint = torch.abs(gradient_sdf.norm(dim=-1) - 1)

    embeddings_constraint = torch.mean(embeddings ** 2)

    # -----------------
    return {'sdf': torch.abs(sdf_constraint).mean() * 3e3,
            'inter': inter_constraint.mean() * 5e2,
            'normal_constraint': normal_constraint.mean() * 1e2,
            'grad_constraint': grad_constraint.mean() * 5e1, 
            'embeddings_constraint': embeddings_constraint.mean() * 1e6}