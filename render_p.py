#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
from scene_p import Scene_p

import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from scene_p.gaussian_model import GaussianModel as GaussianModel_p
import debug_config
from utils.system_utils import searchForMaxIteration
from partition.partition import generate_block_masks
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False
import debug_config
import torch


def merge(render_list, depth_list, alphaLeft_list, eps=1e-10):
    """
    Multi-block compositing for partitioned Gaussian rendering.

    Combines per-block RGB, depth, and alpha outputs using depth-aware
    alpha blending to reconstruct the final image.

    Args:
        render_list (list[Tensor]): List of [3, H, W] RGB tensors.
        depth_list (list[Tensor]): List of [1, H, W] depth tensors.
        alphaLeft_list (list[Tensor]): List of [1, H, W] alpha/transmittance tensors.

    Returns:
        final_rgb (Tensor): Composited RGB image of shape [3, H, W].
        bg_rgb (Tensor): Background RGB residual of shape [3, H, W].

    Author: Jian Xu
    Date:   2025-11-30
    Email:  jxx3451@mavs.uta.edu
    """



    # ------------------------
    # 0. stack tensors
    # ------------------------
    renders = torch.stack(render_list, dim=0)       # [K,3,H,W]
    alphas  = torch.stack(alphaLeft_list, dim=0)    # [K,1,H,W]
    invD    = torch.stack(depth_list, dim=0)  # [K,1,H,W]



    K, C, H, W = renders.shape

    # ------------------------
    # 1. sort pixels along depth
    # ------------------------
    sort_idx = torch.argsort(invD.squeeze(1), dim=0, descending=True)  # [K,H,W]
    idx_rgb = sort_idx.unsqueeze(1).expand(-1, C, -1, -1)    # [K,3,H,W]
    idx_alpha = sort_idx.unsqueeze(1).expand(-1, 1, -1, -1)  # [K,1,H,W]

    front_rgbs = torch.gather(renders, 0, idx_rgb)     # [K,3,H,W]
    # alpha 原始是 [K,1,H,W] → squeeze 变 [K,H,W]
    alphas_squeezed = alphas.squeeze(1)  # [K,H,W]
    # gather 排序
    front_alphas_squeezed = torch.gather(alphas_squeezed, 0, sort_idx)  # [K,H,W]
    # 再加回 channel 维度
    front_alphas = front_alphas_squeezed.unsqueeze(1)  # [K,1,H,W] # [K,1,H,W]



    # ------------------------
    # 2. forward compositing
    # ------------------------
    cumT = torch.cumprod(front_alphas, dim=0)                  # [K,1,H,W]
    prefix_T = torch.cat([torch.ones_like(cumT[:1]), cumT[:-1]], dim=0)  # [K,1,H,W]
    final_rgb = (prefix_T * front_rgbs).sum(dim=0)             # [3,H,W]



    # ------------------------
    # 3. background color
    # ------------------------
    log_front_Ts = torch.log(front_alphas.clamp(min=eps))      # [K,1,H,W]
    log_post_prod_inc = torch.cumsum(log_front_Ts.flip(0), dim=0).flip(0)
    log_post_prod_shift = torch.cat([log_post_prod_inc[1:], torch.zeros_like(log_post_prod_inc[:1])], dim=0)

    inv_scale = torch.exp(-log_post_prod_inc).clamp(max=1e6)
    C_scaled = front_rgbs * inv_scale
    suffix_sum_C = torch.cumsum(C_scaled.flip(0), dim=0).flip(0) - C_scaled
    scale = torch.exp(log_post_prod_shift)
    suffix_color = scale * suffix_sum_C
    bg_rgb = suffix_color[0]


    return final_rgb, bg_rgb

def render_set(model_path, name, iteration, views, gaussians_list, pipeline, background, train_test_exp, separate_sh):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    debug_path = os.path.join(model_path, name, "ours_{}".format(iteration), "blockwise_renders")
    
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(debug_path, exist_ok=True)
    render_list = []
    depth_list = []
    alphaLeft_list = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        for block_idx in range(len(gaussians_list)):
            
            curr_gaussian = gaussians_list[block_idx]
            curr_gaussian.to_gpu()
            out = render(view, curr_gaussian, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)
            curr_gaussian.to_cpu()
            
            rendered = out["render"]
            depth = out["depth"]
            alphaLeft = out["alphaLeft"]
            
            render_list.append(rendered.cpu())
            depth_list.append(depth.cpu())
            alphaLeft_list.append(alphaLeft.cpu())
            
            if debug_config.DEBUG:
                torchvision.utils.save_image(rendered, os.path.join(debug_path, 'view_{0:05d}_block_{1:03d}_render.png'.format(idx, block_idx)))
                
        blockwise_composited, bg_rgb = merge(render_list, depth_list, alphaLeft_list)
        blockwise_composited = blockwise_composited.clamp(0, 1)
        torchvision.utils.save_image(blockwise_composited, os.path.join(render_path, '{0:05d}'.format(idx) + '.png'))
        
        
        gt = view.original_image[0:3, :, :]
        # 左右拼接 gt | rendered for comparison
        if args.train_test_exp:
            blockwise_composited = blockwise_composited[..., blockwise_composited.shape[-1] // 2:]
            gt = gt[..., gt.shape[-1] // 2:]
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

        if debug_config.DEBUG:
            directly_blending = None
            for rendered, depth_image, alphaLeft in zip(render_list, depth_list, alphaLeft_list):
                if directly_blending is None:
                        directly_blending = rendered.clone()
                else:
                    directly_blending += rendered
                directly_blending = directly_blending.clamp(0, 1)
            torchvision.utils.save_image(directly_blending, os.path.join(debug_path,'{0:05d}'.format(idx)+'_directly_blending' + ".png"))
            
        render_list.clear()
        depth_list.clear()
        alphaLeft_list.clear()
        torch.cuda.empty_cache()

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, separate_sh: bool):
    with torch.no_grad():
        # 装载了视角信息的scene
        scene = Scene(dataset, GaussianModel(dataset.sh_degree), load_iteration=iteration, shuffle=False)
        
        
        gaussians_list = []
        scene_list = []
        loaded_iter = searchForMaxIteration(os.path.join(dataset.model_path, "point_cloud"))
        ply_path = os.path.join(dataset.model_path,"point_cloud", "iteration_" + str(loaded_iter), "point_cloud.ply")
        xyz, block_masks = generate_block_masks(ply_path, max_size = 500_000)
        total = len(block_masks)

        with tqdm(total=total, desc="Building blocks", ncols=100) as pbar:
            for i in range(total):
                block = block_masks[i]
                block_size = len(block)
                pbar.set_postfix({"block_id": i,"pts": f"{block_size:,}"})
                curr_gaussians = GaussianModel_p(dataset.sh_degree)
                gaussians_list.append(curr_gaussians)
                scene_list.append(Scene_p(dataset, curr_gaussians, load_iteration=iteration, shuffle=False, block_mask=block))
                pbar.update(1)
            
            
            
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians_list, pipeline, background, dataset.train_test_exp, separate_sh)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians_list, pipeline, background, dataset.train_test_exp, separate_sh)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE)