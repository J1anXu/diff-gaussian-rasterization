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

from utils.system_utils import searchForMaxIteration
from partition.partition import generate_block_masks
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

import torch

def composite_multi_block_debug(render_list, depth_list, alphaLeft_list, eps=1e-10):
    """
    Multi-block compositing with debug info, fully aligned with GPU gather logic.
    
    Args:
        render_list: list of [3,H,W] tensors, length K
        depth_list:  list of [1,H,W] tensors, length K
        alphaLeft_list: list of [1,H,W] tensors, length K

    Returns:
        final_rgb: composited RGB image [3,H,W]
        bg_rgb: background RGB image [3,H,W]
    """
    def info(name, x):
        print(f"{name}: shape={x.shape}, min={x.min().item():.6f}, max={x.max().item():.6f}, mean={x.mean().item():.6f}")

    # ------------------------
    # 0. stack tensors
    # ------------------------
    renders = torch.stack(render_list, dim=0)       # [K,3,H,W]
    alphas  = torch.stack(alphaLeft_list, dim=0)    # [K,1,H,W]
    invD    = torch.stack(depth_list, dim=0)  # [K,1,H,W]

    info("renders", renders)
    info("alphas", alphas)
    info("invD", invD)

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

    info("front_rgbs (sorted)", front_rgbs)
    info("front_alphas (sorted)", front_alphas)

    # ------------------------
    # 2. forward compositing
    # ------------------------
    cumT = torch.cumprod(front_alphas, dim=0)                  # [K,1,H,W]
    prefix_T = torch.cat([torch.ones_like(cumT[:1]), cumT[:-1]], dim=0)  # [K,1,H,W]
    final_rgb = (prefix_T * front_rgbs).sum(dim=0)             # [3,H,W]

    info("cumT", cumT)
    info("prefix_T", prefix_T)
    info("final_rgb (pre-clamp)", final_rgb)

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

    info("log_front_Ts", log_front_Ts)
    info("inv_scale", inv_scale)
    info("C_scaled", C_scaled)
    info("suffix_sum_C", suffix_sum_C)
    info("suffix_color", suffix_color)
    info("bg_rgb", bg_rgb)

    return final_rgb, bg_rgb

def render_set(model_path, name, iteration, views, gaussians_list, pipeline, background, train_test_exp, separate_sh):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    render_list = []
    depth_list = []
    alphaLeft_list = []
    debug_dir = os.path.join(render_path, "debug_blocks")
    os.makedirs(debug_dir, exist_ok=True)
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        for block_idx in range(len(gaussians_list)):
            out = render(view, gaussians_list[block_idx], pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)
            rendered = out["render"]
            depth = out["depth"]
            alphaLeft = out["alphaLeft"]
        
             # 创建调试目录
            render_list.append(rendered)
            depth_list.append(depth)
            alphaLeft_list.append(alphaLeft)
            
            # 保存为 .pt 数据文件
            torch.save(
                {
                    "render": rendered.detach().cpu(),
                    "depth": depth.detach().cpu(),
                    "alphaLeft": alphaLeft.detach().cpu(),
                    "block_idx": block_idx,
                    "frame_idx": idx,  # 假设外层循环是 idx
                },
                os.path.join(debug_dir, f"{idx:05d}_block_{block_idx}.pt")
            )

            torchvision.utils.save_image(rendered, os.path.join(render_path,'{0:05d}'.format(idx) + f'_block_{block_idx}' + ".png"))

        
        # TODO block 按照深度排序然后用alphaLeft_list来融合 注意确实是用depth图来排序的
        final_rendered, bg_rgb = composite_multi_block_debug(render_list, depth_list, alphaLeft_list)
        final_rendered = final_rendered.clamp(0, 1)
        torchvision.utils.save_image(final_rendered, os.path.join(render_path, '{0:05d}'.format(idx) + '_final.png'))
        
        
        gt = view.original_image[0:3, :, :]
        if args.train_test_exp:
            rendered = rendered[..., rendered.shape[-1] // 2:]
            gt = gt[..., gt.shape[-1] // 2:]
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))


        directly_blending = None
        for rendered, depth_image, alphaLeft in zip(render_list, depth_list, alphaLeft_list):
            if directly_blending is None:
                    directly_blending = rendered.clone()
            else:
                directly_blending += rendered
            directly_blending = directly_blending.clamp(0, 1)
        torchvision.utils.save_image(directly_blending, os.path.join(render_path,'{0:05d}'.format(idx)+'_directly_blending' + ".png"))

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