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
from gaussian_renderer import render, merge
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from scene_p.gaussian_model import GaussianModel as GaussianModel_p
import config
from utils.system_utils import searchForMaxIteration
from partition.partition import generate_block_masks
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False
DEBUG = False
import torch




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
            
            if DEBUG:
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

        if DEBUG:
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