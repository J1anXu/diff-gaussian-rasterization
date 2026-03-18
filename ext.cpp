/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <torch/extension.h>
#include "rasterize_points.h"
#include "cpu_adam.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rasterize_gaussians", &RasterizeGaussiansCUDA);
  m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA);
  m.def("mark_visible", &markVisible);
  m.def("adamUpdate", &adamUpdate);
  m.def("fusedssim", &fusedssim);
  m.def("fusedssim_backward", &fusedssim_backward);
  m.def("quat_to_rotmat", &quat_to_rotmat);
  m.def("persp_proj", &persp_proj);
  m.def("world_to_cam", &world_to_cam);
  m.def("calculate_update_ids", &calculate_update_ids);
  m.def("update_counter", &update_counter);
  m.def("adam_deferred_update", &adam_deferred_update);
  m.def("adam_for_next_with_counter", &adam_for_next_with_counter);
  m.def("sparse_adam", &sparse_adam);
  m.def("adam_for_next", &adam_for_next);
  m.def("index_copy", &index_copy);
  m.def("packed_sparse_adam", &packed_sparse_adam);
  m.def("frustum_culling_idx",  &frustum_culling_idx);
  m.def("frustum_culling_mask", &frustum_culling_mask);
  m.def("frustum_culling_gaussian_idx", &frustum_culling_gaussian_idx);
  m.def("merge_blocks", &merge_blocks_cuda);
}