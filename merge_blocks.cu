/*
 * Fused multi-block merge kernel for partitioned 3D Gaussian Splatting.
 *
 * Replaces ~15 PyTorch kernel launches (argsort, gather, cumprod, cumsum, ...)
 * with a single CUDA kernel where each thread handles one pixel.
 *
 * Supports K <= 16 blocks (compile-time MAX_K).
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

#define MAX_K 16

/* ------------------------------------------------------------------ */
/* Sorting network for K elements (register-level, index + value)     */
/* ------------------------------------------------------------------ */

// Compare-and-swap: sort descending by depth
__device__ __forceinline__
void cas(float *d, int *idx, int i, int j) {
    if (d[i] < d[j]) {            // descending: swap if d[i] < d[j]
        float td = d[i]; d[i] = d[j]; d[j] = td;
        int   ti = idx[i]; idx[i] = idx[j]; idx[j] = ti;
    }
}

// Bitonic merge for a subsequence [lo, lo+cnt) — descending
__device__ __forceinline__
void bitonic_merge(float *d, int *idx, int lo, int cnt) {
    for (int k = cnt >> 1; k > 0; k >>= 1) {
        for (int j = 0; j < cnt; j++) {
            int ij = lo + (j ^ k);
            int ii = lo + j;
            if (ij > ii && ij < lo + cnt)
                cas(d, idx, ii, ij);
        }
    }
}

// Full bitonic sort for K elements — descending
__device__ __forceinline__
void bitonic_sort(float *d, int *idx, int K) {
    for (int size = 2; size <= K; size <<= 1) {
        for (int j = 0; j < K; j += size) {
            int cnt = min(size, K - j);
            bitonic_merge(d, idx, j, cnt);
        }
    }
}

// Optimized fixed-size sorting networks for small K
__device__ __forceinline__
void sort2(float *d, int *idx) {
    cas(d, idx, 0, 1);
}

__device__ __forceinline__
void sort3(float *d, int *idx) {
    cas(d, idx, 0, 1); cas(d, idx, 0, 2); cas(d, idx, 1, 2);
}

__device__ __forceinline__
void sort4(float *d, int *idx) {
    cas(d, idx, 0, 1); cas(d, idx, 2, 3);
    cas(d, idx, 0, 2); cas(d, idx, 1, 3);
    cas(d, idx, 1, 2);
}

// Optimal 9-comparator network for 5 elements (descending)
__device__ __forceinline__
void sort5(float *d, int *idx) {
    cas(d, idx, 0, 1); cas(d, idx, 3, 4);
    cas(d, idx, 2, 4); cas(d, idx, 2, 3);
    cas(d, idx, 0, 3); cas(d, idx, 0, 2);
    cas(d, idx, 1, 4); cas(d, idx, 1, 3);
    cas(d, idx, 1, 2);
}

// Optimal 12-comparator network for 6 elements
__device__ __forceinline__
void sort6(float *d, int *idx) {
    cas(d, idx, 0, 1); cas(d, idx, 2, 3); cas(d, idx, 4, 5);
    cas(d, idx, 0, 2); cas(d, idx, 1, 3);
    cas(d, idx, 1, 2); cas(d, idx, 0, 4); cas(d, idx, 3, 5);
    cas(d, idx, 2, 4); cas(d, idx, 1, 2); cas(d, idx, 3, 4);
    cas(d, idx, 2, 3);
}

// 16-comparator network for 7 elements
__device__ __forceinline__
void sort7(float *d, int *idx) {
    cas(d, idx, 0, 1); cas(d, idx, 2, 3); cas(d, idx, 4, 5);
    cas(d, idx, 0, 2); cas(d, idx, 1, 3); cas(d, idx, 4, 6);
    cas(d, idx, 1, 2); cas(d, idx, 5, 6);
    cas(d, idx, 0, 4); cas(d, idx, 1, 5); cas(d, idx, 2, 6);
    cas(d, idx, 1, 4); cas(d, idx, 3, 6);
    cas(d, idx, 2, 4); cas(d, idx, 3, 5);
    cas(d, idx, 3, 4);
}

// 19-comparator network for 8 elements (Batcher's odd-even merge)
__device__ __forceinline__
void sort8(float *d, int *idx) {
    cas(d, idx, 0, 1); cas(d, idx, 2, 3); cas(d, idx, 4, 5); cas(d, idx, 6, 7);
    cas(d, idx, 0, 2); cas(d, idx, 1, 3); cas(d, idx, 4, 6); cas(d, idx, 5, 7);
    cas(d, idx, 1, 2); cas(d, idx, 5, 6); cas(d, idx, 0, 4); cas(d, idx, 3, 7);
    cas(d, idx, 1, 5); cas(d, idx, 2, 6);
    cas(d, idx, 1, 4); cas(d, idx, 3, 6);
    cas(d, idx, 2, 4); cas(d, idx, 3, 5);
    cas(d, idx, 3, 4);
}

__device__ __forceinline__
void sort_dispatch(float *d, int *idx, int K) {
    switch (K) {
        case 1: break;
        case 2: sort2(d, idx); break;
        case 3: sort3(d, idx); break;
        case 4: sort4(d, idx); break;
        case 5: sort5(d, idx); break;
        case 6: sort6(d, idx); break;
        case 7: sort7(d, idx); break;
        case 8: sort8(d, idx); break;
        default: bitonic_sort(d, idx, K); break;
    }
}

/* ------------------------------------------------------------------ */
/* Main fused merge kernel                                            */
/* ------------------------------------------------------------------ */

__global__ void mergeBlocksKernel(
    const float* __restrict__ renders,    // [K, 3, H, W]
    const float* __restrict__ depths,     // [K, H, W]    (already squeezed)
    const float* __restrict__ alphas,     // [K, H, W]    (already squeezed)
    float* __restrict__ out_final_rgb,    // [3, H, W]
    float* __restrict__ out_bg_rgb,       // [3, H, W]
    float* __restrict__ out_front_rgbs,   // [K, 3, H, W]
    float* __restrict__ out_prefix_T,     // [K, H, W]
    int32_t* __restrict__ out_block_rank, // [K, H, W]
    const int K,
    const int H,
    const int W,
    const float eps)
{
    const int px = blockIdx.x * blockDim.x + threadIdx.x;
    const int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= W || py >= H) return;

    const int pid = py * W + px;
    const int HW = H * W;

    // ---- 1. Load per-block data into registers ----
    float d[MAX_K];
    float a[MAX_K];
    float r[MAX_K][3];
    int   order[MAX_K];

    for (int k = 0; k < K; k++) {
        d[k] = depths[k * HW + pid];
        a[k] = alphas[k * HW + pid];
        // renders layout: [K, 3, H, W] → renders[k * 3*HW + c*HW + pid]
        r[k][0] = renders[k * 3 * HW + 0 * HW + pid];
        r[k][1] = renders[k * 3 * HW + 1 * HW + pid];
        r[k][2] = renders[k * 3 * HW + 2 * HW + pid];
        order[k] = k;
    }

    // ---- 2. Sort descending by depth (far-to-near) ----
    sort_dispatch(d, order, K);

    // Reorder rgb and alpha by the sorted permutation
    // (need temp arrays since sort was done on d[]/order[] but r[]/a[] still in original order)
    float sorted_r[MAX_K][3];
    float sorted_a[MAX_K];
    for (int k = 0; k < K; k++) {
        int src = order[k];
        sorted_r[k][0] = r[src][0];
        sorted_r[k][1] = r[src][1];
        sorted_r[k][2] = r[src][2];
        sorted_a[k] = a[src];
    }

    // ---- 3. Prefix transmittance + compositing ----
    float T = 1.0f;
    float final_r = 0.f, final_g = 0.f, final_b = 0.f;
    float pT[MAX_K];  // prefix transmittance per sorted slot

    for (int k = 0; k < K; k++) {
        pT[k] = T;
        final_r += T * sorted_r[k][0];
        final_g += T * sorted_r[k][1];
        final_b += T * sorted_r[k][2];
        T *= sorted_a[k];
    }

    // Clamp final_rgb to [0, 1]
    final_r = fminf(1.f, fmaxf(0.f, final_r));
    final_g = fminf(1.f, fmaxf(0.f, final_g));
    final_b = fminf(1.f, fmaxf(0.f, final_b));

    // ---- 4. bg_rgb: leave-one-out for sorted index 0 ----
    //
    // Composite blocks 1..K-1 as if block 0 doesn't exist.
    // Transmittance restarts from 1.0 at block 1 (block 0's alpha excluded).
    //
    float bg_r = 0.f, bg_g = 0.f, bg_b = 0.f;
    float T_excl = 1.0f;
    for (int k = 1; k < K; k++) {
        bg_r += T_excl * sorted_r[k][0];
        bg_g += T_excl * sorted_r[k][1];
        bg_b += T_excl * sorted_r[k][2];
        T_excl *= sorted_a[k];
    }

    // ---- 5. Block rank (inverse permutation) ----
    int rank[MAX_K];
    for (int k = 0; k < K; k++) {
        rank[order[k]] = k;   // order[k] = original block id at sorted position k
    }

    // ---- 6. Write outputs ----
    // final_rgb [3, H, W]
    out_final_rgb[0 * HW + pid] = final_r;
    out_final_rgb[1 * HW + pid] = final_g;
    out_final_rgb[2 * HW + pid] = final_b;

    // bg_rgb [3, H, W]
    out_bg_rgb[0 * HW + pid] = bg_r;
    out_bg_rgb[1 * HW + pid] = bg_g;
    out_bg_rgb[2 * HW + pid] = bg_b;

    // front_rgbs [K, 3, H, W]
    for (int k = 0; k < K; k++) {
        out_front_rgbs[k * 3 * HW + 0 * HW + pid] = sorted_r[k][0];
        out_front_rgbs[k * 3 * HW + 1 * HW + pid] = sorted_r[k][1];
        out_front_rgbs[k * 3 * HW + 2 * HW + pid] = sorted_r[k][2];
    }

    // prefix_T [K, H, W]  (stored without the channel dim; Python will unsqueeze)
    for (int k = 0; k < K; k++) {
        out_prefix_T[k * HW + pid] = pT[k];
    }

    // block_rank [K, H, W]
    for (int k = 0; k < K; k++) {
        out_block_rank[k * HW + pid] = rank[k];
    }
}

/* ------------------------------------------------------------------ */
/* C++ wrapper callable from Python via pybind11                      */
/* ------------------------------------------------------------------ */

std::vector<torch::Tensor> merge_blocks_cuda(
    torch::Tensor renders,    // [K, 3, H, W]  float32 cuda
    torch::Tensor depths,     // [K, 1, H, W]  float32 cuda
    torch::Tensor alphas,     // [K, 1, H, W]  float32 cuda
    float eps)
{
    TORCH_CHECK(renders.is_cuda(), "renders must be on CUDA");
    TORCH_CHECK(depths.is_cuda(),  "depths must be on CUDA");
    TORCH_CHECK(alphas.is_cuda(),  "alphas must be on CUDA");
    TORCH_CHECK(renders.is_contiguous(), "renders must be contiguous");
    TORCH_CHECK(depths.is_contiguous(),  "depths must be contiguous");
    TORCH_CHECK(alphas.is_contiguous(),  "alphas must be contiguous");

    const int K = renders.size(0);
    const int H = renders.size(2);
    const int W = renders.size(3);
    TORCH_CHECK(K <= MAX_K, "merge_blocks supports at most ", MAX_K, " blocks, got ", K);
    TORCH_CHECK(renders.size(1) == 3, "renders must have 3 channels");

    auto opts_f = torch::TensorOptions().dtype(torch::kFloat32).device(renders.device());
    auto opts_i = torch::TensorOptions().dtype(torch::kInt32).device(renders.device());

    auto out_final_rgb  = torch::empty({3, H, W}, opts_f);
    auto out_bg_rgb     = torch::empty({3, H, W}, opts_f);
    auto out_front_rgbs = torch::empty({K, 3, H, W}, opts_f);
    auto out_prefix_T   = torch::empty({K, H, W}, opts_f);
    auto out_block_rank = torch::empty({K, H, W}, opts_i);

    // Squeeze depth and alpha from [K, 1, H, W] to [K, H, W]
    auto depths_sq = depths.squeeze(1).contiguous();
    auto alphas_sq = alphas.squeeze(1).contiguous();

    // Launch config: one thread per pixel, 16x16 thread blocks
    dim3 block(16, 16);
    dim3 grid((W + block.x - 1) / block.x,
              (H + block.y - 1) / block.y);

    mergeBlocksKernel<<<grid, block>>>(
        renders.data_ptr<float>(),
        depths_sq.data_ptr<float>(),
        alphas_sq.data_ptr<float>(),
        out_final_rgb.data_ptr<float>(),
        out_bg_rgb.data_ptr<float>(),
        out_front_rgbs.data_ptr<float>(),
        out_prefix_T.data_ptr<float>(),
        out_block_rank.data_ptr<int32_t>(),
        K, H, W, eps);

    return { out_final_rgb, out_bg_rgb, out_front_rgbs, out_prefix_T, out_block_rank };
}
