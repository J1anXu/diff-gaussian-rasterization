#include <torch/all.h>
#include <torch/csrc/autograd/function.h>
#include <torch/serialize/tensor.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>
#include <tuple>
#include "cpu_adam.h"
#include <cmath>
#include <ATen/record_function.h>
#include <omp.h>
#include <cstdlib>
#include <Python.h>
#include <pybind11/pybind11.h>
#include <xmmintrin.h>
#include <pmmintrin.h>


at::Tensor quat_to_rotmat(at::Tensor& quats) {
    int N = quats.size(0);
    at::Tensor rotmats = torch::empty({N, 3, 3}, quats.options());

    auto quats_a = quats.accessor<float, 2>();
    auto rotmats_a = rotmats.accessor<float, 3>();

    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

    #pragma omp parallel for
    for (int n = 0; n < N; ++n) {
        _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
        _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

        float w = quats_a[n][0];
        float x = quats_a[n][1];
        float y = quats_a[n][2];
        float z = quats_a[n][3];

        // Normalize quaternion
        float norm = std::sqrt(w*w + x*x + y*y + z*z);
        w /= norm;
        x /= norm;
        y /= norm;
        z /= norm;

        // Compute rotation matrix
        rotmats_a[n][0][0] = 1.f - 2.f * (y * y + z * z);
        rotmats_a[n][0][1] = 2.f * (x * y - w * z);
        rotmats_a[n][0][2] = 2.f * (x * z + w * y);

        rotmats_a[n][1][0] = 2.f * (x * y + w * z);
        rotmats_a[n][1][1] = 1.f - 2.f * (x * x + z * z);
        rotmats_a[n][1][2] = 2.f * (y * z - w * x);

        rotmats_a[n][2][0] = 2.f * (x * z - w * y);
        rotmats_a[n][2][1] = 2.f * (y * z + w * x);
        rotmats_a[n][2][2] = 1.f - 2.f * (x * x + y * y);
    }

    return rotmats;
}

std::tuple<at::Tensor, at::Tensor> persp_proj(
    at::Tensor& means,       // [C, N, 3]
    at::Tensor& covars,      // [C, N, 3, 3]
    at::Tensor& Ks,          // [C, 3, 3]
    int width,
    int height
) {
    int C = means.size(0);
    int N = means.size(1);

    at::Tensor means2d = torch::empty({C, N, 2}, means.options());
    at::Tensor covars2d = torch::empty({C, N, 2, 2}, means.options());
    
    auto means_a = means.accessor<float,3>();
    auto covars_a = covars.accessor<float,4>();
    auto Ks_a = Ks.accessor<float,3>();
    auto means2d_a = means2d.accessor<float,3>();
    auto covars2d_a = covars2d.accessor<float,4>();

    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

    #pragma omp parallel for collapse(2)
    for (int c = 0; c < C; ++c) {
        for (int n = 0; n < N; ++n) {
            _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
            _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

            float tx = means_a[c][n][0];
            float ty = means_a[c][n][1];
            float tz = means_a[c][n][2];
            float tz2 = tz * tz;

            float fx = Ks_a[c][0][0];
            float fy = Ks_a[c][1][1];
            float cx = Ks_a[c][0][2];
            float cy = Ks_a[c][1][2];

            float tan_fovx = 0.5f * width / fx;
            float tan_fovy = 0.5f * height / fy;

            float lim_x_pos = (width - cx) / fx + 0.3f * tan_fovx;
            float lim_x_neg = cx / fx + 0.3f * tan_fovx;
            float lim_y_pos = (height - cy) / fy + 0.3f * tan_fovy;
            float lim_y_neg = cy / fy + 0.3f * tan_fovy;

            float clamped_tx = tz * std::min(std::max(tx / tz, -lim_x_neg), lim_x_pos);
            float clamped_ty = tz * std::min(std::max(ty / tz, -lim_y_neg), lim_y_pos);

            float J[2][3] = {
                {fx / tz, 0.f, -fx * clamped_tx / tz2},
                {0.f, fy / tz, -fy * clamped_ty / tz2}
            };

            // Projected mean
            float proj_x = (Ks_a[c][0][0] * tx + Ks_a[c][0][1] * ty + Ks_a[c][0][2] * tz) / tz;
            float proj_y = (Ks_a[c][1][0] * tx + Ks_a[c][1][1] * ty + Ks_a[c][1][2] * tz) / tz;
            means2d_a[c][n][0] = proj_x;
            means2d_a[c][n][1] = proj_y;

            // Projected covariance: cov2d = J * cov * J^T
            float tmp[2][3] = {0};
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 3; ++j)
                    for (int k = 0; k < 3; ++k)
                        tmp[i][j] += J[i][k] * covars_a[c][n][k][j];

            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 2; ++j) {
                    float val = 0.f;
                    for (int k = 0; k < 3; ++k)
                        val += tmp[i][k] * J[j][k];
                    covars2d_a[c][n][i][j] = val;
                }
        }
    }

    return std::make_tuple(means2d, covars2d);
}



std::tuple<at::Tensor, at::Tensor> world_to_cam(
    at::Tensor& means,
    at::Tensor& covars,
    at::Tensor& viewmats
) {
    int N = means.size(0);
    int C = viewmats.size(0);

    at::Tensor means_c = torch::empty({C, N, 3}, means.options());
    at::Tensor covars_c = torch::empty({C, N, 3, 3}, means.options());

    auto means_a = means.accessor<float,2>();
    auto covars_a = covars.accessor<float,3>();
    auto viewmats_a = viewmats.accessor<float,3>();
    auto means_c_a = means_c.accessor<float,3>();
    auto covars_c_a = covars_c.accessor<float,4>();

    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

    #pragma omp parallel for collapse(2)
    for (int c = 0; c < C; ++c) {
        for (int n = 0; n < N; ++n) {
            _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
            _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

            float R[3][3];
            float t[3];
            for (int i = 0; i < 3; ++i) {
                t[i] = viewmats_a[c][i][3];
                for (int j = 0; j < 3; ++j) {
                    R[i][j] = viewmats_a[c][i][j];
                }
            }

            // Transform mean
            for (int i = 0; i < 3; ++i) {
                means_c_a[c][n][i] = R[i][0] * means_a[n][0] +
                                     R[i][1] * means_a[n][1] +
                                     R[i][2] * means_a[n][2] + t[i];
            }

            // Transform covariance: R * cov * R^T
            float tmp[3][3] = {0};
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    for (int k = 0; k < 3; ++k)
                        tmp[i][j] += R[i][k] * covars_a[n][k][j];

            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j) {
                    float val = 0.f;
                    for (int k = 0; k < 3; ++k)
                        val += tmp[i][k] * R[j][k];
                    covars_c_a[c][n][i][j] = val;
                }
        }
    }

    return std::make_tuple(means_c, covars_c);
}


at::Tensor calculate_update_ids(at::Tensor valid_ids, at::Tensor counter, int total_num) {

    RECORD_FUNCTION("calculate_update_ids", {valid_ids, counter});
 
    int nvalid = valid_ids.size(0);
    int *valid_ids_ptr = valid_ids.data_ptr<int>();
    int8_t *counter_ptr = counter.data_ptr<int8_t>();
    
    char *bitmap = (char*)malloc(total_num);

    omp_set_num_threads(64);
    #pragma omp parallel for
    for (int idx = 0; idx < total_num; ++idx) {
        int val = counter_ptr[idx];
        if (val == 15)
            bitmap[idx] = 1;
        else
            bitmap[idx] = 0;
    }

    #pragma omp parallel for
    for (int idx = 0; idx < nvalid; ++idx) {
        int val = valid_ids_ptr[idx];
        bitmap[val] = 1;
    }

    int nupdate = 0;
    #pragma omp parallel for reduction(+:nupdate)
    for (int idx = 0; idx < total_num; ++idx) {
        if (bitmap[idx] == 1)
            nupdate += 1;
    }

    at::Tensor update_ids_tensor = at::empty({nupdate}, valid_ids.options().dtype(torch::kInt));
    int *update_ids = update_ids_tensor.data_ptr<int>();

    int num_threads = omp_get_max_threads();
    std::vector<std::vector<int>> update_locals(num_threads);

    // Local collection
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        auto& vbuf = update_locals[tid];
        vbuf.reserve(total_num / 5);

        #pragma omp for
        for (int64_t i = 0; i < total_num; ++i) {
            if (bitmap[i] == 1)
                vbuf.push_back(i);
        }
    }

    std::vector<int64_t> v_offsets(num_threads + 1, 0);
    for (int t = 0; t < num_threads; ++t) {
        v_offsets[t + 1] = v_offsets[t] + update_locals[t].size();
    }

    #pragma omp parallel for
    for (int t = 0; t < num_threads; ++t) {
        std::copy(update_locals[t].begin(), update_locals[t].end(), update_ids + v_offsets[t]);
    }

    free(bitmap);

    return update_ids_tensor;
}


void update_counter(at::Tensor counter, at::Tensor update_ids) {

    RECORD_FUNCTION("update_counter", {counter, update_ids});

    int total_num = counter.size(0);
    int nupdate = update_ids.size(0);
    int *update_ids_ptr = update_ids.data_ptr<int>();
    int8_t *counter_ptr = counter.data_ptr<int8_t>();

    omp_set_num_threads(64);
    #pragma omp parallel for
    for (int idx = 0; idx < total_num; ++idx) {
        counter_ptr[idx] += 1;
    }

    #pragma omp parallel for
    for (int idx = 0; idx < nupdate; ++idx) {
        int i = update_ids_ptr[idx];
        counter_ptr[i] = 0;
    }
}


void adam_deferred_update(at::Tensor weight, at::Tensor grad, at::Tensor exp_avg, at::Tensor exp_avg_sq, at::Tensor valid_ids, at::Tensor counter,
    int step, float lr, float beta1, float beta2, float eps) {

    RECORD_FUNCTION("adam_deferred_update", {weight, grad, exp_avg, exp_avg_sq, valid_ids, counter});

    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

    int nvalid = valid_ids.size(0);
    int row_size = 1;
    for (int i = 1; i < weight.dim(); ++i) {
        row_size *= weight.size(i);
    }

    const float bias_correction1 = 1 - std::pow(beta1, step);
    const float bias_correction2 = 1 - std::pow(beta2, step);
    const float step_size = lr / bias_correction1;
    const float bias_correction2_sqrt = std::sqrt(bias_correction2);

    int *valid_ids_ptr = valid_ids.data_ptr<int>();
    float *w_ptr_base = weight.data_ptr<float>();
    float *g_ptr_base = grad.data_ptr<float>();
    float *m_ptr_base = exp_avg.data_ptr<float>();
    float *v_ptr_base = exp_avg_sq.data_ptr<float>();
    int8_t *counter_ptr = counter.data_ptr<int8_t>();

    omp_set_num_threads(64);

    // Precomputed corrections and power values.
    float scale = beta1 / std::sqrt(beta2);
    float correction[16];
    float beta1_pow[16];
    float beta2_pow[16];
    beta1_pow[0] = beta1;
    beta1_pow[1] = beta1 * beta1;
    beta2_pow[0] = beta2;
    beta2_pow[1] = beta2 * beta2;
    correction[0] = 0;
    correction[1] = (lr * beta1) / ((std::sqrt(beta2 / (1 - std::pow(beta2, step - 1)))) * (1 - std::pow(beta1, step - 1)));
    for (int i = 2; i < 16; i++) {
        correction[i] = scale * correction[i-1] + (lr * beta1) / ((std::sqrt(beta2 / (1 - std::pow(beta2, step - i)))) * (1 - std::pow(beta1, step - i)));
        beta1_pow[i] = beta1 * beta1_pow[i-1];
        beta2_pow[i] = beta2 * beta2_pow[i-1];
    }

    #pragma omp parallel for
    for (int64_t idx = 0; idx < nvalid; ++idx) {
        _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
        _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

        int64_t i = valid_ids_ptr[idx];
        int counter_val = counter_ptr[i];

        float* w_ptr = w_ptr_base + i * row_size;
        float* g_ptr = g_ptr_base + i * row_size;
        float* m_ptr = m_ptr_base + i * row_size;
        float* v_ptr = v_ptr_base + i * row_size;

        float correction_val = correction[counter_val];
        float beta1_pow_val = beta1_pow[counter_val];
        float beta2_pow_val = beta2_pow[counter_val];

        for (int64_t d = 0; d < row_size; ++d) {
            // Restore weight and optimizer states
            float w = w_ptr[d];
            float g = g_ptr[d];
            float m_prev = m_ptr[d];
            float v_prev = v_ptr[d];
            float m = beta1_pow_val * m_prev + (1 - beta1) * g;
            float v = beta2_pow_val * v_prev + (1 - beta2) * g * g;
            m_ptr[d] = m;
            v_ptr[d] = v;
            w -= (correction_val * m_prev) / (std::sqrt(v_prev) + eps);

            // Update weight and optimizer states
            float denom = std::sqrt(v) / bias_correction2_sqrt + eps;
            w_ptr[d] = w - step_size * m / denom;
        }
    }
}


void adam_for_next_with_counter(at::Tensor weight, at::Tensor grad, at::Tensor exp_avg, at::Tensor exp_avg_sq, at::Tensor valid_ids, at::Tensor weight_new, at::Tensor counter,
    int step, float lr, float beta1, float beta2, float eps) {

    RECORD_FUNCTION("adam_for_next_with_counter", {weight, grad, exp_avg, exp_avg_sq, valid_ids, weight_new});

    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

    int nvalid = valid_ids.size(0);
    int row_size = 1;
    for (int i = 1; i < weight.dim(); ++i) {
        row_size *= weight.size(i);
    }

    step += 1;
    const float bias_correction1 = 1 - std::pow(beta1, step);
    const float bias_correction2 = 1 - std::pow(beta2, step);
    const float step_size = lr / bias_correction1;
    const float bias_correction2_sqrt = std::sqrt(bias_correction2);

    long int *valid_ids_ptr = valid_ids.data_ptr<int64_t>();
    float *w_ptr_base = weight.data_ptr<float>();
    float *g_ptr_base = grad.data_ptr<float>();
    float *m_ptr_base = exp_avg.data_ptr<float>();
    float *v_ptr_base = exp_avg_sq.data_ptr<float>();
    float *w_new_ptr_base = weight_new.data_ptr<float>();
    int8_t *counter_ptr = counter.data_ptr<int8_t>();

    // Precomputed corrections and power values.
    float scale = beta1 / std::sqrt(beta2);
    float correction[16];
    float beta1_pow[16];
    float beta2_pow[16];
    beta1_pow[0] = beta1;
    beta1_pow[1] = beta1 * beta1;
    beta2_pow[0] = beta2;
    beta2_pow[1] = beta2 * beta2;
    correction[0] = 0;
    correction[1] = (lr * beta1) / ((std::sqrt(beta2 / (1 - std::pow(beta2, step - 1)))) * (1 - std::pow(beta1, step - 1)));
    for (int i = 2; i < 16; i++) {
        correction[i] = scale * correction[i-1] + (lr * beta1) / ((std::sqrt(beta2 / (1 - std::pow(beta2, step - i)))) * (1 - std::pow(beta1, step - i)));
        beta1_pow[i] = beta1 * beta1_pow[i-1];
        beta2_pow[i] = beta2 * beta2_pow[i-1];
    }

    omp_set_num_threads(64);
    #pragma omp parallel for
    for (int64_t idx = 0; idx < nvalid; ++idx) {
        _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
        _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

        int64_t i = valid_ids_ptr[idx];
        int counter_val = counter_ptr[i];

        float* w_ptr = w_ptr_base + i * row_size;
        float* w_new_ptr = w_new_ptr_base + idx * row_size;
        float* g_ptr = g_ptr_base + i * row_size;
        float* m_ptr = m_ptr_base + i * row_size;
        float* v_ptr = v_ptr_base + i * row_size;
        float correction_val = correction[counter_val];
        float beta1_pow_val = beta1_pow[counter_val];
        float beta2_pow_val = beta2_pow[counter_val];

        for (int64_t d = 0; d < row_size; ++d) {
            // Restore weight and optimizer states
            float w = w_ptr[d];
            float g = g_ptr[d];
            float m_prev = m_ptr[d];
            float v_prev = v_ptr[d];
            float m = beta1_pow_val * m_prev + (1 - beta1) * g;
            float v = beta2_pow_val * v_prev + (1 - beta2) * g * g;
            w -= (correction_val * m_prev) / (std::sqrt(v_prev) + eps);

            // Update weight and optimizer states
            float denom = std::sqrt(v) / bias_correction2_sqrt + eps;
            w_new_ptr[d] = w - step_size * m / denom;
        }
    }
}


void sparse_adam(at::Tensor weight, at::Tensor grad, at::Tensor exp_avg, at::Tensor exp_avg_sq, at::Tensor valid_ids,
    int step, float lr, float beta1, float beta2, float eps) {

    RECORD_FUNCTION("sparse_adam", {weight, grad, exp_avg, exp_avg_sq, valid_ids});

    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

    int nvalid = valid_ids.size(0);
    int row_size = 1;
    for (int i = 1; i < weight.dim(); ++i) {
        row_size *= weight.size(i);
    }

    const float bias_correction1 = 1 - std::pow(beta1, step);
    const float bias_correction2 = 1 - std::pow(beta2, step);
    const float step_size = lr / bias_correction1;
    const float bias_correction2_sqrt = std::sqrt(bias_correction2);

    long int *valid_ids_ptr = valid_ids.data_ptr<int64_t>();
    float *w_ptr_base = weight.data_ptr<float>();
    float *g_ptr_base = grad.data_ptr<float>();
    float *m_ptr_base = exp_avg.data_ptr<float>();
    float *v_ptr_base = exp_avg_sq.data_ptr<float>();

    omp_set_num_threads(64);
    #pragma omp parallel for
    for (int64_t idx = 0; idx < nvalid; ++idx) {
        _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
        _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

        int64_t i = valid_ids_ptr[idx];

        float* w_ptr = w_ptr_base + i * row_size;
        float* g_ptr = g_ptr_base + i * row_size;
        float* m_ptr = m_ptr_base + i * row_size;
        float* v_ptr = v_ptr_base + i * row_size;

        for (int64_t d = 0; d < row_size; ++d) {
            float g = g_ptr[d];
            float m = beta1 * m_ptr[d] + (1 - beta1) * g;
            float v = beta2 * v_ptr[d] + (1 - beta2) * g * g;
            m_ptr[d] = m;
            v_ptr[d] = v;

            float denom = std::sqrt(v) / bias_correction2_sqrt + eps;
            w_ptr[d] = w_ptr[d] - step_size * m / denom;
        }
    }
}


void adam_for_next(at::Tensor weight, at::Tensor grad, at::Tensor exp_avg, at::Tensor exp_avg_sq, at::Tensor valid_ids, at::Tensor weight_new,
    int step, float lr, float beta1, float beta2, float eps) {

    RECORD_FUNCTION("adam_for_next", {weight, grad, exp_avg, exp_avg_sq, valid_ids, weight_new});

    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

    int nvalid = valid_ids.size(0);
    int row_size = 1;
    for (int i = 1; i < weight.dim(); ++i) {
        row_size *= weight.size(i);
    }

    step += 1;
    const float bias_correction1 = 1 - std::pow(beta1, step);
    const float bias_correction2 = 1 - std::pow(beta2, step);
    const float step_size = lr / bias_correction1;
    const float bias_correction2_sqrt = std::sqrt(bias_correction2);

    long int *valid_ids_ptr = valid_ids.data_ptr<int64_t>();
    float *w_ptr_base = weight.data_ptr<float>();
    float *w_new_ptr_base = weight_new.data_ptr<float>();
    float *g_ptr_base = grad.data_ptr<float>();
    float *m_ptr_base = exp_avg.data_ptr<float>();
    float *v_ptr_base = exp_avg_sq.data_ptr<float>();

    omp_set_num_threads(64);
    #pragma omp parallel for
    for (int64_t idx = 0; idx < nvalid; ++idx) {
        _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
        _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

        int64_t i = valid_ids_ptr[idx];

        float* w_ptr = w_ptr_base + i * row_size;
        float* w_new_ptr = w_new_ptr_base + idx * row_size;
        float* g_ptr = g_ptr_base + i * row_size;
        float* m_ptr = m_ptr_base + i * row_size;
        float* v_ptr = v_ptr_base + i * row_size;

        for (int64_t d = 0; d < row_size; ++d) {
            float g = g_ptr[d];
            float m = beta1 * m_ptr[d] + (1 - beta1) * g;
            float v = beta2 * v_ptr[d] + (1 - beta2) * g * g;

            float denom = std::sqrt(v) / bias_correction2_sqrt + eps;
            w_new_ptr[d] = w_ptr[d] - step_size * m / denom;
        }
    }
}


void packed_sparse_adam(
    at::Tensor packed,
    at::Tensor grad_subset,
    at::Tensor exp_avg,
    at::Tensor exp_avg_sq,
    at::Tensor valid_ids,
    at::Tensor lr_per_col,
    int step,
    float beta1,
    float beta2,
    float eps)
{
    RECORD_FUNCTION("packed_sparse_adam", {packed, grad_subset, exp_avg, exp_avg_sq, valid_ids});

    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

    int64_t n_vis = valid_ids.size(0);
    int64_t D = packed.size(1);

    const float bias_correction1 = 1.0f - std::pow(beta1, step);
    const float bias_correction2 = 1.0f - std::pow(beta2, step);
    const float bias_correction2_sqrt = std::sqrt(bias_correction2);

    long int *valid_ids_ptr = valid_ids.data_ptr<int64_t>();
    float *packed_ptr = packed.data_ptr<float>();
    float *grad_ptr = grad_subset.data_ptr<float>();
    float *m_ptr = exp_avg.data_ptr<float>();
    float *v_ptr = exp_avg_sq.data_ptr<float>();
    float *lr_ptr = lr_per_col.data_ptr<float>();

    omp_set_num_threads(64);
    #pragma omp parallel for
    for (int64_t vi = 0; vi < n_vis; ++vi) {
        _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
        _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

        int64_t row = valid_ids_ptr[vi];
        float* p = packed_ptr + row * D;
        float* g = grad_ptr + vi * D;
        float* m = m_ptr + row * D;
        float* vv = v_ptr + row * D;

        for (int64_t d = 0; d < D; ++d) {
            float grad = g[d];
            m[d] = beta1 * m[d] + (1.0f - beta1) * grad;
            vv[d] = beta2 * vv[d] + (1.0f - beta2) * grad * grad;
            float denom = std::sqrt(vv[d]) / bias_correction2_sqrt + eps;
            p[d] -= (lr_ptr[d] / bias_correction1) * m[d] / denom;
        }
    }
}


void index_copy(at::Tensor src, at::Tensor indices, at::Tensor dest) {

    RECORD_FUNCTION("index_copy", {src, indices, dest});
    int nvalid = indices.size(0);
    int row_size = 1;
    for (int i = 1; i < src.dim(); ++i) {
        row_size *= src.size(i);
    }

    TORCH_CHECK(dest.size(0) == nvalid, "size(0) of indices and dest must be the same")
    TORCH_CHECK(src.is_contiguous(), "src is not contiguous")
    TORCH_CHECK(dest.is_contiguous(), "dest is not contiguous")
    TORCH_CHECK(indices.is_contiguous(), "indices is not contiguous")

    float *src_base_ptr = src.data_ptr<float>();
    float *dest_base_ptr = dest.data_ptr<float>();
    long int *indices_ptr = indices.data_ptr<int64_t>();

    #pragma omp parallel for
    for (int64_t idx = 0; idx < nvalid; ++idx) {
        int64_t i = indices_ptr[idx];
        float* src_ptr = src_base_ptr + i * row_size;
        float* dest_ptr = dest_base_ptr + idx * row_size;

        for (int64_t d = 0; d < row_size; ++d) {
            dest_ptr[d] = src_ptr[d];
        }
    }
}



// ---------------------------------------------------------------------------
// Frustum culling  (C + OpenMP + AVX-512 via -march=native)
// ---------------------------------------------------------------------------

at::Tensor frustum_culling_idx(at::Tensor xyz, at::Tensor M, float inflate_ratio)
{
    TORCH_CHECK(xyz.is_contiguous(),  "xyz must be contiguous");
    TORCH_CHECK(M.is_contiguous(),    "M must be contiguous");
    TORCH_CHECK(xyz.dim()==2 && xyz.size(1)==3, "xyz must be [N,3]");
    TORCH_CHECK(M.dim()==2 && M.size(0)==4 && M.size(1)==4, "M must be [4,4]");
    TORCH_CHECK(!xyz.is_cuda() && !M.is_cuda(), "inputs must be on CPU");

    const int64_t N = xyz.size(0);
    const float* __restrict__ p = xyz.data_ptr<float>();
    const float* __restrict__ m = M.data_ptr<float>();

    const float m00=m[0], m01=m[1], m02=m[2],  m03=m[3];
    const float m10=m[4], m11=m[5], m12=m[6],  m13=m[7];
    const float m20=m[8], m21=m[9], m22=m[10], m23=m[11];
    const float b0=m[12], b1=m[13], b2=m[14],  b3=m[15];

    // Pass 1: write bool mask — sequential store, compiler can auto-vectorize
    // with AVX-512 (-march=native). No branch in write path.
    std::vector<uint8_t> mask(N);
    uint8_t* __restrict__ mk = mask.data();

    #pragma omp parallel for schedule(static) num_threads(16)
    for (int64_t i = 0; i < N; i++) {
        float x0 = p[i*3+0], x1 = p[i*3+1], x2 = p[i*3+2];
        float cx = x0*m00 + x1*m10 + x2*m20 + b0;
        float cy = x0*m01 + x1*m11 + x2*m21 + b1;
        float cz = x0*m02 + x1*m12 + x2*m22 + b2;
        float cw = x0*m03 + x1*m13 + x2*m23 + b3;
        float inf = inflate_ratio * cw;
        float wpi = cw + inf, wmi = -cw - inf;
        mk[i] = (uint8_t)(
            (cw > 0.f) & (cx >= wmi) & (cx <= wpi) &
            (cy >= wmi) & (cy <= wpi) &
            (cz >= 0.f) & (cz <= wpi));
    }

    // Pass 2: prefix-sum per thread chunk, then compact indices
    const int NT = 16;
    int64_t chunk = (N + NT - 1) / NT;
    std::vector<int64_t> thread_counts(NT, 0);

    #pragma omp parallel num_threads(NT)
    {
        int tid = omp_get_thread_num();
        int64_t lo = tid * chunk;
        int64_t hi = std::min(lo + chunk, N);
        int64_t cnt = 0;
        for (int64_t i = lo; i < hi; i++) cnt += mk[i];
        thread_counts[tid] = cnt;
    }

    // Compute per-thread output offsets
    std::vector<int64_t> offsets(NT + 1, 0);
    for (int t = 0; t < NT; t++) offsets[t+1] = offsets[t] + thread_counts[t];
    int64_t total = offsets[NT];

    auto out = torch::empty({total},
        torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
    int64_t* __restrict__ op = out.data_ptr<int64_t>();

    #pragma omp parallel num_threads(NT)
    {
        int tid = omp_get_thread_num();
        int64_t lo = tid * chunk;
        int64_t hi = std::min(lo + chunk, N);
        int64_t pos = offsets[tid];
        for (int64_t i = lo; i < hi; i++) {
            if (mk[i]) op[pos++] = i;
        }
    }

    return out;
}

at::Tensor frustum_culling_mask(at::Tensor xyz, at::Tensor M, float inflate_ratio)
{
    TORCH_CHECK(xyz.is_contiguous(),  "xyz must be contiguous");
    TORCH_CHECK(M.is_contiguous(),    "M must be contiguous");
    TORCH_CHECK(!xyz.is_cuda() && !M.is_cuda(), "inputs must be on CPU");

    const int64_t N = xyz.size(0);
    const float* __restrict__ p = xyz.data_ptr<float>();
    const float* __restrict__ m = M.data_ptr<float>();

    const float m00=m[0], m01=m[1], m02=m[2],  m03=m[3];
    const float m10=m[4], m11=m[5], m12=m[6],  m13=m[7];
    const float m20=m[8], m21=m[9], m22=m[10], m23=m[11];
    const float b0=m[12], b1=m[13], b2=m[14],  b3=m[15];

    auto mask_out = torch::empty({N},
        torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    bool* __restrict__ out = mask_out.data_ptr<bool>();

    #pragma omp parallel for schedule(static) num_threads(16)
    for (int64_t i = 0; i < N; i++) {
        float x0 = p[i*3+0], x1 = p[i*3+1], x2 = p[i*3+2];
        float cx = x0*m00 + x1*m10 + x2*m20 + b0;
        float cy = x0*m01 + x1*m11 + x2*m21 + b1;
        float cz = x0*m02 + x1*m12 + x2*m22 + b2;
        float cw = x0*m03 + x1*m13 + x2*m23 + b3;
        float inf = inflate_ratio * cw;
        float wpi = cw + inf, wmi = -cw - inf;
        out[i] = (cw > 0.f) & (cx >= wmi) & (cx <= wpi) &
                 (cy >= wmi) & (cy <= wpi) &
                 (cz >= 0.f) & (cz <= wpi);
    }
    return mask_out;
}
