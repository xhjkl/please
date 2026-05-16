#include <metal_stdlib>
using namespace metal;

constant uint GPTOSS_HIDDEN_SIZE = 2880u;
constant uint GPTOSS_GATE_UP_VALUES = 5760u;
constant uint GPTOSS_EXPERTS = 32u;
constant uint GPTOSS_MXFP4_GROUPS = 90u;
constant uint GPTOSS_MXFP4_BYTES_PER_GROUP = 16u;
constant float GPTOSS_FP4_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

inline float bf16_to_float(ushort value) {
    uint bits = ((uint)value) << 16;
    return as_type<float>(bits);
}

inline float fp4_to_float(uint value) {
    return GPTOSS_FP4_LUT[value & 15u];
}

inline float mxfp4_scale(uchar exponent) {
    uint exponent_bits = uint(exponent);
    if (exponent_bits == 0u) {
        return as_type<float>(0x00400000u);
    }
    return as_type<float>(exponent_bits << 23);
}

inline float rope_concentration() {
    return 0.1f * log(32.0f) + 1.0f;
}

inline float rope_inv_freq(uint dim) {
    float d_half = 32.0f;
    float two_pi = 6.2831853071795864769f;
    float low = d_half * log(4096.0f / (32.0f * two_pi)) / log(150000.0f);
    float high = d_half * log(4096.0f / (1.0f * two_pi)) / log(150000.0f);
    float freq = pow(150000.0f, float(dim * 2u) / 64.0f);
    float interpolation = 1.0f / (32.0f * freq);
    float extrapolation = 1.0f / freq;
    float ramp = (float(dim) - low) / (high - low);
    float mask = 1.0f - clamp(ramp, 0.0f, 1.0f);
    return interpolation * (1.0f - mask) + extrapolation * mask;
}

kernel void partial_sum_squares(
    device const float* x [[buffer(0)]],
    device float* partial [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]]
) {
    threadgroup float scratch[256];
    float value = 0.0f;
    if (gid < n) {
        value = x[gid] * x[gid];
    }
    scratch[tid] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        partial[group_id] = scratch[0];
    }
}

kernel void apply_rms_norm(
    device const float* x [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant float& scale [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < n) {
        out[gid] = x[gid] * scale * weight[gid];
    }
}

kernel void apply_rms_norm_from_partials(
    device const float* x [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* partial [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& groups [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) {
        return;
    }

    float sum_squares = 0.0f;
    for (uint group = 0u; group < groups; group++) {
        sum_squares += partial[group];
    }
    float mean_square = sum_squares / float(n);
    float scale = rsqrt(mean_square + 1.0e-5f);
    out[gid] = x[gid] * scale * weight[gid];
}

kernel void rms_norm_batch(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]]
) {
    threadgroup float scratch[256];
    if (row >= rows) {
        return;
    }

    uint row_start = row * cols;
    float sum = 0.0f;
    for (uint col = tid; col < cols; col += 256u) {
        float value = input[row_start + col];
        sum += value * value;
    }
    scratch[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float scale = rsqrt(scratch[0] / float(cols) + 1.0e-5f);
    for (uint col = tid; col < cols; col += 256u) {
        out[row_start + col] = input[row_start + col] * scale * weight[col];
    }
}

kernel void embedding_lookup_bf16(
    device const ushort* weight [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& token [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < cols) {
        out[gid] = bf16_to_float(weight[token * cols + gid]);
    }
}

kernel void embedding_lookup_bf16_batch(
    device const ushort* weight [[buffer(0)]],
    device const uint* tokens [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant uint& token_count [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = cols * token_count;
    if (gid >= total) {
        return;
    }

    uint row = gid / cols;
    uint col = gid - row * cols;
    uint token = tokens[row];
    out[row * cols + col] = bf16_to_float(weight[token * cols + col]);
}

kernel void bf16_matvec(
    device const ushort* weight [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& cols [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]]
) {
    threadgroup float scratch[256];
    float sum = 0.0f;
    if (row < rows) {
        uint row_start = row * cols;
        for (uint col = tid; col < cols; col += 256) {
            sum += bf16_to_float(weight[row_start + col]) * input[col];
        }
    }
    scratch[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0 && row < rows) {
        out[row] = scratch[0] + bias[row];
    }
}

kernel void bf16_matvec_tiled4(
    device const ushort* weight [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& cols [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tile [[threadgroup_position_in_grid]]
) {
    threadgroup float scratch0[256];
    threadgroup float scratch1[256];
    threadgroup float scratch2[256];
    threadgroup float scratch3[256];

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    for (uint col = tid; col < cols; col += 256u) {
        float x = input[col];
        uint weight_start = (tile * cols + col) * 4u;
        sum0 += bf16_to_float(weight[weight_start]) * x;
        sum1 += bf16_to_float(weight[weight_start + 1u]) * x;
        sum2 += bf16_to_float(weight[weight_start + 2u]) * x;
        sum3 += bf16_to_float(weight[weight_start + 3u]) * x;
    }

    scratch0[tid] = sum0;
    scratch1[tid] = sum1;
    scratch2[tid] = sum2;
    scratch3[tid] = sum3;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            scratch0[tid] += scratch0[tid + stride];
            scratch1[tid] += scratch1[tid + stride];
            scratch2[tid] += scratch2[tid + stride];
            scratch3[tid] += scratch3[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        uint row = tile * 4u;
        if (row < rows) {
            out[row] = scratch0[0] + bias[row];
        }
        if (row + 1u < rows) {
            out[row + 1u] = scratch1[0] + bias[row + 1u];
        }
        if (row + 2u < rows) {
            out[row + 2u] = scratch2[0] + bias[row + 2u];
        }
        if (row + 3u < rows) {
            out[row + 3u] = scratch3[0] + bias[row + 3u];
        }
    }
}

kernel void bf16_qkv_matvec_tiled4(
    device const ushort* q_weight [[buffer(0)]],
    device const ushort* k_weight [[buffer(1)]],
    device const ushort* v_weight [[buffer(2)]],
    device const float* input [[buffer(3)]],
    device const float* q_bias [[buffer(4)]],
    device const float* k_bias [[buffer(5)]],
    device const float* v_bias [[buffer(6)]],
    device float* q_out [[buffer(7)]],
    device float* k_out [[buffer(8)]],
    device float* v_out [[buffer(9)]],
    constant uint& q_tiles [[buffer(10)]],
    constant uint& kv_tiles [[buffer(11)]],
    constant uint& cols [[buffer(12)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tile [[threadgroup_position_in_grid]]
) {
    threadgroup float scratch0[256];
    threadgroup float scratch1[256];
    threadgroup float scratch2[256];
    threadgroup float scratch3[256];

    device const ushort* weight = q_weight;
    device const float* bias = q_bias;
    device float* out = q_out;
    uint local_tile = tile;
    uint row_limit = q_tiles * 4u;
    if (tile >= q_tiles && tile < q_tiles + kv_tiles) {
        weight = k_weight;
        bias = k_bias;
        out = k_out;
        local_tile = tile - q_tiles;
        row_limit = kv_tiles * 4u;
    } else if (tile >= q_tiles + kv_tiles) {
        weight = v_weight;
        bias = v_bias;
        out = v_out;
        local_tile = tile - q_tiles - kv_tiles;
        row_limit = kv_tiles * 4u;
    }

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    for (uint col = tid; col < cols; col += 256u) {
        float x = input[col];
        uint weight_start = (local_tile * cols + col) * 4u;
        sum0 += bf16_to_float(weight[weight_start]) * x;
        sum1 += bf16_to_float(weight[weight_start + 1u]) * x;
        sum2 += bf16_to_float(weight[weight_start + 2u]) * x;
        sum3 += bf16_to_float(weight[weight_start + 3u]) * x;
    }

    scratch0[tid] = sum0;
    scratch1[tid] = sum1;
    scratch2[tid] = sum2;
    scratch3[tid] = sum3;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            scratch0[tid] += scratch0[tid + stride];
            scratch1[tid] += scratch1[tid + stride];
            scratch2[tid] += scratch2[tid + stride];
            scratch3[tid] += scratch3[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        uint row = local_tile * 4u;
        if (row < row_limit) {
            out[row] = scratch0[0] + bias[row];
        }
        if (row + 1u < row_limit) {
            out[row + 1u] = scratch1[0] + bias[row + 1u];
        }
        if (row + 2u < row_limit) {
            out[row + 2u] = scratch2[0] + bias[row + 2u];
        }
        if (row + 3u < row_limit) {
            out[row + 3u] = scratch3[0] + bias[row + 3u];
        }
    }
}

kernel void bf16_matvec_batch(
    device const ushort* weight [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& cols [[buffer(5)]],
    constant uint& batch_rows [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]
) {
    threadgroup float scratch[256];
    uint out_row = group.x;
    uint batch_row = group.y;
    if (out_row >= rows || batch_row >= batch_rows) {
        return;
    }

    uint weight_start = out_row * cols;
    uint input_start = batch_row * cols;
    float sum = 0.0f;
    for (uint col = tid; col < cols; col += 256u) {
        sum += bf16_to_float(weight[weight_start + col]) * input[input_start + col];
    }
    scratch[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        out[batch_row * rows + out_row] = scratch[0] + bias[out_row];
    }
}

kernel void bf16_matvec_batch_tiled4(
    device const ushort* weight [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& cols [[buffer(5)]],
    constant uint& batch_rows [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]
) {
    threadgroup float scratch0[256];
    threadgroup float scratch1[256];
    threadgroup float scratch2[256];
    threadgroup float scratch3[256];

    uint tile = group.x;
    uint batch_row = group.y;
    if (batch_row >= batch_rows) {
        return;
    }

    uint input_start = batch_row * cols;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    for (uint col = tid; col < cols; col += 256u) {
        float x = input[input_start + col];
        uint weight_start = (tile * cols + col) * 4u;
        sum0 += bf16_to_float(weight[weight_start]) * x;
        sum1 += bf16_to_float(weight[weight_start + 1u]) * x;
        sum2 += bf16_to_float(weight[weight_start + 2u]) * x;
        sum3 += bf16_to_float(weight[weight_start + 3u]) * x;
    }

    scratch0[tid] = sum0;
    scratch1[tid] = sum1;
    scratch2[tid] = sum2;
    scratch3[tid] = sum3;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            scratch0[tid] += scratch0[tid + stride];
            scratch1[tid] += scratch1[tid + stride];
            scratch2[tid] += scratch2[tid + stride];
            scratch3[tid] += scratch3[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        uint row = tile * 4u;
        uint out_start = batch_row * rows;
        if (row < rows) {
            out[out_start + row] = scratch0[0] + bias[row];
        }
        if (row + 1u < rows) {
            out[out_start + row + 1u] = scratch1[0] + bias[row + 1u];
        }
        if (row + 2u < rows) {
            out[out_start + row + 2u] = scratch2[0] + bias[row + 2u];
        }
        if (row + 3u < rows) {
            out[out_start + row + 3u] = scratch3[0] + bias[row + 3u];
        }
    }
}

kernel void bf16_matvec_logits(
    device const ushort* weight [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]]
) {
    threadgroup float scratch[256];
    float sum = 0.0f;
    if (row < rows) {
        uint row_start = row * cols;
        for (uint col = tid; col < cols; col += 256u) {
            sum += bf16_to_float(weight[row_start + col]) * input[col];
        }
    }
    scratch[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u && row < rows) {
        out[row] = scratch[0];
    }
}

kernel void topk_logits(
    device const float* logits [[buffer(0)]],
    device uint* out_indices [[buffer(1)]],
    device float* out_logits [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant uint& k [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0u) {
        return;
    }

    float top_values[8];
    uint top_indices[8];
    for (uint slot = 0u; slot < 8u; slot++) {
        top_values[slot] = -3.402823466e+38F;
        top_indices[slot] = 0xffffffffu;
    }

    uint capped_k = min(k, 8u);
    for (uint index = 0u; index < n; index++) {
        float value = logits[index];
        for (uint slot = 0u; slot < capped_k; slot++) {
            bool better = value > top_values[slot] ||
                (value == top_values[slot] && index < top_indices[slot]);
            if (better) {
                for (uint move = capped_k - 1u; move > slot; move--) {
                    top_values[move] = top_values[move - 1u];
                    top_indices[move] = top_indices[move - 1u];
                }
                top_values[slot] = value;
                top_indices[slot] = index;
                break;
            }
        }
    }

    for (uint slot = 0u; slot < capped_k; slot++) {
        out_indices[slot] = top_indices[slot];
        out_logits[slot] = top_values[slot];
    }
}

kernel void top1_logits_blocks(
    device const float* logits [[buffer(0)]],
    device uint* block_indices [[buffer(1)]],
    device float* block_values [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint block [[threadgroup_position_in_grid]]
) {
    threadgroup float values[256];
    threadgroup uint indices[256];

    uint index = block * 256u + tid;
    float value = -3.402823466e+38F;
    uint token = 0xffffffffu;
    if (index < n) {
        value = logits[index];
        token = index;
    }

    values[tid] = value;
    indices[tid] = token;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            float right_value = values[tid + stride];
            uint right_index = indices[tid + stride];
            bool better = right_value > values[tid] ||
                (right_value == values[tid] && right_index < indices[tid]);
            if (better) {
                values[tid] = right_value;
                indices[tid] = right_index;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        block_values[block] = values[0];
        block_indices[block] = indices[0];
    }
}

kernel void top1_logits_final(
    device const uint* block_indices [[buffer(0)]],
    device const float* block_values [[buffer(1)]],
    device uint* out_index [[buffer(2)]],
    device float* out_value [[buffer(3)]],
    constant uint& blocks [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float values[256];
    threadgroup uint indices[256];

    float best_value = -3.402823466e+38F;
    uint best_index = 0xffffffffu;
    for (uint block = tid; block < blocks; block += 256u) {
        float value = block_values[block];
        uint token = block_indices[block];
        bool better = value > best_value ||
            (value == best_value && token < best_index);
        if (better) {
            best_value = value;
            best_index = token;
        }
    }

    values[tid] = best_value;
    indices[tid] = best_index;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            float right_value = values[tid + stride];
            uint right_index = indices[tid + stride];
            bool better = right_value > values[tid] ||
                (right_value == values[tid] && right_index < indices[tid]);
            if (better) {
                values[tid] = right_value;
                indices[tid] = right_index;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        out_index[0] = indices[0];
        out_value[0] = values[0];
    }
}

kernel void rope_row(
    device const float* input [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& heads [[buffer(2)]],
    constant uint& position [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = heads * 64u;
    if (gid >= total) {
        return;
    }

    uint head_offset = (gid / 64u) * 64u;
    uint dim = gid % 64u;
    uint pair_dim = dim % 32u;
    float theta = float(position) * rope_inv_freq(pair_dim);
    float c = cos(theta) * rope_concentration();
    float s = sin(theta) * rope_concentration();

    if (dim < 32u) {
        float x1 = input[head_offset + dim];
        float x2 = input[head_offset + dim + 32u];
        out[gid] = x1 * c - x2 * s;
    } else {
        float x1 = input[head_offset + dim - 32u];
        float x2 = input[head_offset + dim];
        out[gid] = x2 * c + x1 * s;
    }
}

kernel void rope_batch(
    device const float* input [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& heads [[buffer(2)]],
    constant uint& start_position [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint width = heads * 64u;
    uint total = rows * width;
    if (gid >= total) {
        return;
    }

    uint row = gid / width;
    uint local = gid - row * width;
    uint head_offset = row * width + (local / 64u) * 64u;
    uint dim = local % 64u;
    uint pair_dim = dim % 32u;
    float theta = float(start_position + row) * rope_inv_freq(pair_dim);
    float c = cos(theta) * rope_concentration();
    float s = sin(theta) * rope_concentration();

    if (dim < 32u) {
        float x1 = input[head_offset + dim];
        float x2 = input[head_offset + dim + 32u];
        out[gid] = x1 * c - x2 * s;
    } else {
        float x1 = input[head_offset + dim - 32u];
        float x2 = input[head_offset + dim];
        out[gid] = x2 * c + x1 * s;
    }
}

kernel void single_token_attention(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device const float* sinks [[buffer(3)]],
    device float* out [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint head [[threadgroup_position_in_grid]]
) {
    threadgroup float scratch[256];
    uint kv_head = head / 8u;
    uint q_start = head * 64u;
    uint kv_start = kv_head * 64u;

    float sum = 0.0f;
    for (uint dim = tid; dim < 64u; dim += 256u) {
        sum += q[q_start + dim] * k[kv_start + dim];
    }
    scratch[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        float score = scratch[0] * 0.125f;
        float sink = sinks[head];
        float max_value = max(score, sink);
        float exp_score = exp(score - max_value);
        float exp_sink = exp(sink - max_value);
        float data_weight = exp_score / (exp_score + exp_sink);
        for (uint dim = 0; dim < 64u; dim++) {
            out[q_start + dim] = data_weight * v[kv_start + dim];
        }
    }
}

kernel void sequence_attention(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device const float* sinks [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant uint& seq_len [[buffer(5)]],
    constant uint& layer [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]
) {
    threadgroup float scratch[256];
    threadgroup float scores[256];
    threadgroup float norm[2];

    uint head = group.x;
    uint query_position = group.y;
    if (head >= 64u || query_position >= seq_len) {
        return;
    }

    uint key_start = 0u;
    if ((layer & 1u) == 0u && query_position + 1u > 128u) {
        key_start = query_position + 1u - 128u;
    }
    uint key_count = query_position + 1u - key_start;
    uint kv_head = head / 8u;
    uint q_start = query_position * 4096u + head * 64u;
    uint kv_start = kv_head * 64u;

    if (key_count > 256u) {
        if (tid != 0u) {
            return;
        }

        float max_value = sinks[head];
        float denom = 1.0f;
        float values[64];
        for (uint dim = 0u; dim < 64u; dim++) {
            values[dim] = 0.0f;
        }

        for (uint key_offset = 0u; key_offset < key_count; key_offset++) {
            uint key_position = key_start + key_offset;
            uint k_start = key_position * 512u + kv_start;
            uint v_start = key_position * 512u + kv_start;
            float sum = 0.0f;
            for (uint dim = 0u; dim < 64u; dim++) {
                sum += q[q_start + dim] * k[k_start + dim];
            }
            float score = sum * 0.125f;
            float next_max = max(max_value, score);
            float old_scale = exp(max_value - next_max);
            float new_scale = exp(score - next_max);
            for (uint dim = 0u; dim < 64u; dim++) {
                values[dim] = values[dim] * old_scale + v[v_start + dim] * new_scale;
            }
            denom = denom * old_scale + new_scale;
            max_value = next_max;
        }

        for (uint dim = 0u; dim < 64u; dim++) {
            out[q_start + dim] = values[dim] / denom;
        }
        return;
    }

    for (uint key_offset = 0u; key_offset < key_count; key_offset++) {
        uint key_position = key_start + key_offset;
        uint k_start = key_position * 512u + kv_start;

        float sum = 0.0f;
        for (uint dim = tid; dim < 64u; dim += 256u) {
            sum += q[q_start + dim] * k[k_start + dim];
        }
        scratch[tid] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = 128u; stride > 0u; stride >>= 1u) {
            if (tid < stride) {
                scratch[tid] += scratch[tid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (tid == 0u) {
            scores[key_offset] = scratch[0] * 0.125f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        float max_value = sinks[head];
        for (uint key_offset = 0u; key_offset < key_count; key_offset++) {
            max_value = max(max_value, scores[key_offset]);
        }

        float denom = exp(sinks[head] - max_value);
        for (uint key_offset = 0u; key_offset < key_count; key_offset++) {
            denom += exp(scores[key_offset] - max_value);
        }
        norm[0] = max_value;
        norm[1] = denom;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint dim = tid; dim < 64u; dim += 256u) {
        float value = 0.0f;
        for (uint key_offset = 0u; key_offset < key_count; key_offset++) {
            uint key_position = key_start + key_offset;
            uint v_start = key_position * 512u + kv_start;
            float weight = exp(scores[key_offset] - norm[0]) / norm[1];
            value += weight * v[v_start + dim];
        }
        out[q_start + dim] = value;
    }
}

kernel void suffix_sequence_attention(
    device const float* q [[buffer(0)]],
    device const float* k_cache [[buffer(1)]],
    device const float* v_cache [[buffer(2)]],
    device const float* sinks [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant uint& start_position [[buffer(5)]],
    constant uint& suffix_len [[buffer(6)]],
    constant uint& layer [[buffer(7)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]
) {
    threadgroup float scratch[256];
    threadgroup float scores[256];
    threadgroup float norm[2];

    uint head = group.x;
    uint query_row = group.y;
    if (head >= 64u || query_row >= suffix_len) {
        return;
    }

    uint query_position = start_position + query_row;
    uint key_start = 0u;
    if ((layer & 1u) == 0u && query_position + 1u > 128u) {
        key_start = query_position + 1u - 128u;
    }
    uint key_count = query_position + 1u - key_start;
    uint kv_head = head / 8u;
    uint q_start = query_row * 4096u + head * 64u;
    uint kv_start = kv_head * 64u;

    if (key_count > 256u) {
        if (tid != 0u) {
            return;
        }

        float max_value = sinks[head];
        float denom = 1.0f;
        float values[64];
        for (uint dim = 0u; dim < 64u; dim++) {
            values[dim] = 0.0f;
        }

        for (uint key_offset = 0u; key_offset < key_count; key_offset++) {
            uint key_position = key_start + key_offset;
            uint k_start = key_position * 512u + kv_start;
            uint v_start = key_position * 512u + kv_start;
            float sum = 0.0f;
            for (uint dim = 0u; dim < 64u; dim++) {
                sum += q[q_start + dim] * k_cache[k_start + dim];
            }
            float score = sum * 0.125f;
            float next_max = max(max_value, score);
            float old_scale = exp(max_value - next_max);
            float new_scale = exp(score - next_max);
            for (uint dim = 0u; dim < 64u; dim++) {
                values[dim] = values[dim] * old_scale + v_cache[v_start + dim] * new_scale;
            }
            denom = denom * old_scale + new_scale;
            max_value = next_max;
        }

        for (uint dim = 0u; dim < 64u; dim++) {
            out[q_start + dim] = values[dim] / denom;
        }
        return;
    }

    for (uint key_offset = 0u; key_offset < key_count; key_offset++) {
        uint key_position = key_start + key_offset;
        uint k_start = key_position * 512u + kv_start;

        float sum = 0.0f;
        for (uint dim = tid; dim < 64u; dim += 256u) {
            sum += q[q_start + dim] * k_cache[k_start + dim];
        }
        scratch[tid] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = 128u; stride > 0u; stride >>= 1u) {
            if (tid < stride) {
                scratch[tid] += scratch[tid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (tid == 0u) {
            scores[key_offset] = scratch[0] * 0.125f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        float max_value = sinks[head];
        for (uint key_offset = 0u; key_offset < key_count; key_offset++) {
            max_value = max(max_value, scores[key_offset]);
        }

        float denom = exp(sinks[head] - max_value);
        for (uint key_offset = 0u; key_offset < key_count; key_offset++) {
            denom += exp(scores[key_offset] - max_value);
        }
        norm[0] = max_value;
        norm[1] = denom;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint dim = tid; dim < 64u; dim += 256u) {
        float value = 0.0f;
        for (uint key_offset = 0u; key_offset < key_count; key_offset++) {
            uint key_position = key_start + key_offset;
            uint v_start = key_position * 512u + kv_start;
            float weight = exp(scores[key_offset] - norm[0]) / norm[1];
            value += weight * v_cache[v_start + dim];
        }
        out[q_start + dim] = value;
    }
}

kernel void kv_cache_decode_attention(
    device const float* q [[buffer(0)]],
    device const float* k_cache [[buffer(1)]],
    device const float* v_cache [[buffer(2)]],
    device const float* sinks [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant uint& layer [[buffer(5)]],
    constant uint& query_position [[buffer(6)]],
    constant uint& cache_start_position [[buffer(7)]],
    constant uint& cache_len [[buffer(8)]],
    uint tid [[thread_index_in_threadgroup]],
    uint head [[threadgroup_position_in_grid]]
) {
    threadgroup float scratch[256];
    threadgroup float scores[128];
    threadgroup float norm[2];

    if (head >= 64u) {
        return;
    }

    uint effective_key_start = cache_start_position;
    if ((layer & 1u) == 0u && query_position + 1u > 128u) {
        effective_key_start = max(effective_key_start, query_position + 1u - 128u);
    }
    uint key_count = query_position + 1u - effective_key_start;
    if (key_count > cache_len) {
        return;
    }

    uint kv_head = head / 8u;
    uint q_start = head * 64u;
    uint kv_start = kv_head * 64u;

    if (key_count > 128u) {
        if (tid != 0u) {
            return;
        }

        float max_value = sinks[head];
        float denom = 1.0f;
        float values[64];
        for (uint dim = 0u; dim < 64u; dim++) {
            values[dim] = 0.0f;
        }

        for (uint key_offset = 0u; key_offset < key_count; key_offset++) {
            uint key_position = effective_key_start + key_offset;
            uint cache_offset = key_position - cache_start_position;
            uint k_start = cache_offset * 512u + kv_start;
            uint v_start = cache_offset * 512u + kv_start;
            float sum = 0.0f;
            for (uint dim = 0u; dim < 64u; dim++) {
                sum += q[q_start + dim] * k_cache[k_start + dim];
            }
            float score = sum * 0.125f;
            float next_max = max(max_value, score);
            float old_scale = exp(max_value - next_max);
            float new_scale = exp(score - next_max);
            for (uint dim = 0u; dim < 64u; dim++) {
                values[dim] = values[dim] * old_scale + v_cache[v_start + dim] * new_scale;
            }
            denom = denom * old_scale + new_scale;
            max_value = next_max;
        }

        for (uint dim = 0u; dim < 64u; dim++) {
            out[q_start + dim] = values[dim] / denom;
        }
        return;
    }

    for (uint key_offset = 0u; key_offset < key_count; key_offset++) {
        uint key_position = effective_key_start + key_offset;
        uint cache_offset = key_position - cache_start_position;
        uint k_start = cache_offset * 512u + kv_start;

        float sum = 0.0f;
        for (uint dim = tid; dim < 64u; dim += 256u) {
            sum += q[q_start + dim] * k_cache[k_start + dim];
        }
        scratch[tid] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = 128u; stride > 0u; stride >>= 1u) {
            if (tid < stride) {
                scratch[tid] += scratch[tid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (tid == 0u) {
            scores[key_offset] = scratch[0] * 0.125f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        float max_value = sinks[head];
        for (uint key_offset = 0u; key_offset < key_count; key_offset++) {
            max_value = max(max_value, scores[key_offset]);
        }

        float denom = exp(sinks[head] - max_value);
        for (uint key_offset = 0u; key_offset < key_count; key_offset++) {
            denom += exp(scores[key_offset] - max_value);
        }
        norm[0] = max_value;
        norm[1] = denom;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint dim = tid; dim < 64u; dim += 256u) {
        float value = 0.0f;
        for (uint key_offset = 0u; key_offset < key_count; key_offset++) {
            uint key_position = effective_key_start + key_offset;
            uint cache_offset = key_position - cache_start_position;
            uint v_start = cache_offset * 512u + kv_start;
            float weight = exp(scores[key_offset] - norm[0]) / norm[1];
            value += weight * v_cache[v_start + dim];
        }
        out[q_start + dim] = value;
    }
}

kernel void vector_add(
    device const float* left [[buffer(0)]],
    device const float* right [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < n) {
        out[gid] = left[gid] + right[gid];
    }
}

kernel void write_f32_slot(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& slot [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < width) {
        output[slot * width + gid] = input[gid];
    }
}

kernel void write_f32_slots_batch(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& start_slot [[buffer(2)]],
    constant uint& slots [[buffer(3)]],
    constant uint& width [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = slots * width;
    if (gid >= total) {
        return;
    }

    uint slot = gid / width;
    uint col = gid - slot * width;
    output[(start_slot + slot) * width + col] = input[gid];
}

kernel void read_f32_slot(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& slot [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < width) {
        output[gid] = input[slot * width + gid];
    }
}

kernel void top4_softmax(
    device const float* logits [[buffer(0)]],
    device uint* out_indices [[buffer(1)]],
    device float* out_logits [[buffer(2)]],
    device float* out_weights [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0u) {
        return;
    }

    float best_logits[4] = {
        -INFINITY,
        -INFINITY,
        -INFINITY,
        -INFINITY,
    };
    uint best_indices[4] = { 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu };

    for (uint i = 0; i < n; i++) {
        float value = logits[i];
        for (uint rank = 0; rank < 4u; rank++) {
            bool better = value > best_logits[rank]
                || (value == best_logits[rank] && i < best_indices[rank]);
            if (!better) {
                continue;
            }
            for (uint move = 3u; move > rank; move--) {
                best_logits[move] = best_logits[move - 1u];
                best_indices[move] = best_indices[move - 1u];
            }
            best_logits[rank] = value;
            best_indices[rank] = i;
            break;
        }
    }

    float max_value = best_logits[0];
    float denom = 0.0f;
    for (uint rank = 0; rank < 4u; rank++) {
        denom += exp(best_logits[rank] - max_value);
    }

    for (uint rank = 0; rank < 4u; rank++) {
        out_indices[rank] = best_indices[rank];
        out_logits[rank] = best_logits[rank];
        out_weights[rank] = exp(best_logits[rank] - max_value) / denom;
    }
}

kernel void top4_softmax_batch(
    device const float* logits [[buffer(0)]],
    device uint* out_indices [[buffer(1)]],
    device float* out_logits [[buffer(2)]],
    device float* out_weights [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& n [[buffer(5)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= rows) {
        return;
    }

    float best_logits[4] = {
        -INFINITY,
        -INFINITY,
        -INFINITY,
        -INFINITY,
    };
    uint best_indices[4] = { 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu };
    uint logits_start = row * n;

    for (uint i = 0; i < n; i++) {
        float value = logits[logits_start + i];
        for (uint rank = 0; rank < 4u; rank++) {
            bool better = value > best_logits[rank]
                || (value == best_logits[rank] && i < best_indices[rank]);
            if (!better) {
                continue;
            }
            for (uint move = 3u; move > rank; move--) {
                best_logits[move] = best_logits[move - 1u];
                best_indices[move] = best_indices[move - 1u];
            }
            best_logits[rank] = value;
            best_indices[rank] = i;
            break;
        }
    }

    float max_value = best_logits[0];
    float denom = 0.0f;
    for (uint rank = 0; rank < 4u; rank++) {
        denom += exp(best_logits[rank] - max_value);
    }

    uint out_start = row * 4u;
    for (uint rank = 0; rank < 4u; rank++) {
        out_indices[out_start + rank] = best_indices[rank];
        out_logits[out_start + rank] = best_logits[rank];
        out_weights[out_start + rank] = exp(best_logits[rank] - max_value) / denom;
    }
}

kernel void mxfp4_matvec(
    device const uchar* blocks [[buffer(0)]],
    device const uchar* scales [[buffer(1)]],
    device const float* input [[buffer(2)]],
    device const float* bias [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    constant uint& groups [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]]
) {
    threadgroup float scratch[256];
    threadgroup float row_scales[90];
    float sum = 0.0f;
    uint values_per_row = groups * 16u;
    uint row_scale_start = row * groups;
    bool cache_scales = row < rows && groups <= GPTOSS_MXFP4_GROUPS;

    if (cache_scales && tid < groups) {
        row_scales[tid] = mxfp4_scale(scales[row_scale_start + tid]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row < rows) {
        uint row_block_start = row * values_per_row;
        for (uint packed_index = tid; packed_index < values_per_row; packed_index += 256u) {
            uint group = packed_index / 16u;
            uint byte_in_group = packed_index - group * 16u;
            uchar packed = blocks[row_block_start + packed_index];
            float scale = cache_scales ? row_scales[group] : mxfp4_scale(scales[row_scale_start + group]);
            uint input_start = group * 32u + byte_in_group * 2u;
            sum += fp4_to_float(uint(packed & 0x0fu)) * scale * input[input_start];
            sum += fp4_to_float(uint(packed >> 4)) * scale * input[input_start + 1u];
        }
    }

    scratch[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0 && row < rows) {
        out[row] = scratch[0] + bias[row];
    }
}

kernel void mxfp4_top4_gate_swiglu(
    device const uchar* blocks [[buffer(0)]],
    device const uchar* scales [[buffer(1)]],
    device const ushort* bias [[buffer(2)]],
    device const float* input [[buffer(3)]],
    device const uint* top_indices [[buffer(4)]],
    device float* out [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]]
) {
    threadgroup float gate_scratch[256];
    threadgroup float linear_scratch[256];
    threadgroup float gate_scales[90];
    threadgroup float linear_scales[90];

    uint row = group.x;
    uint slot = group.y;
    uint expert = top_indices[slot];
    float gate_sum = 0.0f;
    float linear_sum = 0.0f;
    bool active = row < GPTOSS_HIDDEN_SIZE && slot < 4u && expert < GPTOSS_EXPERTS;

    uint gate_row = row * 2u;
    uint linear_row = gate_row + 1u;
    uint values_per_row = GPTOSS_MXFP4_GROUPS * GPTOSS_MXFP4_BYTES_PER_GROUP;
    uint expert_block_start = expert * GPTOSS_GATE_UP_VALUES * values_per_row;
    uint expert_scale_start = expert * GPTOSS_GATE_UP_VALUES * GPTOSS_MXFP4_GROUPS;
    uint gate_block_start = expert_block_start + gate_row * values_per_row;
    uint linear_block_start = expert_block_start + linear_row * values_per_row;
    uint gate_scale_start = expert_scale_start + gate_row * GPTOSS_MXFP4_GROUPS;
    uint linear_scale_start = expert_scale_start + linear_row * GPTOSS_MXFP4_GROUPS;

    if (active && tid < GPTOSS_MXFP4_GROUPS) {
        gate_scales[tid] = mxfp4_scale(scales[gate_scale_start + tid]);
        linear_scales[tid] = mxfp4_scale(scales[linear_scale_start + tid]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (active) {
        for (uint packed_index = tid; packed_index < values_per_row; packed_index += 256u) {
            uint group_index = packed_index / GPTOSS_MXFP4_BYTES_PER_GROUP;
            uint byte_in_group = packed_index - group_index * GPTOSS_MXFP4_BYTES_PER_GROUP;
            uint input_start = group_index * 32u + byte_in_group * 2u;

            uchar gate_packed = blocks[gate_block_start + packed_index];
            float gate_scale = gate_scales[group_index];
            gate_sum += fp4_to_float(uint(gate_packed & 0x0fu)) * gate_scale * input[input_start];
            gate_sum += fp4_to_float(uint(gate_packed >> 4)) * gate_scale * input[input_start + 1u];

            uchar linear_packed = blocks[linear_block_start + packed_index];
            float linear_scale = linear_scales[group_index];
            linear_sum += fp4_to_float(uint(linear_packed & 0x0fu)) * linear_scale * input[input_start];
            linear_sum += fp4_to_float(uint(linear_packed >> 4)) * linear_scale * input[input_start + 1u];
        }
    }

    gate_scratch[tid] = gate_sum;
    linear_scratch[tid] = linear_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            gate_scratch[tid] += gate_scratch[tid + stride];
            linear_scratch[tid] += linear_scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u && row < GPTOSS_HIDDEN_SIZE && slot < 4u && expert < GPTOSS_EXPERTS) {
        uint bias_start = expert * GPTOSS_GATE_UP_VALUES;
        float gate = gate_scratch[0] + bf16_to_float(bias[bias_start + row * 2u]);
        float linear = linear_scratch[0] + bf16_to_float(bias[bias_start + row * 2u + 1u]);
        float x_glu = min(gate, 7.0f);
        float x_linear = clamp(linear, -7.0f, 7.0f);
        float out_glu = x_glu / (1.0f + exp(-1.702f * x_glu));
        out[slot * GPTOSS_HIDDEN_SIZE + row] = out_glu * (x_linear + 1.0f);
    }
}

kernel void mxfp4_top4_gate_swiglu_batch(
    device const uchar* blocks [[buffer(0)]],
    device const uchar* scales [[buffer(1)]],
    device const ushort* bias [[buffer(2)]],
    device const float* input [[buffer(3)]],
    device const uint* top_indices [[buffer(4)]],
    device float* out [[buffer(5)]],
    constant uint& row_offset [[buffer(6)]],
    constant uint& rows [[buffer(7)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]
) {
    threadgroup float gate_scratch[256];
    threadgroup float linear_scratch[256];
    threadgroup float gate_scales[90];
    threadgroup float linear_scales[90];

    uint row = group.x;
    uint slot = group.y;
    uint batch_row = group.z;
    uint source_row = row_offset + batch_row;
    uint top_start = source_row * 4u;
    uint expert = top_indices[top_start + slot];
    float gate_sum = 0.0f;
    float linear_sum = 0.0f;
    bool active = row < GPTOSS_HIDDEN_SIZE && slot < 4u && batch_row < rows && expert < GPTOSS_EXPERTS;

    uint gate_row = row * 2u;
    uint linear_row = gate_row + 1u;
    uint values_per_row = GPTOSS_MXFP4_GROUPS * GPTOSS_MXFP4_BYTES_PER_GROUP;
    uint expert_block_start = expert * GPTOSS_GATE_UP_VALUES * values_per_row;
    uint expert_scale_start = expert * GPTOSS_GATE_UP_VALUES * GPTOSS_MXFP4_GROUPS;
    uint gate_block_start = expert_block_start + gate_row * values_per_row;
    uint linear_block_start = expert_block_start + linear_row * values_per_row;
    uint gate_scale_start = expert_scale_start + gate_row * GPTOSS_MXFP4_GROUPS;
    uint linear_scale_start = expert_scale_start + linear_row * GPTOSS_MXFP4_GROUPS;
    uint input_row_start = source_row * GPTOSS_HIDDEN_SIZE;

    if (active && tid < GPTOSS_MXFP4_GROUPS) {
        gate_scales[tid] = mxfp4_scale(scales[gate_scale_start + tid]);
        linear_scales[tid] = mxfp4_scale(scales[linear_scale_start + tid]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (active) {
        for (uint packed_index = tid; packed_index < values_per_row; packed_index += 256u) {
            uint group_index = packed_index / GPTOSS_MXFP4_BYTES_PER_GROUP;
            uint byte_in_group = packed_index - group_index * GPTOSS_MXFP4_BYTES_PER_GROUP;
            uint input_start = input_row_start + group_index * 32u + byte_in_group * 2u;

            uchar gate_packed = blocks[gate_block_start + packed_index];
            float gate_scale = gate_scales[group_index];
            gate_sum += fp4_to_float(uint(gate_packed & 0x0fu)) * gate_scale * input[input_start];
            gate_sum += fp4_to_float(uint(gate_packed >> 4)) * gate_scale * input[input_start + 1u];

            uchar linear_packed = blocks[linear_block_start + packed_index];
            float linear_scale = linear_scales[group_index];
            linear_sum += fp4_to_float(uint(linear_packed & 0x0fu)) * linear_scale * input[input_start];
            linear_sum += fp4_to_float(uint(linear_packed >> 4)) * linear_scale * input[input_start + 1u];
        }
    }

    gate_scratch[tid] = gate_sum;
    linear_scratch[tid] = linear_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            gate_scratch[tid] += gate_scratch[tid + stride];
            linear_scratch[tid] += linear_scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u && row < GPTOSS_HIDDEN_SIZE && slot < 4u && batch_row < rows && expert < GPTOSS_EXPERTS) {
        uint bias_start = expert * GPTOSS_GATE_UP_VALUES;
        float gate = gate_scratch[0] + bf16_to_float(bias[bias_start + row * 2u]);
        float linear = linear_scratch[0] + bf16_to_float(bias[bias_start + row * 2u + 1u]);
        float x_glu = min(gate, 7.0f);
        float x_linear = clamp(linear, -7.0f, 7.0f);
        float out_glu = x_glu / (1.0f + exp(-1.702f * x_glu));
        uint out_start = (batch_row * 4u + slot) * GPTOSS_HIDDEN_SIZE;
        out[out_start + row] = out_glu * (x_linear + 1.0f);
    }
}

kernel void mxfp4_top4_down_weighted(
    device const uchar* blocks [[buffer(0)]],
    device const uchar* scales [[buffer(1)]],
    device const ushort* bias [[buffer(2)]],
    device const float* expert_acts [[buffer(3)]],
    device const uint* top_indices [[buffer(4)]],
    device const float* top_weights [[buffer(5)]],
    device const float* residual [[buffer(6)]],
    device float* out [[buffer(7)]],
    uint tid [[thread_index_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]]
) {
    threadgroup float scratch[256];
    threadgroup float selected_scales[360];
    float sum = 0.0f;
    uint values_per_row = GPTOSS_MXFP4_GROUPS * GPTOSS_MXFP4_BYTES_PER_GROUP;

    for (uint scale_index = tid; scale_index < 4u * GPTOSS_MXFP4_GROUPS; scale_index += 256u) {
        uint scale_slot = scale_index / GPTOSS_MXFP4_GROUPS;
        uint scale_group = scale_index - scale_slot * GPTOSS_MXFP4_GROUPS;
        uint expert = top_indices[scale_slot];
        if (row < GPTOSS_HIDDEN_SIZE && expert < GPTOSS_EXPERTS) {
            uint expert_scale_start = expert * GPTOSS_HIDDEN_SIZE * GPTOSS_MXFP4_GROUPS;
            uint row_scale_start = expert_scale_start + row * GPTOSS_MXFP4_GROUPS;
            selected_scales[scale_index] = mxfp4_scale(scales[row_scale_start + scale_group]);
        } else {
            selected_scales[scale_index] = 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row < GPTOSS_HIDDEN_SIZE) {
        for (uint slot = 0u; slot < 4u; slot++) {
            uint expert = top_indices[slot];
            if (expert >= GPTOSS_EXPERTS) {
                continue;
            }
            float weight = top_weights[slot];
            uint expert_block_start = expert * GPTOSS_HIDDEN_SIZE * values_per_row;
            uint row_block_start = expert_block_start + row * values_per_row;
            uint input_slot_start = slot * GPTOSS_HIDDEN_SIZE;

            for (uint packed_index = tid; packed_index < values_per_row; packed_index += 256u) {
                uint group_index = packed_index / GPTOSS_MXFP4_BYTES_PER_GROUP;
                uint byte_in_group = packed_index - group_index * GPTOSS_MXFP4_BYTES_PER_GROUP;
                uchar packed = blocks[row_block_start + packed_index];
                float scale = selected_scales[slot * GPTOSS_MXFP4_GROUPS + group_index];
                uint input_start = input_slot_start + group_index * 32u + byte_in_group * 2u;
                sum += weight * fp4_to_float(uint(packed & 0x0fu)) * scale * expert_acts[input_start];
                sum += weight * fp4_to_float(uint(packed >> 4)) * scale * expert_acts[input_start + 1u];
            }
        }
    }

    scratch[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u && row < GPTOSS_HIDDEN_SIZE) {
        float bias_sum = 0.0f;
        for (uint slot = 0u; slot < 4u; slot++) {
            uint expert = top_indices[slot];
            if (expert < GPTOSS_EXPERTS) {
                bias_sum += top_weights[slot] * bf16_to_float(bias[expert * GPTOSS_HIDDEN_SIZE + row]);
            }
        }
        out[row] = residual[row] + scratch[0] + bias_sum;
    }
}

kernel void mxfp4_top4_down_weighted_batch(
    device const uchar* blocks [[buffer(0)]],
    device const uchar* scales [[buffer(1)]],
    device const ushort* bias [[buffer(2)]],
    device const float* expert_acts [[buffer(3)]],
    device const uint* top_indices [[buffer(4)]],
    device const float* top_weights [[buffer(5)]],
    device const float* residual [[buffer(6)]],
    device float* out [[buffer(7)]],
    constant uint& row_offset [[buffer(8)]],
    constant uint& rows [[buffer(9)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]]
) {
    threadgroup float scratch[256];
    threadgroup float selected_scales[360];
    uint row = group.x;
    uint batch_row = group.y;
    uint source_row = row_offset + batch_row;
    float sum = 0.0f;
    uint values_per_row = GPTOSS_MXFP4_GROUPS * GPTOSS_MXFP4_BYTES_PER_GROUP;
    uint top_start = source_row * 4u;

    for (uint scale_index = tid; scale_index < 4u * GPTOSS_MXFP4_GROUPS; scale_index += 256u) {
        uint scale_slot = scale_index / GPTOSS_MXFP4_GROUPS;
        uint scale_group = scale_index - scale_slot * GPTOSS_MXFP4_GROUPS;
        uint expert = top_indices[top_start + scale_slot];
        if (row < GPTOSS_HIDDEN_SIZE && batch_row < rows && expert < GPTOSS_EXPERTS) {
            uint expert_scale_start = expert * GPTOSS_HIDDEN_SIZE * GPTOSS_MXFP4_GROUPS;
            uint row_scale_start = expert_scale_start + row * GPTOSS_MXFP4_GROUPS;
            selected_scales[scale_index] = mxfp4_scale(scales[row_scale_start + scale_group]);
        } else {
            selected_scales[scale_index] = 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row < GPTOSS_HIDDEN_SIZE && batch_row < rows) {
        for (uint slot = 0u; slot < 4u; slot++) {
            uint expert = top_indices[top_start + slot];
            if (expert >= GPTOSS_EXPERTS) {
                continue;
            }
            float weight = top_weights[top_start + slot];
            uint expert_block_start = expert * GPTOSS_HIDDEN_SIZE * values_per_row;
            uint row_block_start = expert_block_start + row * values_per_row;
            uint input_slot_start = (batch_row * 4u + slot) * GPTOSS_HIDDEN_SIZE;

            for (uint packed_index = tid; packed_index < values_per_row; packed_index += 256u) {
                uint group_index = packed_index / GPTOSS_MXFP4_BYTES_PER_GROUP;
                uint byte_in_group = packed_index - group_index * GPTOSS_MXFP4_BYTES_PER_GROUP;
                uchar packed = blocks[row_block_start + packed_index];
                float scale = selected_scales[slot * GPTOSS_MXFP4_GROUPS + group_index];
                uint input_start = input_slot_start + group_index * 32u + byte_in_group * 2u;
                sum += weight * fp4_to_float(uint(packed & 0x0fu)) * scale * expert_acts[input_start];
                sum += weight * fp4_to_float(uint(packed >> 4)) * scale * expert_acts[input_start + 1u];
            }
        }
    }

    scratch[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u && row < GPTOSS_HIDDEN_SIZE && batch_row < rows) {
        uint top_start = source_row * 4u;
        float bias_sum = 0.0f;
        for (uint slot = 0u; slot < 4u; slot++) {
            uint expert = top_indices[top_start + slot];
            if (expert < GPTOSS_EXPERTS) {
                bias_sum += top_weights[top_start + slot] * bf16_to_float(bias[expert * GPTOSS_HIDDEN_SIZE + row]);
            }
        }
        uint hidden_index = source_row * GPTOSS_HIDDEN_SIZE + row;
        out[hidden_index] = residual[hidden_index] + scratch[0] + bias_sum;
    }
}

kernel void swiglu(
    device const float* input [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < n) {
        float x_glu = min(input[gid * 2u], 7.0f);
        float x_linear = clamp(input[gid * 2u + 1u], -7.0f, 7.0f);
        float out_glu = x_glu / (1.0f + exp(-1.702f * x_glu));
        out[gid] = out_glu * (x_linear + 1.0f);
    }
}

kernel void weighted_sum4(
    device const float* vectors [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < n) {
        out[gid] =
            vectors[gid] * weights[0] +
            vectors[n + gid] * weights[1] +
            vectors[n * 2u + gid] * weights[2] +
            vectors[n * 3u + gid] * weights[3];
    }
}
