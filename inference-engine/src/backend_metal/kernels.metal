#include <metal_stdlib>
using namespace metal;

constant uint GPTOSS_HIDDEN_SIZE = 2880u;
constant uint GPTOSS_GATE_UP_VALUES = 5760u;
constant uint GPTOSS_EXPERTS = 32u;
constant uint GPTOSS_MXFP4_GROUPS = 90u;
constant uint GPTOSS_MXFP4_BYTES_PER_GROUP = 16u;
constant uint GPTOSS_DECODE_ATTENTION_THREADS = 64u;
constant uint GPTOSS_Q8_PAIR_SIMDGROUPS = 4u;
constant float GPTOSS_FP4_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

inline float bf16_to_float(ushort value) {
    uint bits = ((uint)value) << 16;
    return as_type<float>(bits);
}

inline float f16_to_float(ushort value) {
    return float(as_type<half>(value));
}

inline float fp4_to_float(uint value) {
    return GPTOSS_FP4_LUT[value & 15u];
}

inline float q8_0_to_float(device const uchar* weight, uint row, uint col, uint cols) {
    uint blocks_per_row = cols / 32u;
    uint block = col / 32u;
    uint lane = col - block * 32u;
    uint offset = (row * blocks_per_row + block) * 34u;
    ushort scale_bits = ushort(weight[offset]) | (ushort(weight[offset + 1u]) << 8);
    float scale = f16_to_float(scale_bits);
    uchar raw = weight[offset + 2u + lane];
    int quant = raw < 128u ? int(raw) : int(raw) - 256;
    return scale * float(quant);
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

kernel void embedding_lookup_q8_0(
    device const uchar* weight [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& token [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < cols) {
        out[gid] = q8_0_to_float(weight, token, gid, cols);
    }
}

kernel void embedding_lookup_q8_0_batch(
    device const uchar* weight [[buffer(0)]],
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
    out[row * cols + col] = q8_0_to_float(weight, token, col, cols);
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

kernel void f32_matvec(
    device const float* weight [[buffer(0)]],
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
        for (uint col = tid; col < cols; col += 256u) {
            sum += weight[row_start + col] * input[col];
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
        out[row] = scratch[0] + bias[row];
    }
}

kernel void f32_matvec_batch(
    device const float* weight [[buffer(0)]],
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
        sum += weight[weight_start + col] * input[input_start + col];
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

kernel void bf16_matvec_add(
    device const ushort* weight [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device const float* residual [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    constant uint& cols [[buffer(6)]],
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
        out[row] = residual[row] + scratch[0] + bias[row];
    }
}

kernel void bf16_matvec_add_tiled4(
    device const ushort* weight [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device const float* residual [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    constant uint& cols [[buffer(6)]],
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
            out[row] = residual[row] + scratch0[0] + bias[row];
        }
        if (row + 1u < rows) {
            out[row + 1u] = residual[row + 1u] + scratch1[0] + bias[row + 1u];
        }
        if (row + 2u < rows) {
            out[row + 2u] = residual[row + 2u] + scratch2[0] + bias[row + 2u];
        }
        if (row + 3u < rows) {
            out[row + 3u] = residual[row + 3u] + scratch3[0] + bias[row + 3u];
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

kernel void q8_0_matvec_logits(
    device const uchar* weight [[buffer(0)]],
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
        for (uint col = tid; col < cols; col += 256u) {
            sum += q8_0_to_float(weight, row, col, cols) * input[col];
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

kernel void q8_0_matvec_logits_pair(
    device const uchar* weight [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simdgroup [[simdgroup_index_in_threadgroup]],
    uint row_pair [[threadgroup_position_in_grid]]
) {
    threadgroup float partial0[32];
    threadgroup float partial1[32];

    const uint row0 = row_pair * 2u;
    const uint row1 = row0 + 1u;
    const uint blocks = cols / 32u;
    const uint n_quant = 8u;
    const uint lanes_per_block = 32u / n_quant;
    const uint ix = uint(lane) / lanes_per_block;
    const uint il = uint(lane) - ix * lanes_per_block;
    const uint first_block = uint(simdgroup) * n_quant + ix;

    float sum0 = 0.0f;
    float sum1 = 0.0f;

    for (uint block = first_block; block < blocks; block += GPTOSS_Q8_PAIR_SIMDGROUPS * n_quant) {
        float values[8];
        uint input_start = block * 32u + il * n_quant;
        for (uint i = 0u; i < n_quant; i++) {
            values[i] = input[input_start + i];
        }

        if (row0 < rows) {
            uint offset = (row0 * blocks + block) * 34u;
            float scale = float(*(device const half*)(weight + offset));
            float sumq = 0.0f;
            device const char* q = (device const char*)(weight + offset + 2u + il * n_quant);
            for (uint i = 0u; i < n_quant; i++) {
                sumq += float(q[i]) * values[i];
            }
            sum0 += sumq * scale;
        }

        if (row1 < rows) {
            uint offset = (row1 * blocks + block) * 34u;
            float scale = float(*(device const half*)(weight + offset));
            float sumq = 0.0f;
            device const char* q = (device const char*)(weight + offset + 2u + il * n_quant);
            for (uint i = 0u; i < n_quant; i++) {
                sumq += float(q[i]) * values[i];
            }
            sum1 += sumq * scale;
        }
    }

    sum0 = simd_sum(sum0);
    sum1 = simd_sum(sum1);
    if (lane == 0) {
        partial0[simdgroup] = sum0;
        partial1[simdgroup] = sum1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup == 0) {
        float total0 = lane < GPTOSS_Q8_PAIR_SIMDGROUPS ? partial0[lane] : 0.0f;
        float total1 = lane < GPTOSS_Q8_PAIR_SIMDGROUPS ? partial1[lane] : 0.0f;
        total0 = simd_sum(total0);
        total1 = simd_sum(total1);
        if (lane == 0) {
            if (row0 < rows) {
                out[row0] = total0;
            }
            if (row1 < rows) {
                out[row1] = total1;
            }
        }
    }
}

kernel void q8_0_matvec(
    device const uchar* weight [[buffer(0)]],
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
        for (uint col = tid; col < cols; col += 256u) {
            sum += q8_0_to_float(weight, row, col, cols) * input[col];
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
        out[row] = scratch[0] + bias[row];
    }
}

kernel void q8_0_matvec_add(
    device const uchar* weight [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device const float* residual [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    constant uint& cols [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]]
) {
    threadgroup float scratch[256];
    float sum = 0.0f;
    if (row < rows) {
        for (uint col = tid; col < cols; col += 256u) {
            sum += q8_0_to_float(weight, row, col, cols) * input[col];
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
        out[row] = residual[row] + scratch[0] + bias[row];
    }
}

kernel void q8_0_matvec_add_pair(
    device const uchar* weight [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device const float* residual [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    constant uint& cols [[buffer(6)]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simdgroup [[simdgroup_index_in_threadgroup]],
    uint row_pair [[threadgroup_position_in_grid]]
) {
    threadgroup float partial0[32];
    threadgroup float partial1[32];

    const uint row0 = row_pair * 2u;
    const uint row1 = row0 + 1u;
    const uint blocks = cols / 32u;
    const uint n_quant = 8u;
    const uint lanes_per_block = 32u / n_quant;
    const uint ix = uint(lane) / lanes_per_block;
    const uint il = uint(lane) - ix * lanes_per_block;
    const uint first_block = uint(simdgroup) * n_quant + ix;

    float sum0 = 0.0f;
    float sum1 = 0.0f;

    for (uint block = first_block; block < blocks; block += GPTOSS_Q8_PAIR_SIMDGROUPS * n_quant) {
        float values[8];
        uint input_start = block * 32u + il * n_quant;
        for (uint i = 0u; i < n_quant; i++) {
            values[i] = input[input_start + i];
        }

        if (row0 < rows) {
            uint offset = (row0 * blocks + block) * 34u;
            float scale = float(*(device const half*)(weight + offset));
            float sumq = 0.0f;
            device const char* q = (device const char*)(weight + offset + 2u + il * n_quant);
            for (uint i = 0u; i < n_quant; i++) {
                sumq += float(q[i]) * values[i];
            }
            sum0 += sumq * scale;
        }

        if (row1 < rows) {
            uint offset = (row1 * blocks + block) * 34u;
            float scale = float(*(device const half*)(weight + offset));
            float sumq = 0.0f;
            device const char* q = (device const char*)(weight + offset + 2u + il * n_quant);
            for (uint i = 0u; i < n_quant; i++) {
                sumq += float(q[i]) * values[i];
            }
            sum1 += sumq * scale;
        }
    }

    sum0 = simd_sum(sum0);
    sum1 = simd_sum(sum1);
    if (lane == 0) {
        partial0[simdgroup] = sum0;
        partial1[simdgroup] = sum1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup == 0) {
        float total0 = lane < GPTOSS_Q8_PAIR_SIMDGROUPS ? partial0[lane] : 0.0f;
        float total1 = lane < GPTOSS_Q8_PAIR_SIMDGROUPS ? partial1[lane] : 0.0f;
        total0 = simd_sum(total0);
        total1 = simd_sum(total1);
        if (lane == 0) {
            if (row0 < rows) {
                out[row0] = residual[row0] + total0 + bias[row0];
            }
            if (row1 < rows) {
                out[row1] = residual[row1] + total1 + bias[row1];
            }
        }
    }
}

kernel void q8_0_qkv_matvec(
    device const uchar* q_weight [[buffer(0)]],
    device const uchar* k_weight [[buffer(1)]],
    device const uchar* v_weight [[buffer(2)]],
    device const float* input [[buffer(3)]],
    device const float* q_bias [[buffer(4)]],
    device const float* k_bias [[buffer(5)]],
    device const float* v_bias [[buffer(6)]],
    device float* q_out [[buffer(7)]],
    device float* k_out [[buffer(8)]],
    device float* v_out [[buffer(9)]],
    constant uint& q_rows [[buffer(10)]],
    constant uint& kv_rows [[buffer(11)]],
    constant uint& cols [[buffer(12)]],
    uint tid [[thread_index_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]]
) {
    threadgroup float scratch[256];

    device const uchar* weight = q_weight;
    device const float* bias = q_bias;
    device float* out = q_out;
    uint local_row = row;
    uint row_limit = q_rows;
    if (row >= q_rows && row < q_rows + kv_rows) {
        weight = k_weight;
        bias = k_bias;
        out = k_out;
        local_row = row - q_rows;
        row_limit = kv_rows;
    } else if (row >= q_rows + kv_rows) {
        weight = v_weight;
        bias = v_bias;
        out = v_out;
        local_row = row - q_rows - kv_rows;
        row_limit = kv_rows;
    }

    float sum = 0.0f;
    if (local_row < row_limit) {
        for (uint col = tid; col < cols; col += 256u) {
            sum += q8_0_to_float(weight, local_row, col, cols) * input[col];
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

    if (tid == 0u && local_row < row_limit) {
        out[local_row] = scratch[0] + bias[local_row];
    }
}

kernel void q8_0_qkv_matvec_pair(
    device const uchar* q_weight [[buffer(0)]],
    device const uchar* k_weight [[buffer(1)]],
    device const uchar* v_weight [[buffer(2)]],
    device const float* input [[buffer(3)]],
    device const float* q_bias [[buffer(4)]],
    device const float* k_bias [[buffer(5)]],
    device const float* v_bias [[buffer(6)]],
    device float* q_out [[buffer(7)]],
    device float* k_out [[buffer(8)]],
    device float* v_out [[buffer(9)]],
    constant uint& q_rows [[buffer(10)]],
    constant uint& kv_rows [[buffer(11)]],
    constant uint& cols [[buffer(12)]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simdgroup [[simdgroup_index_in_threadgroup]],
    uint row_pair [[threadgroup_position_in_grid]]
) {
    threadgroup float partial0[32];
    threadgroup float partial1[32];

    const uint row0 = row_pair * 2u;
    const uint row1 = row0 + 1u;
    const uint total_rows = q_rows + kv_rows + kv_rows;
    const uint blocks = cols / 32u;
    const uint n_quant = 8u;
    const uint lanes_per_block = 32u / n_quant;
    const uint ix = uint(lane) / lanes_per_block;
    const uint il = uint(lane) - ix * lanes_per_block;
    const uint first_block = uint(simdgroup) * n_quant + ix;

    device const uchar* weight0 = q_weight;
    device const float* bias0 = q_bias;
    device float* out0 = q_out;
    uint local0 = row0;
    uint limit0 = q_rows;
    if (row0 >= q_rows && row0 < q_rows + kv_rows) {
        weight0 = k_weight;
        bias0 = k_bias;
        out0 = k_out;
        local0 = row0 - q_rows;
        limit0 = kv_rows;
    } else if (row0 >= q_rows + kv_rows) {
        weight0 = v_weight;
        bias0 = v_bias;
        out0 = v_out;
        local0 = row0 - q_rows - kv_rows;
        limit0 = kv_rows;
    }

    device const uchar* weight1 = q_weight;
    device const float* bias1 = q_bias;
    device float* out1 = q_out;
    uint local1 = row1;
    uint limit1 = q_rows;
    if (row1 >= q_rows && row1 < q_rows + kv_rows) {
        weight1 = k_weight;
        bias1 = k_bias;
        out1 = k_out;
        local1 = row1 - q_rows;
        limit1 = kv_rows;
    } else if (row1 >= q_rows + kv_rows) {
        weight1 = v_weight;
        bias1 = v_bias;
        out1 = v_out;
        local1 = row1 - q_rows - kv_rows;
        limit1 = kv_rows;
    }

    bool active0 = row0 < total_rows && local0 < limit0;
    bool active1 = row1 < total_rows && local1 < limit1;
    float sum0 = 0.0f;
    float sum1 = 0.0f;

    for (uint block = first_block; block < blocks; block += GPTOSS_Q8_PAIR_SIMDGROUPS * n_quant) {
        float values[8];
        uint input_start = block * 32u + il * n_quant;
        for (uint i = 0u; i < n_quant; i++) {
            values[i] = input[input_start + i];
        }

        if (active0) {
            uint offset = (local0 * blocks + block) * 34u;
            float scale = float(*(device const half*)(weight0 + offset));
            float sumq = 0.0f;
            device const char* q = (device const char*)(weight0 + offset + 2u + il * n_quant);
            for (uint i = 0u; i < n_quant; i++) {
                sumq += float(q[i]) * values[i];
            }
            sum0 += sumq * scale;
        }

        if (active1) {
            uint offset = (local1 * blocks + block) * 34u;
            float scale = float(*(device const half*)(weight1 + offset));
            float sumq = 0.0f;
            device const char* q = (device const char*)(weight1 + offset + 2u + il * n_quant);
            for (uint i = 0u; i < n_quant; i++) {
                sumq += float(q[i]) * values[i];
            }
            sum1 += sumq * scale;
        }
    }

    sum0 = simd_sum(sum0);
    sum1 = simd_sum(sum1);
    if (lane == 0) {
        partial0[simdgroup] = sum0;
        partial1[simdgroup] = sum1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup == 0) {
        float total0 = lane < GPTOSS_Q8_PAIR_SIMDGROUPS ? partial0[lane] : 0.0f;
        float total1 = lane < GPTOSS_Q8_PAIR_SIMDGROUPS ? partial1[lane] : 0.0f;
        total0 = simd_sum(total0);
        total1 = simd_sum(total1);
        if (lane == 0) {
            if (active0) {
                out0[local0] = total0 + bias0[local0];
            }
            if (active1) {
                out1[local1] = total1 + bias1[local1];
            }
        }
    }
}

kernel void q8_0_matvec_add_batch(
    device const uchar* weight [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device const float* residual [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    constant uint& cols [[buffer(6)]],
    constant uint& batch_rows [[buffer(7)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]
) {
    threadgroup float scratch[256];
    uint out_row = group.x;
    uint batch_row = group.y;
    if (out_row >= rows || batch_row >= batch_rows) {
        return;
    }

    uint input_start = batch_row * cols;
    float sum = 0.0f;
    for (uint col = tid; col < cols; col += 256u) {
        sum += q8_0_to_float(weight, out_row, col, cols) * input[input_start + col];
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
        uint out_index = batch_row * rows + out_row;
        out[out_index] = residual[out_index] + scratch[0] + bias[out_row];
    }
}

kernel void q8_0_qkv_matvec_batch(
    device const uchar* q_weight [[buffer(0)]],
    device const uchar* k_weight [[buffer(1)]],
    device const uchar* v_weight [[buffer(2)]],
    device const float* input [[buffer(3)]],
    device const float* q_bias [[buffer(4)]],
    device const float* k_bias [[buffer(5)]],
    device const float* v_bias [[buffer(6)]],
    device float* q_out [[buffer(7)]],
    device float* k_out [[buffer(8)]],
    device float* v_out [[buffer(9)]],
    constant uint& q_rows [[buffer(10)]],
    constant uint& kv_rows [[buffer(11)]],
    constant uint& cols [[buffer(12)]],
    constant uint& batch_rows [[buffer(13)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]
) {
    threadgroup float scratch[256];

    uint row = group.x;
    uint batch_row = group.y;
    if (batch_row >= batch_rows) {
        return;
    }

    device const uchar* weight = q_weight;
    device const float* bias = q_bias;
    device float* out = q_out;
    uint local_row = row;
    uint row_limit = q_rows;
    if (row >= q_rows && row < q_rows + kv_rows) {
        weight = k_weight;
        bias = k_bias;
        out = k_out;
        local_row = row - q_rows;
        row_limit = kv_rows;
    } else if (row >= q_rows + kv_rows) {
        weight = v_weight;
        bias = v_bias;
        out = v_out;
        local_row = row - q_rows - kv_rows;
        row_limit = kv_rows;
    }

    float sum = 0.0f;
    if (local_row < row_limit) {
        uint input_start = batch_row * cols;
        for (uint col = tid; col < cols; col += 256u) {
            sum += q8_0_to_float(weight, local_row, col, cols) * input[input_start + col];
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

    if (tid == 0u && local_row < row_limit) {
        out[batch_row * row_limit + local_row] = scratch[0] + bias[local_row];
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
    device uint* sample_result [[buffer(4)]],
    constant uint& blocks [[buffer(5)]],
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
        sample_result[0] = indices[0];
        sample_result[1] = as_type<uint>(values[0]);
        sample_result[2] = indices[0] == 0xffffffffu ? 1u : 0u;
        sample_result[3] = 0u;
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

kernel void qk_rope_write_cache(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device float* q_out [[buffer(3)]],
    device float* k_cache [[buffer(4)]],
    device float* v_cache [[buffer(5)]],
    constant uint& position [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < 4096u) {
        uint head_offset = (gid / 64u) * 64u;
        uint dim = gid % 64u;
        uint pair_dim = dim % 32u;
        float theta = float(position) * rope_inv_freq(pair_dim);
        float c = cos(theta) * rope_concentration();
        float s = sin(theta) * rope_concentration();

        if (dim < 32u) {
            float x1 = q[head_offset + dim];
            float x2 = q[head_offset + dim + 32u];
            q_out[gid] = x1 * c - x2 * s;
        } else {
            float x1 = q[head_offset + dim - 32u];
            float x2 = q[head_offset + dim];
            q_out[gid] = x2 * c + x1 * s;
        }
    }

    if (gid < 512u) {
        uint head_offset = (gid / 64u) * 64u;
        uint dim = gid % 64u;
        uint pair_dim = dim % 32u;
        float theta = float(position) * rope_inv_freq(pair_dim);
        float c = cos(theta) * rope_concentration();
        float s = sin(theta) * rope_concentration();
        float k_value;

        if (dim < 32u) {
            float x1 = k[head_offset + dim];
            float x2 = k[head_offset + dim + 32u];
            k_value = x1 * c - x2 * s;
        } else {
            float x1 = k[head_offset + dim - 32u];
            float x2 = k[head_offset + dim];
            k_value = x2 * c + x1 * s;
        }

        uint cache_offset = position * 512u + gid;
        k_cache[cache_offset] = k_value;
        v_cache[cache_offset] = v[gid];
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
        float local_max = -3.402823466e+38f;
        float local_denom = 0.0f;
        float values[64];
        for (uint dim = 0u; dim < 64u; dim++) {
            values[dim] = 0.0f;
        }

        for (uint key_offset = tid; key_offset < key_count; key_offset += GPTOSS_DECODE_ATTENTION_THREADS) {
            uint key_position = effective_key_start + key_offset;
            uint cache_offset = key_position - cache_start_position;
            uint k_start = cache_offset * 512u + kv_start;
            uint v_start = cache_offset * 512u + kv_start;
            float sum = 0.0f;
            for (uint dim = 0u; dim < 64u; dim++) {
                sum += q[q_start + dim] * k_cache[k_start + dim];
            }
            float score = sum * 0.125f;
            float next_max = max(local_max, score);
            float old_scale = exp(local_max - next_max);
            float new_scale = exp(score - next_max);
            for (uint dim = 0u; dim < 64u; dim++) {
                values[dim] = values[dim] * old_scale + v_cache[v_start + dim] * new_scale;
            }
            local_denom = local_denom * old_scale + new_scale;
            local_max = next_max;
        }

        scratch[tid] = local_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = GPTOSS_DECODE_ATTENTION_THREADS / 2u; stride > 0u; stride >>= 1u) {
            if (tid < stride) {
                scratch[tid] = max(scratch[tid], scratch[tid + stride]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (tid == 0u) {
            norm[0] = max(sinks[head], scratch[0]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float max_value = norm[0];
        float local_scale = local_denom > 0.0f ? exp(local_max - max_value) : 0.0f;
        scratch[tid] = local_denom * local_scale;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = GPTOSS_DECODE_ATTENTION_THREADS / 2u; stride > 0u; stride >>= 1u) {
            if (tid < stride) {
                scratch[tid] += scratch[tid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (tid == 0u) {
            norm[1] = exp(sinks[head] - max_value) + scratch[0];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint dim = 0u; dim < 64u; dim++) {
            scratch[tid] = values[dim] * local_scale;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint stride = GPTOSS_DECODE_ATTENTION_THREADS / 2u; stride > 0u; stride >>= 1u) {
                if (tid < stride) {
                    scratch[tid] += scratch[tid + stride];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            if (tid == 0u) {
                out[q_start + dim] = scratch[0] / norm[1];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        return;
    }

    for (uint key_offset = 0u; key_offset < key_count; key_offset++) {
        uint key_position = effective_key_start + key_offset;
        uint cache_offset = key_position - cache_start_position;
        uint k_start = cache_offset * 512u + kv_start;

        float sum = 0.0f;
        for (uint dim = tid; dim < 64u; dim += GPTOSS_DECODE_ATTENTION_THREADS) {
            sum += q[q_start + dim] * k_cache[k_start + dim];
        }
        scratch[tid] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = GPTOSS_DECODE_ATTENTION_THREADS / 2u; stride > 0u; stride >>= 1u) {
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

    for (uint dim = tid; dim < 64u; dim += GPTOSS_DECODE_ATTENTION_THREADS) {
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

kernel void copy_f32_slot(
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

kernel void mxfp4_gguf_top4_gate_swiglu(
    device const uchar* gate_weight [[buffer(0)]],
    device const uchar* up_weight [[buffer(1)]],
    device const float* gate_bias [[buffer(2)]],
    device const float* up_bias [[buffer(3)]],
    device const float* input [[buffer(4)]],
    device const uint* top_indices [[buffer(5)]],
    device float* out [[buffer(6)]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simdgroup [[simdgroup_index_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]]
) {
    // Experts carousel hot path: llama.cpp-style MXFP4 matvec geometry.
    // One 64-thread group has two simdgroups; each simdgroup computes two
    // adjacent output rows, so each threadgroup produces four rows for one
    // selected expert slot.
    uint first_row = group.x * 4u + uint(simdgroup) * 2u;
    uint slot = group.y;
    uint expert = top_indices[slot];
    bool active = slot < 4u && expert < GPTOSS_EXPERTS;

    uint ix = uint(lane) / 2u;
    uint it = uint(lane) - ix * 2u;
    uint bytes_per_row = GPTOSS_MXFP4_GROUPS * 17u;

    float gate_sum0 = 0.0f;
    float gate_sum1 = 0.0f;
    float up_sum0 = 0.0f;
    float up_sum1 = 0.0f;

    if (active) {
        for (uint group_index = ix; group_index < GPTOSS_MXFP4_GROUPS; group_index += 16u) {
            device const float4* input4 =
                (device const float4*)(input + group_index * 32u + it * 8u);
            float4 low0 = input4[0];
            float4 high0 = input4[4];
            float4 low1 = input4[1];
            float4 high1 = input4[5];

            for (uint row_offset = 0u; row_offset < 2u; row_offset++) {
                uint row = first_row + row_offset;
                if (row >= GPTOSS_HIDDEN_SIZE) {
                    continue;
                }

                uint row_start = (expert * GPTOSS_HIDDEN_SIZE + row) * bytes_per_row;
                uint packed_start = row_start + group_index * 17u + 1u + 8u * it;

                device const uchar* gate_q = gate_weight + packed_start;
                float gate_scale = mxfp4_scale(gate_weight[row_start + group_index * 17u]);
                float4 gate_acc0 = low0 * float4(
                    GPTOSS_FP4_LUT[gate_q[0] & 0x0f],
                    GPTOSS_FP4_LUT[gate_q[1] & 0x0f],
                    GPTOSS_FP4_LUT[gate_q[2] & 0x0f],
                    GPTOSS_FP4_LUT[gate_q[3] & 0x0f]);
                float4 gate_acc1 = high0 * float4(
                    GPTOSS_FP4_LUT[gate_q[0] >> 4],
                    GPTOSS_FP4_LUT[gate_q[1] >> 4],
                    GPTOSS_FP4_LUT[gate_q[2] >> 4],
                    GPTOSS_FP4_LUT[gate_q[3] >> 4]);
                float4 gate_acc2 = low1 * float4(
                    GPTOSS_FP4_LUT[gate_q[4] & 0x0f],
                    GPTOSS_FP4_LUT[gate_q[5] & 0x0f],
                    GPTOSS_FP4_LUT[gate_q[6] & 0x0f],
                    GPTOSS_FP4_LUT[gate_q[7] & 0x0f]);
                float4 gate_acc3 = high1 * float4(
                    GPTOSS_FP4_LUT[gate_q[4] >> 4],
                    GPTOSS_FP4_LUT[gate_q[5] >> 4],
                    GPTOSS_FP4_LUT[gate_q[6] >> 4],
                    GPTOSS_FP4_LUT[gate_q[7] >> 4]);
                float4 gate_acc = (gate_acc0 + gate_acc2) + (gate_acc1 + gate_acc3);
                float gate_dot = gate_scale
                    * ((gate_acc[0] + gate_acc[1]) + (gate_acc[2] + gate_acc[3]));

                device const uchar* up_q = up_weight + packed_start;
                float up_scale = mxfp4_scale(up_weight[row_start + group_index * 17u]);
                float4 up_acc0 = low0 * float4(
                    GPTOSS_FP4_LUT[up_q[0] & 0x0f],
                    GPTOSS_FP4_LUT[up_q[1] & 0x0f],
                    GPTOSS_FP4_LUT[up_q[2] & 0x0f],
                    GPTOSS_FP4_LUT[up_q[3] & 0x0f]);
                float4 up_acc1 = high0 * float4(
                    GPTOSS_FP4_LUT[up_q[0] >> 4],
                    GPTOSS_FP4_LUT[up_q[1] >> 4],
                    GPTOSS_FP4_LUT[up_q[2] >> 4],
                    GPTOSS_FP4_LUT[up_q[3] >> 4]);
                float4 up_acc2 = low1 * float4(
                    GPTOSS_FP4_LUT[up_q[4] & 0x0f],
                    GPTOSS_FP4_LUT[up_q[5] & 0x0f],
                    GPTOSS_FP4_LUT[up_q[6] & 0x0f],
                    GPTOSS_FP4_LUT[up_q[7] & 0x0f]);
                float4 up_acc3 = high1 * float4(
                    GPTOSS_FP4_LUT[up_q[4] >> 4],
                    GPTOSS_FP4_LUT[up_q[5] >> 4],
                    GPTOSS_FP4_LUT[up_q[6] >> 4],
                    GPTOSS_FP4_LUT[up_q[7] >> 4]);
                float4 up_acc = (up_acc0 + up_acc2) + (up_acc1 + up_acc3);
                float up_dot =
                    up_scale * ((up_acc[0] + up_acc[1]) + (up_acc[2] + up_acc[3]));

                if (row_offset == 0u) {
                    gate_sum0 += gate_dot;
                    up_sum0 += up_dot;
                } else {
                    gate_sum1 += gate_dot;
                    up_sum1 += up_dot;
                }
            }
        }
    }

    gate_sum0 = simd_sum(gate_sum0);
    gate_sum1 = simd_sum(gate_sum1);
    up_sum0 = simd_sum(up_sum0);
    up_sum1 = simd_sum(up_sum1);

    if (lane == 0 && active) {
        for (uint row_offset = 0u; row_offset < 2u; row_offset++) {
            uint row = first_row + row_offset;
            if (row >= GPTOSS_HIDDEN_SIZE) {
                continue;
            }
            uint bias_start = expert * GPTOSS_HIDDEN_SIZE + row;
            float gate = (row_offset == 0u ? gate_sum0 : gate_sum1) + gate_bias[bias_start];
            float up = (row_offset == 0u ? up_sum0 : up_sum1) + up_bias[bias_start];
            float x_glu = min(gate, 7.0f);
            float x_linear = clamp(up, -7.0f, 7.0f);
            float out_glu = x_glu / (1.0f + exp(-1.702f * x_glu));
            out[slot * GPTOSS_HIDDEN_SIZE + row] = out_glu * (x_linear + 1.0f);
        }
    }
}

kernel void mxfp4_gguf_top4_down_weighted(
    device const uchar* down_weight [[buffer(0)]],
    device const float* down_bias [[buffer(1)]],
    device const float* expert_acts [[buffer(2)]],
    device const uint* top_indices [[buffer(3)]],
    device const float* top_weights [[buffer(4)]],
    device const float* residual [[buffer(5)]],
    device float* out [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]]
) {
    threadgroup float scratch[256];
    threadgroup float selected_scales[360];
    threadgroup float fp4_values[16];
    float sum = 0.0f;
    uint groups = GPTOSS_MXFP4_GROUPS;
    uint bytes_per_row = groups * 17u;

    if (tid < 16u) {
        fp4_values[tid] = GPTOSS_FP4_LUT[tid];
    }
    for (uint scale_index = tid; scale_index < 4u * groups; scale_index += 256u) {
        uint slot = scale_index / groups;
        uint group_index = scale_index - slot * groups;
        uint expert = top_indices[slot];
        if (row < GPTOSS_HIDDEN_SIZE && expert < GPTOSS_EXPERTS) {
            uint row_start = (expert * GPTOSS_HIDDEN_SIZE + row) * bytes_per_row;
            selected_scales[scale_index] = mxfp4_scale(down_weight[row_start + group_index * 17u]);
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
            uint row_start = (expert * GPTOSS_HIDDEN_SIZE + row) * bytes_per_row;
            uint input_slot_start = slot * GPTOSS_HIDDEN_SIZE;

            for (uint packed_index = tid; packed_index < groups * 16u; packed_index += 256u) {
                uint group_index = packed_index / 16u;
                uint byte_in_group = packed_index - group_index * 16u;
                uint packed_offset = group_index * 17u + 1u + byte_in_group;
                uchar packed = down_weight[row_start + packed_offset];
                float scale = selected_scales[slot * groups + group_index];
                uint input_start = input_slot_start + group_index * 32u;
                sum += weight * fp4_values[uint(packed & 0x0fu)] * scale * expert_acts[input_start + byte_in_group];
                sum += weight * fp4_values[uint(packed >> 4)] * scale * expert_acts[input_start + 16u + byte_in_group];
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
                bias_sum += top_weights[slot] * down_bias[expert * GPTOSS_HIDDEN_SIZE + row];
            }
        }
        out[row] = residual[row] + scratch[0] + bias_sum;
    }
}

kernel void mxfp4_gguf_top4_down_slots(
    device const uchar* down_weight [[buffer(0)]],
    device const float* down_bias [[buffer(1)]],
    device const float* expert_acts [[buffer(2)]],
    device const uint* top_indices [[buffer(3)]],
    device float* out [[buffer(4)]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simdgroup [[simdgroup_index_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]]
) {
    uint first_row = group.x * 4u + uint(simdgroup) * 2u;
    uint slot = group.y;
    uint expert = top_indices[slot];
    bool active = slot < 4u && expert < GPTOSS_EXPERTS;

    uint ix = uint(lane) / 2u;
    uint it = uint(lane) - ix * 2u;
    uint bytes_per_row = GPTOSS_MXFP4_GROUPS * 17u;
    uint input_slot_start = slot * GPTOSS_HIDDEN_SIZE;

    float sum0 = 0.0f;
    float sum1 = 0.0f;

    if (active) {
        for (uint group_index = ix; group_index < GPTOSS_MXFP4_GROUPS; group_index += 16u) {
            device const float4* input4 =
                (device const float4*)(expert_acts + input_slot_start + group_index * 32u + it * 8u);
            float4 low0 = input4[0];
            float4 high0 = input4[4];
            float4 low1 = input4[1];
            float4 high1 = input4[5];

            for (uint row_offset = 0u; row_offset < 2u; row_offset++) {
                uint row = first_row + row_offset;
                if (row >= GPTOSS_HIDDEN_SIZE) {
                    continue;
                }

                uint row_start = (expert * GPTOSS_HIDDEN_SIZE + row) * bytes_per_row;
                uint packed_start = row_start + group_index * 17u + 1u + 8u * it;
                device const uchar* q = down_weight + packed_start;
                float scale = mxfp4_scale(down_weight[row_start + group_index * 17u]);

                float4 acc0 = low0 * float4(
                    GPTOSS_FP4_LUT[q[0] & 0x0f],
                    GPTOSS_FP4_LUT[q[1] & 0x0f],
                    GPTOSS_FP4_LUT[q[2] & 0x0f],
                    GPTOSS_FP4_LUT[q[3] & 0x0f]);
                float4 acc1 = high0 * float4(
                    GPTOSS_FP4_LUT[q[0] >> 4],
                    GPTOSS_FP4_LUT[q[1] >> 4],
                    GPTOSS_FP4_LUT[q[2] >> 4],
                    GPTOSS_FP4_LUT[q[3] >> 4]);
                float4 acc2 = low1 * float4(
                    GPTOSS_FP4_LUT[q[4] & 0x0f],
                    GPTOSS_FP4_LUT[q[5] & 0x0f],
                    GPTOSS_FP4_LUT[q[6] & 0x0f],
                    GPTOSS_FP4_LUT[q[7] & 0x0f]);
                float4 acc3 = high1 * float4(
                    GPTOSS_FP4_LUT[q[4] >> 4],
                    GPTOSS_FP4_LUT[q[5] >> 4],
                    GPTOSS_FP4_LUT[q[6] >> 4],
                    GPTOSS_FP4_LUT[q[7] >> 4]);
                float4 acc = (acc0 + acc2) + (acc1 + acc3);
                float dot = scale * ((acc[0] + acc[1]) + (acc[2] + acc[3]));

                if (row_offset == 0u) {
                    sum0 += dot;
                } else {
                    sum1 += dot;
                }
            }
        }
    }

    sum0 = simd_sum(sum0);
    sum1 = simd_sum(sum1);

    if (lane == 0 && active) {
        for (uint row_offset = 0u; row_offset < 2u; row_offset++) {
            uint row = first_row + row_offset;
            if (row >= GPTOSS_HIDDEN_SIZE) {
                continue;
            }
            float dot = row_offset == 0u ? sum0 : sum1;
            out[slot * GPTOSS_HIDDEN_SIZE + row] =
                dot + down_bias[expert * GPTOSS_HIDDEN_SIZE + row];
        }
    }
}

kernel void weighted_sum4_residual(
    device const float* vectors [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const float* residual [[buffer(2)]],
    device float* out [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < GPTOSS_HIDDEN_SIZE) {
        out[gid] =
            residual[gid] +
            vectors[gid] * weights[0] +
            vectors[GPTOSS_HIDDEN_SIZE + gid] * weights[1] +
            vectors[GPTOSS_HIDDEN_SIZE * 2u + gid] * weights[2] +
            vectors[GPTOSS_HIDDEN_SIZE * 3u + gid] * weights[3];
    }
}

kernel void mxfp4_gguf_top4_gate_swiglu_batch(
    device const uchar* gate_weight [[buffer(0)]],
    device const uchar* up_weight [[buffer(1)]],
    device const float* gate_bias [[buffer(2)]],
    device const float* up_bias [[buffer(3)]],
    device const float* input [[buffer(4)]],
    device const uint* top_indices [[buffer(5)]],
    device float* out [[buffer(6)]],
    constant uint& row_offset [[buffer(7)]],
    constant uint& rows [[buffer(8)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]
) {
    threadgroup float gate_scratch[256];
    threadgroup float up_scratch[256];
    threadgroup float gate_scales[90];
    threadgroup float up_scales[90];

    uint row = group.x;
    uint slot = group.y;
    uint batch_row = group.z;
    uint source_row = row_offset + batch_row;
    uint top_start = source_row * 4u;
    uint expert = top_indices[top_start + slot];
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    bool active = row < GPTOSS_HIDDEN_SIZE && slot < 4u && batch_row < rows && expert < GPTOSS_EXPERTS;

    uint groups = GPTOSS_MXFP4_GROUPS;
    uint bytes_per_row = groups * 17u;
    uint gate_row_start = (expert * GPTOSS_HIDDEN_SIZE + row) * bytes_per_row;
    uint up_row_start = (expert * GPTOSS_HIDDEN_SIZE + row) * bytes_per_row;
    uint input_row_start = source_row * GPTOSS_HIDDEN_SIZE;

    if (active && tid < groups) {
        gate_scales[tid] = mxfp4_scale(gate_weight[gate_row_start + tid * 17u]);
        up_scales[tid] = mxfp4_scale(up_weight[up_row_start + tid * 17u]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (active) {
        for (uint packed_index = tid; packed_index < groups * 16u; packed_index += 256u) {
            uint group_index = packed_index / 16u;
            uint byte_in_group = packed_index - group_index * 16u;
            uint packed_offset = group_index * 17u + 1u + byte_in_group;
            uint input_start = input_row_start + group_index * 32u;

            uchar gate_packed = gate_weight[gate_row_start + packed_offset];
            float gate_scale = gate_scales[group_index];
            gate_sum += fp4_to_float(uint(gate_packed & 0x0fu)) * gate_scale * input[input_start + byte_in_group];
            gate_sum += fp4_to_float(uint(gate_packed >> 4)) * gate_scale * input[input_start + 16u + byte_in_group];

            uchar up_packed = up_weight[up_row_start + packed_offset];
            float up_scale = up_scales[group_index];
            up_sum += fp4_to_float(uint(up_packed & 0x0fu)) * up_scale * input[input_start + byte_in_group];
            up_sum += fp4_to_float(uint(up_packed >> 4)) * up_scale * input[input_start + 16u + byte_in_group];
        }
    }

    gate_scratch[tid] = gate_sum;
    up_scratch[tid] = up_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            gate_scratch[tid] += gate_scratch[tid + stride];
            up_scratch[tid] += up_scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u && active) {
        uint bias_start = expert * GPTOSS_HIDDEN_SIZE + row;
        float gate = gate_scratch[0] + gate_bias[bias_start];
        float up = up_scratch[0] + up_bias[bias_start];
        float x_glu = min(gate, 7.0f);
        float x_linear = clamp(up, -7.0f, 7.0f);
        float out_glu = x_glu / (1.0f + exp(-1.702f * x_glu));
        uint out_start = (batch_row * 4u + slot) * GPTOSS_HIDDEN_SIZE;
        out[out_start + row] = out_glu * (x_linear + 1.0f);
    }
}

kernel void mxfp4_gguf_top4_down_weighted_batch(
    device const uchar* down_weight [[buffer(0)]],
    device const float* down_bias [[buffer(1)]],
    device const float* expert_acts [[buffer(2)]],
    device const uint* top_indices [[buffer(3)]],
    device const float* top_weights [[buffer(4)]],
    device const float* residual [[buffer(5)]],
    device float* out [[buffer(6)]],
    constant uint& row_offset [[buffer(7)]],
    constant uint& rows [[buffer(8)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]]
) {
    threadgroup float scratch[256];
    threadgroup float selected_scales[360];
    uint row = group.x;
    uint batch_row = group.y;
    uint source_row = row_offset + batch_row;
    float sum = 0.0f;
    uint groups = GPTOSS_MXFP4_GROUPS;
    uint bytes_per_row = groups * 17u;
    uint top_start = source_row * 4u;

    for (uint scale_index = tid; scale_index < 4u * groups; scale_index += 256u) {
        uint scale_slot = scale_index / groups;
        uint scale_group = scale_index - scale_slot * groups;
        uint expert = top_indices[top_start + scale_slot];
        if (row < GPTOSS_HIDDEN_SIZE && batch_row < rows && expert < GPTOSS_EXPERTS) {
            uint row_start = (expert * GPTOSS_HIDDEN_SIZE + row) * bytes_per_row;
            selected_scales[scale_index] = mxfp4_scale(down_weight[row_start + scale_group * 17u]);
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
            uint row_start = (expert * GPTOSS_HIDDEN_SIZE + row) * bytes_per_row;
            uint input_slot_start = (batch_row * 4u + slot) * GPTOSS_HIDDEN_SIZE;

            for (uint packed_index = tid; packed_index < groups * 16u; packed_index += 256u) {
                uint group_index = packed_index / 16u;
                uint byte_in_group = packed_index - group_index * 16u;
                uint packed_offset = group_index * 17u + 1u + byte_in_group;
                uchar packed = down_weight[row_start + packed_offset];
                float scale = selected_scales[slot * groups + group_index];
                uint input_start = input_slot_start + group_index * 32u;
                sum += weight * fp4_to_float(uint(packed & 0x0fu)) * scale * expert_acts[input_start + byte_in_group];
                sum += weight * fp4_to_float(uint(packed >> 4)) * scale * expert_acts[input_start + 16u + byte_in_group];
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
        float bias_sum = 0.0f;
        for (uint slot = 0u; slot < 4u; slot++) {
            uint expert = top_indices[top_start + slot];
            if (expert < GPTOSS_EXPERTS) {
                bias_sum += top_weights[top_start + slot] * down_bias[expert * GPTOSS_HIDDEN_SIZE + row];
            }
        }
        uint hidden_index = source_row * GPTOSS_HIDDEN_SIZE + row;
        out[hidden_index] = residual[hidden_index] + scratch[0] + bias_sum;
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
