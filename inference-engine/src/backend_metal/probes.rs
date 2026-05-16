use eyre::{Result, eyre};

use super::{
    GATE_UP_VALUES, HIDDEN_SIZE, KV_HEADS, LAYERS, MAX_KV_CACHE_PROBE_TOKENS,
    MAX_PREFILL_PROBE_TOKENS, MXFP4_BYTES_PER_GROUP, MXFP4_GROUPS, MetalRuntime, Q_HEADS,
    profile::ProfileDelta,
    weights::{bf16_linear_profile_name, mxfp4_profile_name},
};
use crate::backend_cpu;
use crate::harmony_adapter::HarmonyAdapter;
use crate::model_store::{self, SourceModelReport};
use crate::runtime_core::kv_cache::{KvCachePlan, PlannedKvCache};
use crate::runtime_core::sampler::{SampleCandidate, Sampler};
use crate::runtime_core::{
    ExpertScore, GenerationReport, GreedyTextProbeReport, GreedyTokenReport, LmHeadTopKProbeReport,
    MetalMatvecProbeReport, MetalRmsNormProbeReport, MetalSelectedLogitsProbeReport,
    MetalTopKProbeReport, MetalVectorProbeReport, SamplingConfig, SelectedLogit, StopReason,
};
use std::mem::size_of;
use std::sync::Arc;
#[cfg(feature = "profile")]
use std::time::Instant;

impl MetalRuntime {
    fn rms_norm_profiled(&self, name: &str, x: &[f32], weight: &[f32]) -> Result<Vec<f32>> {
        let groups = x.len().div_ceil(256);
        let stats = ProfileDelta {
            command_buffers: 2,
            upload_bytes: (x.len() + weight.len()) * size_of::<f32>()
                + size_of::<u32>()
                + size_of::<f32>(),
            readback_bytes: (groups + x.len()) * size_of::<f32>(),
            ..ProfileDelta::default()
        };
        self.profile_op(name, stats, || self.platform.rms_norm(x, weight))
    }

    fn rope_row_profiled(&self, row: &[f32], heads: usize, position: usize) -> Result<Vec<f32>> {
        let _ = (heads, position);
        let stats = ProfileDelta {
            command_buffers: 1,
            upload_bytes: row.len() * size_of::<f32>() + 2 * size_of::<u32>(),
            readback_bytes: row.len() * size_of::<f32>(),
            ..ProfileDelta::default()
        };
        self.profile_op("op.rope", stats, || {
            self.platform.rope_row(row, heads, position)
        })
    }

    fn single_token_attention_profiled(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        sinks: &[f32],
    ) -> Result<Vec<f32>> {
        let stats = ProfileDelta {
            command_buffers: 1,
            upload_bytes: (q.len() + k.len() + v.len() + sinks.len()) * size_of::<f32>(),
            readback_bytes: q.len() * size_of::<f32>(),
            ..ProfileDelta::default()
        };
        self.profile_op("op.attention.single_token", stats, || {
            self.platform.single_token_attention(q, k, v, sinks)
        })
    }

    fn kv_cache_decode_attention_profiled(
        &self,
        layer: usize,
        query_position: usize,
        cache_start_position: usize,
        q: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        sinks: &[f32],
    ) -> Result<Vec<f32>> {
        let stats = ProfileDelta {
            command_buffers: 1,
            upload_bytes: (q.len() + k_cache.len() + v_cache.len() + sinks.len())
                * size_of::<f32>()
                + 4 * size_of::<u32>(),
            readback_bytes: q.len() * size_of::<f32>(),
            ..ProfileDelta::default()
        };
        self.profile_op("op.attention.kv_decode", stats, || {
            self.platform.kv_cache_decode_attention(
                layer,
                query_position,
                cache_start_position,
                q,
                k_cache,
                v_cache,
                sinks,
            )
        })
    }

    fn vector_add_profiled(&self, name: &str, left: &[f32], right: &[f32]) -> Result<Vec<f32>> {
        let stats = ProfileDelta {
            command_buffers: 1,
            upload_bytes: (left.len() + right.len()) * size_of::<f32>() + size_of::<u32>(),
            readback_bytes: left.len() * size_of::<f32>(),
            ..ProfileDelta::default()
        };
        self.profile_op(name, stats, || self.platform.vector_add(left, right))
    }

    fn top4_softmax_profiled(&self, logits: &[f32]) -> Result<Vec<ExpertScore>> {
        let stats = ProfileDelta {
            command_buffers: 1,
            upload_bytes: logits.len() * size_of::<f32>() + size_of::<u32>(),
            readback_bytes: 4 * (size_of::<u32>() + 2 * size_of::<f32>()),
            ..ProfileDelta::default()
        };
        self.profile_op("op.router.top4", stats, || {
            self.platform.top4_softmax(logits)
        })
    }

    fn swiglu_profiled(&self, values: &[f32]) -> Result<Vec<f32>> {
        let stats = ProfileDelta {
            command_buffers: 1,
            upload_bytes: values.len() * size_of::<f32>() + size_of::<u32>(),
            readback_bytes: (values.len() / 2) * size_of::<f32>(),
            ..ProfileDelta::default()
        };
        self.profile_op("op.swiglu", stats, || self.platform.swiglu(values))
    }

    fn weighted_sum4_profiled(&self, vectors: [&[f32]; 4], weights: [f32; 4]) -> Result<Vec<f32>> {
        let n = vectors[0].len();
        let stats = ProfileDelta {
            command_buffers: 1,
            upload_bytes: n * 4 * size_of::<f32>()
                + weights.len() * size_of::<f32>()
                + size_of::<u32>(),
            readback_bytes: n * size_of::<f32>(),
            ..ProfileDelta::default()
        };
        self.profile_op("op.moe.weighted_sum4", stats, || {
            self.platform.weighted_sum4(vectors, weights)
        })
    }

    fn bf16_matrix(
        &self,
        report: &SourceModelReport,
        tensor_name: &str,
    ) -> Result<Arc<model_store::Bf16Matrix>> {
        let Some(weights) = &self.weights else {
            return Ok(Arc::new(model_store::read_bf16_matrix(
                report,
                tensor_name,
            )?));
        };
        weights.bf16_matrix(report, tensor_name)
    }

    fn bf16_vector(&self, report: &SourceModelReport, tensor_name: &str) -> Result<Arc<Vec<f32>>> {
        let Some(weights) = &self.weights else {
            return Ok(Arc::new(model_store::read_bf16_vector(
                report,
                tensor_name,
            )?));
        };
        weights.bf16_vector(report, tensor_name)
    }

    fn bf16_matrix_row(
        &self,
        report: &SourceModelReport,
        tensor_name: &str,
        row: usize,
    ) -> Result<Arc<Vec<f32>>> {
        let Some(weights) = &self.weights else {
            return Ok(Arc::new(model_store::read_bf16_matrix_row(
                report,
                tensor_name,
                row,
            )?));
        };
        weights.bf16_matrix_row(report, tensor_name, row)
    }

    fn u8_tensor_slice(
        &self,
        report: &SourceModelReport,
        tensor_name: &str,
        element_offset: usize,
        element_len: usize,
    ) -> Result<Arc<Vec<u8>>> {
        let Some(weights) = &self.weights else {
            return Ok(Arc::new(model_store::read_u8_tensor_slice(
                report,
                tensor_name,
                element_offset,
                element_len,
            )?));
        };
        weights.u8_tensor_slice(report, tensor_name, element_offset, element_len)
    }

    fn bf16_linear_matvec(
        &self,
        report: &SourceModelReport,
        weight_name: &str,
        bias_name: &str,
        input: &[f32],
    ) -> Result<Vec<f32>> {
        let op_name = bf16_linear_profile_name(weight_name);
        self.profile_op(&op_name, ProfileDelta::default(), || {
            self.bf16_linear_matvec_inner(report, weight_name, bias_name, input, &op_name)
        })
    }

    fn bf16_linear_matvec_inner(
        &self,
        report: &SourceModelReport,
        weight_name: &str,
        bias_name: &str,
        input: &[f32],
        op_name: &str,
    ) -> Result<Vec<f32>> {
        if self.weights.is_none() {
            let bias = self.bf16_vector(report, bias_name)?;
            let weight = self.bf16_matrix(report, weight_name)?;
            return self.platform.bf16_matvec(
                &weight.values,
                weight.rows,
                weight.cols,
                input,
                &bias,
            );
        }

        let mut cache = self.gpu_bf16_matrices.lock().unwrap();
        if !cache.contains_key(weight_name) {
            let weight = self.bf16_matrix(report, weight_name)?;
            self.record_profile(
                op_name,
                ProfileDelta {
                    upload_bytes: weight.values.len() * size_of::<u16>(),
                    cache_misses: 1,
                    ..ProfileDelta::default()
                },
            );
            let weight =
                self.platform
                    .upload_bf16_matrix(&weight.values, weight.rows, weight.cols)?;
            cache.insert(weight_name.to_string(), weight);
        } else {
            self.record_profile(
                op_name,
                ProfileDelta {
                    cache_hits: 1,
                    ..ProfileDelta::default()
                },
            );
        }
        let Some(weight) = cache.get(weight_name) else {
            return Err(eyre!("cached BF16 matrix is missing for {weight_name}"));
        };

        let mut bias_cache = self.gpu_bf16_vectors.lock().unwrap();
        if !bias_cache.contains_key(bias_name) {
            let bias = self.bf16_vector(report, bias_name)?;
            self.record_profile(
                op_name,
                ProfileDelta {
                    upload_bytes: bias.len() * size_of::<f32>(),
                    cache_misses: 1,
                    ..ProfileDelta::default()
                },
            );
            let bias = self.platform.upload_f32_vector(&bias)?;
            bias_cache.insert(bias_name.to_string(), bias);
        } else {
            self.record_profile(
                op_name,
                ProfileDelta {
                    cache_hits: 1,
                    ..ProfileDelta::default()
                },
            );
        }
        let Some(bias) = bias_cache.get(bias_name) else {
            return Err(eyre!("cached BF16 bias is missing for {bias_name}"));
        };
        self.record_profile(
            op_name,
            ProfileDelta {
                command_buffers: 1,
                upload_bytes: input.len() * size_of::<f32>() + 2 * size_of::<u32>(),
                readback_bytes: weight.rows() * size_of::<f32>(),
                ..ProfileDelta::default()
            },
        );
        self.platform.bf16_matrix_matvec(weight, input, bias)
    }

    fn mxfp4_expert_matvec(
        &self,
        report: &SourceModelReport,
        blocks_name: &str,
        scales_name: &str,
        bias_name: &str,
        expert: usize,
        rows: usize,
        input: &[f32],
    ) -> Result<Vec<f32>> {
        let op_name = mxfp4_profile_name(bias_name);
        self.profile_op(&op_name, ProfileDelta::default(), || {
            self.mxfp4_expert_matvec_inner(
                report,
                blocks_name,
                scales_name,
                bias_name,
                expert,
                rows,
                input,
                &op_name,
            )
        })
    }

    fn mxfp4_expert_matvec_inner(
        &self,
        report: &SourceModelReport,
        blocks_name: &str,
        scales_name: &str,
        bias_name: &str,
        expert: usize,
        rows: usize,
        input: &[f32],
        op_name: &str,
    ) -> Result<Vec<f32>> {
        if self.weights.is_none() {
            let blocks = read_mxfp4_expert_blocks_metal(self, report, blocks_name, expert, rows)?;
            let scales = read_mxfp4_expert_scales_metal(self, report, scales_name, expert, rows)?;
            let bias = self.bf16_matrix_row(report, bias_name, expert)?;
            return self
                .platform
                .mxfp4_matvec(&blocks, &scales, rows, input, &bias);
        }
        #[cfg(not(feature = "profile"))]
        let _ = op_name;

        let blocks_per_expert = rows
            .checked_mul(MXFP4_GROUPS)
            .and_then(|value| value.checked_mul(MXFP4_BYTES_PER_GROUP))
            .ok_or_else(|| eyre!("MXFP4 block slice size overflow"))?;
        let blocks_offset = expert
            .checked_mul(blocks_per_expert)
            .ok_or_else(|| eyre!("MXFP4 block offset overflow"))?;
        let scales_per_expert = rows
            .checked_mul(MXFP4_GROUPS)
            .ok_or_else(|| eyre!("MXFP4 scale slice size overflow"))?;
        let scales_offset = expert
            .checked_mul(scales_per_expert)
            .ok_or_else(|| eyre!("MXFP4 scale offset overflow"))?;

        let blocks_key = (blocks_name.to_string(), blocks_offset, blocks_per_expert);
        let scales_key = (scales_name.to_string(), scales_offset, scales_per_expert);
        #[cfg(feature = "profile")]
        let page_started = Instant::now();
        #[cfg(feature = "profile")]
        let mut page_upload_bytes = 0usize;
        #[cfg(feature = "profile")]
        let mut page_missed = false;
        let mut u8_cache = self.gpu_u8_slices.lock().unwrap();
        if !u8_cache.contains_key(&blocks_key) {
            let blocks =
                self.u8_tensor_slice(report, blocks_name, blocks_offset, blocks_per_expert)?;
            #[cfg(feature = "profile")]
            {
                page_upload_bytes = page_upload_bytes.saturating_add(blocks.len());
                page_missed = true;
                self.record_profile(
                    op_name,
                    ProfileDelta {
                        upload_bytes: blocks.len(),
                        cache_misses: 1,
                        ..ProfileDelta::default()
                    },
                );
            }
            let blocks = self.platform.upload_u8_buffer(&blocks)?;
            u8_cache.insert(blocks_key.clone(), blocks);
        } else {
            #[cfg(feature = "profile")]
            self.record_profile(
                op_name,
                ProfileDelta {
                    cache_hits: 1,
                    ..ProfileDelta::default()
                },
            );
        }
        if !u8_cache.contains_key(&scales_key) {
            let scales =
                self.u8_tensor_slice(report, scales_name, scales_offset, scales_per_expert)?;
            #[cfg(feature = "profile")]
            {
                page_upload_bytes = page_upload_bytes.saturating_add(scales.len());
                page_missed = true;
                self.record_profile(
                    op_name,
                    ProfileDelta {
                        upload_bytes: scales.len(),
                        cache_misses: 1,
                        ..ProfileDelta::default()
                    },
                );
            }
            let scales = self.platform.upload_u8_buffer(&scales)?;
            u8_cache.insert(scales_key.clone(), scales);
        } else {
            #[cfg(feature = "profile")]
            self.record_profile(
                op_name,
                ProfileDelta {
                    cache_hits: 1,
                    ..ProfileDelta::default()
                },
            );
        }
        let Some(blocks) = u8_cache.get(&blocks_key) else {
            return Err(eyre!(
                "cached MXFP4 block slice is missing for {blocks_name} expert {expert}"
            ));
        };
        let Some(scales) = u8_cache.get(&scales_key) else {
            return Err(eyre!(
                "cached MXFP4 scale slice is missing for {scales_name} expert {expert}"
            ));
        };

        let bias_key = (bias_name.to_string(), expert);
        let mut row_cache = self.gpu_bf16_rows.lock().unwrap();
        if !row_cache.contains_key(&bias_key) {
            let bias = self.bf16_matrix_row(report, bias_name, expert)?;
            #[cfg(feature = "profile")]
            {
                page_upload_bytes = page_upload_bytes.saturating_add(bias.len() * size_of::<f32>());
                page_missed = true;
                self.record_profile(
                    op_name,
                    ProfileDelta {
                        upload_bytes: bias.len() * size_of::<f32>(),
                        cache_misses: 1,
                        ..ProfileDelta::default()
                    },
                );
            }
            let bias = self.platform.upload_f32_vector(&bias)?;
            row_cache.insert(bias_key.clone(), bias);
        } else {
            #[cfg(feature = "profile")]
            self.record_profile(
                op_name,
                ProfileDelta {
                    cache_hits: 1,
                    ..ProfileDelta::default()
                },
            );
        }
        let Some(bias) = row_cache.get(&bias_key) else {
            return Err(eyre!(
                "cached MXFP4 bias row is missing for {bias_name} expert {expert}"
            ));
        };
        #[cfg(feature = "profile")]
        if page_missed {
            // Experts carousel slow path: the production resident path should
            // normally have the selected expert slabs already warm. A miss here
            // is tracked as carousel page-spill work, not regular decode work.
            self.record_profile(
                "metric.experts_carousel_page",
                ProfileDelta {
                    wall: page_started.elapsed(),
                    upload_bytes: page_upload_bytes,
                    cache_misses: 1,
                    ..ProfileDelta::default()
                },
            );
        }

        #[cfg(feature = "profile")]
        self.record_profile(
            op_name,
            ProfileDelta {
                command_buffers: 1,
                upload_bytes: input.len() * size_of::<f32>() + 2 * size_of::<u32>(),
                readback_bytes: rows * size_of::<f32>(),
                ..ProfileDelta::default()
            },
        );
        self.platform
            .mxfp4_matvec_resident(blocks, scales, rows, input, bias)
    }
}

pub fn probe_rms_norm_embedding(
    ctx: &MetalRuntime,
    report: &SourceModelReport,
    token: u32,
) -> Result<MetalRmsNormProbeReport> {
    let embedding =
        model_store::read_bf16_matrix_row(report, "model.embed_tokens.weight", token as usize)?;
    let weight = model_store::read_bf16_vector(report, "model.layers.0.input_layernorm.weight")?;
    let cpu = backend_cpu::rms_norm_reference(&embedding, &weight)?;
    let metal = ctx.platform.rms_norm(&embedding, &weight)?;
    if cpu.len() != metal.len() {
        return Err(eyre!(
            "Metal RMSNorm returned {} values, CPU returned {}",
            metal.len(),
            cpu.len()
        ));
    }

    let mut max_abs_delta = 0.0f32;
    let mut sum_abs_delta = 0.0f64;
    for (cpu, metal) in cpu.iter().copied().zip(metal.iter().copied()) {
        let delta = (cpu - metal).abs();
        max_abs_delta = max_abs_delta.max(delta);
        sum_abs_delta += delta as f64;
    }
    let mean_abs_delta = if cpu.is_empty() {
        0.0
    } else {
        (sum_abs_delta / cpu.len() as f64) as f32
    };

    Ok(MetalRmsNormProbeReport {
        token,
        values: cpu.len(),
        max_abs_delta,
        mean_abs_delta,
        cpu_first8: cpu.iter().copied().take(8).collect(),
        metal_first8: metal.into_iter().take(8).collect(),
    })
}

pub fn probe_layer0_q_proj(
    ctx: &MetalRuntime,
    report: &SourceModelReport,
    token: u32,
) -> Result<MetalMatvecProbeReport> {
    probe_layer0_projection(ctx, report, token, "q_proj")
}

pub fn probe_layer0_k_proj(
    ctx: &MetalRuntime,
    report: &SourceModelReport,
    token: u32,
) -> Result<MetalMatvecProbeReport> {
    probe_layer0_projection(ctx, report, token, "k_proj")
}

pub fn probe_layer0_v_proj(
    ctx: &MetalRuntime,
    report: &SourceModelReport,
    token: u32,
) -> Result<MetalMatvecProbeReport> {
    probe_layer0_projection(ctx, report, token, "v_proj")
}

fn probe_layer0_projection(
    ctx: &MetalRuntime,
    report: &SourceModelReport,
    token: u32,
    projection: &str,
) -> Result<MetalMatvecProbeReport> {
    let embedding =
        model_store::read_bf16_matrix_row(report, "model.embed_tokens.weight", token as usize)?;
    let norm_weight =
        model_store::read_bf16_vector(report, "model.layers.0.input_layernorm.weight")?;
    let input = backend_cpu::rms_norm_reference(&embedding, &norm_weight)?;

    let weight_name = format!("model.layers.0.self_attn.{projection}.weight");
    let bias_name = format!("model.layers.0.self_attn.{projection}.bias");
    let report_name = format!("layer0.self_attn.{projection}");

    let weight = model_store::read_bf16_matrix(report, &weight_name)?;
    let bias = model_store::read_bf16_vector(report, &bias_name)?;

    let mut cpu = model_store::matvec_bf16(report, &weight_name, &input)?;
    model_store::add_in_place(&mut cpu, &bias, &report_name)?;
    let metal =
        ctx.platform
            .bf16_matvec(&weight.values, weight.rows, weight.cols, &input, &bias)?;

    matvec_report(&report_name, token, weight.rows, weight.cols, &cpu, metal)
}

pub fn probe_layer0_q_rope(
    ctx: &MetalRuntime,
    report: &SourceModelReport,
    token: u32,
    position: usize,
) -> Result<MetalVectorProbeReport> {
    probe_layer0_rope(ctx, report, token, position, "q_proj", Q_HEADS)
}

pub fn probe_layer0_k_rope(
    ctx: &MetalRuntime,
    report: &SourceModelReport,
    token: u32,
    position: usize,
) -> Result<MetalVectorProbeReport> {
    probe_layer0_rope(ctx, report, token, position, "k_proj", KV_HEADS)
}

fn probe_layer0_rope(
    ctx: &MetalRuntime,
    report: &SourceModelReport,
    token: u32,
    position: usize,
    projection: &str,
    heads: usize,
) -> Result<MetalVectorProbeReport> {
    let input = layer0_projection_cpu(report, token, projection)?;
    let cpu = backend_cpu::apply_rope_reference(&input, heads, position)?;
    let metal = ctx.platform.rope_row(&input, heads, position)?;
    vector_report(
        &format!("layer0.self_attn.{projection}.rope"),
        token,
        position,
        &cpu,
        metal,
    )
}

pub fn probe_layer0_single_token_attention(
    ctx: &MetalRuntime,
    report: &SourceModelReport,
    token: u32,
) -> Result<MetalVectorProbeReport> {
    let q = layer0_projection_cpu(report, token, "q_proj")?;
    let k = layer0_projection_cpu(report, token, "k_proj")?;
    let v = layer0_projection_cpu(report, token, "v_proj")?;
    let q = backend_cpu::apply_rope_reference(&q, Q_HEADS, 0)?;
    let k = backend_cpu::apply_rope_reference(&k, KV_HEADS, 0)?;
    let sinks = model_store::read_bf16_vector(report, "model.layers.0.self_attn.sinks")?;
    let cpu = backend_cpu::single_token_attention_from_rope_reference(&q, &k, &v, &sinks)?;
    let metal = ctx.platform.single_token_attention(&q, &k, &v, &sinks)?;
    vector_report(
        "layer0.self_attn.single_token_attention",
        token,
        0,
        &cpu,
        metal,
    )
}

pub fn probe_layer0_sequence_attention(
    ctx: &MetalRuntime,
    report: &SourceModelReport,
    tokens: &[u32],
) -> Result<MetalVectorProbeReport> {
    let Some(token) = tokens.first().copied() else {
        return Err(eyre!("sequence attention probe needs at least one token"));
    };
    if tokens.len() > MAX_PREFILL_PROBE_TOKENS {
        return Err(eyre!(
            "sequence attention probe supports at most {MAX_PREFILL_PROBE_TOKENS} tokens, got {}",
            tokens.len()
        ));
    }

    let (q, k, v) = layer0_rope_qkv_cpu(report, tokens)?;
    let sinks = model_store::read_bf16_vector(report, "model.layers.0.self_attn.sinks")?;
    let cpu = backend_cpu::sequence_attention_from_rope_reference(0, &q, &k, &v, &sinks)?;
    let cpu = flatten_rows(&cpu);
    let q = flatten_rows(&q);
    let k = flatten_rows(&k);
    let v = flatten_rows(&v);
    let metal = ctx
        .platform
        .sequence_attention(0, &q, &k, &v, &sinks, tokens.len())?;
    vector_report(
        &format!(
            "layer0.self_attn.sequence_attention.prefill{}",
            tokens.len()
        ),
        token,
        tokens.len() - 1,
        &cpu,
        metal,
    )
}

pub fn probe_layer0_kv_cache_decode_attention(
    ctx: &MetalRuntime,
    report: &SourceModelReport,
    tokens: &[u32],
) -> Result<MetalVectorProbeReport> {
    let Some(token) = tokens.last().copied() else {
        return Err(eyre!("KV-cache decode probe needs at least one token"));
    };
    if tokens.len() > MAX_PREFILL_PROBE_TOKENS {
        return Err(eyre!(
            "KV-cache decode probe supports at most {MAX_PREFILL_PROBE_TOKENS} tokens, got {}",
            tokens.len()
        ));
    }

    let (q, k, v) = layer0_rope_qkv_cpu(report, tokens)?;
    let sinks = model_store::read_bf16_vector(report, "model.layers.0.self_attn.sinks")?;
    let cpu = backend_cpu::sequence_attention_from_rope_reference(0, &q, &k, &v, &sinks)?;
    let Some(cpu) = cpu.last() else {
        return Err(eyre!("sequence attention returned no rows"));
    };
    let query_position = tokens.len() - 1;
    let mut kv_cache = PlannedKvCache::new(KvCachePlan::gpt_oss_20b(tokens.len())?);
    let inputs = layer0_embedding_rows(report, tokens)?;
    let q = prefill_layer_attention_cache_metal(ctx, report, 0, &inputs, 0, &mut kv_cache)?;
    let view = kv_cache
        .layer(0)?
        .contiguous_view_for_query(query_position)?;
    let metal = ctx.platform.kv_cache_decode_attention(
        0,
        query_position,
        view.start_position,
        &q[query_position],
        &view.k,
        &view.v,
        &sinks,
    )?;
    vector_report(
        &format!(
            "layer0.self_attn.kv_cache_prefill_decode.prefill{}",
            tokens.len()
        ),
        token,
        query_position,
        cpu,
        metal,
    )
}

pub fn probe_kv_cache_window_rollover_attention(
    ctx: &MetalRuntime,
    report: &SourceModelReport,
    tokens: &[u32],
) -> Result<MetalVectorProbeReport> {
    kv_cache_attention_probe(
        ctx,
        report,
        tokens,
        0,
        &format!("kv_cache.window_rollover.layer0.prefill{}", tokens.len()),
    )
}

pub fn probe_kv_cache_dense_accumulation_attention(
    ctx: &MetalRuntime,
    report: &SourceModelReport,
    tokens: &[u32],
) -> Result<MetalVectorProbeReport> {
    kv_cache_attention_probe(
        ctx,
        report,
        tokens,
        1,
        &format!("kv_cache.dense_accumulation.layer1.prefill{}", tokens.len()),
    )
}

pub fn probe_prefill_layers_output(
    ctx: &MetalRuntime,
    report: &SourceModelReport,
    tokens: &[u32],
    layers: usize,
) -> Result<MetalVectorProbeReport> {
    let Some(token) = tokens.last().copied() else {
        return Err(eyre!("prefill layer-output probe needs at least one token"));
    };
    if tokens.len() > MAX_PREFILL_PROBE_TOKENS {
        return Err(eyre!(
            "prefill layer-output probe supports at most {MAX_PREFILL_PROBE_TOKENS} tokens, got {}",
            tokens.len()
        ));
    }
    if layers > LAYERS {
        return Err(eyre!(
            "requested {layers} layers, but gpt-oss-20b has {LAYERS}"
        ));
    }

    let cpu = backend_cpu::sequence_layers_reference(report, tokens, layers)?;
    let cpu = flatten_rows(&cpu);
    let metal = prefill_layers_metal(ctx, report, tokens, layers)?;
    let metal = flatten_rows(&metal);
    vector_report(
        &format!("prefill.layers{layers}.output.tokens{}", tokens.len()),
        token,
        tokens.len() - 1,
        &cpu,
        metal,
    )
}

pub fn probe_prefill_final_norm(
    ctx: &MetalRuntime,
    report: &SourceModelReport,
    tokens: &[u32],
    layers: usize,
) -> Result<MetalVectorProbeReport> {
    let Some(token) = tokens.last().copied() else {
        return Err(eyre!("prefill final-norm probe needs at least one token"));
    };
    if tokens.len() > MAX_PREFILL_PROBE_TOKENS {
        return Err(eyre!(
            "prefill final-norm probe supports at most {MAX_PREFILL_PROBE_TOKENS} tokens, got {}",
            tokens.len()
        ));
    }

    let cpu = prefill_final_norm_cpu(report, tokens, layers)?;
    let metal = prefill_final_norm_metal(ctx, report, tokens, layers)?;
    vector_report(
        &format!("prefill.layers{layers}.final_norm.tokens{}", tokens.len()),
        token,
        tokens.len() - 1,
        &cpu,
        metal,
    )
}

pub fn probe_prefill_selected_logits(
    ctx: &MetalRuntime,
    report: &SourceModelReport,
    tokens: &[u32],
    layers: usize,
    logit_tokens: &[u32],
) -> Result<MetalSelectedLogitsProbeReport> {
    let Some(token) = tokens.last().copied() else {
        return Err(eyre!(
            "prefill selected-logits probe needs at least one token"
        ));
    };
    if tokens.len() > MAX_PREFILL_PROBE_TOKENS {
        return Err(eyre!(
            "prefill selected-logits probe supports at most {MAX_PREFILL_PROBE_TOKENS} tokens, got {}",
            tokens.len()
        ));
    }

    let cpu_final_hidden = prefill_final_norm_cpu(report, tokens, layers)?;
    let cpu = backend_cpu::selected_logits_reference(report, &cpu_final_hidden, logit_tokens)?;
    let metal_final_hidden = prefill_final_norm_metal(ctx, report, tokens, layers)?;
    let metal = selected_logits_metal(ctx, report, &metal_final_hidden, logit_tokens)?;
    selected_logits_report(
        &format!(
            "prefill.layers{layers}.selected_logits.tokens{}",
            tokens.len()
        ),
        token,
        layers,
        cpu,
        metal,
    )
}

pub fn probe_decode_one_final_norm(
    ctx: &MetalRuntime,
    report: &SourceModelReport,
    prefill_tokens: &[u32],
    decode_token: u32,
    layers: usize,
) -> Result<MetalVectorProbeReport> {
    let total_tokens = decode_total_tokens(prefill_tokens)?;
    let cpu = decode_one_final_norm_cpu(report, prefill_tokens, decode_token, layers)?;
    let metal = decode_one_final_norm_metal(ctx, report, prefill_tokens, decode_token, layers)?;
    vector_report(
        &format!("decode_one.layers{layers}.final_norm.tokens{total_tokens}"),
        decode_token,
        prefill_tokens.len(),
        &cpu,
        metal,
    )
}

pub fn probe_decode_one_selected_logits(
    ctx: &MetalRuntime,
    report: &SourceModelReport,
    prefill_tokens: &[u32],
    decode_token: u32,
    layers: usize,
    logit_tokens: &[u32],
) -> Result<MetalSelectedLogitsProbeReport> {
    let total_tokens = decode_total_tokens(prefill_tokens)?;
    let cpu_final_hidden = decode_one_final_norm_cpu(report, prefill_tokens, decode_token, layers)?;
    let cpu = backend_cpu::selected_logits_reference(report, &cpu_final_hidden, logit_tokens)?;
    let metal_final_hidden =
        decode_one_final_norm_metal(ctx, report, prefill_tokens, decode_token, layers)?;
    let metal = selected_logits_metal(ctx, report, &metal_final_hidden, logit_tokens)?;
    selected_logits_report(
        &format!("decode_one.layers{layers}.selected_logits.tokens{total_tokens}"),
        decode_token,
        layers,
        cpu,
        metal,
    )
}

pub fn probe_decode_one_greedy_text(
    ctx: &MetalRuntime,
    report: &SourceModelReport,
    harmony: &HarmonyAdapter,
    prefill_tokens: &[u32],
    decode_token: u32,
    layers: usize,
) -> Result<GreedyTextProbeReport> {
    let total_tokens = decode_total_tokens(prefill_tokens)?;
    let cpu_hidden = decode_one_final_norm_cpu(report, prefill_tokens, decode_token, layers)?;
    let metal_hidden =
        decode_one_final_norm_metal(ctx, report, prefill_tokens, decode_token, layers)?;
    let cpu = greedy_top1_text(report, harmony, &cpu_hidden)?;
    let metal = greedy_top1_text(report, harmony, &metal_hidden)?;
    Ok(GreedyTextProbeReport {
        name: format!("decode_one.layers{layers}.greedy_text.tokens{total_tokens}"),
        position: prefill_tokens.len(),
        layers,
        scorer: "streaming CPU BF16 lm_head top-1 for each hidden state".to_string(),
        token_match: cpu.token == metal.token,
        logit_delta: (cpu.logit - metal.logit).abs(),
        cpu,
        metal,
    })
}

pub fn probe_decode_one_lm_head_topk(
    ctx: &MetalRuntime,
    report: &SourceModelReport,
    harmony: &HarmonyAdapter,
    prefill_tokens: &[u32],
    decode_token: u32,
    layers: usize,
    k: usize,
) -> Result<LmHeadTopKProbeReport> {
    let total_tokens = decode_total_tokens(prefill_tokens)?;
    let cpu_hidden = decode_one_final_norm_cpu(report, prefill_tokens, decode_token, layers)?;
    let metal_hidden =
        decode_one_final_norm_metal(ctx, report, prefill_tokens, decode_token, layers)?;
    let cpu = lm_head_topk_cpu(report, harmony, &cpu_hidden, k)?;
    let metal = lm_head_topk_metal(ctx, harmony, &metal_hidden, k)?;
    lm_head_topk_report(
        &format!("decode_one.layers{layers}.lm_head_top{k}.tokens{total_tokens}"),
        prefill_tokens.len(),
        layers,
        k,
        cpu,
        metal,
    )
}

pub fn probe_greedy_decode(
    ctx: &MetalRuntime,
    report: &SourceModelReport,
    harmony: &HarmonyAdapter,
    prompt_tokens: &[u32],
    layers: usize,
    max_new_tokens: usize,
) -> Result<GenerationReport> {
    probe_sample_decode(
        ctx,
        report,
        harmony,
        prompt_tokens,
        layers,
        max_new_tokens,
        SamplingConfig {
            temperature: 0.0,
            ..SamplingConfig::default()
        },
    )
}

pub fn probe_sample_decode(
    ctx: &MetalRuntime,
    report: &SourceModelReport,
    harmony: &HarmonyAdapter,
    prompt_tokens: &[u32],
    layers: usize,
    max_new_tokens: usize,
    sampling: SamplingConfig,
) -> Result<GenerationReport> {
    if prompt_tokens.is_empty() {
        return Err(eyre!("sample decode needs at least one prompt token"));
    }
    if layers > LAYERS {
        return Err(eyre!(
            "requested {layers} layers, but gpt-oss-20b has {LAYERS}"
        ));
    }
    if max_new_tokens == 0 {
        return Err(eyre!("sample decode needs at least one new token"));
    }
    if prompt_tokens.len() > MAX_PREFILL_PROBE_TOKENS {
        return Err(eyre!(
            "sample decode probe supports at most {MAX_PREFILL_PROBE_TOKENS} prompt tokens, got {}",
            prompt_tokens.len()
        ));
    }
    if sampling.repetition_penalty != 1.0 {
        return Err(eyre!(
            "repetition_penalty is not implemented yet; got {}",
            sampling.repetition_penalty
        ));
    }

    let (hidden, mut kv_cache) = prefill_layers_with_cache_metal(
        ctx,
        report,
        prompt_tokens,
        layers,
        prompt_tokens
            .len()
            .checked_add(max_new_tokens)
            .ok_or_else(|| eyre!("greedy decode context length overflow"))?,
    )?;
    let Some(hidden) = hidden.last() else {
        return Err(eyre!("greedy decode prefill returned no hidden states"));
    };

    let norm_weight = ctx.bf16_vector(report, "model.norm.weight")?;
    let stop_tokens = harmony.stop_tokens()?;
    let mut final_hidden = ctx.rms_norm_profiled("op.rms_norm.final", hidden, &norm_weight)?;
    let mut generated = Vec::with_capacity(max_new_tokens);
    let mut stop_reason = StopReason::MaxGeneratedTokens;
    let mut sampler = Sampler::new(sampling.clone());

    for step in 0..max_new_tokens {
        let candidates =
            lm_head_topk_metal(ctx, harmony, &final_hidden, sampler.candidate_count())?;
        let sample_candidates = candidates
            .iter()
            .map(|token| SampleCandidate {
                token: token.token,
                logit: token.logit,
                probability: 0.0,
            })
            .collect::<Vec<_>>();
        let sampled = sampler.choose(&sample_candidates)?;
        let token_id = sampled.token;
        let text = candidates
            .iter()
            .find(|token| token.token == token_id)
            .map(|token| token.text.clone())
            .unwrap_or(decode_token_text(harmony, token_id)?);
        let token = GreedyTokenReport {
            token: token_id,
            logit: sampled.logit,
            text,
        };
        generated.push(token);

        if stop_tokens.contains(&token_id) {
            stop_reason = StopReason::EndOfGeneration;
            break;
        }
        if step + 1 == max_new_tokens {
            break;
        }

        let mut x = ctx
            .bf16_matrix_row(report, "model.embed_tokens.weight", token_id as usize)?
            .as_ref()
            .clone();
        let position = prompt_tokens.len() + step;
        for layer in 0..layers {
            x = decode_layer_metal(ctx, report, layer, &x, position, &mut kv_cache)?;
        }
        final_hidden = ctx.rms_norm_profiled("op.rms_norm.final", &x, &norm_weight)?;
    }

    let token_ids = generated
        .iter()
        .map(|token| token.token)
        .collect::<Vec<_>>();
    let text = decode_tokens_text(harmony, &token_ids)?;
    Ok(GenerationReport {
        name: format!(
            "sample_decode.layers{layers}.prompt{}.new{}",
            prompt_tokens.len(),
            max_new_tokens
        ),
        backend: "metal".to_string(),
        scorer: metal_sampler_description(&sampling),
        layers,
        prompt_tokens: prompt_tokens.len(),
        max_new_tokens,
        stop_reason,
        generated,
        text,
    })
}

pub fn probe_layer0_o_proj(
    ctx: &MetalRuntime,
    report: &SourceModelReport,
    token: u32,
) -> Result<MetalMatvecProbeReport> {
    let attention = layer0_single_token_attention_cpu(report, token)?;
    let weight_name = "model.layers.0.self_attn.o_proj.weight";
    let bias_name = "model.layers.0.self_attn.o_proj.bias";
    let report_name = "layer0.self_attn.o_proj";

    let weight = model_store::read_bf16_matrix(report, weight_name)?;
    let bias = model_store::read_bf16_vector(report, bias_name)?;

    let mut cpu = model_store::matvec_bf16(report, weight_name, &attention)?;
    model_store::add_in_place(&mut cpu, &bias, report_name)?;
    let metal =
        ctx.platform
            .bf16_matvec(&weight.values, weight.rows, weight.cols, &attention, &bias)?;

    matvec_report(report_name, token, weight.rows, weight.cols, &cpu, metal)
}

pub fn probe_layer0_attention_residual(
    ctx: &MetalRuntime,
    report: &SourceModelReport,
    token: u32,
) -> Result<MetalVectorProbeReport> {
    let embedding =
        model_store::read_bf16_matrix_row(report, "model.embed_tokens.weight", token as usize)?;
    let o_proj = layer0_o_proj_cpu(report, token)?;
    let cpu = add_vectors(&embedding, &o_proj)?;
    let metal = ctx.platform.vector_add(&embedding, &o_proj)?;
    vector_report("layer0.attention_residual", token, 0, &cpu, metal)
}

pub fn probe_layer0_post_attention_rms_norm(
    ctx: &MetalRuntime,
    report: &SourceModelReport,
    token: u32,
) -> Result<MetalVectorProbeReport> {
    let residual = layer0_attention_residual_cpu(report, token)?;
    let weight =
        model_store::read_bf16_vector(report, "model.layers.0.post_attention_layernorm.weight")?;
    let cpu = backend_cpu::rms_norm_reference(&residual, &weight)?;
    let metal = ctx.platform.rms_norm(&residual, &weight)?;
    vector_report("layer0.post_attention_layernorm", token, 0, &cpu, metal)
}

pub fn probe_layer0_router(
    ctx: &MetalRuntime,
    report: &SourceModelReport,
    token: u32,
) -> Result<MetalMatvecProbeReport> {
    let residual = layer0_attention_residual_cpu(report, token)?;
    let norm_weight =
        model_store::read_bf16_vector(report, "model.layers.0.post_attention_layernorm.weight")?;
    let input = backend_cpu::rms_norm_reference(&residual, &norm_weight)?;
    let weight_name = "model.layers.0.mlp.router.weight";
    let bias_name = "model.layers.0.mlp.router.bias";
    let report_name = "layer0.mlp.router";

    let weight = model_store::read_bf16_matrix(report, weight_name)?;
    let bias = model_store::read_bf16_vector(report, bias_name)?;
    let mut cpu = model_store::matvec_bf16(report, weight_name, &input)?;
    model_store::add_in_place(&mut cpu, &bias, report_name)?;
    let metal =
        ctx.platform
            .bf16_matvec(&weight.values, weight.rows, weight.cols, &input, &bias)?;

    matvec_report(report_name, token, weight.rows, weight.cols, &cpu, metal)
}

pub fn probe_layer0_router_top4(
    ctx: &MetalRuntime,
    report: &SourceModelReport,
    token: u32,
) -> Result<MetalTopKProbeReport> {
    let router = layer0_router_cpu(report, token)?;
    let cpu = backend_cpu::top_k_softmax_reference(&router, 4);
    let metal = ctx.platform.top4_softmax(&router)?;
    top_k_report("layer0.mlp.router.top4", token, cpu, metal)
}

pub fn probe_layer0_top_expert_gate_up(
    ctx: &MetalRuntime,
    report: &SourceModelReport,
    token: u32,
) -> Result<MetalVectorProbeReport> {
    let (expert, input, bias, cpu) = layer0_top_expert_gate_up_parts(report, token)?;
    let prefix = "model.layers.0.mlp.experts";
    let blocks_name = format!("{prefix}.gate_up_proj_blocks");
    let scales_name = format!("{prefix}.gate_up_proj_scales");

    let blocks = read_mxfp4_expert_blocks(report, &blocks_name, expert.index, GATE_UP_VALUES)?;
    let scales = read_mxfp4_expert_scales(report, &scales_name, expert.index, GATE_UP_VALUES)?;
    let metal = ctx
        .platform
        .mxfp4_matvec(&blocks, &scales, GATE_UP_VALUES, &input, &bias)?;
    vector_report(
        &format!("layer0.mlp.expert{}.gate_up_proj", expert.index),
        token,
        0,
        &cpu,
        metal,
    )
}

pub fn probe_layer0_top_expert_swiglu(
    ctx: &MetalRuntime,
    report: &SourceModelReport,
    token: u32,
) -> Result<MetalVectorProbeReport> {
    let (expert, _, _, gate_up) = layer0_top_expert_gate_up_parts(report, token)?;
    let cpu = backend_cpu::swiglu_reference(&gate_up)?;
    let metal = ctx.platform.swiglu(&gate_up)?;
    vector_report(
        &format!("layer0.mlp.expert{}.swiglu", expert.index),
        token,
        0,
        &cpu,
        metal,
    )
}

pub fn probe_layer0_top_expert_down_proj(
    ctx: &MetalRuntime,
    report: &SourceModelReport,
    token: u32,
) -> Result<MetalVectorProbeReport> {
    let (expert, _, _, gate_up) = layer0_top_expert_gate_up_parts(report, token)?;
    let input = backend_cpu::swiglu_reference(&gate_up)?;
    let prefix = "model.layers.0.mlp.experts";
    let blocks_name = format!("{prefix}.down_proj_blocks");
    let scales_name = format!("{prefix}.down_proj_scales");
    let bias_name = format!("{prefix}.down_proj_bias");

    let mut cpu = backend_cpu::mxfp4_expert_matvec_reference(
        report,
        &blocks_name,
        &scales_name,
        expert.index,
        HIDDEN_SIZE,
        &input,
    )?;
    let bias = model_store::read_bf16_matrix_row(report, &bias_name, expert.index)?;
    model_store::add_in_place(&mut cpu, &bias, &format!("{prefix}.down_proj"))?;

    let blocks = read_mxfp4_expert_blocks(report, &blocks_name, expert.index, HIDDEN_SIZE)?;
    let scales = read_mxfp4_expert_scales(report, &scales_name, expert.index, HIDDEN_SIZE)?;
    let metal = ctx
        .platform
        .mxfp4_matvec(&blocks, &scales, HIDDEN_SIZE, &input, &bias)?;
    vector_report(
        &format!("layer0.mlp.expert{}.down_proj", expert.index),
        token,
        0,
        &cpu,
        metal,
    )
}

pub fn probe_layer0_moe_top4(
    ctx: &MetalRuntime,
    report: &SourceModelReport,
    token: u32,
) -> Result<MetalVectorProbeReport> {
    let input = layer0_router_input_cpu(report, token)?;
    let router = layer0_router_cpu(report, token)?;
    let top_experts = backend_cpu::top_k_softmax_reference(&router, 4);
    if top_experts.len() != 4 {
        return Err(eyre!(
            "router returned {} experts, expected 4",
            top_experts.len()
        ));
    }

    let cpu = backend_cpu::layer_moe_reference(report, 0, &input, &top_experts)?;
    let metal = layer_moe_top4_metal(ctx, report, 0, &input, &router)?;
    vector_report("layer0.mlp.top4", token, 0, &cpu, metal)
}

pub fn probe_layer0_output(
    ctx: &MetalRuntime,
    report: &SourceModelReport,
    token: u32,
) -> Result<MetalVectorProbeReport> {
    let embedding =
        model_store::read_bf16_matrix_row(report, "model.embed_tokens.weight", token as usize)?;
    let cpu = backend_cpu::single_token_layer_reference(report, 0, &embedding)?;
    let metal = single_token_layer_metal(ctx, report, 0, &embedding, 0)?;

    vector_report("layer0.output", token, 0, &cpu, metal)
}

pub fn probe_single_token_final_norm(
    ctx: &MetalRuntime,
    report: &SourceModelReport,
    token: u32,
    layers: usize,
) -> Result<MetalVectorProbeReport> {
    let cpu = backend_cpu::single_token_final_norm_reference(report, token, layers)?;
    let metal = single_token_final_norm_metal(ctx, report, token, layers)?;
    vector_report(
        &format!("single_token.layers{layers}.final_norm"),
        token,
        0,
        &cpu,
        metal,
    )
}

pub fn probe_single_token_selected_logits(
    ctx: &MetalRuntime,
    report: &SourceModelReport,
    token: u32,
    layers: usize,
    logit_tokens: &[u32],
) -> Result<MetalSelectedLogitsProbeReport> {
    let cpu_final_hidden = backend_cpu::single_token_final_norm_reference(report, token, layers)?;
    let cpu = backend_cpu::selected_logits_reference(report, &cpu_final_hidden, logit_tokens)?;

    let metal_final_hidden = single_token_final_norm_metal(ctx, report, token, layers)?;
    let metal = selected_logits_metal(ctx, report, &metal_final_hidden, logit_tokens)?;
    selected_logits_report(
        &format!("single_token.layers{layers}.selected_logits"),
        token,
        layers,
        cpu,
        metal,
    )
}

fn single_token_final_norm_metal(
    ctx: &MetalRuntime,
    report: &SourceModelReport,
    token: u32,
    layers: usize,
) -> Result<Vec<f32>> {
    if layers > LAYERS {
        return Err(eyre!(
            "requested {layers} layers, but gpt-oss-20b has {LAYERS}"
        ));
    }

    let mut x =
        model_store::read_bf16_matrix_row(report, "model.embed_tokens.weight", token as usize)?;
    for layer in 0..layers {
        x = single_token_layer_metal(ctx, report, layer, &x, 0)?;
    }

    let norm_weight = ctx.bf16_vector(report, "model.norm.weight")?;
    ctx.rms_norm_profiled("op.rms_norm.final", &x, &norm_weight)
}

fn single_token_layer_metal(
    ctx: &MetalRuntime,
    report: &SourceModelReport,
    layer: usize,
    input: &[f32],
    position: usize,
) -> Result<Vec<f32>> {
    let prefix = format!("model.layers.{layer}");
    let (q, k, v) = layer_attention_qkv_metal(ctx, report, layer, input, position)?;
    let sinks = ctx.bf16_vector(report, &format!("{prefix}.self_attn.sinks"))?;
    let attn = ctx.single_token_attention_profiled(&q, &k, &v, &sinks)?;
    let projected = layer_projection_metal(ctx, report, layer, "o_proj", &attn)?;
    let residual = ctx.vector_add_profiled("op.residual.attention", input, &projected)?;

    let post_norm_weight =
        ctx.bf16_vector(report, &format!("{prefix}.post_attention_layernorm.weight"))?;
    let router_input =
        ctx.rms_norm_profiled("op.rms_norm.post_attention", &residual, &post_norm_weight)?;
    let router = layer_router_metal(ctx, report, layer, &router_input)?;
    let moe = layer_moe_top4_metal(ctx, report, layer, &router_input, &router)?;
    ctx.vector_add_profiled("op.residual.moe", &residual, &moe)
}

fn prefill_layers_metal(
    ctx: &MetalRuntime,
    report: &SourceModelReport,
    tokens: &[u32],
    layers: usize,
) -> Result<Vec<Vec<f32>>> {
    let (x, _) = prefill_layers_with_cache_metal(ctx, report, tokens, layers, tokens.len())?;
    Ok(x)
}

fn prefill_layers_with_cache_metal(
    ctx: &MetalRuntime,
    report: &SourceModelReport,
    tokens: &[u32],
    layers: usize,
    context_tokens: usize,
) -> Result<(Vec<Vec<f32>>, PlannedKvCache)> {
    if layers > LAYERS {
        return Err(eyre!(
            "requested {layers} layers, but gpt-oss-20b has {LAYERS}"
        ));
    }

    let mut x = layer0_embedding_rows_metal(ctx, report, tokens)?;
    let mut kv_cache = PlannedKvCache::new(KvCachePlan::gpt_oss_20b(context_tokens)?);
    for layer in 0..layers {
        x = prefill_layer_metal(ctx, report, layer, &x, 0, &mut kv_cache)?;
    }
    Ok((x, kv_cache))
}

fn prefill_final_norm_cpu(
    report: &SourceModelReport,
    tokens: &[u32],
    layers: usize,
) -> Result<Vec<f32>> {
    let hidden = backend_cpu::sequence_layers_reference(report, tokens, layers)?;
    let Some(hidden) = hidden.last() else {
        return Err(eyre!("prefill final norm needs at least one token"));
    };
    let norm_weight = model_store::read_bf16_vector(report, "model.norm.weight")?;
    backend_cpu::rms_norm_reference(hidden, &norm_weight)
}

fn prefill_final_norm_metal(
    ctx: &MetalRuntime,
    report: &SourceModelReport,
    tokens: &[u32],
    layers: usize,
) -> Result<Vec<f32>> {
    let hidden = prefill_layers_metal(ctx, report, tokens, layers)?;
    let Some(hidden) = hidden.last() else {
        return Err(eyre!("prefill final norm needs at least one token"));
    };
    let norm_weight = ctx.bf16_vector(report, "model.norm.weight")?;
    ctx.rms_norm_profiled("op.rms_norm.final", hidden, &norm_weight)
}

fn decode_one_final_norm_cpu(
    report: &SourceModelReport,
    prefill_tokens: &[u32],
    decode_token: u32,
    layers: usize,
) -> Result<Vec<f32>> {
    let tokens = decode_sequence_tokens(prefill_tokens, decode_token)?;
    prefill_final_norm_cpu(report, &tokens, layers)
}

fn decode_one_final_norm_metal(
    ctx: &MetalRuntime,
    report: &SourceModelReport,
    prefill_tokens: &[u32],
    decode_token: u32,
    layers: usize,
) -> Result<Vec<f32>> {
    let hidden = decode_one_metal(ctx, report, prefill_tokens, decode_token, layers)?;
    let norm_weight = ctx.bf16_vector(report, "model.norm.weight")?;
    ctx.rms_norm_profiled("op.rms_norm.final", &hidden, &norm_weight)
}

fn decode_one_metal(
    ctx: &MetalRuntime,
    report: &SourceModelReport,
    prefill_tokens: &[u32],
    decode_token: u32,
    layers: usize,
) -> Result<Vec<f32>> {
    let total_tokens = decode_total_tokens(prefill_tokens)?;
    let (_, mut kv_cache) =
        prefill_layers_with_cache_metal(ctx, report, prefill_tokens, layers, total_tokens)?;
    let mut x = ctx
        .bf16_matrix_row(report, "model.embed_tokens.weight", decode_token as usize)?
        .as_ref()
        .clone();
    let position = prefill_tokens.len();
    for layer in 0..layers {
        x = decode_layer_metal(ctx, report, layer, &x, position, &mut kv_cache)?;
    }
    Ok(x)
}

fn decode_layer_metal(
    ctx: &MetalRuntime,
    report: &SourceModelReport,
    layer: usize,
    input: &[f32],
    position: usize,
    kv_cache: &mut PlannedKvCache,
) -> Result<Vec<f32>> {
    let prefix = format!("model.layers.{layer}");
    let (q, k, v) = layer_attention_qkv_metal(ctx, report, layer, input, position)?;
    kv_cache.layer_mut(layer)?.push(position, &k, &v)?;
    let view = kv_cache.layer(layer)?.contiguous_view_for_query(position)?;
    let sinks = ctx.bf16_vector(report, &format!("{prefix}.self_attn.sinks"))?;
    let attn = ctx.kv_cache_decode_attention_profiled(
        layer,
        position,
        view.start_position,
        &q,
        &view.k,
        &view.v,
        &sinks,
    )?;
    let projected = layer_projection_metal(ctx, report, layer, "o_proj", &attn)?;
    let residual = ctx.vector_add_profiled("op.residual.attention", input, &projected)?;

    let post_norm_weight =
        ctx.bf16_vector(report, &format!("{prefix}.post_attention_layernorm.weight"))?;
    let router_input =
        ctx.rms_norm_profiled("op.rms_norm.post_attention", &residual, &post_norm_weight)?;
    let router = layer_router_metal(ctx, report, layer, &router_input)?;
    let moe = layer_moe_top4_metal(ctx, report, layer, &router_input, &router)?;
    ctx.vector_add_profiled("op.residual.moe", &residual, &moe)
}

fn prefill_layer_metal(
    ctx: &MetalRuntime,
    report: &SourceModelReport,
    layer: usize,
    inputs: &[Vec<f32>],
    start_position: usize,
    kv_cache: &mut PlannedKvCache,
) -> Result<Vec<Vec<f32>>> {
    let prefix = format!("model.layers.{layer}");
    let sinks = ctx.bf16_vector(report, &format!("{prefix}.self_attn.sinks"))?;
    let q_rows =
        prefill_layer_attention_cache_metal(ctx, report, layer, inputs, start_position, kv_cache)?;

    let mut residuals = Vec::with_capacity(inputs.len());
    for (offset, (input, q)) in inputs.iter().zip(q_rows.iter()).enumerate() {
        let position = start_position + offset;
        let view = kv_cache.layer(layer)?.contiguous_view_for_query(position)?;
        let attn = ctx.kv_cache_decode_attention_profiled(
            layer,
            position,
            view.start_position,
            q,
            &view.k,
            &view.v,
            &sinks,
        )?;
        let projected = layer_projection_metal(ctx, report, layer, "o_proj", &attn)?;
        residuals.push(ctx.vector_add_profiled("op.residual.attention", input, &projected)?);
    }

    let post_norm_weight =
        ctx.bf16_vector(report, &format!("{prefix}.post_attention_layernorm.weight"))?;
    let mut out = Vec::with_capacity(residuals.len());
    for residual in residuals {
        let router_input =
            ctx.rms_norm_profiled("op.rms_norm.post_attention", &residual, &post_norm_weight)?;
        let router = layer_router_metal(ctx, report, layer, &router_input)?;
        let moe = layer_moe_top4_metal(ctx, report, layer, &router_input, &router)?;
        out.push(ctx.vector_add_profiled("op.residual.moe", &residual, &moe)?);
    }

    Ok(out)
}

fn prefill_layer_attention_cache_metal(
    ctx: &MetalRuntime,
    report: &SourceModelReport,
    layer: usize,
    inputs: &[Vec<f32>],
    start_position: usize,
    kv_cache: &mut PlannedKvCache,
) -> Result<Vec<Vec<f32>>> {
    let mut q_rows = Vec::with_capacity(inputs.len());
    for (offset, input) in inputs.iter().enumerate() {
        let position = start_position + offset;
        let (q, k, v) = layer_attention_qkv_metal(ctx, report, layer, input, position)?;
        kv_cache.layer_mut(layer)?.push(position, &k, &v)?;
        q_rows.push(q);
    }
    Ok(q_rows)
}

fn layer_attention_qkv_metal(
    ctx: &MetalRuntime,
    report: &SourceModelReport,
    layer: usize,
    input: &[f32],
    position: usize,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    if input.len() != HIDDEN_SIZE {
        return Err(eyre!(
            "layer {layer} attention input has {} values, expected {HIDDEN_SIZE}",
            input.len()
        ));
    }

    let prefix = format!("model.layers.{layer}");
    let input_norm_weight = ctx.bf16_vector(report, &format!("{prefix}.input_layernorm.weight"))?;
    let attn_input = ctx.rms_norm_profiled("op.rms_norm.input", input, &input_norm_weight)?;

    let q = layer_projection_metal(ctx, report, layer, "q_proj", &attn_input)?;
    let k = layer_projection_metal(ctx, report, layer, "k_proj", &attn_input)?;
    let v = layer_projection_metal(ctx, report, layer, "v_proj", &attn_input)?;
    let q = ctx.rope_row_profiled(&q, Q_HEADS, position)?;
    let k = ctx.rope_row_profiled(&k, KV_HEADS, position)?;
    Ok((q, k, v))
}

fn layer_projection_metal(
    ctx: &MetalRuntime,
    report: &SourceModelReport,
    layer: usize,
    projection: &str,
    input: &[f32],
) -> Result<Vec<f32>> {
    let weight_name = format!("model.layers.{layer}.self_attn.{projection}.weight");
    let bias_name = format!("model.layers.{layer}.self_attn.{projection}.bias");
    ctx.bf16_linear_matvec(report, &weight_name, &bias_name, input)
}

fn layer_router_metal(
    ctx: &MetalRuntime,
    report: &SourceModelReport,
    layer: usize,
    input: &[f32],
) -> Result<Vec<f32>> {
    let weight_name = format!("model.layers.{layer}.mlp.router.weight");
    let bias_name = format!("model.layers.{layer}.mlp.router.bias");
    ctx.bf16_linear_matvec(report, &weight_name, &bias_name, input)
}

fn layer_moe_top4_metal(
    ctx: &MetalRuntime,
    report: &SourceModelReport,
    layer: usize,
    input: &[f32],
    router: &[f32],
) -> Result<Vec<f32>> {
    let top_experts = ctx.top4_softmax_profiled(router)?;
    if top_experts.len() != 4 {
        return Err(eyre!(
            "Metal router returned {} experts, expected 4",
            top_experts.len()
        ));
    }

    let mut downs = Vec::with_capacity(4);
    for expert in &top_experts {
        downs.push(layer_expert_down_proj_metal(
            ctx,
            report,
            layer,
            expert.index,
            input,
        )?);
    }

    ctx.weighted_sum4_profiled(
        [&downs[0], &downs[1], &downs[2], &downs[3]],
        [
            top_experts[0].weight,
            top_experts[1].weight,
            top_experts[2].weight,
            top_experts[3].weight,
        ],
    )
}

fn layer0_top_expert_gate_up_parts(
    report: &SourceModelReport,
    token: u32,
) -> Result<(ExpertScore, Vec<f32>, Vec<f32>, Vec<f32>)> {
    let router = layer0_router_cpu(report, token)?;
    let Some(expert) = backend_cpu::top_k_softmax_reference(&router, 4)
        .first()
        .cloned()
    else {
        return Err(eyre!("router top-4 returned no experts"));
    };
    let input = layer0_router_input_cpu(report, token)?;
    let prefix = "model.layers.0.mlp.experts";
    let blocks_name = format!("{prefix}.gate_up_proj_blocks");
    let scales_name = format!("{prefix}.gate_up_proj_scales");
    let bias_name = format!("{prefix}.gate_up_proj_bias");

    let mut gate_up = backend_cpu::mxfp4_expert_matvec_reference(
        report,
        &blocks_name,
        &scales_name,
        expert.index,
        GATE_UP_VALUES,
        &input,
    )?;
    let bias = model_store::read_bf16_matrix_row(report, &bias_name, expert.index)?;
    model_store::add_in_place(&mut gate_up, &bias, &format!("{prefix}.gate_up_proj"))?;

    Ok((expert, input, bias, gate_up))
}

fn layer_expert_down_proj_metal(
    ctx: &MetalRuntime,
    report: &SourceModelReport,
    layer: usize,
    expert: usize,
    input: &[f32],
) -> Result<Vec<f32>> {
    let prefix = format!("model.layers.{layer}.mlp.experts");

    let gate_up_blocks_name = format!("{prefix}.gate_up_proj_blocks");
    let gate_up_scales_name = format!("{prefix}.gate_up_proj_scales");
    let gate_up_bias_name = format!("{prefix}.gate_up_proj_bias");
    let gate_up = ctx.mxfp4_expert_matvec(
        report,
        &gate_up_blocks_name,
        &gate_up_scales_name,
        &gate_up_bias_name,
        expert,
        GATE_UP_VALUES,
        input,
    )?;
    let swiglu = ctx.swiglu_profiled(&gate_up)?;

    let down_blocks_name = format!("{prefix}.down_proj_blocks");
    let down_scales_name = format!("{prefix}.down_proj_scales");
    let down_bias_name = format!("{prefix}.down_proj_bias");
    ctx.mxfp4_expert_matvec(
        report,
        &down_blocks_name,
        &down_scales_name,
        &down_bias_name,
        expert,
        HIDDEN_SIZE,
        &swiglu,
    )
}

fn selected_logits_metal(
    ctx: &MetalRuntime,
    report: &SourceModelReport,
    final_hidden: &[f32],
    logit_tokens: &[u32],
) -> Result<Vec<SelectedLogit>> {
    let mut rows = Vec::with_capacity(logit_tokens.len() * final_hidden.len());
    for token in logit_tokens {
        let row =
            model_store::read_bf16_matrix_row_bits(report, "lm_head.weight", *token as usize)?;
        if row.len() != final_hidden.len() {
            return Err(eyre!(
                "lm_head row {} has {} values, but final hidden has {} values",
                token,
                row.len(),
                final_hidden.len()
            ));
        }
        rows.extend(row);
    }

    let bias = vec![0.0f32; logit_tokens.len()];
    let logits = ctx.platform.bf16_matvec(
        &rows,
        logit_tokens.len(),
        final_hidden.len(),
        final_hidden,
        &bias,
    )?;
    Ok(logit_tokens
        .iter()
        .copied()
        .zip(logits)
        .map(|(token, logit)| SelectedLogit { token, logit })
        .collect())
}

fn greedy_top1_text(
    report: &SourceModelReport,
    harmony: &HarmonyAdapter,
    final_hidden: &[f32],
) -> Result<GreedyTokenReport> {
    let Some((token, logit)) =
        model_store::top_k_matvec_bf16(report, "lm_head.weight", final_hidden, 1)?
            .into_iter()
            .next()
    else {
        return Err(eyre!("lm_head top-1 returned no tokens"));
    };
    let token = token as u32;
    Ok(GreedyTokenReport {
        token,
        logit,
        text: decode_token_text(harmony, token)?,
    })
}

fn lm_head_topk_cpu(
    report: &SourceModelReport,
    harmony: &HarmonyAdapter,
    final_hidden: &[f32],
    k: usize,
) -> Result<Vec<GreedyTokenReport>> {
    model_store::top_k_matvec_bf16(report, "lm_head.weight", final_hidden, k)?
        .into_iter()
        .map(|(token, logit)| {
            let token = token as u32;
            Ok(GreedyTokenReport {
                token,
                logit,
                text: decode_token_text(harmony, token)?,
            })
        })
        .collect()
}

fn lm_head_topk_metal(
    ctx: &MetalRuntime,
    harmony: &HarmonyAdapter,
    final_hidden: &[f32],
    k: usize,
) -> Result<Vec<GreedyTokenReport>> {
    ctx.lm_head_topk(final_hidden, k)?
        .into_iter()
        .map(|(token, logit)| {
            let token = token as u32;
            Ok(GreedyTokenReport {
                token,
                logit,
                text: decode_token_text(harmony, token)?,
            })
        })
        .collect()
}

pub(crate) fn decode_token_text(harmony: &HarmonyAdapter, token: u32) -> Result<String> {
    match harmony.decode_utf8(&[token]) {
        Ok(text) => Ok(text),
        Err(_) => {
            let bytes = harmony.decode_bytes(&[token])?;
            Ok(format!("<bytes {bytes:?}>"))
        }
    }
}

pub(crate) fn decode_tokens_text(harmony: &HarmonyAdapter, tokens: &[u32]) -> Result<String> {
    match harmony.decode_utf8(tokens) {
        Ok(text) => Ok(text),
        Err(_) => {
            let bytes = harmony.decode_bytes(tokens)?;
            Ok(format!("<bytes {bytes:?}>"))
        }
    }
}

pub(crate) fn metal_sampler_description(sampling: &SamplingConfig) -> String {
    if sampling.temperature <= 0.0 {
        "cached Metal BF16 lm_head logits + Metal top-1".to_string()
    } else if sampling.top_k == 0 && sampling.top_p < 1.0 {
        format!(
            "cached Metal BF16 full-vocab lm_head logits + seeded full-vocab top-p sampler (seed {}, temperature {:.4}, top_p {:.4})",
            sampling.seed, sampling.temperature, sampling.top_p
        )
    } else {
        format!(
            "cached Metal BF16 lm_head logits + seeded temperature/top-k/top-p sampler (seed {}, temperature {:.4}, top_k {}, top_p {:.4})",
            sampling.seed, sampling.temperature, sampling.top_k, sampling.top_p
        )
    }
}

fn lm_head_topk_report(
    name: &str,
    position: usize,
    layers: usize,
    k: usize,
    cpu: Vec<GreedyTokenReport>,
    metal: Vec<GreedyTokenReport>,
) -> Result<LmHeadTopKProbeReport> {
    if cpu.len() != metal.len() {
        return Err(eyre!(
            "Metal lm_head top-k returned {} tokens, CPU returned {}",
            metal.len(),
            cpu.len()
        ));
    }

    let mut tokens_match = true;
    let mut max_abs_delta = 0.0f32;
    for (cpu, metal) in cpu.iter().zip(&metal) {
        tokens_match &= cpu.token == metal.token;
        max_abs_delta = max_abs_delta.max((cpu.logit - metal.logit).abs());
    }

    Ok(LmHeadTopKProbeReport {
        name: name.to_string(),
        position,
        layers,
        k,
        tokens_match,
        max_abs_delta,
        cpu,
        metal,
    })
}

fn layer0_projection_cpu(
    report: &SourceModelReport,
    token: u32,
    projection: &str,
) -> Result<Vec<f32>> {
    let embedding =
        model_store::read_bf16_matrix_row(report, "model.embed_tokens.weight", token as usize)?;
    let norm_weight =
        model_store::read_bf16_vector(report, "model.layers.0.input_layernorm.weight")?;
    let input = backend_cpu::rms_norm_reference(&embedding, &norm_weight)?;
    let weight_name = format!("model.layers.0.self_attn.{projection}.weight");
    let bias_name = format!("model.layers.0.self_attn.{projection}.bias");
    let report_name = format!("model.layers.0.self_attn.{projection}");
    let mut out = model_store::matvec_bf16(report, &weight_name, &input)?;
    let bias = model_store::read_bf16_vector(report, &bias_name)?;
    model_store::add_in_place(&mut out, &bias, &report_name)?;
    Ok(out)
}

fn layer0_embedding_rows(report: &SourceModelReport, tokens: &[u32]) -> Result<Vec<Vec<f32>>> {
    let mut rows = Vec::with_capacity(tokens.len());
    for token in tokens {
        rows.push(model_store::read_bf16_matrix_row(
            report,
            "model.embed_tokens.weight",
            *token as usize,
        )?);
    }
    Ok(rows)
}

fn layer0_embedding_rows_metal(
    ctx: &MetalRuntime,
    report: &SourceModelReport,
    tokens: &[u32],
) -> Result<Vec<Vec<f32>>> {
    let mut rows = Vec::with_capacity(tokens.len());
    for token in tokens {
        rows.push(
            ctx.bf16_matrix_row(report, "model.embed_tokens.weight", *token as usize)?
                .as_ref()
                .clone(),
        );
    }
    Ok(rows)
}

fn decode_sequence_tokens(prefill_tokens: &[u32], decode_token: u32) -> Result<Vec<u32>> {
    decode_total_tokens(prefill_tokens)?;
    let mut tokens = Vec::with_capacity(prefill_tokens.len() + 1);
    tokens.extend_from_slice(prefill_tokens);
    tokens.push(decode_token);
    Ok(tokens)
}

fn decode_total_tokens(prefill_tokens: &[u32]) -> Result<usize> {
    let total_tokens = prefill_tokens
        .len()
        .checked_add(1)
        .ok_or_else(|| eyre!("decode token count overflow"))?;
    if total_tokens > MAX_PREFILL_PROBE_TOKENS {
        return Err(eyre!(
            "decode-one probe supports at most {MAX_PREFILL_PROBE_TOKENS} total tokens, got {total_tokens}"
        ));
    }
    Ok(total_tokens)
}

fn layer0_sequence_projection_cpu(
    report: &SourceModelReport,
    tokens: &[u32],
    projection: &str,
) -> Result<Vec<Vec<f32>>> {
    let mut rows = Vec::with_capacity(tokens.len());
    for token in tokens {
        rows.push(layer0_projection_cpu(report, *token, projection)?);
    }
    Ok(rows)
}

fn layer0_rope_qkv_cpu(
    report: &SourceModelReport,
    tokens: &[u32],
) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>)> {
    let mut q = layer0_sequence_projection_cpu(report, tokens, "q_proj")?;
    let mut k = layer0_sequence_projection_cpu(report, tokens, "k_proj")?;
    let v = layer0_sequence_projection_cpu(report, tokens, "v_proj")?;
    for (position, row) in q.iter_mut().enumerate() {
        *row = backend_cpu::apply_rope_reference(row, Q_HEADS, position)?;
    }
    for (position, row) in k.iter_mut().enumerate() {
        *row = backend_cpu::apply_rope_reference(row, KV_HEADS, position)?;
    }
    Ok((q, k, v))
}

fn kv_cache_attention_probe(
    ctx: &MetalRuntime,
    report: &SourceModelReport,
    tokens: &[u32],
    layer: usize,
    name: &str,
) -> Result<MetalVectorProbeReport> {
    let Some(token) = tokens.last().copied() else {
        return Err(eyre!("KV-cache attention probe needs at least one token"));
    };
    if tokens.len() > MAX_KV_CACHE_PROBE_TOKENS {
        return Err(eyre!(
            "KV-cache attention probe supports at most {MAX_KV_CACHE_PROBE_TOKENS} tokens, got {}",
            tokens.len()
        ));
    }

    let (q, k, v) = layer0_rope_qkv_cpu(report, tokens)?;
    let sinks = model_store::read_bf16_vector(report, "model.layers.0.self_attn.sinks")?;
    let cpu = backend_cpu::sequence_attention_from_rope_reference(layer, &q, &k, &v, &sinks)?;
    let query_position = tokens.len() - 1;
    let Some(cpu) = cpu.get(query_position) else {
        return Err(eyre!("sequence attention returned no row {query_position}"));
    };

    let mut kv_cache = PlannedKvCache::new(KvCachePlan::gpt_oss_20b(tokens.len())?);
    for position in 0..tokens.len() {
        kv_cache
            .layer_mut(layer)?
            .push(position, &k[position], &v[position])?;
    }
    let view = kv_cache
        .layer(layer)?
        .contiguous_view_for_query(query_position)?;
    let metal = ctx.platform.kv_cache_decode_attention(
        layer,
        query_position,
        view.start_position,
        &q[query_position],
        &view.k,
        &view.v,
        &sinks,
    )?;

    vector_report(name, token, query_position, cpu, metal)
}

fn flatten_rows(rows: &[Vec<f32>]) -> Vec<f32> {
    let values = rows.iter().map(Vec::len).sum();
    let mut flat = Vec::with_capacity(values);
    for row in rows {
        flat.extend_from_slice(row);
    }
    flat
}

fn layer0_single_token_attention_cpu(report: &SourceModelReport, token: u32) -> Result<Vec<f32>> {
    let q = layer0_projection_cpu(report, token, "q_proj")?;
    let k = layer0_projection_cpu(report, token, "k_proj")?;
    let v = layer0_projection_cpu(report, token, "v_proj")?;
    let q = backend_cpu::apply_rope_reference(&q, Q_HEADS, 0)?;
    let k = backend_cpu::apply_rope_reference(&k, KV_HEADS, 0)?;
    let sinks = model_store::read_bf16_vector(report, "model.layers.0.self_attn.sinks")?;
    backend_cpu::single_token_attention_from_rope_reference(&q, &k, &v, &sinks)
}

fn layer0_o_proj_cpu(report: &SourceModelReport, token: u32) -> Result<Vec<f32>> {
    let attention = layer0_single_token_attention_cpu(report, token)?;
    let weight_name = "model.layers.0.self_attn.o_proj.weight";
    let bias_name = "model.layers.0.self_attn.o_proj.bias";
    let mut out = model_store::matvec_bf16(report, weight_name, &attention)?;
    let bias = model_store::read_bf16_vector(report, bias_name)?;
    model_store::add_in_place(&mut out, &bias, "model.layers.0.self_attn.o_proj")?;
    Ok(out)
}

fn layer0_attention_residual_cpu(report: &SourceModelReport, token: u32) -> Result<Vec<f32>> {
    let embedding =
        model_store::read_bf16_matrix_row(report, "model.embed_tokens.weight", token as usize)?;
    let o_proj = layer0_o_proj_cpu(report, token)?;
    add_vectors(&embedding, &o_proj)
}

fn layer0_router_cpu(report: &SourceModelReport, token: u32) -> Result<Vec<f32>> {
    let input = layer0_router_input_cpu(report, token)?;
    let weight_name = "model.layers.0.mlp.router.weight";
    let bias_name = "model.layers.0.mlp.router.bias";
    let mut router = model_store::matvec_bf16(report, weight_name, &input)?;
    let bias = model_store::read_bf16_vector(report, bias_name)?;
    model_store::add_in_place(&mut router, &bias, "model.layers.0.mlp.router")?;
    Ok(router)
}

fn layer0_router_input_cpu(report: &SourceModelReport, token: u32) -> Result<Vec<f32>> {
    let residual = layer0_attention_residual_cpu(report, token)?;
    let norm_weight =
        model_store::read_bf16_vector(report, "model.layers.0.post_attention_layernorm.weight")?;
    backend_cpu::rms_norm_reference(&residual, &norm_weight)
}

fn read_mxfp4_expert_blocks(
    report: &SourceModelReport,
    tensor_name: &str,
    expert: usize,
    rows: usize,
) -> Result<Vec<u8>> {
    let blocks_per_expert = rows
        .checked_mul(MXFP4_GROUPS)
        .and_then(|value| value.checked_mul(MXFP4_BYTES_PER_GROUP))
        .ok_or_else(|| eyre!("MXFP4 block slice size overflow"))?;
    let offset = expert
        .checked_mul(blocks_per_expert)
        .ok_or_else(|| eyre!("MXFP4 block offset overflow"))?;
    model_store::read_u8_tensor_slice(report, tensor_name, offset, blocks_per_expert)
}

pub(crate) fn read_mxfp4_expert_blocks_metal(
    ctx: &MetalRuntime,
    report: &SourceModelReport,
    tensor_name: &str,
    expert: usize,
    rows: usize,
) -> Result<Arc<Vec<u8>>> {
    let blocks_per_expert = rows
        .checked_mul(MXFP4_GROUPS)
        .and_then(|value| value.checked_mul(MXFP4_BYTES_PER_GROUP))
        .ok_or_else(|| eyre!("MXFP4 block slice size overflow"))?;
    let offset = expert
        .checked_mul(blocks_per_expert)
        .ok_or_else(|| eyre!("MXFP4 block offset overflow"))?;
    ctx.u8_tensor_slice(report, tensor_name, offset, blocks_per_expert)
}

fn read_mxfp4_expert_scales(
    report: &SourceModelReport,
    tensor_name: &str,
    expert: usize,
    rows: usize,
) -> Result<Vec<u8>> {
    let scales_per_expert = rows
        .checked_mul(MXFP4_GROUPS)
        .ok_or_else(|| eyre!("MXFP4 scale slice size overflow"))?;
    let offset = expert
        .checked_mul(scales_per_expert)
        .ok_or_else(|| eyre!("MXFP4 scale offset overflow"))?;
    model_store::read_u8_tensor_slice(report, tensor_name, offset, scales_per_expert)
}

pub(crate) fn read_mxfp4_expert_scales_metal(
    ctx: &MetalRuntime,
    report: &SourceModelReport,
    tensor_name: &str,
    expert: usize,
    rows: usize,
) -> Result<Arc<Vec<u8>>> {
    let scales_per_expert = rows
        .checked_mul(MXFP4_GROUPS)
        .ok_or_else(|| eyre!("MXFP4 scale slice size overflow"))?;
    let offset = expert
        .checked_mul(scales_per_expert)
        .ok_or_else(|| eyre!("MXFP4 scale offset overflow"))?;
    ctx.u8_tensor_slice(report, tensor_name, offset, scales_per_expert)
}

fn add_vectors(left: &[f32], right: &[f32]) -> Result<Vec<f32>> {
    if left.len() != right.len() {
        return Err(eyre!(
            "vector add length mismatch: left {}, right {}",
            left.len(),
            right.len()
        ));
    }
    Ok(left
        .iter()
        .copied()
        .zip(right.iter().copied())
        .map(|(left, right)| left + right)
        .collect())
}

fn top_k_report(
    name: &str,
    token: u32,
    cpu: Vec<ExpertScore>,
    metal: Vec<ExpertScore>,
) -> Result<MetalTopKProbeReport> {
    if cpu.len() != metal.len() {
        return Err(eyre!(
            "Metal top-k returned {} experts, CPU returned {}",
            metal.len(),
            cpu.len()
        ));
    }

    let mut indices_match = true;
    let mut max_logit_delta = 0.0f32;
    let mut max_weight_delta = 0.0f32;
    for (cpu, metal) in cpu.iter().zip(&metal) {
        indices_match &= cpu.index == metal.index;
        max_logit_delta = max_logit_delta.max((cpu.logit - metal.logit).abs());
        max_weight_delta = max_weight_delta.max((cpu.weight - metal.weight).abs());
    }

    Ok(MetalTopKProbeReport {
        name: name.to_string(),
        token,
        indices_match,
        max_logit_delta,
        max_weight_delta,
        cpu,
        metal,
    })
}

fn selected_logits_report(
    name: &str,
    token: u32,
    layers: usize,
    cpu: Vec<SelectedLogit>,
    metal: Vec<SelectedLogit>,
) -> Result<MetalSelectedLogitsProbeReport> {
    if cpu.len() != metal.len() {
        return Err(eyre!(
            "Metal selected-logits probe returned {} logits, CPU returned {}",
            metal.len(),
            cpu.len()
        ));
    }

    let mut max_abs_delta = 0.0f32;
    let mut sum_abs_delta = 0.0f64;
    for (cpu, metal) in cpu.iter().zip(&metal) {
        if cpu.token != metal.token {
            return Err(eyre!(
                "selected-logit token mismatch: CPU {}, Metal {}",
                cpu.token,
                metal.token
            ));
        }
        let delta = (cpu.logit - metal.logit).abs();
        max_abs_delta = max_abs_delta.max(delta);
        sum_abs_delta += delta as f64;
    }
    let mean_abs_delta = if cpu.is_empty() {
        0.0
    } else {
        (sum_abs_delta / cpu.len() as f64) as f32
    };

    Ok(MetalSelectedLogitsProbeReport {
        name: name.to_string(),
        token,
        layers,
        max_abs_delta,
        mean_abs_delta,
        cpu,
        metal,
    })
}

fn matvec_report(
    name: &str,
    token: u32,
    rows: usize,
    cols: usize,
    cpu: &[f32],
    metal: Vec<f32>,
) -> Result<MetalMatvecProbeReport> {
    if cpu.len() != metal.len() {
        return Err(eyre!(
            "Metal matvec returned {} values, CPU returned {}",
            metal.len(),
            cpu.len()
        ));
    }

    let mut max_abs_delta = 0.0f32;
    let mut sum_abs_delta = 0.0f64;
    for (cpu, metal) in cpu.iter().copied().zip(metal.iter().copied()) {
        let delta = (cpu - metal).abs();
        max_abs_delta = max_abs_delta.max(delta);
        sum_abs_delta += delta as f64;
    }
    let mean_abs_delta = if cpu.is_empty() {
        0.0
    } else {
        (sum_abs_delta / cpu.len() as f64) as f32
    };

    Ok(MetalMatvecProbeReport {
        name: name.to_string(),
        token,
        rows,
        cols,
        max_abs_delta,
        mean_abs_delta,
        cpu_first8: cpu.iter().copied().take(8).collect(),
        metal_first8: metal.into_iter().take(8).collect(),
    })
}

fn vector_report(
    name: &str,
    token: u32,
    position: usize,
    cpu: &[f32],
    metal: Vec<f32>,
) -> Result<MetalVectorProbeReport> {
    if cpu.len() != metal.len() {
        return Err(eyre!(
            "Metal vector probe returned {} values, CPU returned {}",
            metal.len(),
            cpu.len()
        ));
    }

    let mut max_abs_delta = 0.0f32;
    let mut sum_abs_delta = 0.0f64;
    for (cpu, metal) in cpu.iter().copied().zip(metal.iter().copied()) {
        let delta = (cpu - metal).abs();
        max_abs_delta = max_abs_delta.max(delta);
        sum_abs_delta += delta as f64;
    }
    let mean_abs_delta = if cpu.is_empty() {
        0.0
    } else {
        (sum_abs_delta / cpu.len() as f64) as f32
    };

    Ok(MetalVectorProbeReport {
        name: name.to_string(),
        token,
        position,
        values: cpu.len(),
        max_abs_delta,
        mean_abs_delta,
        cpu_first8: cpu.iter().copied().take(8).collect(),
        metal_first8: metal.into_iter().take(8).collect(),
    })
}
