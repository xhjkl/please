use eyre::{Result, eyre};

use super::{EXPERTS, MXFP4_BYTES_PER_GROUP, MXFP4_GROUPS, MetalOracleContext, platform};
use crate::model_store::{self, SafeTensorMap, SourceModelReport};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

pub(crate) fn bf16_linear_profile_name(weight_name: &str) -> String {
    for projection in ["q_proj", "k_proj", "v_proj", "o_proj"] {
        if weight_name.contains(projection) {
            return format!("op.bf16.{projection}");
        }
    }
    if weight_name.contains(".mlp.router.") {
        return "op.bf16.router".to_string();
    }
    "op.bf16.matvec".to_string()
}

pub(crate) fn mxfp4_profile_name(bias_name: &str) -> String {
    if bias_name.contains("gate_up_proj") {
        "op.mxfp4.gate_up".to_string()
    } else if bias_name.contains("down_proj") {
        "op.mxfp4.down".to_string()
    } else {
        "op.mxfp4.matvec".to_string()
    }
}

pub(crate) fn mxfp4_slab_blocks_len(rows: usize) -> Result<usize> {
    EXPERTS
        .checked_mul(rows)
        .and_then(|value| value.checked_mul(MXFP4_GROUPS))
        .and_then(|value| value.checked_mul(MXFP4_BYTES_PER_GROUP))
        .ok_or_else(|| eyre!("MXFP4 expert slab block length overflow"))
}

pub(crate) fn mxfp4_slab_scales_len(rows: usize) -> Result<usize> {
    EXPERTS
        .checked_mul(rows)
        .and_then(|value| value.checked_mul(MXFP4_GROUPS))
        .ok_or_else(|| eyre!("MXFP4 expert slab scale length overflow"))
}

#[derive(Clone)]
pub(crate) struct ResidentLayerExpertSlabs {
    pub(crate) gate_up_blocks: platform::U8Buffer,
    pub(crate) gate_up_scales: platform::U8Buffer,
    pub(crate) gate_up_bias: platform::Bf16MatrixBuffer,
    pub(crate) down_blocks: platform::U8Buffer,
    pub(crate) down_scales: platform::U8Buffer,
    pub(crate) down_bias: platform::Bf16MatrixBuffer,
}

pub(crate) struct GptOssWeights {
    pub(crate) embed: platform::Bf16MatrixBuffer,
    pub(crate) final_norm: platform::F32VectorBuffer,
    pub(crate) lm_head: platform::Bf16MatrixBuffer,
    pub(crate) layers: Vec<GptOssLayerWeights>,
}

pub(crate) struct GptOssLayerWeights {
    pub(crate) input_norm: platform::F32VectorBuffer,
    pub(crate) post_attn_norm: platform::F32VectorBuffer,
    pub(crate) attn: AttentionWeights,
    pub(crate) sparse_mlp: SparseMlpWeights,
}

pub(crate) struct AttentionWeights {
    pub(crate) q: Bf16Linear,
    pub(crate) k: Bf16Linear,
    pub(crate) v: Bf16Linear,
    pub(crate) o: Bf16Linear,
    pub(crate) sinks: platform::F32VectorBuffer,
}

pub(crate) struct SparseMlpWeights {
    pub(crate) router: Bf16Linear,
    pub(crate) experts: ResidentLayerExpertSlabs,
}

pub(crate) struct Bf16Linear {
    pub(crate) weight: platform::Bf16MatrixBuffer,
    pub(crate) bias: platform::F32VectorBuffer,
}

impl GptOssWeights {
    pub(crate) fn load(
        ctx: &MetalOracleContext,
        source: &SafeTensorMap,
        layers: usize,
    ) -> Result<Self> {
        let Some(lm_head) = ctx.lm_head.clone() else {
            return Err(eyre!(
                "Metal lm_head weight is not cached; construct MetalOracleContext::with_lm_head"
            ));
        };
        let embed = ctx.bf16_matrix_buffer_from_map(
            source,
            "model.embed_tokens.weight",
            "op.weight.embed_tokens",
        )?;
        let final_norm =
            ctx.bf16_vector_buffer_from_map(source, "model.norm.weight", "op.weight.final_norm")?;

        let mut layer_weights = Vec::with_capacity(layers);
        for layer in 0..layers {
            layer_weights.push(GptOssLayerWeights::load(ctx, source, layer)?);
        }

        Ok(Self {
            embed,
            final_norm,
            lm_head,
            layers: layer_weights,
        })
    }

    pub(crate) fn layer(&self, layer: usize) -> Result<&GptOssLayerWeights> {
        self.layers
            .get(layer)
            .ok_or_else(|| eyre!("typed gpt-oss weights have no layer {layer}"))
    }
}

impl GptOssLayerWeights {
    fn load(ctx: &MetalOracleContext, source: &SafeTensorMap, layer: usize) -> Result<Self> {
        let prefix = format!("model.layers.{layer}");
        let input_norm = ctx.bf16_vector_buffer_from_map(
            source,
            &format!("{prefix}.input_layernorm.weight"),
            "op.weight.input_layernorm",
        )?;
        let post_attn_norm = ctx.bf16_vector_buffer_from_map(
            source,
            &format!("{prefix}.post_attention_layernorm.weight"),
            "op.weight.post_attention_layernorm",
        )?;
        let attn = AttentionWeights {
            q: Bf16Linear::load(
                ctx,
                source,
                &format!("{prefix}.self_attn.q_proj.weight"),
                &format!("{prefix}.self_attn.q_proj.bias"),
                "op.bf16.q_proj",
            )?,
            k: Bf16Linear::load(
                ctx,
                source,
                &format!("{prefix}.self_attn.k_proj.weight"),
                &format!("{prefix}.self_attn.k_proj.bias"),
                "op.bf16.k_proj",
            )?,
            v: Bf16Linear::load(
                ctx,
                source,
                &format!("{prefix}.self_attn.v_proj.weight"),
                &format!("{prefix}.self_attn.v_proj.bias"),
                "op.bf16.v_proj",
            )?,
            o: Bf16Linear::load(
                ctx,
                source,
                &format!("{prefix}.self_attn.o_proj.weight"),
                &format!("{prefix}.self_attn.o_proj.bias"),
                "op.bf16.o_proj",
            )?,
            sinks: ctx.bf16_vector_buffer_from_map(
                source,
                &format!("{prefix}.self_attn.sinks"),
                "op.weight.attention_sinks",
            )?,
        };
        let sparse_mlp = SparseMlpWeights {
            router: Bf16Linear::load(
                ctx,
                source,
                &format!("{prefix}.mlp.router.weight"),
                &format!("{prefix}.mlp.router.bias"),
                "op.bf16.router",
            )?,
            experts: ctx
                .mxfp4_layer_expert_slabs_from_map(source, &format!("{prefix}.mlp.experts"))?,
        };
        Ok(Self {
            input_norm,
            post_attn_norm,
            attn,
            sparse_mlp,
        })
    }
}

impl Bf16Linear {
    fn load(
        ctx: &MetalOracleContext,
        source: &SafeTensorMap,
        weight_name: &str,
        bias_name: &str,
        op_name: &str,
    ) -> Result<Self> {
        Ok(Self {
            weight: ctx.bf16_matrix_buffer_from_map(source, weight_name, op_name)?,
            bias: ctx.bf16_vector_buffer_from_map(source, bias_name, op_name)?,
        })
    }
}

#[derive(Default)]
pub(crate) struct ResidentWeights {
    bf16_matrices: Mutex<HashMap<String, Arc<model_store::Bf16Matrix>>>,
    bf16_vectors: Mutex<HashMap<String, Arc<Vec<f32>>>>,
    bf16_rows: Mutex<HashMap<(String, usize), Arc<Vec<f32>>>>,
    u8_slices: Mutex<HashMap<(String, usize, usize), Arc<Vec<u8>>>>,
}

impl ResidentWeights {
    pub(crate) fn bf16_matrix(
        &self,
        report: &SourceModelReport,
        tensor_name: &str,
    ) -> Result<Arc<model_store::Bf16Matrix>> {
        if let Some(value) = self.bf16_matrices.lock().unwrap().get(tensor_name).cloned() {
            return Ok(value);
        }
        let value = Arc::new(model_store::read_bf16_matrix(report, tensor_name)?);
        self.bf16_matrices
            .lock()
            .unwrap()
            .insert(tensor_name.to_string(), value.clone());
        Ok(value)
    }

    pub(crate) fn bf16_vector(
        &self,
        report: &SourceModelReport,
        tensor_name: &str,
    ) -> Result<Arc<Vec<f32>>> {
        if let Some(value) = self.bf16_vectors.lock().unwrap().get(tensor_name).cloned() {
            return Ok(value);
        }
        let value = Arc::new(model_store::read_bf16_vector(report, tensor_name)?);
        self.bf16_vectors
            .lock()
            .unwrap()
            .insert(tensor_name.to_string(), value.clone());
        Ok(value)
    }

    pub(crate) fn bf16_vector_from_map(
        &self,
        source: &SafeTensorMap,
        tensor_name: &str,
    ) -> Result<Arc<Vec<f32>>> {
        if let Some(value) = self.bf16_vectors.lock().unwrap().get(tensor_name).cloned() {
            return Ok(value);
        }
        let value = Arc::new(source.read_bf16_vector(tensor_name)?);
        self.bf16_vectors
            .lock()
            .unwrap()
            .insert(tensor_name.to_string(), value.clone());
        Ok(value)
    }

    pub(crate) fn bf16_matrix_row(
        &self,
        report: &SourceModelReport,
        tensor_name: &str,
        row: usize,
    ) -> Result<Arc<Vec<f32>>> {
        let key = (tensor_name.to_string(), row);
        if let Some(value) = self.bf16_rows.lock().unwrap().get(&key).cloned() {
            return Ok(value);
        }
        let value = Arc::new(model_store::read_bf16_matrix_row(report, tensor_name, row)?);
        self.bf16_rows.lock().unwrap().insert(key, value.clone());
        Ok(value)
    }

    pub(crate) fn u8_tensor_slice(
        &self,
        report: &SourceModelReport,
        tensor_name: &str,
        element_offset: usize,
        element_len: usize,
    ) -> Result<Arc<Vec<u8>>> {
        let key = (tensor_name.to_string(), element_offset, element_len);
        if let Some(value) = self.u8_slices.lock().unwrap().get(&key).cloned() {
            return Ok(value);
        }
        let value = Arc::new(model_store::read_u8_tensor_slice(
            report,
            tensor_name,
            element_offset,
            element_len,
        )?);
        self.u8_slices.lock().unwrap().insert(key, value.clone());
        Ok(value)
    }
}
