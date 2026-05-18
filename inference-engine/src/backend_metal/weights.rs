use eyre::{Result, eyre};

use super::{MetalRuntime, platform};
use crate::model_store::gguf::{
    F32MatrixBytes, F32VectorBytes, GptOss20bGguf, GptOss20bGgufLayer, Q8_0MatrixBytes,
};

pub(crate) struct GptOssWeights {
    pub(crate) embed: platform::Q8_0MatrixBuffer,
    pub(crate) final_norm: platform::F32VectorBuffer,
    pub(crate) lm_head: platform::Q8_0MatrixBuffer,
    pub(crate) layers: Vec<GptOssLayerWeights>,
}

pub(crate) struct GptOssLayerWeights {
    pub(crate) input_norm: platform::F32VectorBuffer,
    pub(crate) post_attn_norm: platform::F32VectorBuffer,
    pub(crate) attn: AttentionWeights,
    pub(crate) sparse_mlp: SparseMlpWeights,
}

pub(crate) struct AttentionWeights {
    pub(crate) q: Q8_0Linear,
    pub(crate) k: Q8_0Linear,
    pub(crate) v: Q8_0Linear,
    pub(crate) o: Q8_0Linear,
    pub(crate) sinks: platform::F32VectorBuffer,
}

pub(crate) struct SparseMlpWeights {
    pub(crate) router: F32Linear,
    pub(crate) experts_carousel: ExpertsCarousel,
}

pub(crate) struct Q8_0Linear {
    pub(crate) weight: platform::Q8_0MatrixBuffer,
    pub(crate) bias: platform::F32VectorBuffer,
}

pub(crate) struct F32Linear {
    pub(crate) weight: platform::F32MatrixBuffer,
    pub(crate) bias: platform::F32VectorBuffer,
}

#[derive(Clone)]
pub(crate) struct ExpertsCarousel {
    pub(crate) gate: platform::Mxfp4ExpertTensorBuffer,
    pub(crate) gate_bias: platform::F32MatrixBuffer,
    pub(crate) up: platform::Mxfp4ExpertTensorBuffer,
    pub(crate) up_bias: platform::F32MatrixBuffer,
    pub(crate) down: platform::Mxfp4ExpertTensorBuffer,
    pub(crate) down_bias: platform::F32MatrixBuffer,
}

impl GptOssWeights {
    pub(crate) fn load(
        ctx: &MetalRuntime,
        source: &GptOss20bGguf<'_>,
        layers: usize,
    ) -> Result<Self> {
        let embed = ctx.gguf_q8_0_matrix_buffer(source.embedding_bytes()?, "gguf.embed")?;
        let final_norm =
            ctx.gguf_f32_vector_buffer(source.final_norm_bytes()?, "gguf.final_norm")?;
        let lm_head = ctx.gguf_q8_0_matrix_buffer(source.lm_head_bytes()?, "gguf.lm_head")?;

        let mut layer_weights = Vec::with_capacity(layers);
        for layer in 0..layers {
            layer_weights.push(GptOssLayerWeights::load(ctx, &source.layer(layer)?)?);
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
            .ok_or_else(|| eyre!("typed GGUF gpt-oss weights have no layer {layer}"))
    }
}

impl GptOssLayerWeights {
    fn load(ctx: &MetalRuntime, source: &GptOss20bGgufLayer<'_>) -> Result<Self> {
        let input_norm = ctx.gguf_f32_vector_buffer(source.attn_norm_bytes()?, "gguf.attn_norm")?;
        let post_attn_norm = ctx.gguf_f32_vector_buffer(
            source.post_attention_norm_bytes()?,
            "gguf.post_attention_norm",
        )?;
        let attn = AttentionWeights {
            q: Q8_0Linear::load(
                ctx,
                source.q_bytes()?,
                source.q_bias_bytes()?,
                "gguf.attn_q",
            )?,
            k: Q8_0Linear::load(
                ctx,
                source.k_bytes()?,
                source.k_bias_bytes()?,
                "gguf.attn_k",
            )?,
            v: Q8_0Linear::load(
                ctx,
                source.v_bytes()?,
                source.v_bias_bytes()?,
                "gguf.attn_v",
            )?,
            o: Q8_0Linear::load(
                ctx,
                source.o_bytes()?,
                source.o_bias_bytes()?,
                "gguf.attn_o",
            )?,
            sinks: ctx.gguf_f32_vector_buffer(source.sinks_bytes()?, "gguf.attn_sinks")?,
        };
        let sparse_mlp = SparseMlpWeights {
            router: F32Linear::load(
                ctx,
                source.router_bytes()?,
                source.router_bias_bytes()?,
                "gguf.router",
            )?,
            experts_carousel: ExpertsCarousel::load(ctx, source)?,
        };
        Ok(Self {
            input_norm,
            post_attn_norm,
            attn,
            sparse_mlp,
        })
    }
}

impl Q8_0Linear {
    fn load(
        ctx: &MetalRuntime,
        weight: Q8_0MatrixBytes<'_>,
        bias: F32VectorBytes<'_>,
        op_name: &str,
    ) -> Result<Self> {
        Ok(Self {
            weight: ctx.gguf_q8_0_matrix_buffer(weight, op_name)?,
            bias: ctx.gguf_f32_vector_buffer(bias, op_name)?,
        })
    }
}

impl F32Linear {
    fn load(
        ctx: &MetalRuntime,
        weight: F32MatrixBytes<'_>,
        bias: F32VectorBytes<'_>,
        op_name: &str,
    ) -> Result<Self> {
        Ok(Self {
            weight: ctx.gguf_f32_matrix_buffer(weight, op_name)?,
            bias: ctx.gguf_f32_vector_buffer(bias, op_name)?,
        })
    }
}

impl ExpertsCarousel {
    fn load(ctx: &MetalRuntime, source: &GptOss20bGgufLayer<'_>) -> Result<Self> {
        Ok(Self {
            gate: ctx
                .gguf_mxfp4_expert_tensor_buffer(source.gate_experts_bytes()?, "gguf.mxfp4.gate")?,
            gate_bias: ctx
                .gguf_f32_matrix_buffer(source.gate_experts_bias_bytes()?, "gguf.mxfp4.gate")?,
            up: ctx.gguf_mxfp4_expert_tensor_buffer(source.up_experts_bytes()?, "gguf.mxfp4.up")?,
            up_bias: ctx
                .gguf_f32_matrix_buffer(source.up_experts_bias_bytes()?, "gguf.mxfp4.up")?,
            down: ctx
                .gguf_mxfp4_expert_tensor_buffer(source.down_experts_bytes()?, "gguf.mxfp4.down")?,
            down_bias: ctx
                .gguf_f32_matrix_buffer(source.down_experts_bias_bytes()?, "gguf.mxfp4.down")?,
        })
    }
}
