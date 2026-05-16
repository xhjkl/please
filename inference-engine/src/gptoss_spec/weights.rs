use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

use crate::model_store::{SourceModelReport, TensorHeader};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WeightManifest {
    pub architecture: String,
    pub layout_version: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GptOssSourceValidation {
    pub expected_tensor_count: usize,
    pub actual_tensor_count: usize,
    pub missing_tensors: Vec<String>,
    pub unexpected_tensors: Vec<String>,
    pub shape_mismatches: Vec<TensorShapeMismatch>,
}

impl GptOssSourceValidation {
    pub fn is_ok(&self) -> bool {
        self.missing_tensors.is_empty()
            && self.unexpected_tensors.is_empty()
            && self.shape_mismatches.is_empty()
            && self.expected_tensor_count == self.actual_tensor_count
    }

    pub fn render_for_cli(&self) -> String {
        let mut out = String::new();
        out.push_str("\ngpt-oss-20b source layout:\n");
        out.push_str(&format!(
            "- expected tensors: {}\n",
            self.expected_tensor_count
        ));
        out.push_str(&format!("- actual tensors: {}\n", self.actual_tensor_count));
        out.push_str(&format!(
            "- status: {}\n",
            if self.is_ok() {
                "ok"
            } else {
                "needs attention"
            }
        ));

        if !self.missing_tensors.is_empty() {
            out.push_str("- missing tensors:\n");
            for name in self.missing_tensors.iter().take(20) {
                out.push_str(&format!("  - {name}\n"));
            }
            if self.missing_tensors.len() > 20 {
                out.push_str(&format!(
                    "  - ... {} more\n",
                    self.missing_tensors.len() - 20
                ));
            }
        }

        if !self.unexpected_tensors.is_empty() {
            out.push_str("- unexpected tensors:\n");
            for name in self.unexpected_tensors.iter().take(20) {
                out.push_str(&format!("  - {name}\n"));
            }
            if self.unexpected_tensors.len() > 20 {
                out.push_str(&format!(
                    "  - ... {} more\n",
                    self.unexpected_tensors.len() - 20
                ));
            }
        }

        if !self.shape_mismatches.is_empty() {
            out.push_str("- shape/dtype mismatches:\n");
            for mismatch in self.shape_mismatches.iter().take(20) {
                out.push_str(&format!(
                    "  - {}: expected {} {:?}, got {} {:?}\n",
                    mismatch.name,
                    mismatch.expected_dtype,
                    mismatch.expected_shape,
                    mismatch.actual_dtype,
                    mismatch.actual_shape
                ));
            }
            if self.shape_mismatches.len() > 20 {
                out.push_str(&format!(
                    "  - ... {} more\n",
                    self.shape_mismatches.len() - 20
                ));
            }
        }

        out
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorShapeMismatch {
    pub name: String,
    pub expected_dtype: String,
    pub expected_shape: Vec<u64>,
    pub actual_dtype: String,
    pub actual_shape: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ExpectedTensor {
    dtype: &'static str,
    shape: Vec<u64>,
}

pub fn validate_gpt_oss_20b_source(report: &SourceModelReport) -> GptOssSourceValidation {
    let expected = expected_gpt_oss_20b_tensors();
    let actual = actual_tensors_by_name(report);
    let expected_names: BTreeSet<String> = expected.keys().cloned().collect();
    let actual_names: BTreeSet<String> = actual.keys().cloned().collect();

    let missing_tensors = expected_names.difference(&actual_names).cloned().collect();
    let unexpected_tensors = actual_names.difference(&expected_names).cloned().collect();

    let mut shape_mismatches = Vec::new();
    for (name, expected) in &expected {
        let Some(actual) = actual.get(name) else {
            continue;
        };
        if actual.dtype != expected.dtype || actual.shape != expected.shape {
            shape_mismatches.push(TensorShapeMismatch {
                name: name.clone(),
                expected_dtype: expected.dtype.to_string(),
                expected_shape: expected.shape.clone(),
                actual_dtype: actual.dtype.clone(),
                actual_shape: actual.shape.clone(),
            });
        }
    }

    GptOssSourceValidation {
        expected_tensor_count: expected.len(),
        actual_tensor_count: actual.len(),
        missing_tensors,
        unexpected_tensors,
        shape_mismatches,
    }
}

fn actual_tensors_by_name(report: &SourceModelReport) -> BTreeMap<String, TensorHeader> {
    let mut out = BTreeMap::new();
    for shard in &report.shards {
        for tensor in &shard.tensors {
            out.insert(tensor.name.clone(), tensor.clone());
        }
    }
    out
}

fn expected_gpt_oss_20b_tensors() -> BTreeMap<String, ExpectedTensor> {
    let mut expected = BTreeMap::new();

    insert(
        &mut expected,
        "model.embed_tokens.weight",
        "BF16",
        &[201088, 2880],
    );
    insert(&mut expected, "model.norm.weight", "BF16", &[2880]);
    insert(&mut expected, "lm_head.weight", "BF16", &[201088, 2880]);

    for layer in 0..24 {
        let prefix = format!("model.layers.{layer}");
        insert(
            &mut expected,
            &format!("{prefix}.input_layernorm.weight"),
            "BF16",
            &[2880],
        );
        insert(
            &mut expected,
            &format!("{prefix}.post_attention_layernorm.weight"),
            "BF16",
            &[2880],
        );

        insert(
            &mut expected,
            &format!("{prefix}.self_attn.q_proj.weight"),
            "BF16",
            &[4096, 2880],
        );
        insert(
            &mut expected,
            &format!("{prefix}.self_attn.q_proj.bias"),
            "BF16",
            &[4096],
        );
        insert(
            &mut expected,
            &format!("{prefix}.self_attn.k_proj.weight"),
            "BF16",
            &[512, 2880],
        );
        insert(
            &mut expected,
            &format!("{prefix}.self_attn.k_proj.bias"),
            "BF16",
            &[512],
        );
        insert(
            &mut expected,
            &format!("{prefix}.self_attn.v_proj.weight"),
            "BF16",
            &[512, 2880],
        );
        insert(
            &mut expected,
            &format!("{prefix}.self_attn.v_proj.bias"),
            "BF16",
            &[512],
        );
        insert(
            &mut expected,
            &format!("{prefix}.self_attn.o_proj.weight"),
            "BF16",
            &[2880, 4096],
        );
        insert(
            &mut expected,
            &format!("{prefix}.self_attn.o_proj.bias"),
            "BF16",
            &[2880],
        );
        insert(
            &mut expected,
            &format!("{prefix}.self_attn.sinks"),
            "BF16",
            &[64],
        );

        insert(
            &mut expected,
            &format!("{prefix}.mlp.router.weight"),
            "BF16",
            &[32, 2880],
        );
        insert(
            &mut expected,
            &format!("{prefix}.mlp.router.bias"),
            "BF16",
            &[32],
        );
        insert(
            &mut expected,
            &format!("{prefix}.mlp.experts.gate_up_proj_bias"),
            "BF16",
            &[32, 5760],
        );
        insert(
            &mut expected,
            &format!("{prefix}.mlp.experts.gate_up_proj_blocks"),
            "U8",
            &[32, 5760, 90, 16],
        );
        insert(
            &mut expected,
            &format!("{prefix}.mlp.experts.gate_up_proj_scales"),
            "U8",
            &[32, 5760, 90],
        );
        insert(
            &mut expected,
            &format!("{prefix}.mlp.experts.down_proj_bias"),
            "BF16",
            &[32, 2880],
        );
        insert(
            &mut expected,
            &format!("{prefix}.mlp.experts.down_proj_blocks"),
            "U8",
            &[32, 2880, 90, 16],
        );
        insert(
            &mut expected,
            &format!("{prefix}.mlp.experts.down_proj_scales"),
            "U8",
            &[32, 2880, 90],
        );
    }

    expected
}

fn insert(
    expected: &mut BTreeMap<String, ExpectedTensor>,
    name: &str,
    dtype: &'static str,
    shape: &[u64],
) {
    expected.insert(
        name.to_string(),
        ExpectedTensor {
            dtype,
            shape: shape.to_vec(),
        },
    );
}
