use eyre::{Result, eyre};

use super::{EXPERTS, MXFP4_BYTES_PER_GROUP, MXFP4_GROUPS, platform};
use crate::model_store::{self, SourceModelReport};
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
