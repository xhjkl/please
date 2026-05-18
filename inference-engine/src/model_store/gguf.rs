use eyre::{Result, eyre};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
#[cfg(unix)]
use std::ptr::NonNull;

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const DEFAULT_ALIGNMENT: u64 = 32;
const QK8_0: usize = 32;
const BLOCK_Q8_0_BYTES: usize = 2 + QK8_0;
const QK_MXFP4: usize = 32;
const BLOCK_MXFP4_BYTES: usize = 1 + 16;
const GPT_OSS_20B_LAYERS: usize = 24;
const GPT_OSS_20B_D_MODEL: u64 = 2880;
const GPT_OSS_20B_VOCAB: u64 = 201088;
const GPT_OSS_20B_Q_WIDTH: u64 = 4096;
const GPT_OSS_20B_KV_WIDTH: u64 = 512;
const GPT_OSS_20B_EXPERTS: u64 = 32;
const GPT_OSS_20B_SINKS: u64 = 64;

#[derive(Debug, Clone)]
struct GgufIndex {
    tensors: Vec<GgufTensor>,
}

#[derive(Debug, Clone)]
struct GgufTensor {
    name: String,
    dimensions: Vec<u64>,
    ggml_type: u32,
    absolute_offset: u64,
}

pub struct GgufMap {
    index: GgufIndex,
    bytes: MappedGgufBytes,
    tensors: BTreeMap<String, usize>,
}

#[cfg(unix)]
struct MappedGgufBytes {
    ptr: NonNull<u8>,
    len: usize,
}

#[cfg(not(unix))]
struct MappedGgufBytes {
    bytes: Vec<u8>,
}

#[cfg(unix)]
unsafe impl Send for MappedGgufBytes {}

#[cfg(unix)]
unsafe impl Sync for MappedGgufBytes {}

pub struct F32VectorBytes<'a> {
    pub len: usize,
    pub bytes: &'a [u8],
}

pub struct F32MatrixBytes<'a> {
    pub rows: usize,
    pub cols: usize,
    pub bytes: &'a [u8],
}

pub struct Q8_0MatrixBytes<'a> {
    pub rows: usize,
    pub cols: usize,
    pub bytes: &'a [u8],
}

pub struct Mxfp4ExpertTensorBytes<'a> {
    pub experts: usize,
    pub rows: usize,
    pub cols: usize,
    pub bytes: &'a [u8],
}

#[derive(Debug, Clone)]
struct GptOss20bGgufValidation {
    expected_tensor_count: usize,
    actual_tensor_count: usize,
    missing_tensors: Vec<String>,
    unexpected_tensors: Vec<String>,
    mismatches: Vec<GgufTensorMismatch>,
}

#[derive(Debug, Clone)]
struct GgufTensorMismatch {
    name: String,
    expected_type: u32,
    expected_dimensions: Vec<u64>,
    actual_type: u32,
    actual_dimensions: Vec<u64>,
}

pub struct GptOss20bGguf<'a> {
    map: &'a GgufMap,
}

pub struct GptOss20bGgufLayer<'a> {
    map: &'a GgufMap,
    index: usize,
}

#[derive(Debug, Clone, Copy)]
enum LayerTensor {
    AttnNormWeight,
    PostAttentionNormWeight,
    AttnQBias,
    AttnQWeight,
    AttnKBias,
    AttnKWeight,
    AttnVBias,
    AttnVWeight,
    AttnOutputBias,
    AttnOutputWeight,
    AttnSinksWeight,
    FfnGateInputBias,
    FfnGateInputWeight,
    FfnGateExpertsBias,
    FfnGateExpertsWeight,
    FfnUpExpertsBias,
    FfnUpExpertsWeight,
    FfnDownExpertsBias,
    FfnDownExpertsWeight,
}

impl fmt::Display for LayerTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let suffix = match self {
            Self::AttnNormWeight => "attn_norm.weight",
            Self::PostAttentionNormWeight => "post_attention_norm.weight",
            Self::AttnQBias => "attn_q.bias",
            Self::AttnQWeight => "attn_q.weight",
            Self::AttnKBias => "attn_k.bias",
            Self::AttnKWeight => "attn_k.weight",
            Self::AttnVBias => "attn_v.bias",
            Self::AttnVWeight => "attn_v.weight",
            Self::AttnOutputBias => "attn_output.bias",
            Self::AttnOutputWeight => "attn_output.weight",
            Self::AttnSinksWeight => "attn_sinks.weight",
            Self::FfnGateInputBias => "ffn_gate_inp.bias",
            Self::FfnGateInputWeight => "ffn_gate_inp.weight",
            Self::FfnGateExpertsBias => "ffn_gate_exps.bias",
            Self::FfnGateExpertsWeight => "ffn_gate_exps.weight",
            Self::FfnUpExpertsBias => "ffn_up_exps.bias",
            Self::FfnUpExpertsWeight => "ffn_up_exps.weight",
            Self::FfnDownExpertsBias => "ffn_down_exps.bias",
            Self::FfnDownExpertsWeight => "ffn_down_exps.weight",
        };
        f.write_str(suffix)
    }
}

impl GgufTensor {
    fn nbytes(&self) -> Result<usize> {
        let elements = self
            .dimensions
            .iter()
            .try_fold(1u64, |acc, dim| acc.checked_mul(*dim))
            .ok_or_else(|| eyre!("GGUF tensor {} element count overflow", self.name))?;
        let bytes = match self.ggml_type {
            0 => elements
                .checked_mul(4)
                .ok_or_else(|| eyre!("GGUF tensor {} byte count overflow", self.name))?,
            8 => block_tensor_bytes(&self.name, &self.dimensions, QK8_0, BLOCK_Q8_0_BYTES)?,
            39 => block_tensor_bytes(&self.name, &self.dimensions, QK_MXFP4, BLOCK_MXFP4_BYTES)?,
            value => {
                return Err(eyre!(
                    "GGUF tensor {} has unsupported type {} ({value}) for byte sizing",
                    self.name,
                    ggml_type_name(value)
                ));
            }
        };
        usize::try_from(bytes).map_err(|_| eyre!("GGUF tensor {} is too large", self.name))
    }
}

impl GptOss20bGgufValidation {
    fn is_ok(&self) -> bool {
        self.expected_tensor_count == self.actual_tensor_count
            && self.missing_tensors.is_empty()
            && self.unexpected_tensors.is_empty()
            && self.mismatches.is_empty()
    }
}

impl fmt::Display for GptOss20bGgufValidation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\ngpt-oss-20b GGUF layout:")?;
        writeln!(f, "- expected tensors: {}", self.expected_tensor_count)?;
        writeln!(f, "- actual tensors: {}", self.actual_tensor_count)?;
        writeln!(
            f,
            "- status: {}",
            if self.is_ok() {
                "ok"
            } else {
                "needs attention"
            }
        )?;
        if !self.missing_tensors.is_empty() {
            writeln!(f, "- missing tensors:")?;
            for name in self.missing_tensors.iter().take(20) {
                writeln!(f, "  - {name}")?;
            }
            if self.missing_tensors.len() > 20 {
                writeln!(f, "  - ... {} more\n", self.missing_tensors.len() - 20)?;
            }
        }
        if !self.unexpected_tensors.is_empty() {
            writeln!(f, "- unexpected tensors:")?;
            for name in self.unexpected_tensors.iter().take(20) {
                writeln!(f, "  - {name}")?;
            }
            if self.unexpected_tensors.len() > 20 {
                writeln!(f, "  - ... {} more\n", self.unexpected_tensors.len() - 20)?;
            }
        }
        if !self.mismatches.is_empty() {
            writeln!(f, "- shape/type mismatches:")?;
            for mismatch in self.mismatches.iter().take(20) {
                writeln!(
                    f,
                    "  - {}: expected {} {:?}, got {} {:?}\n",
                    mismatch.name,
                    ggml_type_name(mismatch.expected_type),
                    mismatch.expected_dimensions,
                    ggml_type_name(mismatch.actual_type),
                    mismatch.actual_dimensions
                )?;
            }
            if self.mismatches.len() > 20 {
                writeln!(f, "  - ... {} more", self.mismatches.len() - 20)?;
            }
        }
        Ok(())
    }
}

impl GgufMap {
    pub fn open(path: &Path) -> Result<Self> {
        let index = inspect_gguf(path)?;
        let bytes = MappedGgufBytes::open(path)?;
        let mut tensors = BTreeMap::new();
        for (tensor_index, tensor) in index.tensors.iter().enumerate() {
            if tensors.insert(tensor.name.clone(), tensor_index).is_some() {
                return Err(eyre!("duplicate GGUF tensor {}", tensor.name));
            }
        }
        Ok(Self {
            index,
            bytes,
            tensors,
        })
    }

    pub fn open_canonical() -> Result<Self> {
        Self::open(&canonical_gguf_path())
    }

    fn tensor(&self, name: &str) -> Result<&GgufTensor> {
        let index = self
            .tensors
            .get(name)
            .ok_or_else(|| eyre!("GGUF tensor {name} was not found"))?;
        Ok(&self.index.tensors[*index])
    }

    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8]> {
        let tensor = self.tensor(name)?;
        let len = tensor.nbytes()?;
        let start = usize::try_from(tensor.absolute_offset)
            .map_err(|_| eyre!("GGUF tensor {} offset is too large", tensor.name))?;
        let end = start
            .checked_add(len)
            .ok_or_else(|| eyre!("GGUF tensor {} byte range overflow", tensor.name))?;
        self.bytes
            .as_slice()
            .get(start..end)
            .ok_or_else(|| eyre!("GGUF tensor {} extends past file end", tensor.name))
    }

    pub fn f32_vector_bytes(&self, name: &str) -> Result<F32VectorBytes<'_>> {
        let tensor = self.tensor(name)?;
        if tensor.ggml_type != 0 {
            return Err(eyre!(
                "GGUF tensor {name} has type {}, expected f32",
                ggml_type_name(tensor.ggml_type)
            ));
        }
        if tensor.dimensions.len() != 1 {
            return Err(eyre!("GGUF tensor {name} is not a vector"));
        }
        let len = usize::try_from(tensor.dimensions[0])
            .map_err(|_| eyre!("GGUF tensor {name} vector length is too large"))?;
        Ok(F32VectorBytes {
            len,
            bytes: self.tensor_bytes(name)?,
        })
    }

    pub fn f32_matrix_bytes(&self, name: &str) -> Result<F32MatrixBytes<'_>> {
        let tensor = self.tensor(name)?;
        if tensor.ggml_type != 0 {
            return Err(eyre!(
                "GGUF tensor {name} has type {}, expected f32",
                ggml_type_name(tensor.ggml_type)
            ));
        }
        if tensor.dimensions.len() != 2 {
            return Err(eyre!("GGUF tensor {name} is not a matrix"));
        }
        Ok(F32MatrixBytes {
            rows: usize::try_from(tensor.dimensions[1])
                .map_err(|_| eyre!("GGUF tensor {name} row count is too large"))?,
            cols: usize::try_from(tensor.dimensions[0])
                .map_err(|_| eyre!("GGUF tensor {name} column count is too large"))?,
            bytes: self.tensor_bytes(name)?,
        })
    }

    pub fn q8_0_matrix_bytes(&self, name: &str) -> Result<Q8_0MatrixBytes<'_>> {
        let tensor = self.tensor(name)?;
        if tensor.ggml_type != 8 {
            return Err(eyre!(
                "GGUF tensor {name} has type {}, expected q8_0",
                ggml_type_name(tensor.ggml_type)
            ));
        }
        if tensor.dimensions.len() != 2 {
            return Err(eyre!("GGUF tensor {name} is not a matrix"));
        }
        Ok(Q8_0MatrixBytes {
            rows: usize::try_from(tensor.dimensions[1])
                .map_err(|_| eyre!("GGUF tensor {name} row count is too large"))?,
            cols: usize::try_from(tensor.dimensions[0])
                .map_err(|_| eyre!("GGUF tensor {name} column count is too large"))?,
            bytes: self.tensor_bytes(name)?,
        })
    }

    pub fn mxfp4_expert_tensor_bytes(&self, name: &str) -> Result<Mxfp4ExpertTensorBytes<'_>> {
        let tensor = self.tensor(name)?;
        if tensor.ggml_type != 39 {
            return Err(eyre!(
                "GGUF tensor {name} has type {}, expected mxfp4",
                ggml_type_name(tensor.ggml_type)
            ));
        }
        if tensor.dimensions.len() != 3 {
            return Err(eyre!("GGUF tensor {name} is not an expert tensor"));
        }
        Ok(Mxfp4ExpertTensorBytes {
            experts: usize::try_from(tensor.dimensions[2])
                .map_err(|_| eyre!("GGUF tensor {name} expert count is too large"))?,
            rows: usize::try_from(tensor.dimensions[1])
                .map_err(|_| eyre!("GGUF tensor {name} row count is too large"))?,
            cols: usize::try_from(tensor.dimensions[0])
                .map_err(|_| eyre!("GGUF tensor {name} column count is too large"))?,
            bytes: self.tensor_bytes(name)?,
        })
    }
}

impl<'a> GptOss20bGguf<'a> {
    pub fn new(map: &'a GgufMap) -> Result<Self> {
        let validation = validate_gpt_oss_20b_gguf(&map.index);
        if !validation.is_ok() {
            return Err(eyre!("GGUF does not match gpt-oss-20b layout:{validation}"));
        }
        Ok(Self { map })
    }

    pub fn embedding_bytes(&self) -> Result<Q8_0MatrixBytes<'_>> {
        self.map.q8_0_matrix_bytes("token_embd.weight")
    }

    pub fn lm_head_bytes(&self) -> Result<Q8_0MatrixBytes<'_>> {
        self.map.q8_0_matrix_bytes("output.weight")
    }

    pub fn final_norm_bytes(&self) -> Result<F32VectorBytes<'_>> {
        self.map.f32_vector_bytes("output_norm.weight")
    }

    pub fn layer(&self, index: usize) -> Result<GptOss20bGgufLayer<'a>> {
        if index >= GPT_OSS_20B_LAYERS {
            return Err(eyre!(
                "layer {index} is outside gpt-oss-20b depth {GPT_OSS_20B_LAYERS}"
            ));
        }
        Ok(GptOss20bGgufLayer {
            map: self.map,
            index,
        })
    }
}

impl GptOss20bGgufLayer<'_> {
    pub fn attn_norm_bytes(&self) -> Result<F32VectorBytes<'_>> {
        self.map
            .f32_vector_bytes(&self.tensor_name(LayerTensor::AttnNormWeight))
    }

    pub fn post_attention_norm_bytes(&self) -> Result<F32VectorBytes<'_>> {
        self.map
            .f32_vector_bytes(&self.tensor_name(LayerTensor::PostAttentionNormWeight))
    }

    pub fn q_bias_bytes(&self) -> Result<F32VectorBytes<'_>> {
        self.map
            .f32_vector_bytes(&self.tensor_name(LayerTensor::AttnQBias))
    }

    pub fn q_bytes(&self) -> Result<Q8_0MatrixBytes<'_>> {
        self.map
            .q8_0_matrix_bytes(&self.tensor_name(LayerTensor::AttnQWeight))
    }

    pub fn k_bias_bytes(&self) -> Result<F32VectorBytes<'_>> {
        self.map
            .f32_vector_bytes(&self.tensor_name(LayerTensor::AttnKBias))
    }

    pub fn k_bytes(&self) -> Result<Q8_0MatrixBytes<'_>> {
        self.map
            .q8_0_matrix_bytes(&self.tensor_name(LayerTensor::AttnKWeight))
    }

    pub fn v_bias_bytes(&self) -> Result<F32VectorBytes<'_>> {
        self.map
            .f32_vector_bytes(&self.tensor_name(LayerTensor::AttnVBias))
    }

    pub fn v_bytes(&self) -> Result<Q8_0MatrixBytes<'_>> {
        self.map
            .q8_0_matrix_bytes(&self.tensor_name(LayerTensor::AttnVWeight))
    }

    pub fn o_bias_bytes(&self) -> Result<F32VectorBytes<'_>> {
        self.map
            .f32_vector_bytes(&self.tensor_name(LayerTensor::AttnOutputBias))
    }

    pub fn o_bytes(&self) -> Result<Q8_0MatrixBytes<'_>> {
        self.map
            .q8_0_matrix_bytes(&self.tensor_name(LayerTensor::AttnOutputWeight))
    }

    pub fn sinks_bytes(&self) -> Result<F32VectorBytes<'_>> {
        self.map
            .f32_vector_bytes(&self.tensor_name(LayerTensor::AttnSinksWeight))
    }

    pub fn router_bias_bytes(&self) -> Result<F32VectorBytes<'_>> {
        self.map
            .f32_vector_bytes(&self.tensor_name(LayerTensor::FfnGateInputBias))
    }

    pub fn router_bytes(&self) -> Result<F32MatrixBytes<'_>> {
        self.map
            .f32_matrix_bytes(&self.tensor_name(LayerTensor::FfnGateInputWeight))
    }

    pub fn gate_experts_bytes(&self) -> Result<Mxfp4ExpertTensorBytes<'_>> {
        self.map
            .mxfp4_expert_tensor_bytes(&self.tensor_name(LayerTensor::FfnGateExpertsWeight))
    }

    pub fn gate_experts_bias_bytes(&self) -> Result<F32MatrixBytes<'_>> {
        self.map
            .f32_matrix_bytes(&self.tensor_name(LayerTensor::FfnGateExpertsBias))
    }

    pub fn up_experts_bytes(&self) -> Result<Mxfp4ExpertTensorBytes<'_>> {
        self.map
            .mxfp4_expert_tensor_bytes(&self.tensor_name(LayerTensor::FfnUpExpertsWeight))
    }

    pub fn up_experts_bias_bytes(&self) -> Result<F32MatrixBytes<'_>> {
        self.map
            .f32_matrix_bytes(&self.tensor_name(LayerTensor::FfnUpExpertsBias))
    }

    pub fn down_experts_bytes(&self) -> Result<Mxfp4ExpertTensorBytes<'_>> {
        self.map
            .mxfp4_expert_tensor_bytes(&self.tensor_name(LayerTensor::FfnDownExpertsWeight))
    }

    pub fn down_experts_bias_bytes(&self) -> Result<F32MatrixBytes<'_>> {
        self.map
            .f32_matrix_bytes(&self.tensor_name(LayerTensor::FfnDownExpertsBias))
    }

    fn tensor_name(&self, tensor: LayerTensor) -> String {
        format!("blk.{}.{tensor}", self.index)
    }
}

pub fn canonical_gguf_path() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| String::from("."));
    Path::new(&home)
        .join(".please")
        .join("weights")
        .join("gpt-oss-20b-mxfp4.gguf")
}

fn inspect_gguf(path: &Path) -> Result<GgufIndex> {
    let mut file = File::open(path)
        .map_err(|error| eyre!("could not open GGUF file {}: {error}", path.display()))?;

    let mut magic = [0u8; 4];
    file.read_exact(&mut magic)?;
    if &magic != GGUF_MAGIC {
        return Err(eyre!("{} is not a GGUF file", path.display()));
    }

    let _version = read_u32(&mut file)?;
    let tensor_count = read_u64(&mut file)?;
    let metadata_count = read_u64(&mut file)?;
    let mut alignment = DEFAULT_ALIGNMENT;

    for _ in 0..metadata_count {
        let key = read_string(&mut file)?;
        let value_type = read_u32(&mut file)?;
        if key == "general.alignment" && value_type == 4 {
            alignment = u64::from(read_u32(&mut file)?);
        } else {
            skip_value(&mut file, value_type)?;
        }
    }

    let mut tensors = Vec::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        let name = read_string(&mut file)?;
        let n_dims = read_u32(&mut file)?;
        let mut dimensions = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            dimensions.push(read_u64(&mut file)?);
        }
        let ggml_type = read_u32(&mut file)?;
        let offset = read_u64(&mut file)?;
        tensors.push(GgufTensor {
            name,
            dimensions,
            ggml_type,
            absolute_offset: offset,
        });
    }

    let position = file.stream_position()?;
    let data_start = align_to(position, alignment)?;
    for tensor in &mut tensors {
        tensor.absolute_offset = data_start
            .checked_add(tensor.absolute_offset)
            .ok_or_else(|| eyre!("GGUF tensor {} absolute offset overflow", tensor.name))?;
    }

    Ok(GgufIndex { tensors })
}

fn validate_gpt_oss_20b_gguf(index: &GgufIndex) -> GptOss20bGgufValidation {
    let expected = expected_gpt_oss_20b_gguf_tensors();
    let actual = index
        .tensors
        .iter()
        .map(|tensor| (tensor.name.clone(), tensor))
        .collect::<BTreeMap<_, _>>();
    let expected_names = expected.keys().cloned().collect::<BTreeSet<_>>();
    let actual_names = actual.keys().cloned().collect::<BTreeSet<_>>();
    let missing_tensors = expected_names.difference(&actual_names).cloned().collect();
    let unexpected_tensors = actual_names.difference(&expected_names).cloned().collect();

    let mut mismatches = Vec::new();
    for (name, expected) in &expected {
        let Some(actual) = actual.get(name) else {
            continue;
        };
        if actual.ggml_type != expected.ggml_type || actual.dimensions != expected.dimensions {
            mismatches.push(GgufTensorMismatch {
                name: name.clone(),
                expected_type: expected.ggml_type,
                expected_dimensions: expected.dimensions.clone(),
                actual_type: actual.ggml_type,
                actual_dimensions: actual.dimensions.clone(),
            });
        }
    }

    GptOss20bGgufValidation {
        expected_tensor_count: expected.len(),
        actual_tensor_count: index.tensors.len(),
        missing_tensors,
        unexpected_tensors,
        mismatches,
    }
}

#[derive(Debug, Clone)]
struct ExpectedGgufTensor {
    ggml_type: u32,
    dimensions: Vec<u64>,
}

fn expected_gpt_oss_20b_gguf_tensors() -> BTreeMap<String, ExpectedGgufTensor> {
    let mut expected = BTreeMap::new();
    insert_expected(
        &mut expected,
        "output.weight",
        8,
        &[GPT_OSS_20B_D_MODEL, GPT_OSS_20B_VOCAB],
    );
    insert_expected(
        &mut expected,
        "output_norm.weight",
        0,
        &[GPT_OSS_20B_D_MODEL],
    );
    insert_expected(
        &mut expected,
        "token_embd.weight",
        8,
        &[GPT_OSS_20B_D_MODEL, GPT_OSS_20B_VOCAB],
    );

    for layer in 0..GPT_OSS_20B_LAYERS {
        let prefix = format!("blk.{layer}");
        insert_expected(
            &mut expected,
            &format!("{prefix}.attn_k.bias"),
            0,
            &[GPT_OSS_20B_KV_WIDTH],
        );
        insert_expected(
            &mut expected,
            &format!("{prefix}.attn_k.weight"),
            8,
            &[GPT_OSS_20B_D_MODEL, GPT_OSS_20B_KV_WIDTH],
        );
        insert_expected(
            &mut expected,
            &format!("{prefix}.attn_norm.weight"),
            0,
            &[GPT_OSS_20B_D_MODEL],
        );
        insert_expected(
            &mut expected,
            &format!("{prefix}.attn_output.bias"),
            0,
            &[GPT_OSS_20B_D_MODEL],
        );
        insert_expected(
            &mut expected,
            &format!("{prefix}.attn_output.weight"),
            8,
            &[GPT_OSS_20B_Q_WIDTH, GPT_OSS_20B_D_MODEL],
        );
        insert_expected(
            &mut expected,
            &format!("{prefix}.attn_q.bias"),
            0,
            &[GPT_OSS_20B_Q_WIDTH],
        );
        insert_expected(
            &mut expected,
            &format!("{prefix}.attn_q.weight"),
            8,
            &[GPT_OSS_20B_D_MODEL, GPT_OSS_20B_Q_WIDTH],
        );
        insert_expected(
            &mut expected,
            &format!("{prefix}.attn_sinks.weight"),
            0,
            &[GPT_OSS_20B_SINKS],
        );
        insert_expected(
            &mut expected,
            &format!("{prefix}.attn_v.bias"),
            0,
            &[GPT_OSS_20B_KV_WIDTH],
        );
        insert_expected(
            &mut expected,
            &format!("{prefix}.attn_v.weight"),
            8,
            &[GPT_OSS_20B_D_MODEL, GPT_OSS_20B_KV_WIDTH],
        );
        for expert_part in ["down", "gate", "up"] {
            insert_expected(
                &mut expected,
                &format!("{prefix}.ffn_{expert_part}_exps.bias"),
                0,
                &[GPT_OSS_20B_D_MODEL, GPT_OSS_20B_EXPERTS],
            );
            insert_expected(
                &mut expected,
                &format!("{prefix}.ffn_{expert_part}_exps.weight"),
                39,
                &[
                    GPT_OSS_20B_D_MODEL,
                    GPT_OSS_20B_D_MODEL,
                    GPT_OSS_20B_EXPERTS,
                ],
            );
        }
        insert_expected(
            &mut expected,
            &format!("{prefix}.ffn_gate_inp.bias"),
            0,
            &[GPT_OSS_20B_EXPERTS],
        );
        insert_expected(
            &mut expected,
            &format!("{prefix}.ffn_gate_inp.weight"),
            0,
            &[GPT_OSS_20B_D_MODEL, GPT_OSS_20B_EXPERTS],
        );
        insert_expected(
            &mut expected,
            &format!("{prefix}.post_attention_norm.weight"),
            0,
            &[GPT_OSS_20B_D_MODEL],
        );
    }

    expected
}

fn insert_expected(
    expected: &mut BTreeMap<String, ExpectedGgufTensor>,
    name: &str,
    ggml_type: u32,
    dimensions: &[u64],
) {
    expected.insert(
        name.to_string(),
        ExpectedGgufTensor {
            ggml_type,
            dimensions: dimensions.to_vec(),
        },
    );
}

#[cfg(unix)]
impl MappedGgufBytes {
    fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)
            .map_err(|error| eyre!("could not open GGUF file {}: {error}", path.display()))?;
        let len = file
            .metadata()
            .map_err(|error| eyre!("could not stat GGUF file {}: {error}", path.display()))?
            .len();
        let len =
            usize::try_from(len).map_err(|_| eyre!("GGUF file {} is too large", path.display()))?;
        if len == 0 {
            return Err(eyre!("GGUF file {} is empty", path.display()));
        }

        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_PRIVATE,
                std::os::fd::AsRawFd::as_raw_fd(&file),
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(eyre!(
                "could not mmap GGUF file {}: {}",
                path.display(),
                std::io::Error::last_os_error()
            ));
        }
        let ptr = NonNull::new(ptr.cast::<u8>()).ok_or_else(|| {
            let _ = unsafe { libc::munmap(ptr, len) };
            eyre!("mmap returned null for GGUF file {}", path.display())
        })?;
        Ok(Self { ptr, len })
    }

    fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

#[cfg(unix)]
impl Drop for MappedGgufBytes {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.ptr.as_ptr().cast(), self.len);
        }
    }
}

#[cfg(not(unix))]
impl MappedGgufBytes {
    fn open(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)
            .map_err(|error| eyre!("could not read GGUF file {}: {error}", path.display()))?;
        if bytes.is_empty() {
            return Err(eyre!("GGUF file {} is empty", path.display()));
        }
        Ok(Self { bytes })
    }

    fn as_slice(&self) -> &[u8] {
        &self.bytes
    }
}

fn skip_value(file: &mut File, value_type: u32) -> Result<()> {
    match value_type {
        0 | 1 | 7 => skip(file, 1),
        2 | 3 => skip(file, 2),
        4 | 5 | 6 => skip(file, 4),
        10 | 11 | 12 => skip(file, 8),
        8 => {
            let len = read_u64(file)?;
            skip(file, len)
        }
        9 => {
            let element_type = read_u32(file)?;
            let len = read_u64(file)?;
            for _ in 0..len {
                skip_value(file, element_type)?;
            }
            Ok(())
        }
        _ => Err(eyre!("unsupported GGUF metadata value type {value_type}")),
    }
}

fn skip(file: &mut File, bytes: u64) -> Result<()> {
    let bytes = i64::try_from(bytes).map_err(|_| eyre!("GGUF skip is too large"))?;
    file.seek(SeekFrom::Current(bytes))?;
    Ok(())
}

fn read_string(file: &mut File) -> Result<String> {
    let len = read_u64(file)?;
    let len = usize::try_from(len).map_err(|_| eyre!("GGUF string is too large"))?;
    let mut bytes = vec![0u8; len];
    file.read_exact(&mut bytes)?;
    String::from_utf8(bytes).map_err(|error| eyre!("GGUF string is not UTF-8: {error}"))
}

fn read_u32(file: &mut File) -> Result<u32> {
    let mut bytes = [0u8; 4];
    file.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64(file: &mut File) -> Result<u64> {
    let mut bytes = [0u8; 8];
    file.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn align_to(value: u64, alignment: u64) -> Result<u64> {
    if alignment == 0 {
        return Err(eyre!("GGUF alignment is zero"));
    }
    let remainder = value % alignment;
    if remainder == 0 {
        Ok(value)
    } else {
        value
            .checked_add(alignment - remainder)
            .ok_or_else(|| eyre!("GGUF aligned offset overflow"))
    }
}

fn block_tensor_bytes(
    name: &str,
    dimensions: &[u64],
    qk: usize,
    block_bytes: usize,
) -> Result<u64> {
    let elements = dimensions
        .iter()
        .try_fold(1u64, |acc, dim| acc.checked_mul(*dim))
        .ok_or_else(|| eyre!("GGUF tensor {name} element count overflow"))?;
    let cols = *dimensions
        .first()
        .ok_or_else(|| eyre!("GGUF tensor {name} has no dimensions"))?;
    if cols % qk as u64 != 0 {
        return Err(eyre!(
            "GGUF tensor {name} first dimension {cols} is not block-aligned to {qk}"
        ));
    }
    elements
        .checked_div(qk as u64)
        .and_then(|blocks| blocks.checked_mul(block_bytes as u64))
        .ok_or_else(|| eyre!("GGUF tensor {name} byte count overflow"))
}

fn ggml_type_name(value: u32) -> &'static str {
    match value {
        0 => "f32",
        1 => "f16",
        2 => "q4_0",
        3 => "q4_1",
        6 => "q5_0",
        7 => "q5_1",
        8 => "q8_0",
        9 => "q8_1",
        10 => "q2_k",
        11 => "q3_k",
        12 => "q4_k",
        13 => "q5_k",
        14 => "q6_k",
        15 => "q8_k",
        24 => "i8",
        25 => "i16",
        26 => "i32",
        27 => "i64",
        28 => "f64",
        30 => "bf16",
        34 => "tq1_0",
        35 => "tq2_0",
        39 => "mxfp4",
        40 => "nvfp4",
        41 => "q1_0",
        _ => "unknown",
    }
}
