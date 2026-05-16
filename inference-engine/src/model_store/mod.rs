use eyre::{Result, eyre};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelLocation {
    pub path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceModelReport {
    pub location: ModelLocation,
    pub shards: Vec<SafetensorsShard>,
    pub duplicate_tensors: Vec<String>,
    pub dtype_counts: BTreeMap<String, usize>,
    pub total_tensor_bytes: u64,
    pub total_file_bytes: u64,
}

impl SourceModelReport {
    pub fn tensor_count(&self) -> usize {
        self.shards.iter().map(|shard| shard.tensors.len()).sum()
    }

    pub fn render_for_cli(&self) -> String {
        let mut out = String::new();
        out.push_str("inference-engine probe selected canonical gpt-oss SafeTensors.\n");
        out.push_str("SafeTensors source parse succeeded.\n\n");
        out.push_str(&format!(
            "source: {}\n",
            self.location.path.to_string_lossy()
        ));
        out.push_str(&format!("shards: {}\n", self.shards.len()));
        out.push_str(&format!("tensors: {}\n", self.tensor_count()));
        out.push_str(&format!(
            "tensor bytes declared: {}\n",
            human_bytes(self.total_tensor_bytes)
        ));
        out.push_str(&format!(
            "file bytes on disk: {}\n",
            human_bytes(self.total_file_bytes)
        ));

        if !self.dtype_counts.is_empty() {
            out.push_str("\ndtypes:\n");
            for (dtype, count) in &self.dtype_counts {
                out.push_str(&format!("- {dtype}: {count}\n"));
            }
        }

        out.push_str("\nshards:\n");
        for shard in &self.shards {
            out.push_str(&format!(
                "- {}: {} tensors, header {}, file {}\n",
                shard.file_name,
                shard.tensors.len(),
                human_bytes(shard.header_len),
                human_bytes(shard.file_size)
            ));
        }

        if !self.duplicate_tensors.is_empty() {
            out.push_str("\nwarning: duplicate tensor names found:\n");
            for name in &self.duplicate_tensors {
                out.push_str(&format!("- {name}\n"));
            }
        }

        out
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SafetensorsShard {
    pub path: PathBuf,
    pub file_name: String,
    pub file_size: u64,
    pub header_len: u64,
    pub metadata: BTreeMap<String, String>,
    pub tensors: Vec<TensorHeader>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorHeader {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub data_offsets: (u64, u64),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bf16Matrix {
    pub rows: usize,
    pub cols: usize,
    pub values: Vec<u16>,
}

pub fn canonical_safetensors_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| String::from("."));
    Path::new(&home).join(".please").join("weights")
}

pub fn inspect_canonical_safetensors() -> Result<SourceModelReport> {
    inspect_safetensors_dir(&canonical_safetensors_dir())
}

pub fn inspect_safetensors_dir(path: &Path) -> Result<SourceModelReport> {
    let mut shard_paths = Vec::new();
    let entries = fs::read_dir(path).map_err(|error| {
        eyre!(
            "could not read SafeTensors directory {}: {error}",
            path.display()
        )
    })?;

    for entry in entries {
        let entry = entry?;
        let entry_path = entry.path();
        let Some(file_name) = entry_path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if file_name.starts_with("model-") && file_name.ends_with(".safetensors") {
            shard_paths.push(entry_path);
        }
    }

    shard_paths.sort_by(|left, right| {
        left.file_name()
            .unwrap_or_default()
            .cmp(right.file_name().unwrap_or_default())
    });

    if shard_paths.is_empty() {
        return Err(eyre!(
            "no model-*.safetensors files found in {}",
            path.display()
        ));
    }

    let mut shards = Vec::with_capacity(shard_paths.len());
    for shard_path in shard_paths {
        shards.push(parse_safetensors_shard(&shard_path)?);
    }

    let mut seen = BTreeSet::new();
    let mut duplicates = BTreeSet::new();
    let mut dtype_counts = BTreeMap::new();
    let mut total_tensor_bytes = 0u64;
    let mut total_file_bytes = 0u64;

    for shard in &shards {
        total_file_bytes = total_file_bytes.saturating_add(shard.file_size);
        for tensor in &shard.tensors {
            if !seen.insert(tensor.name.clone()) {
                duplicates.insert(tensor.name.clone());
            }
            *dtype_counts.entry(tensor.dtype.clone()).or_insert(0) += 1;
            total_tensor_bytes = total_tensor_bytes
                .saturating_add(tensor.data_offsets.1.saturating_sub(tensor.data_offsets.0));
        }
    }

    Ok(SourceModelReport {
        location: ModelLocation {
            path: path.to_path_buf(),
        },
        shards,
        duplicate_tensors: duplicates.into_iter().collect(),
        dtype_counts,
        total_tensor_bytes,
        total_file_bytes,
    })
}

pub fn read_bf16_matrix_row(
    report: &SourceModelReport,
    tensor_name: &str,
    row_index: usize,
) -> Result<Vec<f32>> {
    let bytes = read_bf16_matrix_row_bytes(report, tensor_name, row_index)?;
    Ok(bytes
        .chunks_exact(2)
        .map(|bytes| bf16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]])))
        .collect())
}

pub fn read_bf16_matrix_row_bits(
    report: &SourceModelReport,
    tensor_name: &str,
    row_index: usize,
) -> Result<Vec<u16>> {
    let bytes = read_bf16_matrix_row_bytes(report, tensor_name, row_index)?;
    Ok(bytes
        .chunks_exact(2)
        .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]))
        .collect())
}

fn read_bf16_matrix_row_bytes(
    report: &SourceModelReport,
    tensor_name: &str,
    row_index: usize,
) -> Result<Vec<u8>> {
    let Some((shard, tensor)) = report.shards.iter().find_map(|shard| {
        shard
            .tensors
            .iter()
            .find(|tensor| tensor.name == tensor_name)
            .map(|tensor| (shard, tensor))
    }) else {
        return Err(eyre!("tensor {tensor_name} was not found"));
    };

    if tensor.dtype != "BF16" {
        return Err(eyre!(
            "tensor {tensor_name} has dtype {}, expected BF16",
            tensor.dtype
        ));
    }
    if tensor.shape.len() != 2 {
        return Err(eyre!(
            "tensor {tensor_name} has rank {}, expected rank 2",
            tensor.shape.len()
        ));
    }

    let rows = tensor.shape[0] as usize;
    let cols = tensor.shape[1] as usize;
    if row_index >= rows {
        return Err(eyre!(
            "row {row_index} is outside tensor {tensor_name} with {rows} rows"
        ));
    }

    let bytes_per_value = 2usize;
    let row_bytes = cols
        .checked_mul(bytes_per_value)
        .ok_or_else(|| eyre!("row byte length overflow for tensor {tensor_name}"))?;
    let row_offset = row_index
        .checked_mul(row_bytes)
        .ok_or_else(|| eyre!("row offset overflow for tensor {tensor_name}"))?;
    let tensor_bytes = tensor.data_offsets.1.saturating_sub(tensor.data_offsets.0) as usize;
    if row_offset + row_bytes > tensor_bytes {
        return Err(eyre!(
            "row {row_index} for tensor {tensor_name} exceeds tensor data range"
        ));
    }

    let absolute_offset = 8u64
        .saturating_add(shard.header_len)
        .saturating_add(tensor.data_offsets.0)
        .saturating_add(row_offset as u64);
    let mut file = fs::File::open(&shard.path).map_err(|error| {
        eyre!(
            "could not open shard {} for tensor row read: {error}",
            shard.path.display()
        )
    })?;
    file.seek(SeekFrom::Start(absolute_offset))
        .map_err(|error| {
            eyre!(
                "could not seek shard {} for tensor row read: {error}",
                shard.path.display()
            )
        })?;

    let mut bytes = vec![0u8; row_bytes];
    file.read_exact(&mut bytes).map_err(|error| {
        eyre!("could not read row {row_index} from tensor {tensor_name}: {error}")
    })?;

    Ok(bytes)
}

pub fn read_bf16_vector(report: &SourceModelReport, tensor_name: &str) -> Result<Vec<f32>> {
    let (_shard, tensor) = find_tensor(report, tensor_name)?;
    if tensor.dtype != "BF16" {
        return Err(eyre!(
            "tensor {tensor_name} has dtype {}, expected BF16",
            tensor.dtype
        ));
    }
    if tensor.shape.len() != 1 {
        return Err(eyre!(
            "tensor {tensor_name} has rank {}, expected rank 1",
            tensor.shape.len()
        ));
    }
    let bytes = read_tensor_data(report, tensor_name)?;
    Ok(bf16_bytes_to_f32(&bytes))
}

pub fn matvec_bf16(
    report: &SourceModelReport,
    tensor_name: &str,
    input: &[f32],
) -> Result<Vec<f32>> {
    let (_shard, tensor) = find_tensor(report, tensor_name)?;
    if tensor.dtype != "BF16" {
        return Err(eyre!(
            "tensor {tensor_name} has dtype {}, expected BF16",
            tensor.dtype
        ));
    }
    if tensor.shape.len() != 2 {
        return Err(eyre!(
            "tensor {tensor_name} has rank {}, expected rank 2",
            tensor.shape.len()
        ));
    }

    let rows = tensor.shape[0] as usize;
    let cols = tensor.shape[1] as usize;
    if cols != input.len() {
        return Err(eyre!(
            "tensor {tensor_name} has {cols} columns, but input has {} values",
            input.len()
        ));
    }

    let bytes = read_tensor_data(report, tensor_name)?;
    let expected_len = rows
        .checked_mul(cols)
        .and_then(|values| values.checked_mul(2))
        .ok_or_else(|| eyre!("tensor {tensor_name} byte size overflow"))?;
    if bytes.len() != expected_len {
        return Err(eyre!(
            "tensor {tensor_name} has {} bytes, expected {expected_len}",
            bytes.len()
        ));
    }

    let mut out = Vec::with_capacity(rows);
    for row in bytes.chunks_exact(cols * 2) {
        let mut sum = 0.0f32;
        for (value, input) in row.chunks_exact(2).zip(input.iter().copied()) {
            sum += bf16_to_f32(u16::from_le_bytes([value[0], value[1]])) * input;
        }
        out.push(sum);
    }
    Ok(out)
}

pub fn read_bf16_matrix(report: &SourceModelReport, tensor_name: &str) -> Result<Bf16Matrix> {
    let (_shard, tensor) = find_tensor(report, tensor_name)?;
    if tensor.dtype != "BF16" {
        return Err(eyre!(
            "tensor {tensor_name} has dtype {}, expected BF16",
            tensor.dtype
        ));
    }
    if tensor.shape.len() != 2 {
        return Err(eyre!(
            "tensor {tensor_name} has rank {}, expected rank 2",
            tensor.shape.len()
        ));
    }

    let rows = tensor.shape[0] as usize;
    let cols = tensor.shape[1] as usize;
    let bytes = read_tensor_data(report, tensor_name)?;
    let expected_len = rows
        .checked_mul(cols)
        .and_then(|values| values.checked_mul(2))
        .ok_or_else(|| eyre!("tensor {tensor_name} byte size overflow"))?;
    if bytes.len() != expected_len {
        return Err(eyre!(
            "tensor {tensor_name} has {} bytes, expected {expected_len}",
            bytes.len()
        ));
    }

    let values = bytes
        .chunks_exact(2)
        .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]))
        .collect();
    Ok(Bf16Matrix { rows, cols, values })
}

pub fn top_k_matvec_bf16(
    report: &SourceModelReport,
    tensor_name: &str,
    input: &[f32],
    k: usize,
) -> Result<Vec<(usize, f32)>> {
    let (shard, tensor) = find_tensor(report, tensor_name)?;
    if tensor.dtype != "BF16" {
        return Err(eyre!(
            "tensor {tensor_name} has dtype {}, expected BF16",
            tensor.dtype
        ));
    }
    if tensor.shape.len() != 2 {
        return Err(eyre!(
            "tensor {tensor_name} has rank {}, expected rank 2",
            tensor.shape.len()
        ));
    }

    let rows = tensor.shape[0] as usize;
    let cols = tensor.shape[1] as usize;
    if cols != input.len() {
        return Err(eyre!(
            "tensor {tensor_name} has {cols} columns, but input has {} values",
            input.len()
        ));
    }

    let row_bytes = cols
        .checked_mul(2)
        .ok_or_else(|| eyre!("row byte length overflow for tensor {tensor_name}"))?;
    let chunk_rows = 32usize;
    let chunk_bytes = chunk_rows
        .checked_mul(row_bytes)
        .ok_or_else(|| eyre!("chunk byte length overflow for tensor {tensor_name}"))?;
    let absolute_offset = 8u64
        .saturating_add(shard.header_len)
        .saturating_add(tensor.data_offsets.0);
    let mut file = fs::File::open(&shard.path).map_err(|error| {
        eyre!(
            "could not open shard {} for top-k tensor read: {error}",
            shard.path.display()
        )
    })?;
    file.seek(SeekFrom::Start(absolute_offset))
        .map_err(|error| {
            eyre!(
                "could not seek shard {} for top-k tensor read: {error}",
                shard.path.display()
            )
        })?;

    let mut top = TopK::new(k);
    let mut row_index = 0usize;
    let mut buffer = vec![0u8; chunk_bytes];

    while row_index < rows {
        let rows_now = (rows - row_index).min(chunk_rows);
        let bytes_now = rows_now * row_bytes;
        file.read_exact(&mut buffer[..bytes_now]).map_err(|error| {
            eyre!(
                "could not read rows {row_index}..{} from tensor {tensor_name}: {error}",
                row_index + rows_now
            )
        })?;

        for row in buffer[..bytes_now].chunks_exact(row_bytes) {
            let mut sum = 0.0f32;
            for (value, input) in row.chunks_exact(2).zip(input.iter().copied()) {
                sum += bf16_to_f32(u16::from_le_bytes([value[0], value[1]])) * input;
            }
            top.push(row_index, sum);
            row_index += 1;
        }
    }

    Ok(top.finish())
}

pub fn add_in_place(values: &mut [f32], bias: &[f32], tensor_name: &str) -> Result<()> {
    if values.len() != bias.len() {
        return Err(eyre!(
            "{tensor_name} output has {} values but bias has {} values",
            values.len(),
            bias.len()
        ));
    }
    for (value, bias) in values.iter_mut().zip(bias) {
        *value += *bias;
    }
    Ok(())
}

pub fn read_u8_tensor_slice(
    report: &SourceModelReport,
    tensor_name: &str,
    element_offset: usize,
    element_len: usize,
) -> Result<Vec<u8>> {
    let (shard, tensor) = find_tensor(report, tensor_name)?;
    if tensor.dtype != "U8" {
        return Err(eyre!(
            "tensor {tensor_name} has dtype {}, expected U8",
            tensor.dtype
        ));
    }
    let tensor_bytes = tensor.data_offsets.1.saturating_sub(tensor.data_offsets.0) as usize;
    let end = element_offset
        .checked_add(element_len)
        .ok_or_else(|| eyre!("slice overflow for tensor {tensor_name}"))?;
    if end > tensor_bytes {
        return Err(eyre!(
            "slice {element_offset}..{end} exceeds tensor {tensor_name} data length {tensor_bytes}"
        ));
    }

    let absolute_offset = 8u64
        .saturating_add(shard.header_len)
        .saturating_add(tensor.data_offsets.0)
        .saturating_add(element_offset as u64);
    let mut file = fs::File::open(&shard.path).map_err(|error| {
        eyre!(
            "could not open shard {} for tensor slice read: {error}",
            shard.path.display()
        )
    })?;
    file.seek(SeekFrom::Start(absolute_offset))
        .map_err(|error| {
            eyre!(
                "could not seek shard {} for tensor slice read: {error}",
                shard.path.display()
            )
        })?;

    let mut bytes = vec![0u8; element_len];
    file.read_exact(&mut bytes).map_err(|error| {
        eyre!("could not read slice {element_offset}..{end} from tensor {tensor_name}: {error}")
    })?;
    Ok(bytes)
}

fn find_tensor<'a>(
    report: &'a SourceModelReport,
    tensor_name: &str,
) -> Result<(&'a SafetensorsShard, &'a TensorHeader)> {
    report
        .shards
        .iter()
        .find_map(|shard| {
            shard
                .tensors
                .iter()
                .find(|tensor| tensor.name == tensor_name)
                .map(|tensor| (shard, tensor))
        })
        .ok_or_else(|| eyre!("tensor {tensor_name} was not found"))
}

fn read_tensor_data(report: &SourceModelReport, tensor_name: &str) -> Result<Vec<u8>> {
    let (shard, tensor) = find_tensor(report, tensor_name)?;
    let len = tensor.data_offsets.1.saturating_sub(tensor.data_offsets.0) as usize;
    let absolute_offset = 8u64
        .saturating_add(shard.header_len)
        .saturating_add(tensor.data_offsets.0);
    let mut file = fs::File::open(&shard.path).map_err(|error| {
        eyre!(
            "could not open shard {} for tensor read: {error}",
            shard.path.display()
        )
    })?;
    file.seek(SeekFrom::Start(absolute_offset))
        .map_err(|error| {
            eyre!(
                "could not seek shard {} for tensor read: {error}",
                shard.path.display()
            )
        })?;

    let mut bytes = vec![0u8; len];
    file.read_exact(&mut bytes)
        .map_err(|error| eyre!("could not read tensor {tensor_name}: {error}"))?;
    Ok(bytes)
}

fn bf16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|bytes| bf16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]])))
        .collect()
}

fn bf16_to_f32(value: u16) -> f32 {
    f32::from_bits((value as u32) << 16)
}

struct TopK {
    k: usize,
    values: Vec<(usize, f32)>,
}

impl TopK {
    fn new(k: usize) -> Self {
        Self {
            k,
            values: Vec::with_capacity(k),
        }
    }

    fn push(&mut self, index: usize, value: f32) {
        if self.k == 0 {
            return;
        }
        if self.values.len() < self.k {
            self.values.push((index, value));
            self.values.sort_by(compare_desc);
            return;
        }
        let Some((_, current_min)) = self.values.last().copied() else {
            return;
        };
        if value <= current_min {
            return;
        }
        self.values.pop();
        self.values.push((index, value));
        self.values.sort_by(compare_desc);
    }

    fn finish(self) -> Vec<(usize, f32)> {
        self.values
    }
}

fn compare_desc(left: &(usize, f32), right: &(usize, f32)) -> std::cmp::Ordering {
    right
        .1
        .partial_cmp(&left.1)
        .unwrap_or(std::cmp::Ordering::Equal)
        .then_with(|| left.0.cmp(&right.0))
}

fn parse_safetensors_shard(path: &Path) -> Result<SafetensorsShard> {
    let mut file = fs::File::open(path).map_err(|error| {
        eyre!(
            "could not open SafeTensors shard {}: {error}",
            path.display()
        )
    })?;
    let file_size = file.metadata()?.len();
    let mut header_len_bytes = [0u8; 8];
    file.read_exact(&mut header_len_bytes).map_err(|error| {
        eyre!(
            "could not read SafeTensors header length from {}: {error}",
            path.display()
        )
    })?;
    let header_len = u64::from_le_bytes(header_len_bytes);
    if header_len == 0 {
        return Err(eyre!(
            "SafeTensors shard {} has an empty header",
            path.display()
        ));
    }
    if header_len > 256 * 1024 * 1024 {
        return Err(eyre!(
            "SafeTensors shard {} header is too large: {}",
            path.display(),
            header_len
        ));
    }
    if 8 + header_len > file_size {
        return Err(eyre!(
            "SafeTensors shard {} header exceeds file size",
            path.display()
        ));
    }

    let mut header_bytes = vec![0u8; header_len as usize];
    file.read_exact(&mut header_bytes).map_err(|error| {
        eyre!(
            "could not read SafeTensors header from {}: {error}",
            path.display()
        )
    })?;
    let header: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(&header_bytes)
        .map_err(|error| {
            eyre!(
                "could not parse SafeTensors header JSON from {}: {error}",
                path.display()
            )
        })?;

    let data_len = file_size.saturating_sub(8 + header_len);
    let mut metadata = BTreeMap::new();
    let mut tensors = Vec::new();

    for (name, value) in header {
        if name == "__metadata__" {
            if let Some(object) = value.as_object() {
                for (key, value) in object {
                    if let Some(value) = value.as_str() {
                        metadata.insert(key.clone(), value.to_string());
                    }
                }
            }
            continue;
        }

        let tensor = parse_tensor_header(&name, &value, data_len, path)?;
        tensors.push(tensor);
    }

    tensors.sort_by(|left, right| left.name.cmp(&right.name));

    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("<unknown>")
        .to_string();

    Ok(SafetensorsShard {
        path: path.to_path_buf(),
        file_name,
        file_size,
        header_len,
        metadata,
        tensors,
    })
}

fn parse_tensor_header(
    name: &str,
    value: &serde_json::Value,
    data_len: u64,
    path: &Path,
) -> Result<TensorHeader> {
    let object = value.as_object().ok_or_else(|| {
        eyre!(
            "tensor header for {name} in {} is not a JSON object",
            path.display()
        )
    })?;
    let dtype = object
        .get("dtype")
        .and_then(|value| value.as_str())
        .ok_or_else(|| eyre!("tensor {name} in {} is missing dtype", path.display()))?
        .to_string();
    let shape = object
        .get("shape")
        .and_then(|value| value.as_array())
        .ok_or_else(|| eyre!("tensor {name} in {} is missing shape", path.display()))?
        .iter()
        .map(|value| {
            value.as_u64().ok_or_else(|| {
                eyre!(
                    "tensor {name} in {} has a non-integer shape",
                    path.display()
                )
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let offsets = object
        .get("data_offsets")
        .and_then(|value| value.as_array())
        .ok_or_else(|| {
            eyre!(
                "tensor {name} in {} is missing data_offsets",
                path.display()
            )
        })?;
    if offsets.len() != 2 {
        return Err(eyre!(
            "tensor {name} in {} must have exactly two data_offsets",
            path.display()
        ));
    }
    let start = offsets[0].as_u64().ok_or_else(|| {
        eyre!(
            "tensor {name} in {} has a non-integer start offset",
            path.display()
        )
    })?;
    let end = offsets[1].as_u64().ok_or_else(|| {
        eyre!(
            "tensor {name} in {} has a non-integer end offset",
            path.display()
        )
    })?;

    if end < start {
        return Err(eyre!(
            "tensor {name} in {} has descending data_offsets",
            path.display()
        ));
    }
    if end > data_len {
        return Err(eyre!(
            "tensor {name} in {} ends beyond the shard data region",
            path.display()
        ));
    }

    Ok(TensorHeader {
        name: name.to_string(),
        dtype,
        shape,
        data_offsets: (start, end),
    })
}

fn human_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = 0usize;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{} {}", bytes, UNITS[unit])
    } else {
        format!("{value:.2} {}", UNITS[unit])
    }
}
