use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fs::File;
use std::path::{Component, Path};
use std::sync::mpsc;
use std::thread;

#[cfg(unix)]
use std::os::unix::fs::FileExt;

pub const PLE_INDEX_MAGIC: &[u8; 8] = b"SSPLEIDX";
pub const PLE_INDEX_VERSION: u32 = 1;
pub const PLE_INDEX_HEADER_BYTES: usize = 64;
pub const PLE_INDEX_RECORD_BYTES: usize = 40;
pub const PLE_DTYPE_FP8_E4M3: u32 = 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RowAddress {
    pub shard: usize,
    pub byte_offset: u64,
    pub byte_len: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PleShard {
    pub global_row_start: u64,
    pub row_count: u64,
    pub data_offset: u64,
    pub data_bytes: u64,
    pub relative_path: String,
}

impl PleShard {
    fn global_row_end(&self) -> Result<u64, PleIndexError> {
        self.global_row_start
            .checked_add(self.row_count)
            .ok_or(PleIndexError::IntegerOverflow)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PleIndex {
    pub page_bytes: usize,
    pub row_bytes: usize,
    pub dtype: u32,
    pub total_rows: u64,
    pub scale_bf16_bits: u16,
    pub shards: Vec<PleShard>,
}

impl PleIndex {
    pub fn decode(payload: &[u8]) -> Result<Self, PleIndexError> {
        if payload.len() < PLE_INDEX_HEADER_BYTES {
            return Err(PleIndexError::Truncated);
        }
        if &payload[0..8] != PLE_INDEX_MAGIC {
            return Err(PleIndexError::Magic);
        }
        let version = read_u32(payload, 8)?;
        let header_bytes = read_u32(payload, 12)? as usize;
        let page_bytes = read_u32(payload, 16)? as usize;
        let row_bytes = read_u32(payload, 20)? as usize;
        let dtype = read_u32(payload, 24)?;
        let shard_count = read_u32(payload, 28)? as usize;
        let total_rows = read_u64(payload, 32)?;
        let scale_bf16_bits = read_u32(payload, 40)?;
        let records_bytes = read_u32(payload, 44)? as usize;
        let strings_bytes = read_u32(payload, 48)? as usize;
        let stored_crc = read_u32(payload, 60)?;

        if version != PLE_INDEX_VERSION || header_bytes != PLE_INDEX_HEADER_BYTES {
            return Err(PleIndexError::Version(version));
        }
        if dtype != PLE_DTYPE_FP8_E4M3 {
            return Err(PleIndexError::DataType(dtype));
        }
        if page_bytes == 0 || !page_bytes.is_power_of_two() || row_bytes == 0 {
            return Err(PleIndexError::Layout);
        }
        if scale_bf16_bits > u32::from(u16::MAX) {
            return Err(PleIndexError::Layout);
        }
        let expected_records = shard_count
            .checked_mul(PLE_INDEX_RECORD_BYTES)
            .ok_or(PleIndexError::IntegerOverflow)?;
        if records_bytes != expected_records {
            return Err(PleIndexError::RecordTable);
        }
        let expected_size = header_bytes
            .checked_add(records_bytes)
            .and_then(|value| value.checked_add(strings_bytes))
            .ok_or(PleIndexError::IntegerOverflow)?;
        if payload.len() != expected_size {
            return Err(PleIndexError::Truncated);
        }
        let mut checked = payload.to_vec();
        checked[60..64].fill(0);
        if crc32(&checked) != stored_crc {
            return Err(PleIndexError::Checksum);
        }

        let strings_start = header_bytes + records_bytes;
        let mut shards = Vec::with_capacity(shard_count);
        let mut expected_row = 0_u64;
        for ordinal in 0..shard_count {
            let offset = header_bytes + ordinal * PLE_INDEX_RECORD_BYTES;
            let global_row_start = read_u64(payload, offset)?;
            let row_count = read_u64(payload, offset + 8)?;
            let data_offset = read_u64(payload, offset + 16)?;
            let data_bytes = read_u64(payload, offset + 24)?;
            let path_offset = read_u32(payload, offset + 32)? as usize;
            let path_len = read_u32(payload, offset + 36)? as usize;
            if global_row_start != expected_row || row_count == 0 {
                return Err(PleIndexError::RowRange);
            }
            let expected_bytes = row_count
                .checked_mul(row_bytes as u64)
                .ok_or(PleIndexError::IntegerOverflow)?;
            if data_bytes != expected_bytes {
                return Err(PleIndexError::Layout);
            }
            let path_end = path_offset
                .checked_add(path_len)
                .ok_or(PleIndexError::IntegerOverflow)?;
            if path_end > strings_bytes {
                return Err(PleIndexError::Path);
            }
            let raw_path = &payload[strings_start + path_offset..strings_start + path_end];
            let relative_path = std::str::from_utf8(raw_path)
                .map_err(|_| PleIndexError::Path)?
                .to_owned();
            if !safe_relative_path(&relative_path) {
                return Err(PleIndexError::Path);
            }
            let shard = PleShard {
                global_row_start,
                row_count,
                data_offset,
                data_bytes,
                relative_path,
            };
            expected_row = shard.global_row_end()?;
            shards.push(shard);
        }
        if expected_row != total_rows || shards.is_empty() {
            return Err(PleIndexError::RowRange);
        }
        Ok(Self {
            page_bytes,
            row_bytes,
            dtype,
            total_rows,
            scale_bf16_bits: scale_bf16_bits as u16,
            shards,
        })
    }

    pub fn address(&self, global_row: u64) -> Result<RowAddress, PleIndexError> {
        if global_row >= self.total_rows {
            return Err(PleIndexError::RowOutOfRange(global_row));
        }
        let mut low = 0;
        let mut high = self.shards.len();
        while low < high {
            let mid = (low + high) / 2;
            let shard = &self.shards[mid];
            if global_row < shard.global_row_start {
                high = mid;
            } else if global_row >= shard.global_row_end()? {
                low = mid + 1;
            } else {
                let local_row = global_row - shard.global_row_start;
                let row_offset = local_row
                    .checked_mul(self.row_bytes as u64)
                    .and_then(|value| shard.data_offset.checked_add(value))
                    .ok_or(PleIndexError::IntegerOverflow)?;
                return Ok(RowAddress {
                    shard: mid,
                    byte_offset: row_offset,
                    byte_len: self.row_bytes,
                });
            }
        }
        Err(PleIndexError::RowRange)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct PageKey {
    pub shard: usize,
    pub byte_offset: u64,
}

pub trait PageSource {
    fn read_page(&mut self, key: PageKey, page_bytes: usize) -> Result<Vec<u8>, StoreError>;

    fn read_pages(
        &mut self,
        keys: &[PageKey],
        page_bytes: usize,
    ) -> Result<Vec<Vec<u8>>, StoreError> {
        keys.iter()
            .map(|key| self.read_page(*key, page_bytes))
            .collect()
    }
}

pub struct FilePageSource {
    files: Vec<File>,
    file_sizes: Vec<u64>,
    workers: usize,
}

impl FilePageSource {
    pub fn open(index: &PleIndex, model_root: &Path) -> Result<Self, StoreError> {
        Self::open_with_workers(index, model_root, 1)
    }

    pub fn open_with_workers(
        index: &PleIndex,
        model_root: &Path,
        workers: usize,
    ) -> Result<Self, StoreError> {
        if workers == 0 {
            return Err(StoreError::InvalidCache);
        }
        let mut files = Vec::with_capacity(index.shards.len());
        let mut file_sizes = Vec::with_capacity(index.shards.len());
        for shard in &index.shards {
            let path = model_root.join(&shard.relative_path);
            let file = File::open(&path)?;
            let file_size = file.metadata()?.len();
            let tensor_end = shard
                .data_offset
                .checked_add(shard.data_bytes)
                .ok_or(StoreError::IntegerOverflow)?;
            if tensor_end > file_size {
                return Err(StoreError::SourceTruncated(path.display().to_string()));
            }
            files.push(file);
            file_sizes.push(file_size);
        }
        Ok(Self {
            files,
            file_sizes,
            workers,
        })
    }

    #[cfg(unix)]
    fn read_one(&self, key: PageKey, page_bytes: usize) -> Result<Vec<u8>, StoreError> {
        let file = self
            .files
            .get(key.shard)
            .ok_or(StoreError::ShardOutOfRange(key.shard))?;
        let file_size = self.file_sizes[key.shard];
        if key.byte_offset >= file_size {
            return Err(StoreError::PageOutOfRange(key));
        }
        let available = (file_size - key.byte_offset).min(page_bytes as u64) as usize;
        let mut page = vec![0_u8; page_bytes];
        let mut read = 0;
        while read < available {
            let count = file.read_at(&mut page[read..available], key.byte_offset + read as u64)?;
            if count == 0 {
                return Err(StoreError::SourceTruncated(format!(
                    "shard {} at byte {}",
                    key.shard, key.byte_offset
                )));
            }
            read += count;
        }
        Ok(page)
    }
}

impl PageSource for FilePageSource {
    fn read_page(&mut self, key: PageKey, page_bytes: usize) -> Result<Vec<u8>, StoreError> {
        self.read_one(key, page_bytes)
    }

    fn read_pages(
        &mut self,
        keys: &[PageKey],
        page_bytes: usize,
    ) -> Result<Vec<Vec<u8>>, StoreError> {
        if self.workers == 1 || keys.len() <= 1 {
            return keys
                .iter()
                .map(|key| self.read_one(*key, page_bytes))
                .collect();
        }
        let workers = self.workers.min(keys.len());
        let chunk_size = keys.len().div_ceil(workers);
        let mut pages: Vec<Option<Vec<u8>>> = (0..keys.len()).map(|_| None).collect();
        let mut first_error = None;
        thread::scope(|scope| {
            let (sender, receiver) = mpsc::channel();
            let source: &FilePageSource = self;
            for (chunk_number, chunk) in keys.chunks(chunk_size).enumerate() {
                let sender = sender.clone();
                let start = chunk_number * chunk_size;
                scope.spawn(move || {
                    for (offset, key) in chunk.iter().enumerate() {
                        let result = source.read_one(*key, page_bytes);
                        if sender.send((start + offset, result)).is_err() {
                            return;
                        }
                    }
                });
            }
            drop(sender);
            for _ in 0..keys.len() {
                let (ordinal, result) = receiver
                    .recv()
                    .expect("scoped PLE readers retain their sender");
                match result {
                    Ok(page) => pages[ordinal] = Some(page),
                    Err(error) if first_error.is_none() => first_error = Some(error),
                    Err(_) => {}
                }
            }
        });
        if let Some(error) = first_error {
            return Err(error);
        }
        pages
            .into_iter()
            .map(|page| page.ok_or(StoreError::InvalidPageSource))
            .collect()
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CacheStats {
    pub page_hits: u64,
    pub page_misses: u64,
    pub page_reads: u64,
    pub bytes_read: u64,
    pub evictions: u64,
}

struct CacheSlot {
    key: PageKey,
    page: Vec<u8>,
    referenced: bool,
}

pub struct ClockPageCache<S> {
    source: S,
    page_bytes: usize,
    capacity_pages: usize,
    slots: Vec<CacheSlot>,
    index: HashMap<PageKey, usize>,
    hand: usize,
    pub stats: CacheStats,
}

impl<S: PageSource> ClockPageCache<S> {
    pub fn new(source: S, page_bytes: usize, capacity_pages: usize) -> Result<Self, StoreError> {
        if page_bytes == 0 || !page_bytes.is_power_of_two() || capacity_pages == 0 {
            return Err(StoreError::InvalidCache);
        }
        Ok(Self {
            source,
            page_bytes,
            capacity_pages,
            slots: Vec::new(),
            index: HashMap::new(),
            hand: 0,
            stats: CacheStats::default(),
        })
    }

    pub fn fetch_rows(&mut self, index: &PleIndex, rows: &[u64]) -> Result<Vec<u8>, StoreError> {
        if index.page_bytes != self.page_bytes {
            return Err(StoreError::PageSizeMismatch);
        }
        let mut addresses = Vec::with_capacity(rows.len());
        let mut keys = Vec::with_capacity(rows.len() * 2);
        let mut unique = HashSet::with_capacity(rows.len() * 2);
        for row in rows {
            let address = index.address(*row)?;
            let first_offset = address.byte_offset & !(self.page_bytes as u64 - 1);
            let first = PageKey {
                shard: address.shard,
                byte_offset: first_offset,
            };
            if unique.insert(first) {
                keys.push(first);
            }
            if address.byte_offset + address.byte_len as u64 > first_offset + self.page_bytes as u64
            {
                let second = PageKey {
                    shard: address.shard,
                    byte_offset: first_offset + self.page_bytes as u64,
                };
                if unique.insert(second) {
                    keys.push(second);
                }
            }
            addresses.push(address);
        }
        if keys.len() > self.capacity_pages {
            return Err(StoreError::BatchExceedsCache {
                pages: keys.len(),
                capacity: self.capacity_pages,
            });
        }
        self.ensure_pages(&keys)?;

        let output_len = rows
            .len()
            .checked_mul(index.row_bytes)
            .ok_or(StoreError::IntegerOverflow)?;
        let mut output = vec![0_u8; output_len];
        for (ordinal, address) in addresses.iter().enumerate() {
            let page_offset = address.byte_offset & !(self.page_bytes as u64 - 1);
            let within = (address.byte_offset - page_offset) as usize;
            let first_key = PageKey {
                shard: address.shard,
                byte_offset: page_offset,
            };
            let first_slot = *self
                .index
                .get(&first_key)
                .ok_or(StoreError::InvalidPageSource)?;
            let first = &self.slots[first_slot].page;
            let first_len = index.row_bytes.min(self.page_bytes - within);
            let output_start = ordinal * index.row_bytes;
            output[output_start..output_start + first_len]
                .copy_from_slice(&first[within..within + first_len]);
            if first_len < index.row_bytes {
                let second_key = PageKey {
                    shard: address.shard,
                    byte_offset: page_offset + self.page_bytes as u64,
                };
                let second_slot = *self
                    .index
                    .get(&second_key)
                    .ok_or(StoreError::InvalidPageSource)?;
                let remaining = index.row_bytes - first_len;
                output[output_start + first_len..output_start + index.row_bytes]
                    .copy_from_slice(&self.slots[second_slot].page[..remaining]);
            }
        }
        Ok(output)
    }

    fn ensure_pages(&mut self, keys: &[PageKey]) -> Result<(), StoreError> {
        let mut misses = Vec::new();
        for key in keys {
            if let Some(slot) = self.index.get(key).copied() {
                self.slots[slot].referenced = true;
                self.stats.page_hits += 1;
            } else {
                self.stats.page_misses += 1;
                misses.push(*key);
            }
        }
        let pages = self.source.read_pages(&misses, self.page_bytes)?;
        if pages.len() != misses.len() || pages.iter().any(|page| page.len() != self.page_bytes) {
            return Err(StoreError::InvalidPageSource);
        }
        for (key, page) in misses.into_iter().zip(pages) {
            self.stats.page_reads += 1;
            self.stats.bytes_read = self
                .stats
                .bytes_read
                .checked_add(page.len() as u64)
                .ok_or(StoreError::IntegerOverflow)?;
            self.insert(key, page);
        }
        Ok(())
    }

    fn insert(&mut self, key: PageKey, page: Vec<u8>) {
        if self.slots.len() < self.capacity_pages {
            let slot = self.slots.len();
            self.slots.push(CacheSlot {
                key,
                page,
                referenced: true,
            });
            self.index.insert(key, slot);
            return;
        }
        loop {
            if self.slots[self.hand].referenced {
                self.slots[self.hand].referenced = false;
                self.hand = (self.hand + 1) % self.capacity_pages;
                continue;
            }
            let victim = self.slots[self.hand].key;
            self.index.remove(&victim);
            self.slots[self.hand] = CacheSlot {
                key,
                page,
                referenced: true,
            };
            self.index.insert(key, self.hand);
            self.hand = (self.hand + 1) % self.capacity_pages;
            self.stats.evictions += 1;
            return;
        }
    }
}

#[derive(Debug)]
pub enum StoreError {
    Index(PleIndexError),
    Io(std::io::Error),
    InvalidCache,
    InvalidPageSource,
    PageSizeMismatch,
    IntegerOverflow,
    ShardOutOfRange(usize),
    PageOutOfRange(PageKey),
    SourceTruncated(String),
    BatchExceedsCache { pages: usize, capacity: usize },
}

impl fmt::Display for StoreError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Index(error) => write!(formatter, "invalid PLE index: {error}"),
            Self::Io(error) => write!(formatter, "PLE I/O failed: {error}"),
            Self::InvalidCache => write!(formatter, "PLE cache geometry is invalid"),
            Self::InvalidPageSource => write!(formatter, "PLE page source returned invalid data"),
            Self::PageSizeMismatch => write!(formatter, "PLE index and cache page sizes differ"),
            Self::IntegerOverflow => write!(formatter, "PLE storage size overflow"),
            Self::ShardOutOfRange(shard) => write!(formatter, "PLE shard {shard} is out of range"),
            Self::PageOutOfRange(key) => write!(formatter, "PLE page is outside shard: {key:?}"),
            Self::SourceTruncated(path) => write!(formatter, "PLE source is truncated: {path}"),
            Self::BatchExceedsCache { pages, capacity } => write!(
                formatter,
                "PLE batch needs {pages} pages but cache holds {capacity}; reduce the prefill chunk"
            ),
        }
    }
}

impl std::error::Error for StoreError {}

impl From<PleIndexError> for StoreError {
    fn from(value: PleIndexError) -> Self {
        Self::Index(value)
    }
}

impl From<std::io::Error> for StoreError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PleIndexError {
    Truncated,
    Magic,
    Version(u32),
    DataType(u32),
    Layout,
    RecordTable,
    Checksum,
    Path,
    RowRange,
    RowOutOfRange(u64),
    IntegerOverflow,
}

impl fmt::Display for PleIndexError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated => write!(formatter, "truncated PLE index"),
            Self::Magic => write!(formatter, "invalid PLE index magic"),
            Self::Version(version) => write!(formatter, "unsupported PLE index version {version}"),
            Self::DataType(dtype) => write!(formatter, "unsupported PLE data type {dtype}"),
            Self::Layout => write!(formatter, "invalid PLE tensor layout"),
            Self::RecordTable => write!(formatter, "invalid PLE shard record table"),
            Self::Checksum => write!(formatter, "PLE index checksum mismatch"),
            Self::Path => write!(formatter, "invalid PLE source path"),
            Self::RowRange => write!(formatter, "non-contiguous PLE row range"),
            Self::RowOutOfRange(row) => write!(formatter, "PLE row {row} is out of range"),
            Self::IntegerOverflow => write!(formatter, "PLE index integer overflow"),
        }
    }
}

impl std::error::Error for PleIndexError {}

fn safe_relative_path(value: &str) -> bool {
    !value.is_empty()
        && Path::new(value)
            .components()
            .all(|component| matches!(component, Component::Normal(_) | Component::CurDir))
}

fn read_u32(payload: &[u8], offset: usize) -> Result<u32, PleIndexError> {
    let bytes = payload
        .get(offset..offset + 4)
        .ok_or(PleIndexError::Truncated)?;
    Ok(u32::from_le_bytes(
        bytes.try_into().expect("four-byte slice"),
    ))
}

fn read_u64(payload: &[u8], offset: usize) -> Result<u64, PleIndexError> {
    let bytes = payload
        .get(offset..offset + 8)
        .ok_or(PleIndexError::Truncated)?;
    Ok(u64::from_le_bytes(
        bytes.try_into().expect("eight-byte slice"),
    ))
}

fn crc32(payload: &[u8]) -> u32 {
    let mut crc = !0_u32;
    for byte in payload {
        crc ^= u32::from(*byte);
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xedb8_8320 & mask);
        }
    }
    !crc
}

/// Physical layout of one logical embedding table split across safetensors
/// shards. Header offsets are supplied by the validated manifest loader.
#[derive(Debug)]
pub struct RowLayout<'a> {
    pub rows_per_tensor: u64,
    pub row_bytes: usize,
    pub tensors_per_file: usize,
    pub tensor_data_offsets: &'a [u64],
}

impl RowLayout<'_> {
    pub fn address(&self, logical_table: usize, row: u64) -> Result<RowAddress, &'static str> {
        if row >= self.rows_per_tensor || self.tensors_per_file == 0 {
            return Err("PLE row or layout is out of range");
        }
        let shard = logical_table / self.tensors_per_file;
        let base = *self
            .tensor_data_offsets
            .get(logical_table)
            .ok_or("PLE table is out of range")?;
        Ok(RowAddress {
            shard,
            byte_offset: base + row * self.row_bytes as u64,
            byte_len: self.row_bytes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MemoryPageSource {
        shards: Vec<Vec<u8>>,
    }

    impl PageSource for MemoryPageSource {
        fn read_page(&mut self, key: PageKey, page_bytes: usize) -> Result<Vec<u8>, StoreError> {
            let shard = self
                .shards
                .get(key.shard)
                .ok_or(StoreError::ShardOutOfRange(key.shard))?;
            let start = key.byte_offset as usize;
            if start >= shard.len() {
                return Err(StoreError::PageOutOfRange(key));
            }
            let mut page = vec![0_u8; page_bytes];
            let end = (start + page_bytes).min(shard.len());
            page[..end - start].copy_from_slice(&shard[start..end]);
            Ok(page)
        }
    }

    fn encode_test_index(index: &PleIndex) -> Vec<u8> {
        let mut paths = Vec::new();
        let mut records = Vec::new();
        for shard in &index.shards {
            let path_offset = paths.len() as u32;
            paths.extend_from_slice(shard.relative_path.as_bytes());
            records.extend_from_slice(&shard.global_row_start.to_le_bytes());
            records.extend_from_slice(&shard.row_count.to_le_bytes());
            records.extend_from_slice(&shard.data_offset.to_le_bytes());
            records.extend_from_slice(&shard.data_bytes.to_le_bytes());
            records.extend_from_slice(&path_offset.to_le_bytes());
            records.extend_from_slice(&(shard.relative_path.len() as u32).to_le_bytes());
        }
        let mut payload = Vec::new();
        payload.extend_from_slice(PLE_INDEX_MAGIC);
        payload.extend_from_slice(&PLE_INDEX_VERSION.to_le_bytes());
        payload.extend_from_slice(&(PLE_INDEX_HEADER_BYTES as u32).to_le_bytes());
        payload.extend_from_slice(&(index.page_bytes as u32).to_le_bytes());
        payload.extend_from_slice(&(index.row_bytes as u32).to_le_bytes());
        payload.extend_from_slice(&index.dtype.to_le_bytes());
        payload.extend_from_slice(&(index.shards.len() as u32).to_le_bytes());
        payload.extend_from_slice(&index.total_rows.to_le_bytes());
        payload.extend_from_slice(&u32::from(index.scale_bf16_bits).to_le_bytes());
        payload.extend_from_slice(&(records.len() as u32).to_le_bytes());
        payload.extend_from_slice(&(paths.len() as u32).to_le_bytes());
        payload.extend_from_slice(&0_u64.to_le_bytes());
        payload.extend_from_slice(&0_u32.to_le_bytes());
        assert_eq!(payload.len(), PLE_INDEX_HEADER_BYTES);
        payload.extend_from_slice(&records);
        payload.extend_from_slice(&paths);
        let checksum = crc32(&payload);
        payload[60..64].copy_from_slice(&checksum.to_le_bytes());
        payload
    }

    #[test]
    fn maps_a_ple_row_without_touching_weight_data() {
        let offsets = [4096, 400_005_120, 800_006_144];
        let layout = RowLayout {
            rows_per_tensor: 2_500_012,
            row_bytes: 160,
            tensors_per_file: 13,
            tensor_data_offsets: &offsets,
        };
        assert_eq!(
            layout.address(1, 7).expect("valid row"),
            RowAddress {
                shard: 0,
                byte_offset: 400_006_240,
                byte_len: 160,
            }
        );
    }

    #[test]
    fn decodes_binary_index_and_maps_global_rows() {
        let index = PleIndex {
            page_bytes: 4096,
            row_bytes: 160,
            dtype: PLE_DTYPE_FP8_E4M3,
            total_rows: 5,
            scale_bf16_bits: 0x3f80,
            shards: vec![
                PleShard {
                    global_row_start: 0,
                    row_count: 3,
                    data_offset: 8192,
                    data_bytes: 480,
                    relative_path: "model-1.safetensors".into(),
                },
                PleShard {
                    global_row_start: 3,
                    row_count: 2,
                    data_offset: 4096,
                    data_bytes: 320,
                    relative_path: "model-2.safetensors".into(),
                },
            ],
        };
        let decoded = PleIndex::decode(&encode_test_index(&index)).expect("valid index");
        assert_eq!(decoded, index);
        assert_eq!(
            decoded.address(4).expect("valid row"),
            RowAddress {
                shard: 1,
                byte_offset: 4256,
                byte_len: 160,
            }
        );
    }

    #[test]
    fn fetches_exact_rows_across_page_boundaries_and_caches_pages() {
        let source: Vec<u8> = (0..8192).map(|value| (value % 251) as u8).collect();
        let index = PleIndex {
            page_bytes: 4096,
            row_bytes: 160,
            dtype: PLE_DTYPE_FP8_E4M3,
            total_rows: 2,
            scale_bf16_bits: 0x3f80,
            shards: vec![PleShard {
                global_row_start: 0,
                row_count: 2,
                data_offset: 4016,
                data_bytes: 320,
                relative_path: "weights.safetensors".into(),
            }],
        };
        let page_source = MemoryPageSource {
            shards: vec![source.clone()],
        };
        let mut cache = ClockPageCache::new(page_source, 4096, 2).expect("valid cache");
        let actual = cache.fetch_rows(&index, &[0, 1, 0]).expect("rows load");
        let mut expected = Vec::new();
        expected.extend_from_slice(&source[4016..4176]);
        expected.extend_from_slice(&source[4176..4336]);
        expected.extend_from_slice(&source[4016..4176]);
        assert_eq!(actual, expected);
        assert_eq!(cache.stats.page_reads, 2);
        cache.fetch_rows(&index, &[0, 1]).expect("cache hits");
        assert_eq!(cache.stats.page_hits, 2);
        assert_eq!(cache.stats.page_reads, 2);
    }

    #[test]
    fn rejects_corrupt_binary_index() {
        let index = PleIndex {
            page_bytes: 4096,
            row_bytes: 160,
            dtype: PLE_DTYPE_FP8_E4M3,
            total_rows: 1,
            scale_bf16_bits: 0x3f80,
            shards: vec![PleShard {
                global_row_start: 0,
                row_count: 1,
                data_offset: 0,
                data_bytes: 160,
                relative_path: "weights.safetensors".into(),
            }],
        };
        let mut encoded = encode_test_index(&index);
        *encoded.last_mut().expect("nonempty") ^= 1;
        assert_eq!(PleIndex::decode(&encoded), Err(PleIndexError::Checksum));
    }

    #[test]
    fn rejects_prefill_batch_larger_than_fixed_cache() {
        let index = PleIndex {
            page_bytes: 4096,
            row_bytes: 160,
            dtype: PLE_DTYPE_FP8_E4M3,
            total_rows: 3,
            scale_bf16_bits: 0x3f80,
            shards: (0..3)
                .map(|shard| PleShard {
                    global_row_start: shard,
                    row_count: 1,
                    data_offset: shard * 4096,
                    data_bytes: 160,
                    relative_path: "weights.safetensors".into(),
                })
                .collect(),
        };
        let page_source = MemoryPageSource {
            shards: vec![vec![0; 4096 * 3]; 3],
        };
        let mut cache = ClockPageCache::new(page_source, 4096, 2).expect("valid cache");
        assert!(matches!(
            cache.fetch_rows(&index, &[0, 1, 2]),
            Err(StoreError::BatchExceedsCache {
                pages: 3,
                capacity: 2
            })
        ));
    }
}
