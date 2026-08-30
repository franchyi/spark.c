//! Persistent fixed-buffer NVMe reader.
//!
//! The queueing structure is adapted from SGLang's Apache-2.0
//! `sglang-storage` reader at e14d1c3cb62855e774475a55dac80baff45afbd4.
//! Flash removes PyO3 and the copy into `Vec<Vec<u8>>`: callers lend one
//! fixed-address slab, register it once, and reads complete directly into the
//! offsets consumed by CUDA.

use io_uring::{IoUring, opcode, types};
use std::fs::File;
use std::io;
use std::os::fd::AsRawFd;
use std::os::unix::fs::FileExt;
use std::thread;

pub const SGLANG_STORAGE_REVISION: &str = "e14d1c3cb62855e774475a55dac80baff45afbd4";

#[derive(Clone, Copy, Debug)]
pub struct FixedRead<'a> {
    pub file: &'a File,
    pub file_offset: u64,
    pub buffer_offset: usize,
    pub bytes: usize,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct FixedReadStats {
    pub operations: u64,
    pub bytes: u64,
    pub submission_batches: u64,
}

/// One persistent ring borrowing one stable CPU/GPU-visible slab. The slab is
/// registered as a single fixed buffer; individual operations may target
/// disjoint subranges by address while retaining buffer index zero.
pub struct FixedBufferReader<'a> {
    ring: IoUring,
    buffer: &'a mut [u8],
    queue_depth: usize,
    max_batch: usize,
    completion_results: Vec<i32>,
    expected_bytes: Vec<u32>,
    ranges: Vec<(usize, usize)>,
}

impl<'a> FixedBufferReader<'a> {
    pub fn new(buffer: &'a mut [u8], queue_depth: usize, max_batch: usize) -> io::Result<Self> {
        if buffer.is_empty() || queue_depth == 0 || max_batch == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "fixed buffer, queue depth, and max batch must be non-zero",
            ));
        }
        let entries = u32::try_from(queue_depth)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "queue depth exceeds u32"))?;
        let ring = IoUring::new(entries)?;
        let iovec = libc::iovec {
            iov_base: buffer.as_mut_ptr().cast(),
            iov_len: buffer.len(),
        };
        // SAFETY: `buffer` is exclusively borrowed for the lifetime of this
        // reader and the ring is dropped before that borrow can end.
        unsafe { ring.submitter().register_buffers(&[iovec])? };
        Ok(Self {
            ring,
            buffer,
            queue_depth,
            max_batch,
            completion_results: vec![i32::MIN; max_batch],
            expected_bytes: vec![0; max_batch],
            ranges: Vec::with_capacity(max_batch),
        })
    }

    pub fn buffer(&self) -> &[u8] {
        self.buffer
    }

    pub fn buffer_mut(&mut self) -> &mut [u8] {
        self.buffer
    }

    pub fn read(&mut self, reads: &[FixedRead<'_>]) -> io::Result<FixedReadStats> {
        self.validate(reads)?;
        if reads.is_empty() {
            return Ok(FixedReadStats::default());
        }
        self.completion_results[..reads.len()].fill(i32::MIN);
        self.expected_bytes[..reads.len()].fill(0);
        for (index, read) in reads.iter().enumerate() {
            self.buffer[read.buffer_offset..read.buffer_offset + read.bytes].fill(0);
            self.expected_bytes[index] = read.bytes as u32;
        }

        let mut stats = FixedReadStats::default();
        for base in (0..reads.len()).step_by(self.queue_depth) {
            let batch = (reads.len() - base).min(self.queue_depth);
            {
                let mut submission = self.ring.submission();
                for local in 0..batch {
                    let index = base + local;
                    let read = reads[index];
                    // SAFETY: validation proves the subrange belongs to the
                    // registered slab and no two reads overlap. The mutable
                    // slab borrow prevents any concurrent Rust access.
                    let destination = unsafe { self.buffer.as_mut_ptr().add(read.buffer_offset) };
                    let entry = opcode::ReadFixed::new(
                        types::Fd(read.file.as_raw_fd()),
                        destination,
                        read.bytes as u32,
                        0,
                    )
                    .offset(read.file_offset)
                    .build()
                    .user_data((index + 1) as u64);
                    // SAFETY: all pointed-to memory and file descriptors stay
                    // alive until every completion in this batch is drained.
                    unsafe { submission.push(&entry) }.map_err(|_| {
                        io::Error::new(io::ErrorKind::WouldBlock, "io_uring queue is full")
                    })?;
                }
            }

            self.ring.submit_and_wait(batch)?;
            let mut completion = self.ring.completion();
            for _ in 0..batch {
                let entry = completion.next().ok_or_else(|| {
                    io::Error::new(io::ErrorKind::UnexpectedEof, "missing io_uring completion")
                })?;
                let user_data = entry.user_data();
                if user_data == 0 || user_data > reads.len() as u64 {
                    return Err(io::Error::other("invalid io_uring completion user_data"));
                }
                self.completion_results[user_data as usize - 1] = entry.result();
            }
            stats.submission_batches = stats.submission_batches.saturating_add(1);
        }

        for (index, result) in self.completion_results[..reads.len()]
            .iter()
            .copied()
            .enumerate()
        {
            if result < 0 {
                return Err(io::Error::from_raw_os_error(-result));
            }
            let expected = self.expected_bytes[index];
            if result as u32 != expected {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    format!(
                        "short fixed-buffer read at offset {}: expected {expected}, got {result}",
                        reads[index].file_offset
                    ),
                ));
            }
            stats.operations = stats.operations.saturating_add(1);
            stats.bytes = stats.bytes.saturating_add(u64::from(expected));
        }
        Ok(stats)
    }

    /// Issue large warm-cache batches with parallel positional reads while
    /// retaining the same fixed destination slab. GB10 measurements show that
    /// this is substantially faster than a very deep buffered `io_uring`
    /// submission for prefill, whereas `read()` wins for decode-sized batches.
    pub fn read_parallel(
        &mut self,
        reads: &[FixedRead<'_>],
        workers: usize,
    ) -> io::Result<FixedReadStats> {
        if workers == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "parallel fixed-buffer workers must be non-zero",
            ));
        }
        self.validate(reads)?;
        if reads.is_empty() {
            return Ok(FixedReadStats::default());
        }
        for read in reads {
            self.buffer[read.buffer_offset..read.buffer_offset + read.bytes].fill(0);
        }

        let workers = workers.min(reads.len());
        let chunk_size = reads.len().div_ceil(workers);
        let base_address = self.buffer.as_mut_ptr() as usize;
        let results = thread::scope(|scope| {
            let mut handles = Vec::with_capacity(workers);
            for chunk in reads.chunks(chunk_size) {
                handles.push(scope.spawn(move || -> io::Result<(u64, u64)> {
                    let mut operations = 0_u64;
                    let mut bytes = 0_u64;
                    for read in chunk {
                        let mut completed = 0_usize;
                        while completed < read.bytes {
                            // SAFETY: `validate` proved every destination is
                            // within the exclusively borrowed slab and that no
                            // destinations overlap. Scoped workers finish
                            // before this method returns the mutable borrow.
                            let destination = unsafe {
                                std::slice::from_raw_parts_mut(
                                    (base_address + read.buffer_offset + completed) as *mut u8,
                                    read.bytes - completed,
                                )
                            };
                            let count = read.file.read_at(
                                destination,
                                read.file_offset + completed as u64,
                            )?;
                            if count == 0 {
                                return Err(io::Error::new(
                                    io::ErrorKind::UnexpectedEof,
                                    format!(
                                        "short positional read at offset {}: expected {}, got {completed}",
                                        read.file_offset, read.bytes
                                    ),
                                ));
                            }
                            completed += count;
                        }
                        operations = operations.saturating_add(1);
                        bytes = bytes.saturating_add(read.bytes as u64);
                    }
                    Ok((operations, bytes))
                }));
            }
            handles
                .into_iter()
                .map(|handle| {
                    handle
                        .join()
                        .map_err(|_| io::Error::other("parallel PLE reader panicked"))?
                })
                .collect::<io::Result<Vec<_>>>()
        })?;
        Ok(results.into_iter().fold(
            FixedReadStats::default(),
            |mut stats, (operations, bytes)| {
                stats.operations = stats.operations.saturating_add(operations);
                stats.bytes = stats.bytes.saturating_add(bytes);
                stats
            },
        ))
    }

    fn validate(&mut self, reads: &[FixedRead<'_>]) -> io::Result<()> {
        if reads.len() > self.max_batch {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "fixed-buffer read has {} operations but max batch is {}",
                    reads.len(),
                    self.max_batch
                ),
            ));
        }
        self.ranges.clear();
        for read in reads {
            if read.bytes == 0 || read.bytes > u32::MAX as usize {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "fixed-buffer read length must fit a non-zero u32",
                ));
            }
            let end = read
                .buffer_offset
                .checked_add(read.bytes)
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "buffer overflow"))?;
            if end > self.buffer.len() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "fixed-buffer read exceeds the registered slab",
                ));
            }
            self.ranges.push((read.buffer_offset, end));
        }
        self.ranges.sort_unstable();
        if self.ranges.windows(2).any(|pair| pair[0].1 > pair[1].0) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "fixed-buffer read destinations overlap",
            ));
        }
        Ok(())
    }
}

impl Drop for FixedBufferReader<'_> {
    fn drop(&mut self) {
        let _ = self.ring.submitter().unregister_buffers();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::{self, OpenOptions};
    use std::io::Write;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_file() -> (std::path::PathBuf, File) {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "flash-uring-{}-{nonce}.bin",
            std::process::id()
        ));
        let mut file = OpenOptions::new()
            .create_new(true)
            .read(true)
            .write(true)
            .open(&path)
            .expect("temp file");
        file.write_all(&vec![0x31; 4096]).expect("page one");
        file.write_all(&vec![0x72; 4096]).expect("page two");
        file.sync_all().expect("sync");
        (path, file)
    }

    #[test]
    fn reads_directly_into_disjoint_offsets_of_one_registered_slab() {
        let (path, file) = temp_file();
        let mut slab = vec![0_u8; 8192];
        let mut reader = match FixedBufferReader::new(&mut slab, 1, 2) {
            Ok(reader) => reader,
            Err(error) if matches!(error.raw_os_error(), Some(code) if code == libc::ENOSYS || code == libc::EPERM) =>
            {
                fs::remove_file(path).expect("remove");
                return;
            }
            Err(error) => panic!("ring: {error}"),
        };
        let stats = reader
            .read(&[
                FixedRead {
                    file: &file,
                    file_offset: 4096,
                    buffer_offset: 0,
                    bytes: 4096,
                },
                FixedRead {
                    file: &file,
                    file_offset: 0,
                    buffer_offset: 4096,
                    bytes: 4096,
                },
            ])
            .expect("fixed reads");
        assert_eq!(stats.operations, 2);
        assert_eq!(stats.bytes, 8192);
        assert_eq!(stats.submission_batches, 2);
        assert_eq!(&reader.buffer()[..4096], &[0x72; 4096]);
        assert_eq!(&reader.buffer()[4096..], &[0x31; 4096]);
        drop(reader);
        drop(file);
        fs::remove_file(path).expect("remove");
    }

    #[test]
    fn rejects_overlapping_destinations_before_submission() {
        let (path, file) = temp_file();
        let mut slab = vec![0_u8; 8192];
        let mut reader = match FixedBufferReader::new(&mut slab, 2, 2) {
            Ok(reader) => reader,
            Err(error) if matches!(error.raw_os_error(), Some(code) if code == libc::ENOSYS || code == libc::EPERM) =>
            {
                fs::remove_file(path).expect("remove");
                return;
            }
            Err(error) => panic!("ring: {error}"),
        };
        let error = reader
            .read(&[
                FixedRead {
                    file: &file,
                    file_offset: 0,
                    buffer_offset: 0,
                    bytes: 4096,
                },
                FixedRead {
                    file: &file,
                    file_offset: 4096,
                    buffer_offset: 2048,
                    bytes: 4096,
                },
            ])
            .expect_err("overlap");
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        drop(reader);
        drop(file);
        fs::remove_file(path).expect("remove");
    }

    #[test]
    fn parallel_pread_keeps_the_same_fixed_slab_and_byte_contract() {
        let (path, file) = temp_file();
        let mut slab = vec![0_u8; 8192];
        let mut reader = match FixedBufferReader::new(&mut slab, 2, 2) {
            Ok(reader) => reader,
            Err(error) if matches!(error.raw_os_error(), Some(code) if code == libc::ENOSYS || code == libc::EPERM) =>
            {
                fs::remove_file(path).expect("remove");
                return;
            }
            Err(error) => panic!("ring: {error}"),
        };
        let stats = reader
            .read_parallel(
                &[
                    FixedRead {
                        file: &file,
                        file_offset: 4096,
                        buffer_offset: 0,
                        bytes: 4096,
                    },
                    FixedRead {
                        file: &file,
                        file_offset: 0,
                        buffer_offset: 4096,
                        bytes: 4096,
                    },
                ],
                2,
            )
            .expect("parallel reads");
        assert_eq!(stats.operations, 2);
        assert_eq!(stats.bytes, 8192);
        assert_eq!(&reader.buffer()[..4096], &[0x72; 4096]);
        assert_eq!(&reader.buffer()[4096..], &[0x31; 4096]);
        drop(reader);
        drop(file);
        fs::remove_file(path).expect("remove");
    }
}
