use memmap2::{Mmap, MmapMut, MmapOptions};
use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::Path;
use bytemuck::{Pod, Zeroable};

/// Header for the memory-mapped vector file
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct VectorFileHeader {
    magic: u32,           // Magic number to verify file format
    version: u32,         // File format version
    num_vectors: u64,     // Number of vectors in the file
    dimension: u32,       // Dimensionality of vectors
    _padding: u32,        // Padding for alignment
}

const MAGIC_NUMBER: u32 = 0x56454354; // "VECT" in hex
const VERSION: u32 = 1;

/// Memory-mapped vector store
pub struct VectorStore {
    mmap: Mmap,
    num_vectors: usize,
    dimension: usize,
}

impl VectorStore {
    /// Create a new vector store from a file
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };

        // Read and validate header
        if mmap.len() < std::mem::size_of::<VectorFileHeader>() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "File too small for header",
            ));
        }

        let header = unsafe { &*(mmap.as_ptr() as *const VectorFileHeader) };

        if header.magic != MAGIC_NUMBER {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid magic number",
            ));
        }

        if header.version != VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Unsupported file version",
            ));
        }

        Ok(VectorStore {
            mmap,
            num_vectors: header.num_vectors as usize,
            dimension: header.dimension as usize,
        })
    }

    /// Create a new vector store file from a slice of vectors
    pub fn create<P: AsRef<Path>>(path: P, vectors: &[Vec<f32>]) -> io::Result<()> {
        if vectors.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Cannot create empty vector store",
            ));
        }

        let dimension = vectors[0].len();

        // Validate all vectors have the same dimension
        for vec in vectors {
            if vec.len() != dimension {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "All vectors must have the same dimension",
                ));
            }
        }

        let header = VectorFileHeader {
            magic: MAGIC_NUMBER,
            version: VERSION,
            num_vectors: vectors.len() as u64,
            dimension: dimension as u32,
            _padding: 0,
        };

        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;

        // Write header
        let header_bytes = bytemuck::bytes_of(&header);
        file.write_all(header_bytes)?;

        // Write vectors
        for vector in vectors {
            let bytes = bytemuck::cast_slice::<f32, u8>(vector);
            file.write_all(bytes)?;
        }

        file.flush()?;
        Ok(())
    }

    /// Get the number of vectors in the store
    #[inline]
    pub fn len(&self) -> usize {
        self.num_vectors
    }

    /// Check if the store is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.num_vectors == 0
    }

    /// Get the dimension of vectors
    #[inline]
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get a vector by index (zero-copy)
    #[inline]
    pub fn get(&self, index: usize) -> Option<&[f32]> {
        if index >= self.num_vectors {
            return None;
        }

        let header_size = std::mem::size_of::<VectorFileHeader>();
        let vector_size = self.dimension * std::mem::size_of::<f32>();
        let offset = header_size + index * vector_size;

        let slice = &self.mmap[offset..offset + vector_size];
        Some(bytemuck::cast_slice::<u8, f32>(slice))
    }

    /// Get a vector by index without bounds checking (unsafe but fast)
    #[inline]
    pub unsafe fn get_unchecked(&self, index: usize) -> &[f32] {
        let header_size = std::mem::size_of::<VectorFileHeader>();
        let vector_size = self.dimension * std::mem::size_of::<f32>();
        let offset = header_size + index * vector_size;

        let slice = &self.mmap[offset..offset + vector_size];
        bytemuck::cast_slice::<u8, f32>(slice)
    }

    /// Iterator over all vectors
    pub fn iter(&self) -> VectorStoreIter {
        VectorStoreIter {
            store: self,
            index: 0,
        }
    }
}

/// Iterator over vectors in the store
pub struct VectorStoreIter<'a> {
    store: &'a VectorStore,
    index: usize,
}

impl<'a> Iterator for VectorStoreIter<'a> {
    type Item = &'a [f32];

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.store.len() {
            let vec = self.store.get(self.index);
            self.index += 1;
            vec
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.store.len() - self.index;
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for VectorStoreIter<'a> {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_create_and_open() {
        let path = "/tmp/test_vectors.vec";

        // Create test data
        let vectors = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];

        // Create store
        VectorStore::create(path, &vectors).unwrap();

        // Open and verify
        let store = VectorStore::open(path).unwrap();
        assert_eq!(store.len(), 3);
        assert_eq!(store.dimension(), 3);

        let vec0 = store.get(0).unwrap();
        assert_eq!(vec0, &[1.0, 2.0, 3.0]);

        let vec2 = store.get(2).unwrap();
        assert_eq!(vec2, &[7.0, 8.0, 9.0]);

        // Cleanup
        fs::remove_file(path).ok();
    }
}
