mod distance;
mod hnsw;
mod vector_store;

use distance::DistanceMetric;
use hnsw::HNSWIndex;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::sync::Arc;
use vector_store::VectorStore;

/// Python wrapper for the vector store
#[pyclass]
struct PyVectorStore {
    store: Arc<VectorStore>,
}

#[pymethods]
impl PyVectorStore {
    /// Create a new vector store from a file
    #[staticmethod]
    fn open(path: String) -> PyResult<Self> {
        let store = VectorStore::open(path)
            .map_err(|e| PyValueError::new_err(format!("Failed to open vector store: {}", e)))?;

        Ok(PyVectorStore {
            store: Arc::new(store),
        })
    }

    /// Create a new vector store file from numpy array
    #[staticmethod]
    fn create(path: String, vectors: Vec<Vec<f32>>) -> PyResult<()> {
        VectorStore::create(path, &vectors)
            .map_err(|e| PyValueError::new_err(format!("Failed to create vector store: {}", e)))
    }

    /// Get the number of vectors
    fn __len__(&self) -> usize {
        self.store.len()
    }

    /// Get a vector by index
    fn get(&self, index: usize) -> PyResult<Vec<f32>> {
        self.store
            .get(index)
            .map(|v| v.to_vec())
            .ok_or_else(|| PyValueError::new_err("Index out of bounds"))
    }

    /// Get the dimension of vectors
    fn dimension(&self) -> usize {
        self.store.dimension()
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!(
            "VectorStore(num_vectors={}, dimension={})",
            self.store.len(),
            self.store.dimension()
        )
    }
}

/// Python wrapper for HNSW index
#[pyclass]
struct PyHNSWIndex {
    index: HNSWIndex,
}

#[pymethods]
impl PyHNSWIndex {
    /// Create a new HNSW index
    #[new]
    #[pyo3(signature = (vector_store, m=16, ef_construction=200, metric="l2"))]
    fn new(
        vector_store: &PyVectorStore,
        m: usize,
        ef_construction: usize,
        metric: &str,
    ) -> PyResult<Self> {
        let distance_metric = match metric.to_lowercase().as_str() {
            "l2" | "euclidean" => DistanceMetric::L2,
            "cosine" => DistanceMetric::Cosine,
            "dot" | "dotproduct" => DistanceMetric::DotProduct,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unknown metric: {}. Use 'l2', 'cosine', or 'dot'",
                    metric
                )))
            }
        };

        Ok(PyHNSWIndex {
            index: HNSWIndex::new(
                Arc::clone(&vector_store.store),
                m,
                ef_construction,
                distance_metric,
            ),
        })
    }

    /// Build the index
    fn build(&mut self) {
        self.index.build();
    }

    /// Search for k nearest neighbors
    #[pyo3(signature = (query, k=10, ef=None))]
    fn search(&self, query: Vec<f32>, k: usize, ef: Option<usize>) -> PyResult<Vec<(usize, f32)>> {
        let ef_search = ef.unwrap_or(k.max(50));
        Ok(self.index.search(&query, k, ef_search))
    }

    /// Get index statistics
    fn stats(&self) -> PyResult<PyObject> {
        let stats = self.index.stats();

        Python::with_gil(|py| {
            let dict = pyo3::types::PyDict::new_bound(py);
            dict.set_item("num_vectors", stats.num_vectors)?;
            dict.set_item("dimension", stats.dimension)?;
            dict.set_item("num_layers", stats.num_layers)?;
            dict.set_item("total_connections", stats.total_connections)?;
            dict.set_item("max_connections", stats.max_connections)?;
            dict.set_item("avg_connections", stats.avg_connections)?;
            Ok(dict.into())
        })
    }

    /// String representation
    fn __repr__(&self) -> String {
        let stats = self.index.stats();
        format!(
            "HNSWIndex(num_vectors={}, dimension={}, layers={})",
            stats.num_vectors, stats.dimension, stats.num_layers
        )
    }
}

/// Brute-force search for exact nearest neighbors (useful for small datasets or validation)
#[pyfunction]
#[pyo3(signature = (vector_store, query, k=10, metric="l2"))]
fn brute_force_search(
    vector_store: &PyVectorStore,
    query: Vec<f32>,
    k: usize,
    metric: &str,
) -> PyResult<Vec<(usize, f32)>> {
    let distance_fn: Box<dyn Fn(&[f32], &[f32]) -> f32> = match metric.to_lowercase().as_str() {
        "l2" | "euclidean" => Box::new(distance::l2_distance),
        "cosine" => Box::new(distance::cosine_distance),
        "dot" | "dotproduct" => Box::new(|a: &[f32], b: &[f32]| {
            -a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>()
        }),
        _ => {
            return Err(PyValueError::new_err(format!(
                "Unknown metric: {}. Use 'l2', 'cosine', or 'dot'",
                metric
            )))
        }
    };

    let mut results: Vec<(usize, f32)> = (0..vector_store.store.len())
        .map(|i| {
            let vec = vector_store.store.get(i).unwrap();
            let dist = distance_fn(&query, vec);
            (i, dist)
        })
        .collect();

    results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(k);

    Ok(results)
}

/// Calculate distance between two vectors
#[pyfunction]
#[pyo3(signature = (a, b, metric="l2"))]
fn calculate_distance(a: Vec<f32>, b: Vec<f32>, metric: &str) -> PyResult<f32> {
    if a.len() != b.len() {
        return Err(PyValueError::new_err("Vectors must have the same dimension"));
    }

    let dist = match metric.to_lowercase().as_str() {
        "l2" | "euclidean" => distance::l2_distance(&a, &b),
        "cosine" => distance::cosine_distance(&a, &b),
        "dot" | "dotproduct" => -a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>(),
        _ => {
            return Err(PyValueError::new_err(format!(
                "Unknown metric: {}. Use 'l2', 'cosine', or 'dot'",
                metric
            )))
        }
    };

    Ok(dist)
}

/// Python module
#[pymodule]
fn _zero_copy(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyVectorStore>()?;
    m.add_class::<PyHNSWIndex>()?;
    m.add_function(wrap_pyfunction!(brute_force_search, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_distance, m)?)?;
    Ok(())
}
